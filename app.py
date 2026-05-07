from __future__ import annotations

import shlex
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from yolo_dashboard.config import AppConfig
from yolo_dashboard.label_studio import (
    LabelStudioClient,
    LabelStudioConnectionStatus,
    LabelStudioError,
    LabelStudioProject,
    build_task_payload,
)
from yolo_dashboard.storage import (
    get_unsynced_captures,
    list_capture_artifacts,
    mark_captures_synced,
    save_label_studio_export_archive,
    save_capture,
    save_tasks_manifest,
)
from yolo_dashboard.training import (
    PreparedDataset,
    TrainingJobState,
    TrainedModelArtifact,
    discover_trained_models,
    download_base_model,
    get_resource_usage,
    get_training_job,
    inspect_existing_yolo_dataset,
    is_base_model_downloaded,
    list_base_model_names,
    list_training_device_options,
    list_prepared_datasets,
    prepare_label_studio_yolo_dataset,
    start_training_job,
)
from yolo_dashboard.webrtc_processor import LiveYOLOProcessor
from yolo_dashboard.yolo_inference import Detection, YOLOService, parse_selected_labels


MODEL_SOURCE_KEY = "model_source"
BASE_MODEL_KEY = "base_model_name"
TRAINED_MODEL_KEY = "trained_model_path"
CUSTOM_MODEL_KEY = "custom_model_path"
DATASET_KEY = "selected_dataset_yaml"
LOCAL_DATASET_INFO_KEY = "local_dataset_info"
LAST_EXPORTED_DATASET_KEY = "latest_exported_dataset"
LAST_TRAINING_RESULT_KEY = "latest_training_result"
ACTIVE_TRAINING_JOB_KEY = "active_training_job"
TRAINING_URL_KEY = "training_label_studio_url"
TRAINING_API_KEY_KEY = "training_label_studio_api_key"
TRAINING_PROJECT_KEY = "training_label_studio_project"

MODEL_SOURCE_LABELS = {
    "base": "Base YOLO",
    "trained": "Model hasil training",
    "custom": "Path custom",
}


st.set_page_config(
    page_title="YOLO Camera Dashboard",
    page_icon=":camera:",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_yolo_service(model_path: str) -> YOLOService:
    return YOLOService(model_path=model_path)


def detections_to_dataframe(detections: list[Detection]) -> pd.DataFrame:
    rows = [detection.to_row() for detection in detections]
    return pd.DataFrame(rows)


def render_capture_gallery(captures) -> None:
    if not captures:
        st.info("Belum ada capture yang tersimpan.")
        return

    st.caption("Capture terbaru")
    columns = st.columns(3)
    for index, capture in enumerate(captures[:6]):
        column = columns[index % 3]
        with column:
            st.image(str(capture.image_path), use_container_width=True)
            st.caption(capture.image_path.name)
            st.caption(
                f"{capture.source} | {capture.image_width}x{capture.image_height} | "
                f"{len(capture.detections)} deteksi"
            )


def render_detection_table(detections: list[Detection]) -> None:
    if not detections:
        st.caption("Belum ada objek terdeteksi pada frame terakhir.")
        return

    st.dataframe(
        detections_to_dataframe(detections),
        use_container_width=True,
        hide_index=True,
    )


def decode_uploaded_image(uploaded_file) -> np.ndarray | None:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


def _build_shell_path(root_path: Path, shell_name: str) -> str:
    try:
        relative_path = root_path.resolve().relative_to(ROOT_DIR.resolve())
    except ValueError:
        relative_path = None

    if shell_name == "powershell":
        if relative_path is None:
            return str(root_path)
        if not relative_path.parts:
            return "$PWD"
        return "$PWD\\" + "\\".join(relative_path.parts)

    if relative_path is None:
        return shlex.quote(root_path.resolve().as_posix())
    if not relative_path.parts:
        return "$(pwd)"
    return "$(pwd)/" + "/".join(relative_path.parts)


def build_label_studio_start_commands(config: AppConfig) -> dict[str, str]:
    powershell_root = _build_shell_path(
        root_path=config.label_studio_local_root,
        shell_name="powershell",
    )
    bash_root = _build_shell_path(
        root_path=config.label_studio_local_root,
        shell_name="bash",
    )
    return {
        "powershell": (
            '$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"\n'
            f'$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="{powershell_root}"\n'
            "label-studio start"
        ),
        "bash": (
            "export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true\n"
            f"export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={bash_root}\n"
            "label-studio start"
        ),
    }


def initialize_model_selection_state(
    config: AppConfig,
    trained_models: list[TrainedModelArtifact],
) -> None:
    base_models = list_base_model_names()
    trained_lookup = {
        str(artifact.path.resolve()): artifact for artifact in trained_models
    }
    default_model = config.default_model_path.strip()
    resolved_default_model = str(Path(default_model).resolve())

    if BASE_MODEL_KEY not in st.session_state:
        st.session_state[BASE_MODEL_KEY] = (
            default_model if default_model in base_models else base_models[0]
        )
    if CUSTOM_MODEL_KEY not in st.session_state:
        st.session_state[CUSTOM_MODEL_KEY] = default_model
    if TRAINED_MODEL_KEY not in st.session_state:
        st.session_state[TRAINED_MODEL_KEY] = next(iter(trained_lookup), "")

    if trained_lookup:
        if st.session_state[TRAINED_MODEL_KEY] not in trained_lookup:
            st.session_state[TRAINED_MODEL_KEY] = next(iter(trained_lookup))
    else:
        st.session_state[TRAINED_MODEL_KEY] = ""

    if MODEL_SOURCE_KEY not in st.session_state:
        if default_model in base_models:
            st.session_state[MODEL_SOURCE_KEY] = "base"
        elif resolved_default_model in trained_lookup:
            st.session_state[MODEL_SOURCE_KEY] = "trained"
            st.session_state[TRAINED_MODEL_KEY] = resolved_default_model
        else:
            st.session_state[MODEL_SOURCE_KEY] = "custom"


def resolve_selected_model_path(
    trained_models: list[TrainedModelArtifact],
) -> str:
    base_models = list_base_model_names()
    trained_lookup = {
        str(artifact.path.resolve()): artifact for artifact in trained_models
    }
    model_source = st.session_state.get(MODEL_SOURCE_KEY, "base")

    if model_source == "base":
        return str(st.session_state.get(BASE_MODEL_KEY, base_models[0]))
    if model_source == "trained":
        selected_trained_model = str(st.session_state.get(TRAINED_MODEL_KEY, ""))
        if selected_trained_model in trained_lookup:
            return selected_trained_model

    custom_model_path = str(st.session_state.get(CUSTOM_MODEL_KEY, "")).strip()
    if custom_model_path:
        return custom_model_path
    return str(st.session_state.get(BASE_MODEL_KEY, base_models[0]))


def _render_model_download_panel() -> None:
    models = list_base_model_names()
    for model_name in models:
        col_name, col_status, col_btn = st.columns([3, 2, 2])
        downloaded = is_base_model_downloaded(model_name)
        with col_name:
            st.text(model_name)
        with col_status:
            if downloaded:
                st.success("Tersedia", icon="✓")
            else:
                st.warning("Belum download", icon="!")
        with col_btn:
            if not downloaded:
                if st.button("Download", key=f"dl_{model_name}"):
                    with st.spinner(f"Downloading {model_name}..."):
                        try:
                            download_base_model(model_name)
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Gagal: {exc}")


def render_model_selector(
    config: AppConfig,
    trained_models: list[TrainedModelArtifact],
) -> str:
    initialize_model_selection_state(config=config, trained_models=trained_models)

    st.subheader("Model YOLO")
    st.radio(
        "Sumber model",
        options=list(MODEL_SOURCE_LABELS),
        format_func=lambda value: MODEL_SOURCE_LABELS[value],
        key=MODEL_SOURCE_KEY,
    )

    if st.session_state[MODEL_SOURCE_KEY] == "base":
        st.selectbox(
            "Base model",
            options=list_base_model_names(),
            key=BASE_MODEL_KEY,
        )
        with st.expander("Download base model"):
            _render_model_download_panel()
    elif st.session_state[MODEL_SOURCE_KEY] == "trained":
        trained_options = [str(artifact.path.resolve()) for artifact in trained_models]
        trained_lookup = {
            str(artifact.path.resolve()): artifact for artifact in trained_models
        }
        if trained_options:
            if st.session_state[TRAINED_MODEL_KEY] not in trained_lookup:
                st.session_state[TRAINED_MODEL_KEY] = trained_options[0]
            st.selectbox(
                "Pilih model hasil training",
                options=trained_options,
                format_func=lambda value: trained_lookup[value].display_name,
                key=TRAINED_MODEL_KEY,
            )
        else:
            st.info("Belum ada model hasil training. Pilih base model atau isi path custom.")
            st.text_input("Path model custom", key=CUSTOM_MODEL_KEY)
    else:
        st.text_input("Path model custom", key=CUSTOM_MODEL_KEY)

    selected_model_path = resolve_selected_model_path(trained_models=trained_models)
    st.caption(f"Model aktif: {selected_model_path}")
    st.caption(f"Folder hasil training: {config.training_runs_dir}")
    return selected_model_path


def format_dataset_option(dataset: PreparedDataset) -> str:
    split_summary = ", ".join(
        f"{split_name}:{count}"
        for split_name, count in dataset.split_counts.items()
    )
    return (
        f"{dataset.dataset_dir.name} | {len(dataset.class_names)} kelas | "
        f"{split_summary} | labeled:{dataset.labeled_images}"
    )


def serialize_dataset(dataset: PreparedDataset) -> dict[str, object]:
    return {
        "dataset_dir": str(dataset.dataset_dir.resolve()),
        "data_yaml_path": str(dataset.data_yaml_path.resolve()),
        "source_archive_path": str(dataset.source_archive_path.resolve()),
        "source_project_id": dataset.source_project_id,
        "class_names": list(dataset.class_names),
        "split_counts": dict(dataset.split_counts),
        "labeled_images": int(dataset.labeled_images),
        "created_at": dataset.created_at,
    }


def deserialize_dataset(payload: object) -> PreparedDataset | None:
    if not isinstance(payload, dict):
        return None

    try:
        return PreparedDataset(
            dataset_dir=Path(str(payload["dataset_dir"])),
            data_yaml_path=Path(str(payload["data_yaml_path"])),
            source_archive_path=Path(str(payload["source_archive_path"])),
            source_project_id=(
                int(payload["source_project_id"])
                if payload.get("source_project_id") is not None
                else None
            ),
            class_names=[str(item) for item in payload.get("class_names", [])],
            split_counts={
                str(split_name): int(count)
                for split_name, count in payload.get("split_counts", {}).items()
            },
            labeled_images=int(payload.get("labeled_images", 0)),
            created_at=str(payload.get("created_at", "")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def build_training_result_payload(job: TrainingJobState) -> dict[str, str]:
    return {
        "run_dir": str(job.run_dir.resolve()),
        "best_model_path": (
            str(job.best_model_path.resolve()) if job.best_model_path else ""
        ),
        "last_model_path": (
            str(job.last_model_path.resolve()) if job.last_model_path else ""
        ),
        "results_path": (
            str(job.results_path.resolve()) if job.results_path else ""
        ),
        "dataset_yaml_path": str(job.dataset_yaml_path.resolve()),
        "source_model_path": job.source_model_path,
    }


def render_resource_usage() -> None:
    resources = get_resource_usage()

    resource_cols = st.columns(3)
    with resource_cols[0]:
        cpu_percent = resources.get("cpu_percent")
        st.metric(
            "CPU",
            f"{cpu_percent:.1f}%" if cpu_percent is not None else "N/A",
        )
    with resource_cols[1]:
        memory_percent = resources.get("memory_percent")
        st.metric(
            "RAM",
            f"{memory_percent:.1f}%" if memory_percent is not None else "N/A",
        )
    with resource_cols[2]:
        process_memory_gb = resources.get("process_memory_gb")
        st.metric(
            "Proses Streamlit",
            f"{process_memory_gb:.2f} GB" if process_memory_gb is not None else "N/A",
        )

    gpu_resources = resources.get("gpus", [])
    if gpu_resources:
        gpu_cols = st.columns(len(gpu_resources))
        for column, gpu in zip(gpu_cols, gpu_resources):
            with column:
                st.metric(
                    f"GPU {gpu['index']}",
                    f"{gpu['used_gb']:.2f}/{gpu['total_gb']:.2f} GB",
                )
                st.caption(f"{gpu['name']} | mem {gpu['memory_percent']:.1f}%")


def render_training_job_status(job: TrainingJobState) -> None:
    st.markdown("#### Progress Training")
    progress_value = 0.0
    if job.total_epochs > 0:
        progress_value = min(job.current_epoch / job.total_epochs, 1.0)
    st.progress(
        progress_value,
        text=(
            f"Status: {job.status} | Epoch {job.current_epoch}/{job.total_epochs} | "
            f"Device: {job.device_label}"
        ),
    )
    st.caption(f"Run dir: {job.run_dir}")
    st.caption(f"Status detail: {job.message}")

    render_resource_usage()

    if job.results_path and job.results_path.exists():
        try:
            results_dataframe = pd.read_csv(job.results_path)
        except Exception:
            results_dataframe = pd.DataFrame()

        if not results_dataframe.empty:
            st.caption("Metric terakhir")
            st.dataframe(
                results_dataframe.tail(1),
                use_container_width=True,
                hide_index=True,
            )


def render_label_studio_connection_status(
    connection_status: LabelStudioConnectionStatus,
) -> None:
    st.success(
        f"Koneksi ke Label Studio berhasil: {connection_status.base_url} | "
        f"project terdeteksi: {connection_status.project_count}"
    )


def try_list_label_studio_projects(
    label_studio_url: str,
    api_key: str,
) -> tuple[list[LabelStudioProject], str | None]:
    if not label_studio_url.strip() or not api_key.strip():
        return [], None

    try:
        client = LabelStudioClient(
            base_url=label_studio_url,
            api_key=api_key,
        )
        return client.list_projects(), None
    except LabelStudioError as error:
        return [], str(error)


def render_live_camera_tab(
    config: AppConfig,
    service: YOLOService,
    confidence_threshold: float,
    iou_threshold: float,
    image_size: int,
    selected_labels: list[str],
    auto_capture: bool,
    capture_interval: float,
) -> None:
    st.subheader("Live Cam Inference")
    st.write(
        "Izinkan akses kamera browser, lalu klik `START` pada komponen video. "
        "Frame akan diproses dengan YOLO dan bisa disimpan otomatis ke folder capture."
    )
    st.caption(f"Model aktif untuk live inference: {service.model_version}")

    context = webrtc_streamer(
        key="yolo-live-camera",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30},
            },
            "audio": False,
        },
        async_processing=True,
        video_processor_factory=LiveYOLOProcessor,
    )

    snapshot = None
    if context.video_processor is not None:
        context.video_processor.update_settings(
            service=service,
            conf=confidence_threshold,
            iou=iou_threshold,
            image_size=image_size,
            selected_labels=selected_labels,
            auto_capture=auto_capture,
            capture_interval=capture_interval,
            capture_dir=config.capture_dir,
        )
        snapshot = context.video_processor.snapshot()

    controls_col, preview_col = st.columns([1, 2])

    with controls_col:
        st.metric("Auto capture", "Aktif" if auto_capture else "Nonaktif")
        st.metric("Interval", f"{capture_interval:.1f} detik")
        if snapshot and snapshot["last_capture_path"]:
            st.success(f"Capture terakhir: {Path(snapshot['last_capture_path']).name}")
        if snapshot and snapshot["error"]:
            st.error(snapshot["error"])

        manual_capture_clicked = st.button(
            "Simpan snapshot sekarang",
            type="primary",
            disabled=not snapshot or snapshot["frame"] is None,
        )
        if manual_capture_clicked and context.video_processor is not None:
            artifact = context.video_processor.capture_now()
            if artifact is None:
                st.warning("Belum ada frame yang bisa disimpan. Jalankan kamera dulu.")
            else:
                st.success(f"Snapshot tersimpan: {artifact.image_path.name}")

        if snapshot:
            st.metric("Deteksi frame terakhir", len(snapshot["detections"]))
            st.metric("Ukuran frame", f"{snapshot['width']}x{snapshot['height']}")
        else:
            st.info("Start kamera untuk mulai preview dan deteksi.")

    with preview_col:
        if snapshot and snapshot["annotated_frame"] is not None:
            st.image(
                snapshot["annotated_frame"],
                channels="BGR",
                caption="Frame terakhir yang diproses",
                use_container_width=True,
            )
        else:
            st.caption("Preview frame terakhir akan muncul di sini setelah kamera aktif.")

    st.markdown("#### Tabel Deteksi")
    if snapshot:
        render_detection_table(snapshot["detections"])
    else:
        st.caption("Belum ada data deteksi.")

    captures = list_capture_artifacts(config.capture_dir, limit=18)
    st.markdown("#### Capture Tersimpan")
    render_capture_gallery(captures)


def render_image_inference_tab(
    config: AppConfig,
    service: YOLOService,
    confidence_threshold: float,
    iou_threshold: float,
    image_size: int,
    selected_labels: list[str],
) -> None:
    st.subheader("Inference dari Image Upload")
    uploaded_file = st.file_uploader(
        "Upload image untuk dicek dengan model YOLO",
        type=["jpg", "jpeg", "png"],
        key="image-uploader",
    )

    if uploaded_file is None:
        st.info("Upload satu image untuk melihat hasil deteksi statis.")
        return

    image = decode_uploaded_image(uploaded_file)
    if image is None:
        st.error("File image tidak bisa dibaca.")
        return

    try:
        output = service.detect(
            image=image,
            conf=confidence_threshold,
            iou=iou_threshold,
            image_size=image_size,
            selected_labels=selected_labels,
        )
    except Exception as error:  # pragma: no cover - depends on local runtime env
        st.error(f"Gagal menjalankan inference: {error}")
        return

    result_col, info_col = st.columns([2, 1])
    with result_col:
        st.image(
            output.annotated_frame,
            channels="BGR",
            caption="Hasil deteksi image upload",
            use_container_width=True,
        )
    with info_col:
        st.metric("Jumlah objek", len(output.detections))
        st.metric("Ukuran image", f"{output.image_width}x{output.image_height}")
        if st.button("Simpan image ke folder capture", type="primary"):
            artifact = save_capture(
                frame=image,
                capture_dir=config.capture_dir,
                detections=output.detections,
                source=f"upload:{uploaded_file.name}",
            )
            st.success(f"Image tersimpan sebagai {artifact.image_path.name}")

    st.markdown("#### Detail Deteksi")
    render_detection_table(output.detections)


def render_label_studio_tab(
    config: AppConfig,
    service: YOLOService,
    confidence_threshold: float,
    iou_threshold: float,
    image_size: int,
    selected_labels: list[str],
) -> None:
    st.subheader("Sinkron Capture ke Label Studio")
    st.write(
        "Project ini sekarang diasumsikan memakai instalasi Label Studio via `pip`, "
        "bukan workflow Docker wajib. Jalankan server Label Studio di environment Python yang sama "
        "atau environment lain yang punya akses ke folder data ini."
    )
    start_commands = build_label_studio_start_commands(config)
    windows_tab, linux_tab = st.tabs(["Windows", "Linux/macOS"])
    with windows_tab:
        st.code("pip install label-studio label-studio-sdk", language="powershell")
        st.code(start_commands["powershell"], language="powershell")
    with linux_tab:
        st.code("pip install label-studio label-studio-sdk", language="bash")
        st.code(start_commands["bash"], language="bash")

    captures = list_capture_artifacts(config.capture_dir, limit=500)
    if not captures:
        st.info("Belum ada capture yang bisa dikirim ke Label Studio.")
    else:
        st.caption(
            f"Total capture tersedia: {len(captures)} | "
            f"Folder capture: {config.capture_dir}"
        )

    default_project_key = (
        f"{config.label_studio_url.rstrip('/')}/{config.label_studio_project_title.strip()}"
    )
    pending_captures = get_unsynced_captures(
        captures=captures,
        sync_index_path=config.sync_index_path,
        project_key=default_project_key,
    )
    st.caption(
        f"Capture baru untuk project default: {len(pending_captures)} | "
        f"Document root: {config.label_studio_local_root}"
    )

    with st.form("label-studio-sync-form"):
        label_studio_url = st.text_input("Label Studio URL", value=config.label_studio_url)
        api_key = st.text_input(
            "Label Studio API Key",
            value=config.label_studio_api_key,
            type="password",
        )
        project_title = st.text_input(
            "Nama project Label Studio",
            value=config.label_studio_project_title,
        )
        local_root = st.text_input(
            "Document root Label Studio",
            value=str(config.label_studio_local_root),
        )
        include_predictions = st.checkbox(
            "Kirim prediksi YOLO sebagai pre-label",
            value=True,
        )
        force_reimport = st.checkbox(
            "Import ulang semua capture, termasuk yang pernah dikirim",
            value=False,
        )
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            test_connection_clicked = st.form_submit_button("Test koneksi")
        with action_col2:
            submitted = st.form_submit_button("Sync ke Label Studio", type="primary")

    if not test_connection_clicked and not submitted:
        render_capture_gallery(captures)
        return

    if not label_studio_url or not api_key or not project_title:
        st.error("URL, API key, dan nama project Label Studio wajib diisi.")
        return

    try:
        client = LabelStudioClient(base_url=label_studio_url, api_key=api_key)
        connection_status = client.test_connection()
        render_label_studio_connection_status(connection_status)
    except LabelStudioError as error:
        st.error(f"Test koneksi Label Studio gagal: {error}")
        return

    if test_connection_clicked:
        render_capture_gallery(captures)
        return

    if not captures:
        st.error("Belum ada capture yang bisa disinkronkan.")
        return

    project_key = f"{label_studio_url.rstrip('/')}/{project_title.strip()}"
    captures_to_sync = captures if force_reimport else get_unsynced_captures(
        captures=captures,
        sync_index_path=config.sync_index_path,
        project_key=project_key,
    )
    if not captures_to_sync:
        st.info("Tidak ada capture baru untuk dikirim.")
        return

    try:
        labels = service.class_names
        project = client.get_or_create_project(
            title=project_title,
            labels=labels,
            description="Capture kamera dari Streamlit dashboard dengan pre-label YOLO.",
        )

        tasks = []
        synced_image_paths: list[Path] = []
        ordered_captures = list(reversed(captures_to_sync))
        for capture in ordered_captures:
            detections = capture.detections
            if include_predictions and not detections:
                image = cv2.imread(str(capture.image_path))
                if image is not None:
                    output = service.detect(
                        image=image,
                        conf=confidence_threshold,
                        iou=iou_threshold,
                        image_size=image_size,
                        selected_labels=selected_labels,
                    )
                    detections = output.detections

            tasks.append(
                build_task_payload(
                    image_path=capture.image_path,
                    document_root=Path(local_root),
                    image_width=capture.image_width,
                    image_height=capture.image_height,
                    detections=detections if include_predictions else [],
                    model_version=service.model_version,
                )
            )
            synced_image_paths.append(capture.image_path)

        manifest_path = save_tasks_manifest(tasks=tasks, export_dir=config.export_dir)
        client.import_tasks(project_id=project.id, tasks=tasks)
        mark_captures_synced(
            sync_index_path=config.sync_index_path,
            project_key=project_key,
            image_paths=synced_image_paths,
        )
    except LabelStudioError as error:
        st.error(str(error))
        return
    except Exception as error:  # pragma: no cover - depends on local runtime env
        st.error(f"Gagal sinkron ke Label Studio: {error}")
        return

    st.success(
        f"{len(tasks)} task berhasil dikirim ke project `{project.title}` "
        f"(ID: {project.id})."
    )
    st.caption(f"Manifest task tersimpan di: {manifest_path}")
    render_capture_gallery(captures_to_sync[:6])


def render_training_tab(
    config: AppConfig,
    active_model_path: str,
) -> None:
    st.subheader("Export dan Training YOLO")
    st.write(
        "Tab ini bisa memakai ZIP export YOLO yang di-upload manual, export langsung via API Label Studio, "
        "atau folder dataset lokal yang sudah valid, lalu menjalankan training Ultralytics langsung dari Streamlit."
    )

    st.caption(f"Folder dataset siap train: {config.dataset_dir}")
    st.caption(f"Folder output training: {config.training_runs_dir}")

    st.session_state.setdefault(TRAINING_URL_KEY, config.label_studio_url)
    st.session_state.setdefault(TRAINING_API_KEY_KEY, config.label_studio_api_key)

    latest_exported_dataset = st.session_state.get(LAST_EXPORTED_DATASET_KEY)
    latest_training_result = st.session_state.get(LAST_TRAINING_RESULT_KEY)
    active_training_job = get_training_job(
        str(st.session_state.get(ACTIVE_TRAINING_JOB_KEY, ""))
    )

    if active_training_job and active_training_job.status == "completed":
        st.session_state[LAST_TRAINING_RESULT_KEY] = build_training_result_payload(
            active_training_job
        )
        latest_training_result = st.session_state[LAST_TRAINING_RESULT_KEY]
        st.session_state[ACTIVE_TRAINING_JOB_KEY] = ""
        active_training_job = None
    elif active_training_job and active_training_job.status == "failed":
        st.error(
            f"Training gagal di epoch {active_training_job.current_epoch}/{active_training_job.total_epochs}: "
            f"{active_training_job.error or active_training_job.message}"
        )
        st.session_state[ACTIVE_TRAINING_JOB_KEY] = ""

    if latest_exported_dataset:
        st.success(
            "Dataset terbaru siap dipakai: "
            f"{latest_exported_dataset['dataset_dir']}"
        )
    if latest_training_result:
        st.success(
            "Training terakhir selesai. "
            f"Run dir: {latest_training_result['run_dir']}"
        )
        best_model_path = latest_training_result.get("best_model_path")
        if best_model_path:
            st.caption(f"Best model: {best_model_path}")
            if st.button("Pakai best.pt sebagai model aktif"):
                st.session_state[MODEL_SOURCE_KEY] = "trained"
                st.session_state[TRAINED_MODEL_KEY] = str(Path(best_model_path).resolve())
                st.rerun()

    training_running = active_training_job is not None and active_training_job.status in {
        "queued",
        "running",
    }
    if training_running and active_training_job is not None:
        render_training_job_status(active_training_job)

    st.markdown("#### Sumber Dataset Training")
    upload_tab, api_tab, folder_tab = st.tabs(
        ["Upload ZIP", "Export via API", "Folder Dataset"]
    )

    with upload_tab:
        st.write(
            "Upload file ZIP hasil export YOLO dari Label Studio kalau kamu sudah export manual dari UI Label Studio."
        )
        st.caption(
            "Jika ZIP tidak membawa folder `images/`, dashboard akan mencoba memakai "
            f"source image dari `{config.label_studio_local_root}`."
        )
        uploaded_export = st.file_uploader(
            "Upload ZIP export YOLO Label Studio",
            type=["zip"],
            key="label_studio_export_upload",
        )
        upload_train_split = st.slider(
            "Proporsi train split dari ZIP upload",
            min_value=0.50,
            max_value=0.95,
            value=0.80,
            step=0.05,
            key="upload_train_split",
        )

        if st.button(
            "Gunakan ZIP upload",
            type="primary",
            key="prepare_uploaded_export",
            disabled=training_running,
        ):
            if uploaded_export is None:
                st.error("Upload file ZIP hasil export Label Studio dulu.")
            else:
                try:
                    with st.spinner("Menyiapkan dataset dari ZIP upload..."):
                        archive_path = save_label_studio_export_archive(
                            export_dir=config.export_dir,
                            original_name=uploaded_export.name,
                            payload=uploaded_export.getvalue(),
                        )
                        prepared_dataset = prepare_label_studio_yolo_dataset(
                            archive_path=archive_path,
                            dataset_root=config.dataset_dir,
                            train_split=float(upload_train_split),
                            fallback_image_roots=[
                                config.capture_dir,
                                config.label_studio_local_root,
                            ],
                        )
                    st.session_state[DATASET_KEY] = str(prepared_dataset.data_yaml_path.resolve())
                    st.session_state[LAST_EXPORTED_DATASET_KEY] = {
                        "archive_path": str(archive_path.resolve()),
                        "dataset_dir": str(prepared_dataset.dataset_dir.resolve()),
                        "data_yaml_path": str(prepared_dataset.data_yaml_path.resolve()),
                    }
                    st.success(
                        f"ZIP upload berhasil dipakai. Dataset siap train di {prepared_dataset.dataset_dir.name}."
                    )
                except Exception as error:  # pragma: no cover - depends on local runtime env
                    st.error(f"Gagal memproses ZIP upload: {error}")

    with api_tab:
        st.caption(
            "Export via API akan mengutamakan format YOLO yang menyertakan image. "
            "Kalau server hanya mengembalikan label, dashboard tetap mencoba mencari "
            f"source image dari `{config.label_studio_local_root}`."
        )
        label_studio_url = st.text_input(
            "Label Studio URL",
            key=TRAINING_URL_KEY,
        )
        api_key = st.text_input(
            "Label Studio API Key",
            key=TRAINING_API_KEY_KEY,
            type="password",
        )

        projects, project_error = try_list_label_studio_projects(
            label_studio_url=label_studio_url,
            api_key=api_key,
        )

        selected_project_id: int
        if project_error:
            st.warning(project_error)
            selected_project_id = int(
                st.number_input(
                    "Project ID Label Studio",
                    min_value=1,
                    value=1,
                    step=1,
                    key="training_project_id_error",
                )
            )
        elif projects:
            project_lookup = {str(project.id): project for project in projects}
            default_project_id = next(
                (
                    str(project.id)
                    for project in projects
                    if project.title.strip() == config.label_studio_project_title.strip()
                ),
                str(projects[0].id),
            )
            if (
                TRAINING_PROJECT_KEY not in st.session_state
                or st.session_state[TRAINING_PROJECT_KEY] not in project_lookup
            ):
                st.session_state[TRAINING_PROJECT_KEY] = default_project_id
            selected_project_key = st.selectbox(
                "Project Label Studio",
                options=list(project_lookup),
                format_func=lambda value: (
                    f"{project_lookup[value].title} (ID: {project_lookup[value].id})"
                ),
                key=TRAINING_PROJECT_KEY,
            )
            selected_project_id = int(selected_project_key)
        else:
            st.info("Belum ada project terdeteksi. Isi project ID manual jika perlu.")
            selected_project_id = int(
                st.number_input(
                    "Project ID Label Studio",
                    min_value=1,
                    value=1,
                    step=1,
                    key="training_project_id_manual",
                )
            )

        export_col, split_col = st.columns(2)
        with export_col:
            train_split = st.slider(
                "Proporsi train split",
                min_value=0.50,
                max_value=0.95,
                value=0.80,
                step=0.05,
                key="api_train_split",
            )
        with split_col:
            export_timeout = st.number_input(
                "Timeout export (detik)",
                min_value=60,
                max_value=1800,
                value=300,
                step=30,
                key="api_export_timeout",
            )

        if st.button(
            "Export YOLO dan siapkan dataset",
            type="primary",
            key="prepare_api_export",
            disabled=training_running,
        ):
            if not label_studio_url.strip() or not api_key.strip():
                st.error("URL dan API key Label Studio wajib diisi sebelum export.")
            else:
                try:
                    with st.spinner("Mengunduh export YOLO dari Label Studio..."):
                        client = LabelStudioClient(
                            base_url=label_studio_url,
                            api_key=api_key,
                        )
                        export_artifact = client.export_project_to_archive(
                            project_id=selected_project_id,
                            export_dir=config.export_dir,
                            export_type="YOLO",
                            timeout_seconds=int(export_timeout),
                        )
                        prepared_dataset = prepare_label_studio_yolo_dataset(
                            archive_path=export_artifact.archive_path,
                            dataset_root=config.dataset_dir,
                            train_split=float(train_split),
                            project_id=selected_project_id,
                            fallback_image_roots=[
                                config.capture_dir,
                                config.label_studio_local_root,
                            ],
                        )
                    st.session_state[DATASET_KEY] = str(prepared_dataset.data_yaml_path.resolve())
                    st.session_state[LAST_EXPORTED_DATASET_KEY] = {
                        "archive_path": str(export_artifact.archive_path.resolve()),
                        "dataset_dir": str(prepared_dataset.dataset_dir.resolve()),
                        "data_yaml_path": str(prepared_dataset.data_yaml_path.resolve()),
                    }
                    st.success(
                        f"Export YOLO selesai. Dataset siap train di {prepared_dataset.dataset_dir.name}."
                    )
                except LabelStudioError as error:
                    st.error(str(error))
                except Exception as error:  # pragma: no cover - depends on local runtime env
                    st.error(f"Gagal export dataset Label Studio: {error}")

    with folder_tab:
        st.write(
            "Pakai dataset YOLO lokal secara langsung jika folder dataset sudah ada dan valid."
        )
        local_dataset_input = st.text_input(
            "Path folder dataset atau file data.yaml",
            value=str(config.dataset_dir),
            key="local_dataset_input",
            disabled=training_running,
            help="Contoh: data/datasets/my_dataset atau data/datasets/my_dataset/data.yaml",
        )
        if st.button(
            "Gunakan folder dataset",
            type="primary",
            key="prepare_local_dataset",
            disabled=training_running,
        ):
            try:
                local_dataset = inspect_existing_yolo_dataset(Path(local_dataset_input))
                st.session_state[LOCAL_DATASET_INFO_KEY] = serialize_dataset(local_dataset)
                st.session_state[DATASET_KEY] = str(local_dataset.data_yaml_path.resolve())
                st.success(
                    f"Dataset lokal valid dan siap dipakai: {local_dataset.data_yaml_path}"
                )
            except Exception as error:  # pragma: no cover - depends on local runtime env
                st.error(f"Folder dataset belum valid: {error}")

    datasets = list_prepared_datasets(config.dataset_dir)
    local_dataset = deserialize_dataset(st.session_state.get(LOCAL_DATASET_INFO_KEY))
    if local_dataset is not None and local_dataset.data_yaml_path.exists():
        if not any(
            dataset.data_yaml_path.resolve() == local_dataset.data_yaml_path.resolve()
            for dataset in datasets
        ):
            datasets = [local_dataset, *datasets]
    dataset_lookup = {
        str(dataset.data_yaml_path.resolve()): dataset for dataset in datasets
    }

    st.markdown("#### Dataset Siap Train")
    if not dataset_lookup:
        st.info(
            "Belum ada dataset siap train. Siapkan dataset lewat upload ZIP, export API, atau folder dataset lokal."
        )
        return

    if DATASET_KEY not in st.session_state or st.session_state[DATASET_KEY] not in dataset_lookup:
        st.session_state[DATASET_KEY] = next(iter(dataset_lookup))

    selected_dataset_key = st.selectbox(
        "Pilih dataset",
        options=list(dataset_lookup),
        format_func=lambda value: format_dataset_option(dataset_lookup[value]),
        key=DATASET_KEY,
    )
    selected_dataset = dataset_lookup[selected_dataset_key]

    dataset_col, model_col = st.columns(2)
    with dataset_col:
        st.metric("Jumlah kelas", len(selected_dataset.class_names))
        st.metric("Labeled images", selected_dataset.labeled_images)
        st.caption(f"Data YAML: {selected_dataset.data_yaml_path}")
    with model_col:
        st.metric("Train images", int(selected_dataset.split_counts.get("train", 0)))
        st.metric("Val images", int(selected_dataset.split_counts.get("val", 0)))
        st.caption(f"Starting weights: {active_model_path}")

    if selected_dataset.class_names:
        st.caption(f"Classes: {', '.join(selected_dataset.class_names)}")

    st.markdown("#### Konfigurasi Training")
    run_name = st.text_input(
        "Nama run training",
        value=f"train_{Path(active_model_path).stem}",
        disabled=training_running,
    )
    epochs_col, batch_col, img_col, patience_col = st.columns(4)
    with epochs_col:
        epochs = int(
            st.number_input(
                "Epochs",
                min_value=1,
                max_value=1000,
                value=50,
                disabled=training_running,
            )
        )
    with batch_col:
        batch_size = int(
            st.number_input(
                "Batch size",
                min_value=1,
                max_value=256,
                value=16,
                disabled=training_running,
            )
        )
    with img_col:
        image_size = int(
            st.selectbox(
                "Image size",
                options=[320, 416, 512, 640, 768, 960, 1280],
                index=3,
                disabled=training_running,
            )
        )
    with patience_col:
        patience = int(
            st.number_input(
                "Patience",
                min_value=0,
                max_value=200,
                value=20,
                disabled=training_running,
            )
        )

    device_options = list_training_device_options()
    device_option_lookup = {value: label for value, label in device_options}
    extra_param_col1, extra_param_col2, extra_param_col3, extra_param_col4 = st.columns(4)
    with extra_param_col1:
        device_selection = st.selectbox(
            "Device training",
            options=list(device_option_lookup),
            format_func=lambda value: device_option_lookup[value],
            disabled=training_running,
        )
    with extra_param_col2:
        workers = int(
            st.number_input(
                "Workers",
                min_value=0,
                max_value=32,
                value=8,
                disabled=training_running,
            )
        )
    with extra_param_col3:
        optimizer = st.selectbox(
            "Optimizer",
            options=["auto", "SGD", "Adam", "AdamW"],
            disabled=training_running,
        )
    with extra_param_col4:
        learning_rate = float(
            st.number_input(
                "Learning rate",
                min_value=0.0001,
                max_value=1.0,
                value=0.01,
                step=0.0005,
                format="%.4f",
                disabled=training_running,
            )
        )

    if selected_dataset.labeled_images == 0:
        st.warning("Dataset ini belum punya file label. Training tidak dijalankan.")
        return

    if training_running:
        st.info("Training sedang berjalan. Progress di-refresh otomatis setiap 2 detik.")
    if st.button("Mulai training YOLO", type="primary", disabled=training_running):
        try:
            started_job = start_training_job(
                dataset_yaml_path=selected_dataset.data_yaml_path,
                model_path=active_model_path,
                runs_dir=config.training_runs_dir,
                run_name=run_name,
                epochs=epochs,
                image_size=image_size,
                batch_size=batch_size,
                patience=patience,
                device_selection=device_selection,
                workers=workers,
                optimizer=optimizer,
                learning_rate=learning_rate,
            )
            st.session_state[ACTIVE_TRAINING_JOB_KEY] = started_job.job_id
            st.rerun()
        except Exception as error:  # pragma: no cover - depends on local runtime env
            st.error(f"Gagal menjalankan training YOLO: {error}")

    if training_running:
        time.sleep(2)
        st.rerun()


def main() -> None:
    config = AppConfig.from_env()
    config.ensure_directories()
    trained_models = discover_trained_models(config.training_runs_dir)

    st.title("YOLO Camera Dashboard")
    st.write(
        "Dashboard Streamlit untuk capture kamera, pre-label dengan YOLO, "
        "sinkron ke Label Studio, export dataset, dan training ulang model."
    )

    with st.sidebar:
        st.header("Konfigurasi")
        active_model_path = render_model_selector(
            config=config,
            trained_models=trained_models,
        )
        confidence_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.30, 0.05)
        iou_threshold = st.slider("IoU threshold", 0.05, 0.95, 0.45, 0.05)
        image_size = st.select_slider(
            "Inference image size",
            options=[320, 416, 512, 640, 768, 960, 1280],
            value=640,
        )
        selected_label_text = st.text_input(
            "Filter label (pisahkan dengan koma, opsional)",
            value="",
            help="Contoh: person, car",
        )
        auto_capture = st.checkbox("Auto capture kamera", value=True)
        capture_interval = st.number_input(
            "Interval auto capture (detik)",
            min_value=1.0,
            max_value=60.0,
            value=5.0,
            step=1.0,
        )
        st.caption(f"Folder capture: {config.capture_dir}")

    selected_labels = parse_selected_labels(selected_label_text)
    service = get_yolo_service(active_model_path)

    tab_live, tab_image, tab_label, tab_training = st.tabs(
        ["Live Camera", "Image Inference", "Label Studio", "Training"]
    )

    with tab_live:
        render_live_camera_tab(
            config=config,
            service=service,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            image_size=image_size,
            selected_labels=selected_labels,
            auto_capture=auto_capture,
            capture_interval=capture_interval,
        )

    with tab_image:
        render_image_inference_tab(
            config=config,
            service=service,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            image_size=image_size,
            selected_labels=selected_labels,
        )

    with tab_label:
        render_label_studio_tab(
            config=config,
            service=service,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            image_size=image_size,
            selected_labels=selected_labels,
        )

    with tab_training:
        render_training_tab(
            config=config,
            active_model_path=active_model_path,
        )


if __name__ == "__main__":
    main()
