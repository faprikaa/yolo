from __future__ import annotations

import sys
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
    TrainedModelArtifact,
    discover_trained_models,
    list_base_model_names,
    list_prepared_datasets,
    prepare_label_studio_yolo_dataset,
    train_yolo_model,
)
from yolo_dashboard.webrtc_processor import LiveYOLOProcessor
from yolo_dashboard.yolo_inference import Detection, YOLOService, parse_selected_labels


MODEL_SOURCE_KEY = "model_source"
BASE_MODEL_KEY = "base_model_name"
TRAINED_MODEL_KEY = "trained_model_path"
CUSTOM_MODEL_KEY = "custom_model_path"
DATASET_KEY = "selected_dataset_yaml"
LAST_EXPORTED_DATASET_KEY = "latest_exported_dataset"
LAST_TRAINING_RESULT_KEY = "latest_training_result"
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


def build_label_studio_start_command(config: AppConfig) -> str:
    return (
        '$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"\n'
        f'$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="{config.label_studio_local_root}"\n'
        "label-studio start"
    )


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
    st.subheader("Live Camera Detection")
    st.write(
        "Izinkan akses kamera browser, lalu klik `START` pada komponen video. "
        "Frame akan diproses dengan YOLO dan bisa disimpan otomatis ke folder capture."
    )

    context = webrtc_streamer(
        key="yolo-live-camera",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
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
    st.code("pip install label-studio label-studio-sdk", language="powershell")
    st.code(build_label_studio_start_command(config), language="powershell")

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
        submitted = st.form_submit_button("Sync ke Label Studio", type="primary")

    if not submitted:
        render_capture_gallery(captures)
        return

    if not captures:
        st.error("Belum ada capture yang bisa disinkronkan.")
        return
    if not label_studio_url or not api_key or not project_title:
        st.error("URL, API key, dan nama project Label Studio wajib diisi.")
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
        client = LabelStudioClient(base_url=label_studio_url, api_key=api_key)
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
        "Tab ini bisa memakai ZIP export YOLO yang di-upload manual atau export langsung via API Label Studio, "
        "menyiapkan `data.yaml`, lalu menjalankan training Ultralytics langsung dari Streamlit."
    )

    st.caption(f"Folder dataset siap train: {config.dataset_dir}")
    st.caption(f"Folder output training: {config.training_runs_dir}")

    st.session_state.setdefault(TRAINING_URL_KEY, config.label_studio_url)
    st.session_state.setdefault(TRAINING_API_KEY_KEY, config.label_studio_api_key)

    latest_exported_dataset = st.session_state.get(LAST_EXPORTED_DATASET_KEY)
    latest_training_result = st.session_state.get(LAST_TRAINING_RESULT_KEY)

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

    st.markdown("#### Sumber Export Label Studio")
    upload_tab, api_tab = st.tabs(["Upload ZIP", "Export via API"])

    with upload_tab:
        st.write(
            "Upload file ZIP hasil export YOLO dari Label Studio kalau kamu sudah export manual dari UI Label Studio."
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

        if st.button("Gunakan ZIP upload", type="primary", key="prepare_uploaded_export"):
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

        if st.button("Export YOLO dan siapkan dataset", type="primary", key="prepare_api_export"):
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

    datasets = list_prepared_datasets(config.dataset_dir)
    dataset_lookup = {
        str(dataset.data_yaml_path.resolve()): dataset for dataset in datasets
    }

    st.markdown("#### Dataset Siap Train")
    if not dataset_lookup:
        st.info("Belum ada dataset hasil export Label Studio yang siap digunakan.")
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
    )
    epochs_col, batch_col, img_col, patience_col = st.columns(4)
    with epochs_col:
        epochs = int(st.number_input("Epochs", min_value=1, max_value=1000, value=50))
    with batch_col:
        batch_size = int(st.number_input("Batch size", min_value=1, max_value=256, value=16))
    with img_col:
        image_size = int(
            st.selectbox("Image size", options=[320, 416, 512, 640, 768, 960, 1280], index=3)
        )
    with patience_col:
        patience = int(st.number_input("Patience", min_value=0, max_value=200, value=20))

    device = st.text_input(
        "Device training",
        value="auto",
        help="Contoh: auto, cpu, 0, 0,1",
    )

    if selected_dataset.labeled_images == 0:
        st.warning("Dataset ini belum punya file label. Training tidak dijalankan.")
        return

    if st.button("Mulai training YOLO", type="primary"):
        try:
            with st.spinner("Training YOLO sedang berjalan..."):
                training_result = train_yolo_model(
                    dataset_yaml_path=selected_dataset.data_yaml_path,
                    model_path=active_model_path,
                    runs_dir=config.training_runs_dir,
                    run_name=run_name,
                    epochs=epochs,
                    image_size=image_size,
                    batch_size=batch_size,
                    patience=patience,
                    device=device,
                )
            st.session_state[LAST_TRAINING_RESULT_KEY] = {
                "run_dir": str(training_result.run_dir.resolve()),
                "best_model_path": (
                    str(training_result.best_model_path.resolve())
                    if training_result.best_model_path
                    else ""
                ),
                "last_model_path": (
                    str(training_result.last_model_path.resolve())
                    if training_result.last_model_path
                    else ""
                ),
                "results_path": (
                    str(training_result.results_path.resolve())
                    if training_result.results_path
                    else ""
                ),
                "dataset_yaml_path": str(training_result.dataset_yaml_path.resolve()),
                "source_model_path": training_result.source_model_path,
            }
            st.rerun()
        except Exception as error:  # pragma: no cover - depends on local runtime env
            st.error(f"Gagal menjalankan training YOLO: {error}")


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
