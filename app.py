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
    build_task_payload,
)
from yolo_dashboard.storage import (
    get_unsynced_captures,
    list_capture_artifacts,
    mark_captures_synced,
    save_capture,
    save_tasks_manifest,
)
from yolo_dashboard.webrtc_processor import LiveYOLOProcessor
from yolo_dashboard.yolo_inference import Detection, YOLOService, parse_selected_labels


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
    captures = list_capture_artifacts(config.capture_dir, limit=500)
    if not captures:
        st.info("Belum ada capture yang bisa dikirim ke Label Studio.")
        return

    st.write(
        "Dashboard ini mengirim gambar ke Label Studio lewat API dan menggunakan "
        "mode local files serving. Pastikan document root Label Studio menunjuk ke folder data project ini."
    )
    st.code(
        "docker run -it -p 8080:8080 "
        "-v ${PWD}/data:/label-studio/files "
        "--env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true "
        "--env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files "
        "heartexlabs/label-studio:latest label-studio",
        language="powershell",
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
        f"Total capture tersedia: {len(captures)} | "
        f"Capture baru untuk project default: {len(pending_captures)}"
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
        client.import_tasks(project_id=project["id"], tasks=tasks)
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
        f"{len(tasks)} task berhasil dikirim ke project `{project_title}` "
        f"(ID: {project['id']})."
    )
    st.caption(f"Manifest task tersimpan di: {manifest_path}")
    render_capture_gallery(captures_to_sync[:6])


def main() -> None:
    config = AppConfig.from_env()
    config.ensure_directories()

    st.title("YOLO Camera Dashboard")
    st.write(
        "Dashboard Streamlit untuk capture kamera, pre-label dengan YOLO, "
        "dan sinkron task ke Label Studio."
    )

    with st.sidebar:
        st.header("Konfigurasi")
        model_path = st.text_input("Path model YOLO", value=config.default_model_path)
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
    service = get_yolo_service(model_path)

    tab_live, tab_image, tab_label = st.tabs(
        ["Live Camera", "Image Inference", "Label Studio"]
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


if __name__ == "__main__":
    main()
