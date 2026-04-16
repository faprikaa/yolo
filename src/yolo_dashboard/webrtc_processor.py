from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import av
from streamlit_webrtc import VideoProcessorBase

from yolo_dashboard.storage import save_capture
from yolo_dashboard.yolo_inference import YOLOService


class LiveYOLOProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._service: YOLOService | None = None
        self._conf = 0.3
        self._iou = 0.45
        self._image_size = 640
        self._selected_labels: list[str] = []
        self._auto_capture = False
        self._capture_interval = 5.0
        self._capture_dir = Path("data/captures")
        self._last_capture_at = 0.0
        self._latest_frame = None
        self._latest_annotated = None
        self._latest_detections = []
        self._last_capture_path: str | None = None
        self._latest_error = ""
        self._width = 0
        self._height = 0

    def update_settings(
        self,
        service: YOLOService,
        conf: float,
        iou: float,
        image_size: int,
        selected_labels: list[str],
        auto_capture: bool,
        capture_interval: float,
        capture_dir: Path,
    ) -> None:
        with self._lock:
            self._service = service
            self._conf = conf
            self._iou = iou
            self._image_size = image_size
            self._selected_labels = list(selected_labels)
            self._auto_capture = auto_capture
            self._capture_interval = capture_interval
            self._capture_dir = capture_dir

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "frame": None if self._latest_frame is None else self._latest_frame.copy(),
                "annotated_frame": (
                    None if self._latest_annotated is None else self._latest_annotated.copy()
                ),
                "detections": list(self._latest_detections),
                "last_capture_path": self._last_capture_path,
                "error": self._latest_error,
                "width": self._width,
                "height": self._height,
            }

    def capture_now(self):
        with self._lock:
            if self._latest_frame is None:
                return None
            frame = self._latest_frame.copy()
            detections = list(self._latest_detections)
            capture_dir = self._capture_dir

        artifact = save_capture(
            frame=frame,
            capture_dir=capture_dir,
            detections=detections,
            source="manual-camera",
        )
        with self._lock:
            self._last_capture_path = str(artifact.image_path)
            self._last_capture_at = time.monotonic()
        return artifact

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image_height, image_width = image.shape[:2]

        with self._lock:
            service = self._service
            conf = self._conf
            iou = self._iou
            image_size = self._image_size
            selected_labels = list(self._selected_labels)
            auto_capture = self._auto_capture
            capture_interval = self._capture_interval
            capture_dir = self._capture_dir
            last_capture_at = self._last_capture_at

        if service is None:
            with self._lock:
                self._latest_frame = image.copy()
                self._latest_annotated = image.copy()
                self._latest_detections = []
                self._latest_error = ""
                self._width = image_width
                self._height = image_height
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        try:
            output = service.detect(
                image=image,
                conf=conf,
                iou=iou,
                image_size=image_size,
                selected_labels=selected_labels,
            )
            annotated = output.annotated_frame
            detections = output.detections

            capture_path: str | None = None
            current_time = time.monotonic()
            should_capture = (
                auto_capture
                and capture_interval > 0
                and current_time - last_capture_at >= capture_interval
            )
            if should_capture:
                artifact = save_capture(
                    frame=image,
                    capture_dir=capture_dir,
                    detections=detections,
                    source="auto-camera",
                )
                capture_path = str(artifact.image_path)

            with self._lock:
                self._latest_frame = image.copy()
                self._latest_annotated = annotated.copy()
                self._latest_detections = list(detections)
                self._latest_error = ""
                self._width = image_width
                self._height = image_height
                if capture_path:
                    self._last_capture_path = capture_path
                    self._last_capture_at = current_time

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
        except Exception as error:  # pragma: no cover - depends on local runtime env
            with self._lock:
                self._latest_frame = image.copy()
                self._latest_annotated = image.copy()
                self._latest_detections = []
                self._latest_error = str(error)
                self._width = image_width
                self._height = image_height
            return av.VideoFrame.from_ndarray(image, format="bgr24")
