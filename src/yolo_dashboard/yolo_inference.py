from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    label: str
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }

    def to_row(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
            "width": round(self.width, 1),
            "height": round(self.height, 1),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Detection":
        return cls(
            label=str(payload["label"]),
            class_id=int(payload["class_id"]),
            confidence=float(payload["confidence"]),
            x1=float(payload["x1"]),
            y1=float(payload["y1"]),
            x2=float(payload["x2"]),
            y2=float(payload["y2"]),
        )


@dataclass(frozen=True)
class InferenceOutput:
    annotated_frame: np.ndarray
    detections: list[Detection]
    image_width: int
    image_height: int


def parse_selected_labels(raw_value: str) -> list[str]:
    seen: set[str] = set()
    labels: list[str] = []
    for item in raw_value.split(","):
        label = item.strip()
        key = label.casefold()
        if not label or key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return labels


def _class_color(class_id: int) -> tuple[int, int, int]:
    palette = [
        (243, 156, 18),
        (46, 204, 113),
        (52, 152, 219),
        (231, 76, 60),
        (155, 89, 182),
        (26, 188, 156),
        (241, 196, 15),
        (230, 126, 34),
    ]
    return palette[class_id % len(palette)]


def annotate_frame(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = frame.copy()
    for detection in detections:
        color = _class_color(detection.class_id)
        start = (int(detection.x1), int(detection.y1))
        end = (int(detection.x2), int(detection.y2))
        cv2.rectangle(annotated, start, end, color, 2)

        label_text = f"{detection.label} {detection.confidence:.2f}"
        text_size, baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1,
        )
        text_width, text_height = text_size
        text_x = max(0, int(detection.x1))
        text_y = max(text_height + 8, int(detection.y1) - 8)
        cv2.rectangle(
            annotated,
            (text_x, text_y - text_height - 8),
            (text_x + text_width + 8, text_y + baseline - 4),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label_text,
            (text_x + 4, text_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


class YOLOService:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model: Any | None = None
        self._model_lock = threading.Lock()

    @property
    def model_version(self) -> str:
        return Path(self.model_path).name

    def _load_model(self) -> Any:
        with self._model_lock:
            if self._model is None:
                from ultralytics import YOLO

                self._model = YOLO(self.model_path)
            return self._model

    @property
    def class_names(self) -> list[str]:
        model = self._load_model()
        names = model.names
        if isinstance(names, dict):
            return [str(names[index]) for index in sorted(names)]
        return [str(name) for name in names]

    def _resolve_class_ids(self, selected_labels: list[str]) -> list[int] | None:
        if not selected_labels:
            return None

        class_lookup = {
            label.casefold(): index
            for index, label in enumerate(self.class_names)
        }
        unknown_labels = [
            label for label in selected_labels if label.casefold() not in class_lookup
        ]
        if unknown_labels:
            available = ", ".join(self.class_names[:15])
            raise ValueError(
                "Label filter tidak ada di model: "
                f"{', '.join(unknown_labels)}. Label yang tersedia: {available}"
            )

        return [class_lookup[label.casefold()] for label in selected_labels]

    def detect(
        self,
        image: np.ndarray,
        conf: float,
        iou: float,
        image_size: int,
        selected_labels: list[str] | None = None,
    ) -> InferenceOutput:
        model = self._load_model()
        class_ids = self._resolve_class_ids(selected_labels or [])
        results = model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=image_size,
            classes=class_ids,
            verbose=False,
        )
        result = results[0]
        detections = self._result_to_detections(result)
        annotated = annotate_frame(image, detections)
        height, width = image.shape[:2]
        return InferenceOutput(
            annotated_frame=annotated,
            detections=detections,
            image_width=width,
            image_height=height,
        )

    def _result_to_detections(self, result: Any) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        names = result.names if hasattr(result, "names") else {}

        detections: list[Detection] = []
        for coordinates, confidence, class_id in zip(xyxy, confidences, class_ids):
            if isinstance(names, dict):
                label = str(names.get(class_id, f"class_{class_id}"))
            else:
                label = str(names[class_id])

            detections.append(
                Detection(
                    label=label,
                    class_id=int(class_id),
                    confidence=float(confidence),
                    x1=float(coordinates[0]),
                    y1=float(coordinates[1]),
                    x2=float(coordinates[2]),
                    y2=float(coordinates[3]),
                )
            )

        return detections
