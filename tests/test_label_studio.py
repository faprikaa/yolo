from __future__ import annotations

from pathlib import Path

from yolo_dashboard.label_studio import (
    build_label_config,
    build_local_file_url,
    build_task_payload,
)
from yolo_dashboard.yolo_inference import Detection


def test_build_label_config_contains_image_and_labels() -> None:
    config = build_label_config(["person", "car"])

    assert '<Image name="image" value="$image" />' in config
    assert 'Label value="person"' in config
    assert 'Label value="car"' in config


def test_build_local_file_url_uses_relative_path() -> None:
    document_root = Path("D:/datasets")
    image_path = document_root / "captures" / "frame_001.jpg"

    url = build_local_file_url(image_path=image_path, document_root=document_root)

    assert url == "/data/local-files/?d=captures/frame_001.jpg"


def test_build_task_payload_adds_predictions() -> None:
    image_path = Path("D:/datasets/captures/frame_001.jpg")
    document_root = Path("D:/datasets")
    detections = [
        Detection(
            label="person",
            class_id=0,
            confidence=0.88,
            x1=10,
            y1=20,
            x2=110,
            y2=220,
        )
    ]

    task = build_task_payload(
        image_path=image_path,
        document_root=document_root,
        image_width=200,
        image_height=400,
        detections=detections,
        model_version="best.pt",
    )

    prediction = task["predictions"][0]
    rectangle = prediction["result"][0]["value"]

    assert task["data"]["image"] == "/data/local-files/?d=captures/frame_001.jpg"
    assert prediction["model_version"] == "best.pt"
    assert rectangle["x"] == 5.0
    assert rectangle["y"] == 5.0
    assert rectangle["width"] == 50.0
    assert rectangle["height"] == 50.0
    assert rectangle["rectanglelabels"] == ["person"]
