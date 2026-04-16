from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from yolo_dashboard.storage import (
    get_unsynced_captures,
    load_capture_artifact,
    load_sync_index,
    mark_captures_synced,
    save_capture,
)
from yolo_dashboard.yolo_inference import Detection


def test_save_capture_writes_image_and_metadata(tmp_path: Path) -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    detections = [
        Detection(
            label="helmet",
            class_id=1,
            confidence=0.92,
            x1=4,
            y1=6,
            x2=20,
            y2=30,
        )
    ]

    artifact = save_capture(
        frame=frame,
        capture_dir=tmp_path,
        detections=detections,
        source="unit-test",
    )

    assert artifact.image_path.exists()
    assert artifact.metadata_path.exists()

    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    assert metadata["source"] == "unit-test"
    assert metadata["image_width"] == 64
    assert metadata["image_height"] == 48
    assert metadata["detections"][0]["label"] == "helmet"

    loaded = load_capture_artifact(artifact.image_path)
    assert loaded.image_width == 64
    assert loaded.image_height == 48
    assert loaded.detections[0].label == "helmet"


def test_mark_captures_synced_filters_existing_items(tmp_path: Path) -> None:
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    first = save_capture(frame, tmp_path, [], "first")
    second = save_capture(frame, tmp_path, [], "second")
    sync_index_path = tmp_path / "sync_index.json"
    project_key = "http://localhost:8080/project-demo"

    mark_captures_synced(
        sync_index_path=sync_index_path,
        project_key=project_key,
        image_paths=[first.image_path],
    )

    sync_index = load_sync_index(sync_index_path)
    assert str(first.image_path.resolve()) in sync_index[project_key]

    pending = get_unsynced_captures(
        captures=[first, second],
        sync_index_path=sync_index_path,
        project_key=project_key,
    )

    assert [artifact.image_path for artifact in pending] == [second.image_path]
