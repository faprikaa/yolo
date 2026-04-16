from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from yolo_dashboard.yolo_inference import Detection


@dataclass(frozen=True)
class CaptureArtifact:
    image_path: Path
    metadata_path: Path
    captured_at: str
    source: str
    detections: list[Detection]
    image_width: int
    image_height: int


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_capture(
    frame: np.ndarray,
    capture_dir: Path,
    detections: list[Detection],
    source: str,
) -> CaptureArtifact:
    capture_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _timestamp_slug()
    image_path = capture_dir / f"capture_{timestamp}.jpg"
    metadata_path = capture_dir / f"capture_{timestamp}.json"
    image_height, image_width = frame.shape[:2]

    saved = cv2.imwrite(str(image_path), frame)
    if not saved:
        raise RuntimeError(f"Gagal menulis image capture ke {image_path}")

    payload = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "image_width": image_width,
        "image_height": image_height,
        "detections": [detection.as_dict() for detection in detections],
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    return CaptureArtifact(
        image_path=image_path,
        metadata_path=metadata_path,
        captured_at=payload["captured_at"],
        source=source,
        detections=detections,
        image_width=image_width,
        image_height=image_height,
    )


def load_capture_artifact(image_path: Path) -> CaptureArtifact:
    metadata_path = image_path.with_suffix(".json")
    metadata = _read_json(metadata_path, default={})
    image = cv2.imread(str(image_path))
    if image is not None:
        image_height, image_width = image.shape[:2]
    else:
        image_width = int(metadata.get("image_width", 0))
        image_height = int(metadata.get("image_height", 0))

    detection_payloads = metadata.get("detections", [])
    detections = [Detection.from_dict(payload) for payload in detection_payloads]

    return CaptureArtifact(
        image_path=image_path,
        metadata_path=metadata_path,
        captured_at=str(metadata.get("captured_at", "")),
        source=str(metadata.get("source", "unknown")),
        detections=detections,
        image_width=image_width,
        image_height=image_height,
    )


def list_capture_artifacts(capture_dir: Path, limit: int = 24) -> list[CaptureArtifact]:
    capture_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        [
            path
            for path in capture_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if limit > 0:
        image_paths = image_paths[:limit]
    return [load_capture_artifact(path) for path in image_paths]


def save_tasks_manifest(tasks: list[dict[str, Any]], export_dir: Path) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = export_dir / f"label_studio_tasks_{_timestamp_slug()}.json"
    manifest_path.write_text(
        json.dumps(tasks, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def load_sync_index(sync_index_path: Path) -> dict[str, list[str]]:
    return _read_json(sync_index_path, default={})


def get_unsynced_captures(
    captures: list[CaptureArtifact],
    sync_index_path: Path,
    project_key: str,
) -> list[CaptureArtifact]:
    sync_index = load_sync_index(sync_index_path)
    known_paths = set(sync_index.get(project_key, []))
    return [
        capture
        for capture in captures
        if str(capture.image_path.resolve()) not in known_paths
    ]


def mark_captures_synced(
    sync_index_path: Path,
    project_key: str,
    image_paths: list[Path],
) -> None:
    sync_index = load_sync_index(sync_index_path)
    existing_paths = set(sync_index.get(project_key, []))
    updated_paths = list(existing_paths.union(str(path.resolve()) for path in image_paths))
    sync_index[project_key] = sorted(updated_paths)
    sync_index_path.parent.mkdir(parents=True, exist_ok=True)
    sync_index_path.write_text(
        json.dumps(sync_index, indent=2),
        encoding="utf-8",
    )
