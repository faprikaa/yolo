from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote
from xml.sax.saxutils import escape

import requests

from yolo_dashboard.yolo_inference import Detection


class LabelStudioError(RuntimeError):
    """Raised when Label Studio integration fails."""


def build_label_config(labels: list[str]) -> str:
    palette = [
        "#1abc9c",
        "#3498db",
        "#f39c12",
        "#e74c3c",
        "#9b59b6",
        "#2ecc71",
        "#16a085",
        "#d35400",
    ]
    unique_labels: list[str] = []
    seen: set[str] = set()
    for label in labels:
        normalized = label.strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique_labels.append(normalized)

    label_nodes = "\n".join(
        f'    <Label value="{escape(label)}" background="{palette[index % len(palette)]}" />'
        for index, label in enumerate(unique_labels)
    )
    return (
        "<View>\n"
        '  <Image name="image" value="$image" />\n'
        '  <RectangleLabels name="label" toName="image">\n'
        f"{label_nodes}\n"
        "  </RectangleLabels>\n"
        "</View>"
    )


def build_local_file_url(image_path: Path, document_root: Path) -> str:
    try:
        relative_path = image_path.resolve().relative_to(document_root.resolve())
    except ValueError as error:
        raise LabelStudioError(
            f"Image {image_path} tidak berada di bawah document root {document_root}. "
            "Samakan LABEL_STUDIO_LOCAL_ROOT dengan parent folder image capture."
        ) from error

    return f"/data/local-files/?d={quote(relative_path.as_posix())}"


def detection_to_prediction_result(
    detection: Detection,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    return {
        "from_name": "label",
        "to_name": "image",
        "type": "rectanglelabels",
        "origin": "prediction",
        "image_rotation": 0,
        "original_width": image_width,
        "original_height": image_height,
        "score": round(detection.confidence, 6),
        "value": {
            "x": round((detection.x1 / image_width) * 100, 4),
            "y": round((detection.y1 / image_height) * 100, 4),
            "width": round((detection.width / image_width) * 100, 4),
            "height": round((detection.height / image_height) * 100, 4),
            "rotation": 0,
            "rectanglelabels": [detection.label],
        },
    }


def build_task_payload(
    image_path: Path,
    document_root: Path,
    image_width: int,
    image_height: int,
    detections: list[Detection],
    model_version: str,
) -> dict[str, Any]:
    task_payload: dict[str, Any] = {
        "data": {
            "image": build_local_file_url(
                image_path=image_path,
                document_root=document_root,
            )
        },
        "meta": {
            "source_path": str(image_path.resolve()),
            "model_version": model_version,
        },
    }

    if detections:
        task_payload["predictions"] = [
            {
                "model_version": model_version,
                "score": round(
                    max(detection.confidence for detection in detections),
                    6,
                ),
                "result": [
                    detection_to_prediction_result(
                        detection=detection,
                        image_width=image_width,
                        image_height=image_height,
                    )
                    for detection in detections
                ],
            }
        ]

    return task_payload


class LabelStudioClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json",
            }
        )

    def _request(
        self,
        method: str,
        endpoint: str,
        expected_statuses: tuple[int, ...] = (200, 201),
        **kwargs: Any,
    ) -> Any:
        response = self.session.request(
            method=method,
            url=f"{self.base_url}{endpoint}",
            timeout=self.timeout,
            **kwargs,
        )
        if response.status_code not in expected_statuses:
            raise LabelStudioError(
                f"Label Studio API error {response.status_code}: {response.text}"
            )

        if not response.content:
            return {}

        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            return response.json()
        return response.text

    def list_projects(self) -> list[dict[str, Any]]:
        response = self._request("GET", "/api/projects/")
        if isinstance(response, dict) and "results" in response:
            return list(response["results"])
        if isinstance(response, list):
            return response
        return []

    def get_or_create_project(
        self,
        title: str,
        labels: list[str],
        description: str = "",
    ) -> dict[str, Any]:
        projects = self.list_projects()
        for project in projects:
            if str(project.get("title", "")).strip() == title.strip():
                return project

        payload = {
            "title": title,
            "description": description,
            "label_config": build_label_config(labels),
            "show_instruction": True,
        }
        return self._request("POST", "/api/projects/", json=payload)

    def import_tasks(self, project_id: int, tasks: list[dict[str, Any]]) -> Any:
        if not tasks:
            raise LabelStudioError("Tidak ada task yang akan diimport ke Label Studio.")
        return self._request(
            "POST",
            f"/api/projects/{project_id}/import?commit_to_project=true&return_task_ids=true",
            expected_statuses=(200, 201, 202),
            json=tasks,
        )
