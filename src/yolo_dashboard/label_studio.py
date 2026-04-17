from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from xml.sax.saxutils import escape

from yolo_dashboard.yolo_inference import Detection


class LabelStudioError(RuntimeError):
    """Raised when Label Studio integration fails."""


@dataclass(frozen=True)
class LabelStudioProject:
    id: int
    title: str
    description: str


@dataclass(frozen=True)
class LabelStudioExportArtifact:
    project_id: int
    project_title: str
    export_id: int
    archive_path: Path
    export_type: str
    created_at: str


@dataclass(frozen=True)
class LabelStudioConnectionStatus:
    base_url: str
    project_count: int


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
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        if not self.base_url:
            raise LabelStudioError("Label Studio URL wajib diisi.")
        if not self.api_key:
            raise LabelStudioError("Label Studio API key wajib diisi.")
        self.client = self._create_sdk_client()

    def list_projects(self) -> list[LabelStudioProject]:
        try:
            projects = self.client.projects.list()
        except Exception as error:
            raise LabelStudioError(f"Gagal mengambil daftar project Label Studio: {error}") from error

        normalized_projects = [
            self._normalize_project(project)
            for project in self._as_list(projects)
        ]
        return sorted(normalized_projects, key=lambda project: project.title.casefold())

    def test_connection(self) -> LabelStudioConnectionStatus:
        projects = self.list_projects()
        return LabelStudioConnectionStatus(
            base_url=self.base_url,
            project_count=len(projects),
        )

    def get_or_create_project(
        self,
        title: str,
        labels: list[str],
        description: str = "",
    ) -> LabelStudioProject:
        normalized_title = title.strip()
        if not normalized_title:
            raise LabelStudioError("Nama project Label Studio wajib diisi.")

        for project in self.list_projects():
            if project.title.strip() == normalized_title:
                return project

        try:
            project = self.client.projects.create(
                title=normalized_title,
                description=description,
                label_config=build_label_config(labels),
                show_instruction=True,
            )
        except Exception as error:
            raise LabelStudioError(f"Gagal membuat project Label Studio: {error}") from error

        return self._normalize_project(project)

    def import_tasks(self, project_id: int, tasks: list[dict[str, Any]]) -> Any:
        if not tasks:
            raise LabelStudioError("Tidak ada task yang akan diimport ke Label Studio.")

        try:
            return self.client.projects.import_tasks(id=project_id, request=tasks)
        except Exception as error:
            raise LabelStudioError(f"Gagal import task ke Label Studio: {error}") from error

    def export_project_to_archive(
        self,
        project_id: int,
        export_dir: Path,
        export_type: str = "YOLO",
        download_resources: bool = True,
        timeout_seconds: int = 300,
        poll_interval_seconds: float = 1.0,
    ) -> LabelStudioExportArtifact:
        export_dir.mkdir(parents=True, exist_ok=True)
        export_type_candidates = self._resolve_export_type_candidates(
            export_type=export_type,
            download_resources=download_resources,
        )

        project = self._get_project(project_id)
        snapshot_title = (
            f"Streamlit {export_type_candidates[0]} export "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        try:
            export_job = self.client.projects.exports.create(
                id=project_id,
                title=snapshot_title,
            )
        except Exception as error:
            raise LabelStudioError(f"Gagal membuat export snapshot: {error}") from error

        export_id = int(self._read_attr(export_job, "id"))
        self._wait_for_export_completion(
            project_id=project_id,
            export_id=export_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

        last_error: Exception | None = None
        for export_type_name in export_type_candidates:
            try:
                conversion = self.client.projects.exports.convert(
                    id=project_id,
                    export_pk=export_id,
                    export_type=export_type_name,
                    download_resources=download_resources,
                )
            except TypeError:
                try:
                    conversion = self.client.projects.exports.convert(
                        id=project_id,
                        export_pk=export_id,
                        export_type=export_type_name,
                    )
                except Exception as error:
                    last_error = error
                    continue
            except Exception as error:
                last_error = error
                continue

            converted_format_id = self._read_attr(conversion, "converted_format")
            self._wait_for_conversion_completion(
                project_id=project_id,
                export_id=export_id,
                export_type=export_type_name,
                converted_format_id=converted_format_id,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )

            archive_path = export_dir / (
                f"label_studio_project_{project_id}_{export_type_name.lower()}_{_timestamp_slug()}.zip"
            )
            try:
                with archive_path.open("wb") as output_file:
                    for chunk in self.client.projects.exports.download(
                        id=project_id,
                        export_pk=export_id,
                        export_type=export_type_name,
                        request_options={"chunk_size": 1024 * 1024},
                    ):
                        output_file.write(chunk)
            except Exception as error:
                last_error = error
                if archive_path.exists():
                    archive_path.unlink(missing_ok=True)
                continue

            return LabelStudioExportArtifact(
                project_id=project_id,
                project_title=project.title,
                export_id=export_id,
                archive_path=archive_path,
                export_type=export_type_name,
                created_at=datetime.now(timezone.utc).isoformat(),
            )

        raise LabelStudioError(
            "Gagal membuat archive export Label Studio untuk format "
            f"{', '.join(export_type_candidates)}: {last_error}"
        )

    def _resolve_export_type_candidates(
        self,
        export_type: str,
        download_resources: bool,
    ) -> list[str]:
        normalized_export_type = export_type.strip().upper() or "YOLO"
        if normalized_export_type == "YOLO" and download_resources:
            return ["YOLO_WITH_IMAGES", "YOLO"]
        return [normalized_export_type]

    def _create_sdk_client(self) -> Any:
        try:
            from label_studio_sdk import LabelStudio
        except ImportError as error:
            raise LabelStudioError(
                "Package `label-studio-sdk` belum terpasang. "
                "Install dependency project dulu dengan `pip install -r requirements.txt`."
            ) from error

        try:
            return LabelStudio(base_url=self.base_url, api_key=self.api_key)
        except Exception as error:
            raise LabelStudioError(f"Gagal menginisialisasi Label Studio SDK: {error}") from error

    def _get_project(self, project_id: int) -> LabelStudioProject:
        try:
            project = self.client.projects.get(id=project_id)
        except Exception as error:
            raise LabelStudioError(f"Gagal mengambil detail project {project_id}: {error}") from error
        return self._normalize_project(project)

    def _normalize_project(self, project: Any) -> LabelStudioProject:
        return LabelStudioProject(
            id=int(self._read_attr(project, "id")),
            title=str(self._read_attr(project, "title", "")),
            description=str(self._read_attr(project, "description", "")),
        )

    def _as_list(self, payload: Any) -> list[Any]:
        if payload is None:
            return []
        if isinstance(payload, list):
            return payload
        if isinstance(payload, tuple):
            return list(payload)
        if isinstance(payload, dict):
            if isinstance(payload.get("results"), list):
                return list(payload["results"])
            if isinstance(payload.get("items"), list):
                return list(payload["items"])
            return []

        results = self._read_attr(payload, "results")
        if isinstance(results, list):
            return list(results)
        items = self._read_attr(payload, "items")
        if isinstance(items, list):
            return list(items)

        try:
            return list(payload)
        except TypeError:
            return []

    def _wait_for_export_completion(
        self,
        project_id: int,
        export_id: int,
        timeout_seconds: int,
        poll_interval_seconds: float,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                job = self.client.projects.exports.get(id=project_id, export_pk=export_id)
            except Exception as error:
                raise LabelStudioError(f"Gagal mengecek status export snapshot: {error}") from error

            status = str(self._read_attr(job, "status", "")).lower()
            if status == "completed":
                return
            if status == "failed":
                raise LabelStudioError("Export snapshot Label Studio gagal diproses.")
            time.sleep(poll_interval_seconds)

        raise LabelStudioError("Export snapshot Label Studio timeout.")

    def _wait_for_conversion_completion(
        self,
        project_id: int,
        export_id: int,
        export_type: str,
        converted_format_id: Any,
        timeout_seconds: int,
        poll_interval_seconds: float,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        normalized_export_type = export_type.upper()
        while time.monotonic() < deadline:
            try:
                job = self.client.projects.exports.get(id=project_id, export_pk=export_id)
            except Exception as error:
                raise LabelStudioError(f"Gagal mengecek status konversi export: {error}") from error

            converted_formats = list(self._read_attr(job, "converted_formats", []) or [])
            current_format = self._find_converted_format(
                converted_formats=converted_formats,
                converted_format_id=converted_format_id,
                export_type=normalized_export_type,
            )
            if current_format is not None:
                status = str(self._read_attr(current_format, "status", "")).lower()
                if status == "completed":
                    return
                if status == "failed":
                    raise LabelStudioError(
                        f"Konversi export {normalized_export_type} di Label Studio gagal."
                    )

            time.sleep(poll_interval_seconds)

        raise LabelStudioError(f"Konversi export {normalized_export_type} timeout.")

    def _find_converted_format(
        self,
        converted_formats: list[Any],
        converted_format_id: Any,
        export_type: str,
    ) -> Any | None:
        normalized_id = str(converted_format_id) if converted_format_id is not None else None
        normalized_export_type = export_type.upper()

        for item in converted_formats:
            item_id = self._read_attr(item, "id")
            if normalized_id is not None and str(item_id) == normalized_id:
                return item

        for item in converted_formats:
            item_export_type = str(self._read_attr(item, "export_type", "")).upper()
            if item_export_type == normalized_export_type:
                return item
        return None

    @staticmethod
    def _read_attr(payload: Any, key: str, default: Any = None) -> Any:
        if isinstance(payload, dict):
            return payload.get(key, default)
        return getattr(payload, key, default)


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
