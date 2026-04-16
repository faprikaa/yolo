from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    default_model_path: str
    capture_dir: Path
    export_dir: Path
    sync_index_path: Path
    label_studio_url: str
    label_studio_api_key: str
    label_studio_project_title: str
    label_studio_local_root: Path

    @classmethod
    def from_env(cls) -> "AppConfig":
        capture_dir = Path(os.getenv("CAPTURE_DIR", "data/captures")).resolve()
        export_dir = Path(os.getenv("EXPORT_DIR", "data/exports")).resolve()
        local_root = Path(
            os.getenv("LABEL_STUDIO_LOCAL_ROOT", str(capture_dir.parent))
        ).resolve()
        return cls(
            default_model_path=os.getenv("YOLO_MODEL_PATH", "yolo11n.pt"),
            capture_dir=capture_dir,
            export_dir=export_dir,
            sync_index_path=export_dir / "label_studio_sync_index.json",
            label_studio_url=os.getenv("LABEL_STUDIO_URL", "http://localhost:8080"),
            label_studio_api_key=os.getenv("LABEL_STUDIO_API_KEY", ""),
            label_studio_project_title=os.getenv(
                "LABEL_STUDIO_PROJECT_TITLE",
                "YOLO Camera Dashboard",
            ),
            label_studio_local_root=local_root,
        )

    def ensure_directories(self) -> None:
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.label_studio_local_root.mkdir(parents=True, exist_ok=True)
