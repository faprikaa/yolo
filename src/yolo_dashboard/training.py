from __future__ import annotations

import csv
import json
import os
import random
import shutil
import threading
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_METADATA_NAME = "dataset_metadata.json"
DEFAULT_BASE_MODELS = (
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
)
_TRAINING_JOBS: dict[str, "TrainingJobState"] = {}
_TRAINING_JOBS_LOCK = threading.Lock()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def list_base_model_names() -> list[str]:
    return list(DEFAULT_BASE_MODELS)


def list_training_device_options() -> list[tuple[str, str]]:
    options = [("auto", "Auto"), ("cpu", "CPU")]

    try:
        import torch
    except Exception:
        return options

    if not torch.cuda.is_available():
        return options

    for index in range(torch.cuda.device_count()):
        options.append((f"gpu:{index}", f"GPU {index} - {torch.cuda.get_device_name(index)}"))
    return options


def normalize_training_device(selection: str) -> tuple[str, str]:
    normalized = selection.strip().lower()
    if not normalized or normalized == "auto":
        return "", "Auto"
    if normalized == "cpu":
        return "cpu", "CPU"
    if normalized.startswith("gpu:"):
        device_index = normalized.split(":", maxsplit=1)[1].strip()
        if not device_index.isdigit():
            raise RuntimeError(f"Format device GPU tidak valid: {selection}")
        return device_index, f"GPU {device_index}"
    raise RuntimeError(f"Pilihan device tidak dikenali: {selection}")


@dataclass(frozen=True)
class TrainedModelArtifact:
    path: Path
    display_name: str
    modified_at: str


@dataclass(frozen=True)
class PreparedDataset:
    dataset_dir: Path
    data_yaml_path: Path
    source_archive_path: Path
    source_project_id: int | None
    class_names: list[str]
    split_counts: dict[str, int]
    labeled_images: int
    created_at: str


@dataclass(frozen=True)
class TrainingArtifact:
    run_dir: Path
    best_model_path: Path | None
    last_model_path: Path | None
    results_path: Path | None
    dataset_yaml_path: Path
    source_model_path: str


@dataclass
class TrainingJobState:
    job_id: str
    run_name: str
    run_dir: Path
    dataset_yaml_path: Path
    source_model_path: str
    device_label: str
    total_epochs: int
    status: str
    message: str
    current_epoch: int
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    best_model_path: Path | None = None
    last_model_path: Path | None = None
    results_path: Path | None = None


def discover_trained_models(
    runs_dir: Path,
    limit: int = 50,
) -> list[TrainedModelArtifact]:
    if not runs_dir.exists():
        return []

    model_paths = sorted(
        [path for path in runs_dir.rglob("*.pt") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    artifacts: list[TrainedModelArtifact] = []
    for path in model_paths[:limit]:
        try:
            relative_path = path.resolve().relative_to(runs_dir.resolve()).as_posix()
        except ValueError:
            relative_path = path.name
        artifacts.append(
            TrainedModelArtifact(
                path=path,
                display_name=relative_path,
                modified_at=datetime.fromtimestamp(
                    path.stat().st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
            )
        )
    return artifacts


def list_prepared_datasets(dataset_root: Path) -> list[PreparedDataset]:
    if not dataset_root.exists():
        return []

    metadata_paths = sorted(
        dataset_root.rglob(DATASET_METADATA_NAME),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return [_load_prepared_dataset(metadata_path) for metadata_path in metadata_paths]


def start_training_job(
    dataset_yaml_path: Path,
    model_path: str,
    runs_dir: Path,
    run_name: str,
    epochs: int,
    image_size: int,
    batch_size: int,
    patience: int,
    device_selection: str,
    workers: int,
    optimizer: str,
    learning_rate: float,
) -> TrainingJobState:
    resolved_run_name = _ensure_unique_run_name(runs_dir, run_name.strip())
    run_dir = runs_dir / resolved_run_name
    normalized_device, device_label = normalize_training_device(device_selection)
    job_id = _timestamp_slug()

    job_state = TrainingJobState(
        job_id=job_id,
        run_name=resolved_run_name,
        run_dir=run_dir,
        dataset_yaml_path=dataset_yaml_path,
        source_model_path=model_path,
        device_label=device_label,
        total_epochs=int(epochs),
        status="queued",
        message="Menunggu training dimulai",
        current_epoch=0,
        created_at=_iso_now(),
    )
    with _TRAINING_JOBS_LOCK:
        _TRAINING_JOBS[job_id] = job_state

    worker = threading.Thread(
        target=_run_training_job,
        kwargs={
            "job_id": job_id,
            "dataset_yaml_path": dataset_yaml_path,
            "model_path": model_path,
            "runs_dir": runs_dir,
            "run_name": resolved_run_name,
            "epochs": int(epochs),
            "image_size": int(image_size),
            "batch_size": int(batch_size),
            "patience": int(patience),
            "device": normalized_device,
            "workers": int(workers),
            "optimizer": optimizer,
            "learning_rate": float(learning_rate),
        },
        daemon=True,
    )
    worker.start()
    return deepcopy(job_state)


def get_training_job(job_id: str) -> TrainingJobState | None:
    if not job_id:
        return None

    with _TRAINING_JOBS_LOCK:
        job_state = deepcopy(_TRAINING_JOBS.get(job_id))
    if job_state is None:
        return None

    progress_snapshot = read_training_progress(job_state.run_dir)
    job_state.current_epoch = int(progress_snapshot["completed_epochs"])
    job_state.results_path = progress_snapshot["results_path"]
    weights_dir = job_state.run_dir / "weights"
    best_model_path = weights_dir / "best.pt"
    last_model_path = weights_dir / "last.pt"
    job_state.best_model_path = best_model_path if best_model_path.exists() else None
    job_state.last_model_path = last_model_path if last_model_path.exists() else None
    return job_state


def get_resource_usage() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "cpu_percent": None,
        "memory_percent": None,
        "process_memory_gb": None,
        "gpus": [],
    }

    try:
        import psutil

        process = psutil.Process(os.getpid())
        snapshot["cpu_percent"] = float(psutil.cpu_percent(interval=None))
        snapshot["memory_percent"] = float(psutil.virtual_memory().percent)
        snapshot["process_memory_gb"] = float(process.memory_info().rss / (1024 ** 3))
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            gpus: list[dict[str, Any]] = []
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                used_bytes = total_bytes - free_bytes
                gpus.append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "used_gb": round(used_bytes / (1024 ** 3), 2),
                        "total_gb": round(total_bytes / (1024 ** 3), 2),
                        "memory_percent": round((used_bytes / total_bytes) * 100, 2)
                        if total_bytes
                        else 0.0,
                    }
                )
            snapshot["gpus"] = gpus
    except Exception:
        pass

    return snapshot


def read_training_progress(run_dir: Path) -> dict[str, Any]:
    results_path = run_dir / "results.csv"
    return {
        "completed_epochs": _read_completed_epochs(run_dir),
        "results_path": results_path if results_path.exists() else None,
    }


def prepare_label_studio_yolo_dataset(
    archive_path: Path,
    dataset_root: Path,
    train_split: float = 0.8,
    seed: int = 42,
    project_id: int | None = None,
) -> PreparedDataset:
    if not archive_path.exists():
        raise RuntimeError(f"File export Label Studio tidak ditemukan: {archive_path}")
    if not 0.5 <= train_split <= 0.95:
        raise RuntimeError("Train split harus di antara 0.50 sampai 0.95.")

    dataset_root.mkdir(parents=True, exist_ok=True)
    slug = _timestamp_slug()
    extract_dir = dataset_root / f"_extract_{slug}"
    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)

        export_root = _find_label_studio_export_root(extract_dir)
        class_names = _read_class_names(export_root / "classes.txt")
        source_images = _list_source_images(export_root / "images")
        if not source_images:
            raise RuntimeError("Export YOLO tidak berisi image yang bisa dipakai untuk training.")

        dataset_dir = dataset_root / f"prepared_yolo_{slug}"
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        splits = _split_images(source_images, train_split=train_split, seed=seed)
        labeled_relative_paths: set[str] = set()
        split_counts: dict[str, int] = {}

        raw_images_dir = export_root / "images"
        raw_labels_dir = export_root / "labels"
        for split_name, image_paths in splits.items():
            split_counts[split_name] = len(image_paths)
            for image_path in image_paths:
                relative_path = image_path.relative_to(raw_images_dir)
                destination_image = images_dir / split_name / relative_path
                destination_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, destination_image)

                source_label = raw_labels_dir / relative_path.with_suffix(".txt")
                if source_label.exists():
                    destination_label = labels_dir / split_name / relative_path.with_suffix(".txt")
                    destination_label.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_label, destination_label)
                    labeled_relative_paths.add(relative_path.as_posix())

        data_yaml_path = dataset_dir / "data.yaml"
        data_yaml_path.write_text(
            _build_data_yaml(dataset_dir=dataset_dir, class_names=class_names),
            encoding="utf-8",
        )

        metadata = {
            "created_at": _iso_now(),
            "data_yaml_path": str(data_yaml_path.resolve()),
            "dataset_dir": str(dataset_dir.resolve()),
            "source_archive_path": str(archive_path.resolve()),
            "source_project_id": project_id,
            "class_names": class_names,
            "split_counts": split_counts,
            "labeled_images": len(labeled_relative_paths),
        }
        _write_json(dataset_dir / DATASET_METADATA_NAME, metadata)
        return _prepared_dataset_from_payload(metadata)
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def train_yolo_model(
    dataset_yaml_path: Path,
    model_path: str,
    runs_dir: Path,
    run_name: str,
    epochs: int,
    image_size: int,
    batch_size: int,
    patience: int,
    device: str,
    workers: int,
    optimizer: str,
    learning_rate: float,
) -> TrainingArtifact:
    if not dataset_yaml_path.exists():
        raise RuntimeError(f"File dataset YAML tidak ditemukan: {dataset_yaml_path}")

    from ultralytics import YOLO

    runs_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)

    train_kwargs: dict[str, object] = {
        "data": str(dataset_yaml_path),
        "epochs": int(epochs),
        "imgsz": int(image_size),
        "batch": int(batch_size),
        "patience": int(patience),
        "workers": int(workers),
        "project": str(runs_dir),
        "name": run_name.strip() or f"train_{_timestamp_slug()}",
        "exist_ok": False,
        "verbose": False,
    }
    normalized_optimizer = optimizer.strip()
    if normalized_optimizer and normalized_optimizer.lower() != "auto":
        train_kwargs["optimizer"] = normalized_optimizer
    if learning_rate > 0:
        train_kwargs["lr0"] = float(learning_rate)
    normalized_device = device.strip()
    if normalized_device and normalized_device.lower() != "auto":
        train_kwargs["device"] = normalized_device

    result = model.train(**train_kwargs)
    save_dir = getattr(result, "save_dir", None) or getattr(
        getattr(model, "trainer", None),
        "save_dir",
        None,
    )
    if save_dir is None:
        raise RuntimeError("Ultralytics tidak mengembalikan direktori hasil training.")

    run_dir = Path(save_dir)
    weights_dir = run_dir / "weights"
    best_model_path = weights_dir / "best.pt"
    last_model_path = weights_dir / "last.pt"
    results_path = run_dir / "results.csv"

    return TrainingArtifact(
        run_dir=run_dir,
        best_model_path=best_model_path if best_model_path.exists() else None,
        last_model_path=last_model_path if last_model_path.exists() else None,
        results_path=results_path if results_path.exists() else None,
        dataset_yaml_path=dataset_yaml_path,
        source_model_path=model_path,
    )


def _run_training_job(
    job_id: str,
    dataset_yaml_path: Path,
    model_path: str,
    runs_dir: Path,
    run_name: str,
    epochs: int,
    image_size: int,
    batch_size: int,
    patience: int,
    device: str,
    workers: int,
    optimizer: str,
    learning_rate: float,
) -> None:
    _update_training_job(
        job_id,
        status="running",
        message="Training sedang berjalan",
        started_at=_iso_now(),
    )
    try:
        artifact = train_yolo_model(
            dataset_yaml_path=dataset_yaml_path,
            model_path=model_path,
            runs_dir=runs_dir,
            run_name=run_name,
            epochs=epochs,
            image_size=image_size,
            batch_size=batch_size,
            patience=patience,
            device=device,
            workers=workers,
            optimizer=optimizer,
            learning_rate=learning_rate,
        )
        _update_training_job(
            job_id,
            status="completed",
            message="Training selesai",
            current_epoch=int(epochs),
            finished_at=_iso_now(),
            best_model_path=artifact.best_model_path,
            last_model_path=artifact.last_model_path,
            results_path=artifact.results_path,
        )
    except Exception as error:
        _update_training_job(
            job_id,
            status="failed",
            message="Training gagal",
            finished_at=_iso_now(),
            error=str(error),
        )


def _update_training_job(job_id: str, **updates: Any) -> None:
    with _TRAINING_JOBS_LOCK:
        job_state = _TRAINING_JOBS.get(job_id)
        if job_state is None:
            return
        for key, value in updates.items():
            setattr(job_state, key, value)


def _ensure_unique_run_name(runs_dir: Path, requested_name: str) -> str:
    base_name = requested_name or f"train_{_timestamp_slug()}"
    candidate = base_name
    suffix = 2
    while (runs_dir / candidate).exists():
        candidate = f"{base_name}_{suffix}"
        suffix += 1
    return candidate


def _read_completed_epochs(run_dir: Path) -> int:
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        return 0

    with results_path.open("r", encoding="utf-8", newline="") as results_file:
        reader = csv.reader(results_file)
        rows = [row for row in reader if row]
    return max(0, len(rows) - 1)


def _find_label_studio_export_root(root: Path) -> Path:
    if (root / "classes.txt").exists() and (root / "images").exists():
        return root

    for candidate in root.rglob("classes.txt"):
        candidate_root = candidate.parent
        if (candidate_root / "images").exists():
            return candidate_root

    raise RuntimeError(
        "Struktur export YOLO dari Label Studio tidak dikenali. "
        "Pastikan file ZIP berisi classes.txt dan folder images/."
    )


def _read_class_names(classes_path: Path) -> list[str]:
    if not classes_path.exists():
        raise RuntimeError(f"File classes.txt tidak ditemukan di {classes_path}")

    class_names = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not class_names:
        raise RuntimeError("classes.txt kosong, class YOLO tidak bisa ditentukan.")
    return class_names


def _list_source_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise RuntimeError(f"Folder images tidak ditemukan di export: {images_dir}")
    return sorted(
        [
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
    )


def _split_images(
    image_paths: list[Path],
    train_split: float,
    seed: int,
) -> dict[str, list[Path]]:
    shuffled = list(image_paths)
    random.Random(seed).shuffle(shuffled)

    if len(shuffled) == 1:
        return {"train": [shuffled[0]], "val": [shuffled[0]]}

    validation_count = max(1, int(round(len(shuffled) * (1 - train_split))))
    validation_count = min(validation_count, len(shuffled) - 1)
    split_index = len(shuffled) - validation_count
    return {
        "train": shuffled[:split_index],
        "val": shuffled[split_index:],
    }


def _build_data_yaml(dataset_dir: Path, class_names: list[str]) -> str:
    lines = [
        f"path: {json.dumps(dataset_dir.resolve().as_posix())}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for index, class_name in enumerate(class_names):
        lines.append(f"  {index}: {json.dumps(class_name)}")
    lines.append("")
    return "\n".join(lines)


def _load_prepared_dataset(metadata_path: Path) -> PreparedDataset:
    payload = _read_json(metadata_path)
    return _prepared_dataset_from_payload(payload)


def _prepared_dataset_from_payload(payload: dict) -> PreparedDataset:
    return PreparedDataset(
        dataset_dir=Path(payload["dataset_dir"]),
        data_yaml_path=Path(payload["data_yaml_path"]),
        source_archive_path=Path(payload["source_archive_path"]),
        source_project_id=(
            int(payload["source_project_id"])
            if payload.get("source_project_id") is not None
            else None
        ),
        class_names=[str(item) for item in payload.get("class_names", [])],
        split_counts={
            str(split_name): int(count)
            for split_name, count in payload.get("split_counts", {}).items()
        },
        labeled_images=int(payload.get("labeled_images", 0)),
        created_at=str(payload.get("created_at", "")),
    )
