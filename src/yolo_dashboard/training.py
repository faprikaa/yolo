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
from urllib.parse import parse_qs, unquote, urlparse
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
class DatasetSourceAsset:
    image_path: Path
    relative_path: Path
    label_path: Path | None


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


def inspect_existing_yolo_dataset(dataset_path: Path) -> PreparedDataset:
    data_yaml_path = _resolve_data_yaml_path(dataset_path)
    payload = _read_yaml_file(data_yaml_path)
    dataset_dir = _resolve_dataset_root(data_yaml_path=data_yaml_path, payload=payload)
    class_names = _normalize_class_names(payload.get("names"))
    if not class_names:
        raise RuntimeError("Dataset valid harus punya `names` di data.yaml.")

    split_entries_by_name: dict[str, list[Path]] = {}
    split_counts: dict[str, int] = {}
    for split_name in ("train", "val"):
        if split_name not in payload:
            raise RuntimeError(f"Dataset valid harus punya entry `{split_name}` di data.yaml.")

        split_entries = _resolve_dataset_entries(
            split_value=payload[split_name],
            dataset_dir=dataset_dir,
            yaml_dir=data_yaml_path.parent,
        )
        image_count = sum(_count_images_in_entry(path) for path in split_entries)
        if image_count <= 0:
            raise RuntimeError(
                f"Path `{split_name}` pada dataset tidak berisi image yang bisa dipakai."
            )
        split_entries_by_name[split_name] = split_entries
        split_counts[split_name] = image_count

    labeled_images = _count_labeled_images(split_entries_by_name)
    if labeled_images <= 0:
        raise RuntimeError(
            "Dataset ditemukan, tetapi file label YOLO tidak terdeteksi. "
            "Pastikan struktur `images/...` dan `labels/...` tersedia."
        )

    return PreparedDataset(
        dataset_dir=dataset_dir,
        data_yaml_path=data_yaml_path,
        source_archive_path=data_yaml_path,
        source_project_id=None,
        class_names=class_names,
        split_counts=split_counts,
        labeled_images=labeled_images,
        created_at=datetime.fromtimestamp(
            data_yaml_path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat(),
    )


def prepare_label_studio_yolo_dataset(
    archive_path: Path,
    dataset_root: Path,
    train_split: float = 0.8,
    seed: int = 42,
    project_id: int | None = None,
    fallback_image_roots: list[Path] | None = None,
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
        source_assets = _collect_label_studio_source_assets(
            export_root=export_root,
            fallback_image_roots=fallback_image_roots,
        )
        if not source_assets:
            normalized_roots = _normalize_search_roots(fallback_image_roots or [])
            fallback_summary = ", ".join(str(path) for path in normalized_roots) or "-"
            raise RuntimeError(
                "Export YOLO tidak berisi image di dalam ZIP dan image sumber juga tidak "
                "ditemukan dari fallback path. "
                f"Fallback yang dicek: {fallback_summary}"
            )

        dataset_dir = dataset_root / f"prepared_yolo_{slug}"
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        splits = _split_images(source_assets, train_split=train_split, seed=seed)
        labeled_relative_paths: set[str] = set()
        split_counts: dict[str, int] = {}

        for split_name, assets in splits.items():
            split_counts[split_name] = len(assets)
            for asset in assets:
                destination_image = images_dir / split_name / asset.relative_path
                destination_image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(asset.image_path, destination_image)

                if asset.label_path is not None and asset.label_path.exists():
                    destination_label = (
                        labels_dir / split_name / asset.relative_path.with_suffix(".txt")
                    )
                    destination_label.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(asset.label_path, destination_label)
                    labeled_relative_paths.add(asset.relative_path.with_suffix("").as_posix())

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
    if (root / "classes.txt").exists() and (
        (root / "images").exists() or (root / "labels").exists()
    ):
        return root

    for candidate in root.rglob("classes.txt"):
        candidate_root = candidate.parent
        if (candidate_root / "images").exists() or (candidate_root / "labels").exists():
            return candidate_root

    raise RuntimeError(
        "Struktur export YOLO dari Label Studio tidak dikenali. "
        "Pastikan file ZIP berisi classes.txt dan folder labels/."
    )


def _collect_label_studio_source_assets(
    export_root: Path,
    fallback_image_roots: list[Path] | None,
) -> list[DatasetSourceAsset]:
    labels_dir = _find_export_child_dir(export_root, "labels")
    images_dir = _find_export_child_dir(export_root, "images")
    source_assets: list[DatasetSourceAsset] = []
    covered_label_keys: set[str] = set()

    if images_dir is not None:
        for image_path in _list_source_images(images_dir):
            relative_path = image_path.relative_to(images_dir)
            label_path = None
            if labels_dir is not None:
                candidate_label = labels_dir / relative_path.with_suffix(".txt")
                if candidate_label.exists():
                    label_path = candidate_label
                    covered_label_keys.add(_normalized_relative_key(relative_path))
            source_assets.append(
                DatasetSourceAsset(
                    image_path=image_path,
                    relative_path=relative_path,
                    label_path=label_path,
                )
            )

    if labels_dir is None:
        return source_assets

    candidate_index = _build_candidate_image_index(
        search_roots=_normalize_search_roots(fallback_image_roots or []),
        extra_image_paths=_extract_absolute_image_paths_from_export_metadata(export_root),
    )

    for label_path in _list_label_files(labels_dir):
        relative_label_path = label_path.relative_to(labels_dir)
        relative_key = _normalized_relative_key(relative_label_path)
        if relative_key in covered_label_keys:
            continue

        matched_image_path = _resolve_image_for_label(
            relative_label_path=relative_label_path,
            candidate_index=candidate_index,
        )
        if matched_image_path is None:
            continue

        source_assets.append(
            DatasetSourceAsset(
                image_path=matched_image_path,
                relative_path=relative_label_path.with_suffix(matched_image_path.suffix),
                label_path=label_path,
            )
        )
        covered_label_keys.add(relative_key)

    return _deduplicate_source_assets(source_assets)


def _find_export_child_dir(root: Path, folder_name: str) -> Path | None:
    direct_candidate = root / folder_name
    if direct_candidate.is_dir():
        return direct_candidate

    candidates = sorted(
        [
            candidate
            for candidate in root.rglob(folder_name)
            if candidate.is_dir()
        ],
        key=lambda path: path.as_posix().casefold(),
    )
    if not candidates:
        return None
    return candidates[0]


def _list_label_files(labels_dir: Path) -> list[Path]:
    if not labels_dir.exists():
        return []
    return sorted(
        [
            path
            for path in labels_dir.rglob("*.txt")
            if path.is_file()
        ]
    )


def _normalize_search_roots(search_roots: list[Path]) -> list[Path]:
    normalized_roots: list[Path] = []
    seen_roots: set[str] = set()
    for root in search_roots:
        try:
            resolved_root = root.expanduser().resolve()
        except Exception:
            continue
        if not resolved_root.exists() or not resolved_root.is_dir():
            continue
        root_key = str(resolved_root).casefold()
        if root_key in seen_roots:
            continue
        seen_roots.add(root_key)
        normalized_roots.append(resolved_root)
    return normalized_roots


def _build_candidate_image_index(
    search_roots: list[Path],
    extra_image_paths: list[Path],
) -> dict[str, dict[str, list[Path]]]:
    by_relative: dict[str, list[Path]] = {}
    by_stem: dict[str, list[Path]] = {}
    seen_paths: set[str] = set()

    for image_path in extra_image_paths:
        _register_candidate_image(
            image_path=image_path,
            search_roots=search_roots,
            by_relative=by_relative,
            by_stem=by_stem,
            seen_paths=seen_paths,
        )

    for root in search_roots:
        for image_path in root.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            _register_candidate_image(
                image_path=image_path,
                search_roots=search_roots,
                by_relative=by_relative,
                by_stem=by_stem,
                seen_paths=seen_paths,
            )

    return {
        "by_relative": by_relative,
        "by_stem": by_stem,
    }


def _register_candidate_image(
    image_path: Path,
    search_roots: list[Path],
    by_relative: dict[str, list[Path]],
    by_stem: dict[str, list[Path]],
    seen_paths: set[str],
) -> None:
    try:
        resolved_image_path = image_path.expanduser().resolve()
    except Exception:
        return
    if not resolved_image_path.exists() or not resolved_image_path.is_file():
        return
    if resolved_image_path.suffix.lower() not in IMAGE_SUFFIXES:
        return

    image_key = str(resolved_image_path).casefold()
    if image_key in seen_paths:
        return
    seen_paths.add(image_key)

    by_stem.setdefault(resolved_image_path.stem.casefold(), []).append(resolved_image_path)

    for root in search_roots:
        try:
            relative_path = resolved_image_path.relative_to(root)
        except ValueError:
            continue

        relative_key = _normalized_relative_key(relative_path)
        by_relative.setdefault(relative_key, []).append(resolved_image_path)

        if relative_path.parts and relative_path.parts[0].casefold() == "images":
            trimmed_relative = Path(*relative_path.parts[1:]) if len(relative_path.parts) > 1 else Path(relative_path.name)
            trimmed_key = _normalized_relative_key(trimmed_relative)
            by_relative.setdefault(trimmed_key, []).append(resolved_image_path)


def _resolve_image_for_label(
    relative_label_path: Path,
    candidate_index: dict[str, dict[str, list[Path]]],
) -> Path | None:
    relative_key = _normalized_relative_key(relative_label_path)
    by_relative = candidate_index.get("by_relative", {})
    by_stem = candidate_index.get("by_stem", {})

    direct_candidates = by_relative.get(relative_key, [])
    if direct_candidates:
        return sorted(
            direct_candidates,
            key=lambda path: path.as_posix().casefold(),
        )[0]

    stem_candidates = by_stem.get(relative_label_path.stem.casefold(), [])
    if stem_candidates:
        return sorted(stem_candidates, key=lambda path: path.as_posix().casefold())[0]

    return None


def _extract_absolute_image_paths_from_export_metadata(root: Path) -> list[Path]:
    discovered_paths: list[Path] = []
    seen_paths: set[str] = set()
    for json_path in root.rglob("*.json"):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for value in _iter_string_values(payload):
            absolute_image_path = _extract_absolute_image_path(value)
            if absolute_image_path is None:
                continue
            image_key = str(absolute_image_path).casefold()
            if image_key in seen_paths:
                continue
            seen_paths.add(image_key)
            discovered_paths.append(absolute_image_path)
    return discovered_paths


def _iter_string_values(payload: Any) -> list[str]:
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, dict):
        values: list[str] = []
        for value in payload.values():
            values.extend(_iter_string_values(value))
        return values
    if isinstance(payload, list):
        values: list[str] = []
        for item in payload:
            values.extend(_iter_string_values(item))
        return values
    return []


def _extract_absolute_image_path(value: str) -> Path | None:
    raw_value = value.strip()
    if not raw_value:
        return None

    candidate_values = [raw_value]
    if raw_value.startswith("file://"):
        candidate_values.append(unquote(urlparse(raw_value).path))

    query_payload = parse_qs(urlparse(raw_value).query)
    for item in query_payload.get("d", []):
        candidate_values.append(unquote(item))

    for candidate in candidate_values:
        normalized_candidate = candidate.strip()
        if not normalized_candidate:
            continue
        candidate_path = Path(normalized_candidate)
        if (
            candidate_path.is_absolute()
            and candidate_path.exists()
            and candidate_path.is_file()
            and candidate_path.suffix.lower() in IMAGE_SUFFIXES
        ):
            return candidate_path.resolve()
    return None


def _normalized_relative_key(path: Path) -> str:
    return path.with_suffix("").as_posix().casefold()


def _deduplicate_source_assets(source_assets: list[DatasetSourceAsset]) -> list[DatasetSourceAsset]:
    deduplicated_assets: list[DatasetSourceAsset] = []
    seen_keys: set[tuple[str, str]] = set()

    for asset in source_assets:
        asset_key = (
            asset.relative_path.as_posix().casefold(),
            str(asset.image_path.resolve()).casefold(),
        )
        if asset_key in seen_keys:
            continue
        seen_keys.add(asset_key)
        deduplicated_assets.append(asset)

    return deduplicated_assets


def _resolve_data_yaml_path(dataset_path: Path) -> Path:
    candidate_path = dataset_path.expanduser().resolve()
    if candidate_path.is_file():
        if candidate_path.suffix.lower() not in {".yaml", ".yml"}:
            raise RuntimeError("File dataset harus berupa `data.yaml` atau `data.yml`.")
        return candidate_path

    if not candidate_path.exists():
        raise RuntimeError(f"Path dataset tidak ditemukan: {candidate_path}")
    if not candidate_path.is_dir():
        raise RuntimeError(f"Path dataset tidak valid: {candidate_path}")

    for file_name in ("data.yaml", "data.yml"):
        yaml_path = candidate_path / file_name
        if yaml_path.exists():
            return yaml_path

    raise RuntimeError(
        "Folder dataset belum valid. File `data.yaml` atau `data.yml` tidak ditemukan."
    )


def _read_yaml_file(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "Dependency YAML tidak tersedia. Pastikan environment project ter-install lengkap."
        ) from error

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Isi YAML dataset tidak valid: {path}")
    return payload


def _resolve_dataset_root(data_yaml_path: Path, payload: dict[str, Any]) -> Path:
    root_entry = payload.get("path")
    if root_entry:
        root_path = Path(str(root_entry))
        if not root_path.is_absolute():
            root_path = (data_yaml_path.parent / root_path).resolve()
        else:
            root_path = root_path.resolve()
        return root_path
    return data_yaml_path.parent.resolve()


def _normalize_class_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, list):
        return [str(item).strip() for item in raw_names if str(item).strip()]
    if isinstance(raw_names, dict):
        ordered_items = sorted(raw_names.items(), key=lambda item: int(item[0]))
        return [str(value).strip() for _, value in ordered_items if str(value).strip()]
    return []


def _resolve_dataset_entries(
    split_value: Any,
    dataset_dir: Path,
    yaml_dir: Path,
) -> list[Path]:
    if isinstance(split_value, list):
        resolved_paths: list[Path] = []
        for item in split_value:
            resolved_paths.extend(
                _resolve_dataset_entries(
                    split_value=item,
                    dataset_dir=dataset_dir,
                    yaml_dir=yaml_dir,
                )
            )
        return resolved_paths

    if isinstance(split_value, str):
        candidate_path = Path(split_value)
    else:
        candidate_path = Path(str(split_value))

    if candidate_path.is_absolute():
        resolved_path = candidate_path.resolve()
    else:
        dataset_candidate = (dataset_dir / candidate_path).resolve()
        yaml_candidate = (yaml_dir / candidate_path).resolve()
        resolved_path = dataset_candidate if dataset_candidate.exists() else yaml_candidate

    if not resolved_path.exists():
        raise RuntimeError(f"Path dataset tidak ditemukan: {resolved_path}")
    return [resolved_path]


def _count_images_in_entry(path: Path) -> int:
    if path.is_file():
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return 1
        if path.suffix.lower() == ".txt":
            lines = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            return len(lines)
        return 0

    return len(
        [
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        ]
    )


def _count_labeled_images(split_entries_by_name: dict[str, list[Path]]) -> int:
    label_files: set[str] = set()
    for split_entries in split_entries_by_name.values():
        for entry in split_entries:
            for label_dir in _candidate_label_dirs(entry):
                if not label_dir.exists():
                    continue
                for label_path in label_dir.rglob("*.txt"):
                    label_files.add(str(label_path.resolve()))
    return len(label_files)


def _candidate_label_dirs(entry: Path) -> list[Path]:
    candidates: list[Path] = []
    if entry.is_dir():
        if "images" in entry.parts:
            images_index = entry.parts.index("images")
            label_parts = list(entry.parts)
            label_parts[images_index] = "labels"
            candidates.append(Path(*label_parts))
        candidates.append(entry.parent / "labels" / entry.name)
        candidates.append(entry.parent / "labels")
    return list(dict.fromkeys(candidate.resolve() for candidate in candidates))


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
