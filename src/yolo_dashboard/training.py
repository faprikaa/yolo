from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_METADATA_NAME = "dataset_metadata.json"
DEFAULT_BASE_MODELS = (
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
)


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
        "project": str(runs_dir),
        "name": run_name.strip() or f"train_{_timestamp_slug()}",
        "exist_ok": False,
        "verbose": False,
    }
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
