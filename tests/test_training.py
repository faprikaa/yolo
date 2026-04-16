from __future__ import annotations

import os
from pathlib import Path
from zipfile import ZipFile

from yolo_dashboard.training import (
    discover_trained_models,
    list_base_model_names,
    list_prepared_datasets,
    prepare_label_studio_yolo_dataset,
)


def test_prepare_label_studio_yolo_dataset_creates_ultralytics_layout(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "label_studio_yolo_export.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("export-root/classes.txt", "helmet\nperson\n")
        archive.writestr("export-root/images/frame_001.jpg", b"image-1")
        archive.writestr("export-root/images/frame_002.jpg", b"image-2")
        archive.writestr("export-root/labels/frame_001.txt", "0 0.5 0.5 0.2 0.2\n")

    dataset = prepare_label_studio_yolo_dataset(
        archive_path=archive_path,
        dataset_root=tmp_path / "datasets",
        train_split=0.8,
        seed=7,
        project_id=10,
    )

    assert dataset.class_names == ["helmet", "person"]
    assert dataset.source_project_id == 10
    assert dataset.data_yaml_path.exists()
    assert dataset.split_counts == {"train": 1, "val": 1}
    assert dataset.labeled_images == 1

    data_yaml = dataset.data_yaml_path.read_text(encoding="utf-8")
    assert "train: images/train" in data_yaml
    assert "val: images/val" in data_yaml
    assert '0: "helmet"' in data_yaml
    assert '1: "person"' in data_yaml

    prepared_datasets = list_prepared_datasets(tmp_path / "datasets")
    assert prepared_datasets[0].data_yaml_path == dataset.data_yaml_path
    assert (dataset.dataset_dir / "dataset_metadata.json").exists()


def test_discover_trained_models_lists_latest_weights_first(tmp_path: Path) -> None:
    older_model = tmp_path / "runs" / "train_old" / "weights" / "best.pt"
    older_model.parent.mkdir(parents=True, exist_ok=True)
    older_model.write_bytes(b"old")
    os.utime(older_model, (1_700_000_000, 1_700_000_000))

    newer_model = tmp_path / "runs" / "train_new" / "weights" / "last.pt"
    newer_model.parent.mkdir(parents=True, exist_ok=True)
    newer_model.write_bytes(b"new")
    os.utime(newer_model, (1_800_000_000, 1_800_000_000))

    discovered_models = discover_trained_models(tmp_path / "runs")

    assert [artifact.path for artifact in discovered_models] == [newer_model, older_model]
    assert discovered_models[0].display_name == "train_new/weights/last.pt"


def test_base_models_include_yolo11_and_yolov8() -> None:
    base_models = list_base_model_names()

    assert "yolo11n.pt" in base_models
    assert "yolov8n.pt" in base_models
