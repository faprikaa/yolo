"""Standalone training script — runs without Streamlit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from yolo_dashboard.training import (
    DEFAULT_BASE_MODELS,
    TrainingArtifact,
    list_prepared_datasets,
    train_yolo_model,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO model")

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path ke data.yaml dataset (atau nama dataset di DATASET_DIR)",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help=f"Base model atau path ke .pt file. Default: yolo11n.pt. Pilihan: {', '.join(DEFAULT_BASE_MODELS)}",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto", help="cpu | 0 | 0,1 | auto")
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr", type=float, default=0.0, help="lr0, 0 = pakai default ultralytics")
    parser.add_argument(
        "--runs-dir",
        default="data/runs",
        help="Folder output hasil training",
    )
    parser.add_argument("--name", default="", help="Nama run (kosong = auto)")
    parser.add_argument("--list-datasets", action="store_true", help="List dataset yang tersedia lalu keluar")

    return parser.parse_args()


def _resolve_dataset_yaml(dataset_arg: str) -> Path:
    candidate = Path(dataset_arg)
    if candidate.exists():
        return candidate.resolve()

    dataset_dir = Path("data/datasets")
    if dataset_dir.exists():
        datasets = list_prepared_datasets(dataset_dir)
        for ds in datasets:
            if ds.dataset_dir.name == dataset_arg:
                return ds.data_yaml_path
        for ds in datasets:
            if dataset_arg in ds.dataset_dir.name:
                return ds.data_yaml_path

    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {dataset_arg}\n"
        "Gunakan --list-datasets untuk melihat dataset yang tersedia."
    )


def main() -> None:
    args = _parse_args()

    if args.list_datasets:
        dataset_dir = Path("data/datasets")
        datasets = list_prepared_datasets(dataset_dir)
        if not datasets:
            print("Tidak ada dataset di data/datasets/")
            return
        print(f"{'Nama':<40} {'Kelas':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
        print("-" * 75)
        for ds in datasets:
            train = ds.split_counts.get("train", 0)
            val = ds.split_counts.get("val", 0)
            test = ds.split_counts.get("test", 0)
            print(f"{ds.dataset_dir.name:<40} {len(ds.class_names):<8} {train:<8} {val:<8} {test:<8}")
        return

    dataset_yaml = _resolve_dataset_yaml(args.dataset)
    runs_dir = Path(args.runs_dir).resolve()
    device = "" if args.device.lower() == "auto" else args.device

    print(f"Dataset  : {dataset_yaml}")
    print(f"Model    : {args.model}")
    print(f"Epochs   : {args.epochs}")
    print(f"Imgsz    : {args.imgsz}")
    print(f"Batch    : {args.batch}")
    print(f"Device   : {device or 'auto'}")
    print(f"Runs dir : {runs_dir}")
    print()

    artifact: TrainingArtifact = train_yolo_model(
        dataset_yaml_path=dataset_yaml,
        model_path=args.model,
        runs_dir=runs_dir,
        run_name=args.name,
        epochs=args.epochs,
        image_size=args.imgsz,
        batch_size=args.batch,
        patience=args.patience,
        device=device,
        workers=args.workers,
        optimizer=args.optimizer,
        learning_rate=args.lr,
    )

    print()
    print("=== Training selesai ===")
    print(f"Run dir  : {artifact.run_dir}")
    if artifact.best_model_path:
        print(f"Best     : {artifact.best_model_path}")
    if artifact.last_model_path:
        print(f"Last     : {artifact.last_model_path}")
    if artifact.results_path:
        print(f"Results  : {artifact.results_path}")


if __name__ == "__main__":
    main()
