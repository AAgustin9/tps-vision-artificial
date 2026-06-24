"""
Trains a YOLOv8 object detector on the converted PKLot dataset.

Localizes parking spaces directly from a full image and classifies each as
"empty" or "occupied" in a single pass -- no manual ROI calibration needed.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import DATA_YOLO, MODEL_PATH, ensure_dirs


def main():
    from ultralytics import YOLO

    ensure_dirs()

    data_yaml = DATA_YOLO / "data.yaml"
    if not data_yaml.exists():
        print(f"ERROR: {data_yaml} not found. Run src/convert_to_yolo.py first.")
        return

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=40,
        imgsz=640,
        batch=16,
        patience=10,
        project="results/yolo_runs",
        name="parking",
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy2(best_weights, MODEL_PATH)
    print(f"\nBest weights copied to {MODEL_PATH}")

    print("\nValidating on test split...")
    metrics = model.val(data=str(data_yaml), split="test")
    print(f"mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
