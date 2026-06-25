# Parking Space Occupancy Detector

A Computer Vision system that automatically detects and locates free/occupied parking spaces in an image, using a YOLOv8 object detector trained on the PKLot dataset.

## Academic Context

- **Dataset**: PKLot — standard CV benchmark with annotated parking space images across 3 lots and 3 weather conditions, here used via a Roboflow COCO export with `space-empty` / `space-occupied` bounding boxes
- **Model**: YOLOv8n fine-tuned on the PKLot space annotations — localizes spaces and classifies occupancy in a single forward pass, no manual ROI calibration needed
- **Citation**: De Almeida, P.R.L. et al. "PKLot – A robust dataset for parking lot classification." Expert Systems with Applications, 2015

## Setup

```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Get the PKLot Roboflow COCO export and extract it to `data/pklot_raw/` (expects `train/`, `valid/`, `test/` subfolders, each with images + `_annotations.coco.json`)
2. Convert to YOLO format: `python src/convert_to_yolo.py`

## Training

```bash
python src/train_yolo.py
```

Training is GPU-intensive — see `COLAB_GUIDE.md` for running it on a free Colab T4.

## Run the App

```bash
streamlit run app.py
```

## File Structure

```
tpfinal/
├── data/
│   ├── pklot_raw/          # Roboflow COCO export of PKLot (place here)
│   └── yolo/                # Converted YOLO-format dataset
├── models/
│   └── parking_yolo.pt
├── results/
├── src/
│   ├── convert_to_yolo.py
│   ├── train_yolo.py
│   ├── detector.py
│   └── utils.py
├── app.py
├── requirements.txt
└── README.md
```

## Workflow

### 1. Convert Dataset

```bash
python src/convert_to_yolo.py
```

### 2. Train Model

```bash
python src/train_yolo.py
```

Fine-tunes YOLOv8n for up to 50 epochs (early stopping on validation loss). Saves the best weights to `models/parking_yolo.pt` and reports mAP50/mAP50-95 on the test split.

### 3. Run Detection

**Streamlit Web App** (interactive + live camera support):

```bash
streamlit run app.py
```

**Command-line** (single image):

```bash
python src/detector.py --image test.jpg --output result.jpg
```

## API Reference

### `detect`

```python
from src.detector import detect
import numpy as np
from PIL import Image

img = np.array(Image.open("parking.jpg").convert("RGB"))
detections = detect(img, threshold=0.5)
# [{'x': 12, 'y': 34, 'w': 60, 'h': 50, 'label': 'Empty', 'confidence': 0.91}, ...]
```

### `run_detection`

```python
from src.detector import run_detection
from pathlib import Path

annotated_img, stats = run_detection(Path("parking.jpg"), threshold=0.5)
print(stats)
# Output: {'total': 20, 'free': 12, 'occupied': 8, 'occupancy_pct': 40.0}
```

## Notes

- Requires Python 3.9–3.12 (Ultralytics/PyTorch do not yet support 3.13+)
- A GPU speeds up training significantly; CPU is fine for local inference on single images/webcam frames
- Camera mode requires `opencv-python` (headless version used for inference)
- The app supports two workflows: `Automatic` line/grid detection for simple top-down lots, and `Manual Grid` calibration for reliable fixed-camera/difficult images. Manual Grid lets you define the parking area rows/columns, then the app classifies each calibrated slot as `Empty` or `Occupied` using pretrained YOLO (`yolov8n.pt`) plus visual texture/contrast. The old trained `models/parking_yolo.pt` model remains as a fallback when automatic slot lines cannot be found.
