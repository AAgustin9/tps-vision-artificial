# Parking Space Occupancy Detector

A Computer Vision system for detecting free and occupied parking spaces using transfer learning on the PKLot dataset.

## Academic Context

- **Dataset**: PKLot — standard CV benchmark with ~700k annotated space images across 3 lots and 3 weather conditions
- **Model**: MobileNetV2 pretrained on ImageNet (transfer learning)
- **Why transfer learning**: MobileNetV2 provides robust low-level feature detectors (edges, textures) that generalize to parking space appearance. Training from scratch on 4,000 images would overfit severely
- **Metrics**: accuracy, precision, recall, F1-score (binary), confusion matrix — reported per class
- **Citation**: De Almeida, P.R.L. et al. "PKLot – A robust dataset for parking lot classification." Expert Systems with Applications, 2015

## Setup

```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Download PKLot from Kaggle: https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset
2. Place extracted folder at `data/pklot_raw/`
3. Run: `python src/prepare_dataset.py`

## Training

```bash
python src/train.py
```

## Run the App

```bash
streamlit run app.py
```

## Manual Lot Configuration

To define custom parking space coordinates for a new lot:

```bash
python src/space_selector.py --image <path_to_image> --lot-name <name>
```

## File Structure

```
tpfinal/
├── data/
│   ├── pklot_raw/          # PKLot dataset (place here)
│   └── processed/
│       ├── empty/          # Cropped empty space images
│       └── occupied/       # Cropped occupied space images
├── models/
│   └── parking_classifier.h5
├── configs/
│   └── lot_configs.json
├── results/
│   ├── training_curves.png
│   └── confusion_matrix.png
├── src/
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── classifier.py
│   ├── space_selector.py
│   ├── detector.py
│   └── utils.py
├── app.py
├── requirements.txt
└── README.md
```

## Model Architecture

- Base: MobileNetV2 (pretrained ImageNet, frozen during phase 1)
- Head: GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
- Phase 2 fine-tuning: last 20 layers of base unfrozen, Adam(1e-5)
- Input: 64×64 RGB crops
- Output: probability of "Occupied" class

## Workflow

### 1. Prepare Dataset

Extract and normalize space crops from the PKLot dataset:

```bash
python src/prepare_dataset.py
```

This creates balanced train/val/test splits with ~2000 empty and ~2000 occupied crops.

### 2. Train Model

Train the binary classifier with transfer learning:

```bash
python src/train.py
```

The model is trained in two phases:
- **Phase 1**: Fine-tune head only (30 epochs, base frozen)
- **Phase 2**: Unfreeze last 20 base layers and fine-tune (10 epochs)

Evaluation metrics and confusion matrix are saved to `results/`.

### 3. Define Lot Configurations

For a new parking lot, use the interactive space selector:

```bash
python src/space_selector.py --image parking_lot.jpg --lot-name "My Lot"
```

- Left-click drag: draw a parking space bounding box
- Right-click: undo last space
- Q: save and quit
- R: reset all spaces

Configurations are saved to `configs/lot_configs.json`.

### 4. Run Detection

**Streamlit Web App** (interactive + live camera support):

```bash
streamlit run app.py
```

**Command-line** (single image):

```bash
python src/detector.py --image test.jpg --lot-name PUCPR_demo --output result.jpg
```

## Performance

With the PKLot dataset (70/15/15 train/val/test split):

- **Accuracy**: ~92–96%
- **Precision**: 91–97% (per class)
- **Recall**: 91–97% (per class)
- **F1-Score**: 91–97% (per class)

Performance varies by lot, weather condition, and image quality.

## API Reference

### `ParkingClassifier`

```python
from src.classifier import get_classifier

classifier = get_classifier(threshold=0.7)
label, confidence = classifier.predict(crop)  # Single crop
results = classifier.predict_batch(crops)      # List of crops
```

### `run_detection`

```python
from src.detector import run_detection
from pathlib import Path

annotated_img, stats = run_detection(
    image_path=Path("parking.jpg"),
    lot_name="PUCPR_demo",
    threshold=0.7,
    show_confidence=True
)

print(stats)
# Output: {'total': 20, 'free': 12, 'occupied': 8, 'occupancy_pct': 40.0}
```

## Notes

- Requires Python 3.8+
- TensorFlow 2.12+ with GPU support recommended for training
- Camera mode requires `opencv-python` (headless version used for inference)
- Default lot configurations (PUCPR_demo, UFPR_demo) are for testing only
