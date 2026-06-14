from pathlib import Path
import json
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "pklot_raw"
DATA_PROCESSED = ROOT / "data" / "processed"
EMPTY_DIR = DATA_PROCESSED / "empty"
OCCUPIED_DIR = DATA_PROCESSED / "occupied"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "parking_classifier.h5"
CONFIGS_DIR = ROOT / "configs"
LOT_CONFIG_PATH = CONFIGS_DIR / "lot_configs.json"
RESULTS_DIR = ROOT / "results"

IMG_SIZE = (64, 64)
LABELS = {0: "Empty", 1: "Occupied"}

# Colors: (R, G, B) for PIL, (B, G, R) for OpenCV
COLOR_FREE_RGB = (34, 197, 94)    # #22c55e
COLOR_OCCUPIED_RGB = (239, 68, 68) # #ef4444
COLOR_FREE_BGR = (94, 197, 34)
COLOR_OCCUPIED_BGR = (68, 68, 239)

DEFAULT_CONFIGS = {
    "PUCPR_demo": {
        "description": "PUCPR lot demo (approximate geometry for testing)",
        "spaces": [
            {"id": i, "x": 50 + (i % 10) * 65, "y": 80 + (i // 10) * 55, "w": 60, "h": 50}
            for i in range(1, 21)
        ]
    },
    "UFPR_demo": {
        "description": "UFPR lot demo (approximate geometry for testing)",
        "spaces": [
            {"id": i, "x": 30 + (i % 8) * 75, "y": 60 + (i // 8) * 60, "w": 70, "h": 55}
            for i in range(1, 17)
        ]
    }
}


def ensure_dirs() -> None:
    for d in [EMPTY_DIR, OCCUPIED_DIR, MODELS_DIR, CONFIGS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    if not LOT_CONFIG_PATH.exists():
        with open(LOT_CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIGS, f, indent=2)


def load_image_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def resize_crop(crop: np.ndarray, size=IMG_SIZE) -> np.ndarray:
    img = Image.fromarray(crop).resize(size, Image.LANCZOS)
    return np.array(img)
