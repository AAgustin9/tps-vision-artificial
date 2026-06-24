from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "pklot_raw"
DATA_YOLO = ROOT / "data" / "yolo"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "parking_yolo.pt"
RESULTS_DIR = ROOT / "results"

CLASS_NAMES = ["empty", "occupied"]
LABELS = {0: "Empty", 1: "Occupied"}

# Colors: (R, G, B) for PIL, (B, G, R) for OpenCV
COLOR_FREE_RGB = (34, 197, 94)    # #22c55e
COLOR_OCCUPIED_RGB = (239, 68, 68) # #ef4444
COLOR_FREE_BGR = (94, 197, 34)
COLOR_OCCUPIED_BGR = (68, 68, 239)


def ensure_dirs() -> None:
    for d in [MODELS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_image_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))
