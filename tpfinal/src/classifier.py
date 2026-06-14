"""Inference wrapper for the parking space binary classifier."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from utils import MODEL_PATH, IMG_SIZE


class ParkingClassifier:
    def __init__(self, model_path: Path = MODEL_PATH, threshold: float = 0.7):
        self.model_path = model_path
        self.threshold = threshold
        self._model = None
        self._load_model()

    def _load_model(self):
        import tensorflow as tf
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please run `python src/train.py` first."
            )
        self._model = tf.keras.models.load_model(str(self.model_path))

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        from PIL import Image
        img = Image.fromarray(crop.astype(np.uint8)).resize(IMG_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    def predict(self, crop: np.ndarray) -> tuple:
        """Predict a single crop. Returns (label, confidence)."""
        arr = self._preprocess(crop)
        batch = np.expand_dims(arr, 0)
        prob = float(self._model.predict(batch, verbose=0)[0][0])
        if prob >= self.threshold:
            return ("Occupied", prob)
        return ("Empty", 1.0 - prob)

    def predict_batch(self, crops: list) -> list:
        """Predict a list of crops. Returns list of (label, confidence)."""
        if not crops:
            return []
        arrays = np.stack([self._preprocess(c) for c in crops])
        probs = self._model.predict(arrays, verbose=0).flatten()
        results = []
        for prob in probs:
            if prob >= self.threshold:
                results.append(("Occupied", float(prob)))
            else:
                results.append(("Empty", float(1.0 - prob)))
        return results


_instance: "ParkingClassifier | None" = None


def get_classifier(threshold: float = 0.7) -> ParkingClassifier:
    global _instance
    if _instance is None or _instance.threshold != threshold:
        _instance = ParkingClassifier(threshold=threshold)
    return _instance
