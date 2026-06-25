"""Script principal de inferencia: clasifica cada espacio de estacionamiento
de una imagen nueva como libre/ocupado usando el modelo entrenado en Colab,
y dibuja el resultado sobre la imagen."""
import json

import cv2
import numpy as np


def load_spots(spots_path):
    """Carga la lista de espacios calibrados desde spots.json."""
    with open(spots_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["spots"]


def order_points(points):
    """Ordena 4 puntos arbitrarios como top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = pts[:, 0] - pts[:, 1]

    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmax(diffs)]
    bottom_left = pts[np.argmin(diffs)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def crop_spot(image, points, output_size=(224, 224)):
    """Recorta y endereza el cuadrilatero de un espacio via warpPerspective."""
    width, height = output_size
    src = order_points(points)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, transform, (width, height))


def preprocess_crop(crop):
    """Convierte un crop BGR uint8 en el batch float32 normalizado que espera el modelo."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)
