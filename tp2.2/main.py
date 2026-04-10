"""
Clasificador (tp2.2).

Abre la webcam, detecta contornos y los clasifica usando el modelo
entrenado (models/shape_classifier.joblib) en lugar de matchShapes.

Requiere haber ejecutado previamente:
  1. python create_dataset.py   (o generar data/hu_moments.csv manualmente)
  2. python train.py

Teclas:
  q - salir
  s - guardar frame anotado en output/
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from joblib import load

from labels import LABELS


MODEL_PATH = Path("models/shape_classifier.joblib")

COLORS: dict[str, tuple[int, int, int]] = {
    "circle": (0, 200, 255),
    "rectangle": (0, 220, 0),
    "star": (255, 160, 0),
    "unknown": (0, 0, 255),
}

THRESHOLD_VALUE = 140
MIN_AREA = 400
MORPH_KERNEL_SIZE = 3

CONTROL_WINDOW = "Controles"
ANNOTATED_WINDOW = "Webcam clasificada (ML)"
MASK_WINDOW = "Mascara binaria"


def preprocess(frame: np.ndarray, threshold: int, kernel_size: int) -> np.ndarray:
    # Converts frame to grayscale, blurs it, applies inverse binary threshold,
    # and runs morphological close+open to produce a clean binary mask.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    k = max(1, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    kernel = np.ones((k, k), dtype=np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def find_contours(binary: np.ndarray, min_area: int) -> list[np.ndarray]:
    # Returns all external contours in the binary mask whose area is at least min_area.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def compute_hu(contour: np.ndarray) -> list[float]:
    # Computes the 7 Hu moment invariants for a contour and returns them as a flat list.
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    return hu.tolist()


def draw_detection(
    canvas: np.ndarray,
    contour: np.ndarray,
    label: str,
    label_id: int,
) -> None:
    # Draws a colored bounding box and label caption for a detected contour on canvas.
    # Color is determined by the label name; unknown shapes get red.
    x, y, w, h = cv2.boundingRect(contour)
    color = COLORS.get(label, (255, 255, 255))
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
    caption = f"{label} ({label_id})"
    cv2.putText(
        canvas,
        caption,
        (x, max(18, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    # Loads the trained model, opens the webcam, and runs the classification loop in real time.
    # For each detected contour, computes Hu moments, predicts the label with the ML model,
    # and draws the result. Press q to quit, s to save the current frame.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado en {MODEL_PATH}.\n"
            "Ejecuta primero:\n"
            "  python create_dataset.py\n"
            "  python train.py"
        )

    clf = load(MODEL_PATH)
    print(f"Modelo cargado desde {MODEL_PATH}")
    print("Etiquetas:", {k: v for k, v in LABELS.items()})
    print("Teclas: q=salir, s=guardar frame")

    cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW, 440, 160)
    cv2.createTrackbar("Threshold", CONTROL_WINDOW, THRESHOLD_VALUE, 255, lambda _: None)
    cv2.createTrackbar("Kernel", CONTROL_WINDOW, MORPH_KERNEL_SIZE, 25, lambda _: None)
    cv2.createTrackbar("Min area", CONTROL_WINDOW, MIN_AREA, 10000, lambda _: None)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam.")

    cv2.namedWindow(ANNOTATED_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)

    output_dir = Path("output")
    saved_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            threshold = max(0, cv2.getTrackbarPos("Threshold", CONTROL_WINDOW))
            kernel_size = cv2.getTrackbarPos("Kernel", CONTROL_WINDOW)
            min_area = max(1, cv2.getTrackbarPos("Min area", CONTROL_WINDOW))

            binary = preprocess(frame, threshold, kernel_size)
            contours = find_contours(binary, min_area)

            canvas = frame.copy()
            for contour in contours:
                hu = compute_hu(contour)
                label_id = int(clf.predict([hu])[0])
                label = LABELS.get(label_id, "unknown")
                draw_detection(canvas, contour, label, label_id)

            cv2.putText(
                canvas,
                f"thr={threshold} kernel={kernel_size} min_area={min_area}"
                f"  |  q: salir  s: guardar",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(ANNOTATED_WINDOW, canvas)
            cv2.imshow(MASK_WINDOW, binary)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                output_dir.mkdir(parents=True, exist_ok=True)
                frame_path = output_dir / f"frame_{saved_frames:03d}.png"
                mask_path = output_dir / f"frame_{saved_frames:03d}_mask.png"
                cv2.imwrite(str(frame_path), canvas)
                cv2.imwrite(str(mask_path), binary)
                saved_frames += 1
                print(f"Frame guardado en {frame_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
