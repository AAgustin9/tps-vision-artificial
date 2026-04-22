"""
Capturador de imagenes de entrenamiento (tp2.2).

Abre la webcam. Selecciona la clase con 1/2/3, luego presiona
ESPACIO para guardar el frame actual en data/shapes/<label>/.

Workflow:
  1. Mostrar un objeto frente a la camara.
  2. Presionar 1, 2 o 3 para seleccionar la clase.
  3. Presionar ESPACIO para guardar el frame.
  4. Repetir para todas las formas deseadas.
  5. Ejecutar create_dataset.py y train.py.

Teclas:
  1/2/3   - seleccionar clase (circle/rectangle/star)
  ESPACIO - guardar frame en data/shapes/<label>/
  q       - salir
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from labels import LABELS


THRESHOLD_VALUE = 140
MIN_AREA = 400
MORPH_KERNEL_SIZE = 3

CONTROL_WINDOW = "Controles"
VIDEO_WINDOW = "Capturador de entrenamiento"
MASK_WINDOW = "Mascara binaria"

SHAPES_DIR = Path("data/shapes")


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


def count_saved(label: str) -> int:
    # Returns how many PNG images are already saved for a given label (class).
    d = SHAPES_DIR / label
    if not d.exists():
        return 0
    return len(list(d.glob("*.png")))


def main() -> None:
    # Opens the webcam so the user can capture training images.
    # Press 1/2/3 to select the active class (circle/rectangle/star),
    # SPACE to save the current frame to data/shapes/<label>/, and q to quit.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam.")

    print("=" * 60)
    print("Capturador de imagenes de entrenamiento.")
    print("Clases disponibles:")
    for k, v in LABELS.items():
        print(f"  {k}: {v}")
    print()
    print("1/2/3: seleccionar clase | ESPACIO: guardar | q: salir")
    print("=" * 60)

    active_label_id: int = 1

    cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW, 440, 160)
    cv2.createTrackbar("Threshold", CONTROL_WINDOW, THRESHOLD_VALUE, 255, lambda _: None)
    cv2.createTrackbar("Kernel", CONTROL_WINDOW, MORPH_KERNEL_SIZE, 25, lambda _: None)
    cv2.createTrackbar("Min area", CONTROL_WINDOW, MIN_AREA, 10000, lambda _: None)

    cv2.namedWindow(VIDEO_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)

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
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)

            active_name = LABELS.get(active_label_id, "?")
            saved = count_saved(active_name)
            cv2.putText(
                canvas,
                f"Clase: {active_name} ({saved} guardadas) | "
                f"1/2/3: cambiar | ESPACIO: guardar | q: salir",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(VIDEO_WINDOW, canvas)
            cv2.imshow(MASK_WINDOW, binary)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("1"), ord("2"), ord("3")):
                active_label_id = key - ord("0")
                print(f"Clase activa: {LABELS[active_label_id]}")
            if key == ord(" "):
                label_name = LABELS[active_label_id]
                out_dir = SHAPES_DIR / label_name
                out_dir.mkdir(parents=True, exist_ok=True)
                idx = count_saved(label_name)
                out_path = out_dir / f"{label_name}_{idx:03d}.png"
                cv2.imwrite(str(out_path), frame)
                print(f"Guardado: {out_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
