"""
Generador de descriptores (tp2.2).

Abre la webcam, detecta contornos y al presionar ESPACIO imprime en
terminal los invariantes de Hu del contorno más grande detectado.

Workflow sugerido:
  1. Mostrar un objeto frente a la cámara.
  2. Presionar ESPACIO para capturar sus invariantes de Hu.
  3. Copiar los valores impresos en la terminal y pegarlos en
     data/hu_moments.csv junto con la etiqueta correspondiente.
  4. Repetir para todas las formas deseadas.

Diccionario de etiquetas (labels.py):
  1 = circle
  2 = rectangle
  3 = star

Teclas:
  ESPACIO - imprimir invariantes de Hu del contorno mas grande
  q       - salir
"""
from __future__ import annotations

import cv2
import numpy as np

from labels import LABELS


THRESHOLD_VALUE = 140
MIN_AREA = 400
MORPH_KERNEL_SIZE = 3

CONTROL_WINDOW = "Controles"
VIDEO_WINDOW = "Generador de descriptores"
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


def hu_moments(contour: np.ndarray) -> list[float]:
    # Computes the 7 Hu moment invariants for a contour and returns them as a flat list.
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    return hu.tolist()


def main() -> None:
    # Opens the webcam and runs the descriptor capture loop.
    # Each time SPACE is pressed, prints the 7 Hu invariants of the largest detected contour
    # in CSV format so they can be copied into data/hu_moments.csv with the correct label.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam.")

    print("=" * 60)
    print("Generador de descriptores activo.")
    print("Diccionario de etiquetas:")
    for k, v in LABELS.items():
        print(f"  {k}: {v}")
    print()
    print("ESPACIO: capturar invariantes de Hu | q: salir")
    print("=" * 60)

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

            cv2.putText(
                canvas,
                f"Contornos: {len(contours)} | ESPACIO: capturar | q: salir",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(VIDEO_WINDOW, canvas)
            cv2.imshow(MASK_WINDOW, binary)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    hu = hu_moments(largest)
                    print(f"{hu[0]:.8e},{hu[1]:.8e},{hu[2]:.8e},{hu[3]:.8e},"
                          f"{hu[4]:.8e},{hu[5]:.8e},{hu[6]:.8e},<ETIQUETA>")
                else:
                    print("No se detecto ningun contorno.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
