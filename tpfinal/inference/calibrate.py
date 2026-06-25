"""Script de calibracion: marca manualmente las esquinas de cada espacio
de estacionamiento sobre una imagen de referencia y guarda sus coordenadas
en un archivo JSON (spots.json). Se corre una sola vez por layout de camara."""
import argparse
import json
import sys

import cv2
import numpy as np


class SpotCalibrator:
    """Acumula los clicks del usuario y los agrupa en espacios de 4 puntos."""

    def __init__(self):
        self.pending_points = []
        self.spots = []
        self._next_id = 1

    def on_mouse_click(self, x, y):
        """Registra un click. Al llegar al 4to punto, cierra el espacio actual."""
        self.pending_points.append([x, y])
        if len(self.pending_points) == 4:
            spot_id = f"spot_{self._next_id}"
            self.spots.append({"id": spot_id, "points": self.pending_points})
            self._next_id += 1
            self.pending_points = []

    def undo(self):
        """Deshace el ultimo punto pendiente, o el ultimo espacio ya cerrado."""
        if self.pending_points:
            self.pending_points.pop()
        elif self.spots:
            self.spots.pop()
            self._next_id -= 1

    def to_dict(self):
        return {"spots": self.spots}

    def save(self, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def _draw_overlay(image, calibrator):
    """Dibuja los espacios ya cerrados (amarillo) y los puntos pendientes (circulos)."""
    overlay = image.copy()
    for spot in calibrator.spots:
        points = np.array(spot["points"], dtype=np.int32)
        cv2.polylines(overlay, [points], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.putText(
            overlay, spot["id"], tuple(points[0]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
        )
    for point in calibrator.pending_points:
        cv2.circle(overlay, tuple(point), 4, (255, 0, 0), -1)
    return overlay


def _run_window_loop(image, calibrator, window_name="Calibracion de espacios"):
    """Bucle principal de la ventana de calibracion.

    Teclas: 'u' deshacer, 's' guardar y salir, 'q'/ESC salir sin guardar.
    Devuelve True si el usuario pidio guardar, False si salio sin guardar.
    """
    cv2.namedWindow(window_name)

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            calibrator.on_mouse_click(x, y)

    cv2.setMouseCallback(window_name, _on_mouse)

    while True:
        cv2.imshow(window_name, _draw_overlay(image, calibrator))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("u"):
            calibrator.undo()
        elif key == ord("s"):
            cv2.destroyAllWindows()
            return True
        elif key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Marca manualmente los espacios de estacionamiento de una imagen de referencia."
    )
    parser.add_argument("--image", required=True, help="Ruta a la imagen de referencia del estacionamiento")
    parser.add_argument("--output", default="spots.json", help="Ruta del archivo JSON de salida")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: no se pudo leer la imagen '{args.image}'", file=sys.stderr)
        sys.exit(1)

    calibrator = SpotCalibrator()
    saved = _run_window_loop(image, calibrator)

    if saved:
        calibrator.save(args.output)
        print(f"Se guardaron {len(calibrator.spots)} espacios en '{args.output}'")
    else:
        print("Calibracion descartada, no se guardo ningun archivo.")


if __name__ == "__main__":
    main()
