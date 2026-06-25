"""Script de calibracion: marca manualmente las esquinas de cada espacio
de estacionamiento sobre una imagen de referencia y guarda sus coordenadas
en un archivo JSON (spots.json). Se corre una sola vez por layout de camara."""
import json


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
