# PKLot Parking Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-part PKLot parking-spot detection project: a Colab notebook that trains a MobileNetV2 binary classifier (free/occupied) on PKLot patches, and local CLI scripts that calibrate spot coordinates once per camera layout and then classify+annotate new parking-lot photos.

**Architecture:** `/training` holds a single self-contained `.ipynb` for Colab (no local dependency). `/inference` holds two independent CLI scripts (`calibrate.py`, `detect.py`) built from small, unit-testable pure functions (coordinate math, cropping, classification, drawing) wired together by thin `main()`/CLI glue that is the only part touching real OpenCV windows or TensorFlow model loading — that glue is tested by monkeypatching the I/O boundaries.

**Tech Stack:** TensorFlow/Keras + MobileNetV2 (Colab, training only), OpenCV (`opencv-python`) + NumPy + TensorFlow-CPU (local inference), pytest for local script tests.

## Global Constraints

- Comentarios de código en español; nombres de variables/funciones en inglés (spec section "Notas generales").
- `/training` y `/inference` son carpetas hermanas en la raíz del repo (decisión confirmada con el usuario).
- Los archivos del enfoque YOLOv8 anterior (`app.py`, `src/*`, `README.md`, `requirements.txt`, `COLAB_GUIDE.md`) se eliminan (`git rm`) como parte de este trabajo (confirmado con el usuario).
- `detect.py` y `calibrate.py` deben poder correr sin GPU.
- El modelo `.h5` real y el `spots.json` real de cada layout NO se versionan en git; solo se versiona `spots.json.example`.
- `inference/requirements.txt` solo incluye dependencias estrictamente necesarias para correr `calibrate.py`/`detect.py` en producción (pytest va en un `requirements-dev.txt` separado, usado solo para los tests de este plan).
- El notebook no asume una única estructura de carpetas de PKLot; debe explorarla primero.

---

### Task 1: Limpieza del repo y scaffolding de carpetas

**Files:**
- Delete (git rm): `app.py`, `requirements.txt`, `README.md`, `COLAB_GUIDE.md`, `src/convert_to_yolo.py`, `src/detector.py`, `src/train_yolo.py`, `src/utils.py`
- Create: `.gitignore`
- Create: `training/.gitkeep` (placeholder, removed once notebook exists in Task 7)
- Create: `inference/.gitkeep` (placeholder, removed once scripts exist in Task 3/6)

**Interfaces:**
- Produces: clean repo root with empty `training/` and `inference/` directories ready for subsequent tasks.

- [ ] **Step 1: Confirm the deleted files and stage the deletion**

```bash
git -C /Users/agussoul/projects/ua/va/tps/tpfinal status
```

Expected: same 8 files listed as "deleted" that were shown at the start of this conversation (`app.py`, `requirements.txt`, `README.md`, `COLAB_GUIDE.md`, `src/convert_to_yolo.py`, `src/detector.py`, `src/train_yolo.py`, `src/utils.py`).

```bash
git -C /Users/agussoul/projects/ua/va/tps/tpfinal rm app.py requirements.txt README.md COLAB_GUIDE.md src/convert_to_yolo.py src/detector.py src/train_yolo.py src/utils.py
```

Expected: `rm 'app.py'` etc. printed for each file; no errors. If `src/` becomes empty, git will not track the directory itself, which is fine — it will be removed.

- [ ] **Step 2: Create the new directory skeletons**

```bash
mkdir -p /Users/agussoul/projects/ua/va/tps/tpfinal/training /Users/agussoul/projects/ua/va/tps/tpfinal/inference
touch /Users/agussoul/projects/ua/va/tps/tpfinal/training/.gitkeep /Users/agussoul/projects/ua/va/tps/tpfinal/inference/.gitkeep
```

- [ ] **Step 3: Create `.gitignore`**

```
# Modelos entrenados (se descargan de Drive, no se versionan)
*.h5
*.keras
saved_model/

# Calibración real de un layout específico (solo se versiona spots.json.example)
inference/spots.json

# Python
__pycache__/
*.pyc
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# Sistema
.DS_Store
```

Write this to `/Users/agussoul/projects/ua/va/tps/tpfinal/.gitignore`.

- [ ] **Step 4: Verify the resulting tree**

```bash
git -C /Users/agussoul/projects/ua/va/tps/tpfinal status
find /Users/agussoul/projects/ua/va/tps/tpfinal -maxdepth 2 -not -path '*/.git*'
```

Expected: `git status` shows the 8 deletions staged plus `.gitignore`, `training/.gitkeep`, `inference/.gitkeep` as new files. `find` shows `training/` and `inference/` directories present.

- [ ] **Step 5: Commit**

```bash
git -C /Users/agussoul/projects/ua/va/tps/tpfinal add -A
git -C /Users/agussoul/projects/ua/va/tps/tpfinal commit -m "Remove YOLOv8 approach, scaffold training/ and inference/ folders"
```

---

### Task 2: `calibrate.py` — lógica núcleo de calibración (`SpotCalibrator`)

**Files:**
- Create: `inference/calibrate.py`
- Create: `inference/tests/test_calibrate.py`
- Create: `inference/requirements-dev.txt`

**Interfaces:**
- Produces: `class SpotCalibrator` with methods `on_mouse_click(self, x: int, y: int) -> None`, `undo(self) -> None`, `to_dict(self) -> dict`, `save(self, output_path: str) -> None`. Spot dict shape: `{"id": str, "points": list[list[int]]}` (exactly 4 `[x, y]` pairs per spot). `to_dict()` returns `{"spots": [...]}`.

- [ ] **Step 1: Write `requirements-dev.txt`**

```
pytest==8.3.3
```

Write to `/Users/agussoul/projects/ua/va/tps/tpfinal/inference/requirements-dev.txt`.

- [ ] **Step 2: Write the failing tests**

Create `/Users/agussoul/projects/ua/va/tps/tpfinal/inference/tests/test_calibrate.py`:

```python
import json

from calibrate import SpotCalibrator


def test_four_clicks_create_one_spot_with_incremental_id():
    calibrator = SpotCalibrator()
    for point in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        calibrator.on_mouse_click(*point)

    assert calibrator.pending_points == []
    assert len(calibrator.spots) == 1
    assert calibrator.spots[0]["id"] == "spot_1"
    assert calibrator.spots[0]["points"] == [[0, 0], [10, 0], [10, 10], [0, 10]]


def test_second_spot_gets_incremental_id():
    calibrator = SpotCalibrator()
    for point in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        calibrator.on_mouse_click(*point)
    for point in [(20, 20), (30, 20), (30, 30), (20, 30)]:
        calibrator.on_mouse_click(*point)

    assert len(calibrator.spots) == 2
    assert calibrator.spots[1]["id"] == "spot_2"


def test_undo_removes_pending_point_before_spot_is_complete():
    calibrator = SpotCalibrator()
    calibrator.on_mouse_click(0, 0)
    calibrator.on_mouse_click(10, 0)
    calibrator.undo()

    assert calibrator.pending_points == [[0, 0]]
    assert calibrator.spots == []


def test_undo_removes_last_completed_spot_when_no_pending_points():
    calibrator = SpotCalibrator()
    for point in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        calibrator.on_mouse_click(*point)
    calibrator.undo()

    assert calibrator.spots == []

    calibrator.on_mouse_click(0, 0)
    calibrator.on_mouse_click(10, 0)
    calibrator.on_mouse_click(10, 10)
    calibrator.on_mouse_click(0, 10)
    assert calibrator.spots[0]["id"] == "spot_1"


def test_to_dict_returns_spots_wrapper():
    calibrator = SpotCalibrator()
    for point in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        calibrator.on_mouse_click(*point)

    assert calibrator.to_dict() == {
        "spots": [{"id": "spot_1", "points": [[0, 0], [10, 0], [10, 10], [0, 10]]}]
    }


def test_save_writes_json_file(tmp_path):
    calibrator = SpotCalibrator()
    for point in [(0, 0), (10, 0), (10, 10), (0, 10)]:
        calibrator.on_mouse_click(*point)

    output_path = tmp_path / "spots.json"
    calibrator.save(str(output_path))

    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data == calibrator.to_dict()
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /Users/agussoul/projects/ua/va/tps/tpfinal/inference && python -m pytest tests/test_calibrate.py -v
```

Expected: `ModuleNotFoundError: No module named 'calibrate'` (file doesn't exist yet).

- [ ] **Step 4: Implement `SpotCalibrator` in `calibrate.py`**

Create `/Users/agussoul/projects/ua/va/tps/tpfinal/inference/calibrate.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/agussoul/projects/ua/va/tps/tpfinal/inference && python -m pytest tests/test_calibrate.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git -C /Users/agussoul/projects/ua/va/tps/tpfinal add inference/calibrate.py inference/tests/test_calibrate.py inference/requirements-dev.txt
git -C /Users/agussoul/projects/ua/va/tps/tpfinal commit -m "Add SpotCalibrator core logic for spot calibration"
```

---

### Task 3: `calibrate.py` — bucle de ventana OpenCV y CLI

**Files:**
- Modify: `inference/calibrate.py`
- Modify: `inference/tests/test_calibrate.py`

**Interfaces:**
- Consumes: `SpotCalibrator` from Task 2 (`on_mouse_click`, `undo`, `save`).
- Produces: `_draw_overlay(image, calibrator) -> np.ndarray`, `_run_window_loop(image, calibrator, window_name="Calibracion de espacios") -> bool` (returns `True` if user saved, `False` if quit without saving), `parse_args(argv=None) -> argparse.Namespace` (flags `--image`, `--output` default `spots.json`), `main(argv=None) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `/Users/agussoul/projects/ua/va/tps/tpfinal/inference/tests/test_calibrate.py`:

```python
import numpy as np
import cv2

import calibrate
from calibrate import SpotCalibrator, _run_window_loop, parse_args, main


def test_parse_args_defaults_output_to_spots_json():
    args = parse_args(["--image", "ref.jpg"])
    assert args.image == "ref.jpg"
    assert args.output == "spots.json"


def test_run_window_loop_returns_false_on_quit_key(monkeypatch):
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("q"))

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    saved = _run_window_loop(image, SpotCalibrator())

    assert saved is False


def test_run_window_loop_returns_true_on_save_key(monkeypatch):
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("s"))

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    saved = _run_window_loop(image, SpotCalibrator())

    assert saved is True


def test_run_window_loop_undo_key_does_not_exit(monkeypatch):
    keys = iter([ord("u"), ord("q")])
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: next(keys))

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    saved = _run_window_loop(image, SpotCalibrator())

    assert saved is False


def test_main_does_not_save_when_quitting(tmp_path, monkeypatch):
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    monkeypatch.setattr(cv2, "imread", lambda path: image)
    monkeypatch.setattr(calibrate, "_run_window_loop", lambda img, cal: False)

    output_path = tmp_path / "spots.json"
    main(["--image", "ref.jpg", "--output", str(output_path)])

    assert not output_path.exists()


def test_main_saves_when_run_window_loop_returns_true(tmp_path, monkeypatch):
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    monkeypatch.setattr(cv2, "imread", lambda path: image)

    def fake_loop(img, cal):
        cal.on_mouse_click(0, 0)
        cal.on_mouse_click(10, 0)
        cal.on_mouse_click(10, 10)
        cal.on_mouse_click(0, 10)
        return True

    monkeypatch.setattr(calibrate, "_run_window_loop", fake_loop)

    output_path = tmp_path / "spots.json"
    main(["--image", "ref.jpg", "--output", str(output_path)])

    assert output_path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/agussoul/projects/ua/va/tps/tpfinal/inference && python -m pytest tests/test_calibrate.py -v
```

Expected: `ImportError: cannot import name '_run_window_loop'` (and similar) for the 6 new tests; the 6 Task-2 tests still pass.

- [ ] **Step 3: Implement the window loop and CLI**

Append to `/Users/agussoul/projects/ua/va/tps/tpfinal/inference/calibrate.py`:

```python
import argparse
import sys

import cv2
import numpy as np


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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/agussoul/projects/ua/va/tps/tpfinal/inference && python -m pytest tests/test_calibrate.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git -C /Users/agussoul/projects/ua/va/tps/tpfinal add inference/calibrate.py inference/tests/test_calibrate.py
git -C /Users/agussoul/projects/ua/va/tps/tpfinal commit -m "Add OpenCV window loop and CLI to calibrate.py"
```

---

### Task 4: `detect.py` — geometría y preprocesamiento de patches

**Files:**
- Create: `inference/detect.py`
- Create: `inference/tests/test_detect.py`

**Interfaces:**
- Produces: `load_spots(spots_path: str) -> list[dict]`, `order_points(points: list) -> np.ndarray` (4x2 float32, orden tl/tr/br/bl), `crop_spot(image: np.ndarray, points: list, output_size: tuple[int, int] = (224, 224)) -> np.ndarray`, `preprocess_crop(crop: np.ndarray) -> np.ndarray` (shape `(1, 224, 224, 3)` float32 en `[0, 1]`).

- [ ] **Step 1: Write the failing tests**

Create `/Users/agussoul/projects/ua/va/tps/tpfinal/inference/tests/test_detect.py`:

```python
import json

import numpy as np

from detect import crop_spot, load_spots, order_points, preprocess_crop


def test_load_spots_reads_json_file(tmp_path):
    spots_path = tmp_path / "spots.json"
    spots_path.write_text(json.dumps({"spots": [{"id": "spot_1", "points": [[0, 0], [10, 0], [10, 10], [0, 10]]}]}))

    spots = load_spots(str(spots_path))

    assert spots == [{"id": "spot_1", "points": [[0, 0], [10, 0], [10, 10], [0, 10]]}]


def test_order_points_sorts_into_tl_tr_br_bl():
    # Puntos dados en orden "desordenado": bottom-right, top-left, top-right, bottom-left
    shuffled = [[10, 10], [0, 0], [10, 0], [0, 10]]

    ordered = order_points(shuffled)

    assert ordered.tolist() == [[0, 0], [10, 0], [10, 10], [0, 10]]


def test_crop_spot_returns_image_of_requested_output_size():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:50, 20:50] = 255  # cuadrado blanco donde esta el espacio
    points = [[20, 20], [50, 20], [50, 50], [20, 50]]

    crop = crop_spot(image, points, output_size=(32, 32))

    assert crop.shape == (32, 32, 3)
    assert crop.mean() > 200  # deberia ser mayormente blanco


def test_preprocess_crop_normalizes_and_adds_batch_dim():
    crop = np.full((224, 224, 3), 255, dtype=np.uint8)

    batch = preprocess_crop(crop)

    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert np.allclose(batch, 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/agussoul/projects/ua/va/tps/tpfinal/inference && python -m pytest tests/test_detect.py -v
```

Expected: `ModuleNotFoundError: No module named 'detect'`.

- [ ] **Step 3: Implement the geometry/preprocessing functions**

Create `/Users/agussoul/projects/ua/va/tps/tpfinal/inference/detect.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/agussoul/projects/ua/va/tps/tpfinal/inference && python -m pytest tests/test_detect.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git -C /Users/agussoul/projects/ua/va/tps/tpfinal add inference/detect.py inference/tests/test_detect.py
git -C /Users/agussoul/projects/ua/va/tps/tpfinal commit -m "Add spot geometry and preprocessing functions to detect.py"
```

---

### Task 5: `detect.py` — clasificación, dibujo y resumen

**Files:**
- Modify: `inference/detect.py`
- Modify: `inference/tests/test_detect.py`

**Interfaces:**
- Consumes: `preprocess_crop` from Task 4.
- Produces: `classify_spot(model, crop: np.ndarray, threshold: float = 0.5) -> tuple[bool, float]` (`is_occupied`, `probability`), `build_summary_text(results: list[bool]) -> str`, `draw_results(image: np.ndarray, spots: list[dict], results: list[bool]) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

Append to `/Users/agussoul/projects/ua/va/tps/tpfinal/inference/tests/test_detect.py`:

```python
from detect import build_summary_text, classify_spot, draw_results


class FakeModel:
    def __init__(self, probability):
        self.probability = probability

    def predict(self, batch, verbose=0):
        return np.array([[self.probability]], dtype=np.float32)


def test_classify_spot_below_threshold_is_free():
    crop = np.zeros((224, 224, 3), dtype=np.uint8)

    is_occupied, probability = classify_spot(FakeModel(0.2), crop)

    assert is_occupied is False
    assert probability == 0.2


def test_classify_spot_above_threshold_is_occupied():
    crop = np.zeros((224, 224, 3), dtype=np.uint8)

    is_occupied, probability = classify_spot(FakeModel(0.9), crop)

    assert is_occupied is True
    assert probability == 0.9


def test_build_summary_text_counts_free_spots():
    text = build_summary_text([True, False, False, True])

    assert text == "2/4 espacios libres"


def test_build_summary_text_all_occupied():
    text = build_summary_text([True, True])

    assert text == "0/2 espacios libres"


def test_draw_results_returns_same_shape_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    spots = [{"id": "spot_1", "points": [[10, 10], [40, 10], [40, 40], [10, 40]]}]

    annotated = draw_results(image, spots, [True])

    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)  # algo se dibujo encima
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/agussoul/projects/ua/va/tps/tpfinal/inference && python -m pytest tests/test_detect.py -v
```

Expected: the 4 Task-4 tests pass; the 5 new ones fail with `ImportError: cannot import name 'classify_spot'`.

- [ ] **Step 3: Implement classification, drawing and summary**

Append to `/Users/agussoul/projects/ua/va/tps/tpfinal/inference/detect.py`:

```python
def classify_spot(model, crop, threshold=0.5):
    """Clasifica un crop ya recortado: devuelve (is_occupied, probabilidad)."""
    batch = preprocess_crop(crop)
    probability = float(model.predict(batch, verbose=0)[0][0])
    return probability >= threshold, probability


def build_summary_text(results):
    """Construye el texto resumen, ej. '12/40 espacios libres'."""
    total = len(results)
    free = sum(1 for is_occupied in results if not is_occupied)
    return f"{free}/{total} espacios libres"


def draw_results(image, spots, results):
    """Dibuja cada espacio (verde=libre, rojo=ocupado) y el resumen sobre una copia de la imagen."""
    annotated = image.copy()
    for spot, is_occupied in zip(spots, results):
        points = np.array(spot["points"], dtype=np.int32)
        color = (0, 0, 255) if is_occupied else (0, 255, 0)
        cv2.polylines(annotated, [points], isClosed=True, color=color, thickness=2)
        cv2.putText(
            annotated, spot["id"], tuple(points[0]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )

    summary = build_summary_text(results)
    cv2.putText(
        annotated, summary, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
    )
    return annotated
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/agussoul/projects/ua/va/tps/tpfinal/inference && python -m pytest tests/test_detect.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git -C /Users/agussoul/projects/ua/va/tps/tpfinal add inference/detect.py inference/tests/test_detect.py
git -C /Users/agussoul/projects/ua/va/tps/tpfinal commit -m "Add classification, drawing and summary logic to detect.py"
```

---

### Task 6: `detect.py` — CLI, carga de modelo y `requirements.txt`

**Files:**
- Modify: `inference/detect.py`
- Modify: `inference/tests/test_detect.py`
- Create: `inference/requirements.txt`
- Create: `inference/spots.json.example`

**Interfaces:**
- Consumes: `load_spots`, `crop_spot`, `classify_spot`, `draw_results`, `build_summary_text` from Tasks 4–5.
- Produces: `load_model_from_path(model_path: str)` (lazy-imports `tensorflow`, fuerza CPU), `parse_args(argv=None) -> argparse.Namespace` (flags `--image`, `--spots`, `--model`, `--output` opcional), `run(image_path, spots_path, model_path, output_path=None, model_loader=load_model_from_path) -> tuple[np.ndarray, str]` (imagen anotada, texto resumen), `main(argv=None) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `/Users/agussoul/projects/ua/va/tps/tpfinal/inference/tests/test_detect.py`:

```python
import json as _json

import cv2 as _cv2

import detect
from detect import main, parse_args, run


def test_parse_args_output_defaults_to_none():
    args = parse_args(["--image", "lot.jpg", "--spots", "spots.json", "--model", "model.h5"])
    assert args.output is None


def test_run_returns_annotated_image_and_summary(tmp_path, monkeypatch):
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    spots_path = tmp_path / "spots.json"
    spots_path.write_text(_json.dumps({
        "spots": [{"id": "spot_1", "points": [[5, 5], [25, 5], [25, 25], [5, 25]]}]
    }))

    monkeypatch.setattr(_cv2, "imread", lambda path: image)
    monkeypatch.setattr(_cv2, "imwrite", lambda path, img: True)
    monkeypatch.setattr(_cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(_cv2, "waitKey", lambda delay: 0)
    monkeypatch.setattr(_cv2, "destroyAllWindows", lambda: None)

    annotated, summary = run(
        "lot.jpg", str(spots_path), "model.h5",
        model_loader=lambda path: FakeModel(0.1),
    )

    assert annotated.shape == image.shape
    assert summary == "1/1 espacios libres"


def test_main_writes_output_image(tmp_path, monkeypatch):
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    spots_path = tmp_path / "spots.json"
    spots_path.write_text(_json.dumps({
        "spots": [{"id": "spot_1", "points": [[5, 5], [25, 5], [25, 25], [5, 25]]}]
    }))
    output_path = tmp_path / "result.jpg"

    monkeypatch.setattr(_cv2, "imread", lambda path: image)
    written = {}

    def fake_imwrite(path, img):
        written["path"] = path
        return True

    monkeypatch.setattr(_cv2, "imwrite", fake_imwrite)
    monkeypatch.setattr(_cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(_cv2, "waitKey", lambda delay: 0)
    monkeypatch.setattr(_cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(detect, "load_model_from_path", lambda path: FakeModel(0.9))

    main([
        "--image", "lot.jpg",
        "--spots", str(spots_path),
        "--model", "model.h5",
        "--output", str(output_path),
    ])

    assert written["path"] == str(output_path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/agussoul/projects/ua/va/tps/tpfinal/inference && python -m pytest tests/test_detect.py -v
```

Expected: the 9 previous tests pass; the 3 new ones fail with `ImportError: cannot import name 'parse_args'` (or `'run'`/`'main'`).

- [ ] **Step 3: Implement model loading, `run`, and CLI**

Append to `/Users/agussoul/projects/ua/va/tps/tpfinal/inference/detect.py`:

```python
import argparse
import os
import sys


def load_model_from_path(model_path):
    """Carga el modelo Keras forzando ejecucion en CPU (sin GPU disponible)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from tensorflow import keras

    return keras.models.load_model(model_path)


def run(image_path, spots_path, model_path, output_path=None, model_loader=load_model_from_path):
    """Orquesta: lee imagen, carga spots y modelo, clasifica cada espacio y dibuja el resultado.

    Devuelve (imagen_anotada, texto_resumen). Si output_path es provisto, guarda la imagen.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: no se pudo leer la imagen '{image_path}'", file=sys.stderr)
        sys.exit(1)

    spots = load_spots(spots_path)
    model = model_loader(model_path)

    results = []
    for spot in spots:
        crop = crop_spot(image, spot["points"])
        is_occupied, _ = classify_spot(model, crop)
        results.append(is_occupied)

    annotated = draw_results(image, spots, results)
    summary = build_summary_text(results)

    if output_path:
        cv2.imwrite(output_path, annotated)

    return annotated, summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Clasifica cada espacio de estacionamiento de una imagen como libre/ocupado."
    )
    parser.add_argument("--image", required=True, help="Ruta a la imagen del estacionamiento a analizar")
    parser.add_argument("--spots", required=True, help="Ruta al archivo spots.json calibrado")
    parser.add_argument("--model", required=True, help="Ruta al modelo .h5 entrenado")
    parser.add_argument("--output", default=None, help="Ruta de la imagen resultado (default: <imagen>_resultado.jpg)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_path = args.output or f"{os.path.splitext(args.image)[0]}_resultado.jpg"

    annotated, summary = run(args.image, args.spots, args.model, output_path=output_path)

    print(summary)
    cv2.imshow("Resultado", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/agussoul/projects/ua/va/tps/tpfinal/inference && python -m pytest tests/test_detect.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Write `requirements.txt` and `spots.json.example`**

Create `/Users/agussoul/projects/ua/va/tps/tpfinal/inference/requirements.txt`:

```
opencv-python==4.10.0.84
tensorflow-cpu==2.17.0
numpy==1.26.4
```

Create `/Users/agussoul/projects/ua/va/tps/tpfinal/inference/spots.json.example`:

```json
{
  "spots": [
    {
      "id": "spot_1",
      "points": [[120, 80], [200, 80], [200, 160], [120, 160]]
    },
    {
      "id": "spot_2",
      "points": [[210, 80], [290, 80], [290, 160], [210, 160]]
    }
  ]
}
```

- [ ] **Step 6: Commit**

```bash
git -C /Users/agussoul/projects/ua/va/tps/tpfinal add inference/detect.py inference/tests/test_detect.py inference/requirements.txt inference/spots.json.example
git -C /Users/agussoul/projects/ua/va/tps/tpfinal commit -m "Add detect.py CLI, model loading, requirements and spots.json example"
```

---

### Task 7: Notebook de entrenamiento (`training/PKLot_training.ipynb`)

**Files:**
- Create: `training/PKLot_training.ipynb`
- Delete: `training/.gitkeep`

**Interfaces:**
- Produces: a Colab notebook with no Python-importable interface (consumed manually by the user in Colab); the deliverable is verified by JSON/syntax validation, not pytest.

- [ ] **Step 1: Build the notebook as a Jupyter JSON document**

Create `/Users/agussoul/projects/ua/va/tps/tpfinal/training/PKLot_training.ipynb` with the following cells, in order (each `# %%` marks a new cell; markdown cells are written as such):

```
[markdown] # Entrenamiento PKLot: clasificador libre/ocupado con MobileNetV2
Notebook pensado para correr en Google Colab. Entrena un clasificador binario
sobre patches del dataset PKLot y exporta el modelo final a Google Drive.

[code] # Celda 1: Instalacion/import de dependencias
!pip install -q tensorflow scikit-learn matplotlib seaborn

import os
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

print("TensorFlow version:", tf.__version__)

[code] # Celda 2: Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

[code] # Celda 3: Configuracion (editar estas variables antes de correr el notebook)
# Ruta a la carpeta del dataset PKLot dentro de tu Drive.
DATASET_PATH = "/content/drive/MyDrive/PKLot"
# Carpeta de Drive donde se va a guardar el modelo entrenado.
OUTPUT_PATH = "/content/drive/MyDrive/PKLot_output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

[markdown] ## Exploracion de la estructura de carpetas
PKLot puede venir con estructuras distintas segun como se descargue
(por ejemplo `PKLot/PKLotSegmented/UFPR04/Cloudy/2012-09-12/Empty/...`).
Esta celda recorre `DATASET_PATH` y muestra el arbol de carpetas y cuantas
imagenes Empty/Occupied encuentra, para verificar el path ANTES de entrenar.

[code] # Celda 4: Explorar la estructura real del dataset
def print_tree(root, max_depth=4, _depth=0):
    """Imprime el arbol de carpetas hasta max_depth niveles de profundidad."""
    if _depth > max_depth:
        return
    root = Path(root)
    entries = sorted([p for p in root.iterdir() if p.is_dir()])
    for entry in entries:
        print("  " * _depth + f"- {entry.name}/")
        print_tree(entry, max_depth=max_depth, _depth=_depth + 1)

print(f"Explorando: {DATASET_PATH}\n")
print_tree(DATASET_PATH)

empty_count = sum(1 for _ in Path(DATASET_PATH).rglob("Empty/*") if _.is_file())
occupied_count = sum(1 for _ in Path(DATASET_PATH).rglob("Occupied/*") if _.is_file())
empty_count += sum(1 for _ in Path(DATASET_PATH).rglob("empty/*") if _.is_file())
occupied_count += sum(1 for _ in Path(DATASET_PATH).rglob("occupied/*") if _.is_file())
print(f"\nImagenes encontradas -> Empty: {empty_count} | Occupied: {occupied_count}")
assert empty_count > 0 and occupied_count > 0, "No se encontraron imagenes Empty/Occupied en DATASET_PATH. Revisa la ruta."

[markdown] ## Carga del dataset y split train/val/test
Buscamos todas las imagenes bajo carpetas `Empty`/`Occupied` (sin importar
mayusculas) y armamos pares (ruta, label). Si detectamos subcarpetas con
formato de fecha (PKLot trae `Cloudy/2012-09-12/...`), agrupamos el split por
fecha para que todos los frames de un mismo dia queden en el mismo split y
evitar data leakage entre frames del mismo video. Si no hay esa estructura,
hacemos split aleatorio estratificado por clase.

[code] # Celda 5: Recolectar imagenes y labels
def collect_image_label_pairs(dataset_path):
    """Recorre dataset_path y devuelve lista de (ruta, label, date_key).

    date_key es la carpeta inmediatamente superior a Empty/Occupied si su
    nombre tiene forma de fecha (YYYY-MM-DD), o None si no se pudo inferir.
    """
    import re

    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    pairs = []
    for label_name, label in [("Empty", 0), ("empty", 0), ("Occupied", 1), ("occupied", 1)]:
        for image_path in Path(dataset_path).rglob(f"{label_name}/*"):
            if not image_path.is_file():
                continue
            parent_date_dir = image_path.parent.parent.name
            date_key = parent_date_dir if date_pattern.match(parent_date_dir) else None
            pairs.append((str(image_path), label, date_key))
    return pairs


pairs = collect_image_label_pairs(DATASET_PATH)
print(f"Total de imagenes encontradas: {len(pairs)}")

has_date_structure = all(date_key is not None for _, _, date_key in pairs) and len(pairs) > 0
print(f"Estructura por fecha detectada: {has_date_structure}")

[code] # Celda 6: Split train/val/test (70/15/15)
random.seed(RANDOM_SEED)

if has_date_structure:
    # Split por fecha: todos los frames de un mismo dia van al mismo split.
    dates = sorted(set(date_key for _, _, date_key in pairs))
    random.shuffle(dates)
    n = len(dates)
    train_dates = set(dates[: int(0.7 * n)])
    val_dates = set(dates[int(0.7 * n): int(0.85 * n)])
    test_dates = set(dates[int(0.85 * n):])

    train_pairs = [(p, l) for p, l, d in pairs if d in train_dates]
    val_pairs = [(p, l) for p, l, d in pairs if d in val_dates]
    test_pairs = [(p, l) for p, l, d in pairs if d in test_dates]
else:
    # Fallback: split aleatorio estratificado por clase.
    paths = [p for p, l, _ in pairs]
    labels = [l for _, l, _ in pairs]
    train_paths, rest_paths, train_labels, rest_labels = train_test_split(
        paths, labels, test_size=0.3, stratify=labels, random_state=RANDOM_SEED
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        rest_paths, rest_labels, test_size=0.5, stratify=rest_labels, random_state=RANDOM_SEED
    )
    train_pairs = list(zip(train_paths, train_labels))
    val_pairs = list(zip(val_paths, val_labels))
    test_pairs = list(zip(test_paths, test_labels))


def count_by_label(split_pairs):
    counts = defaultdict(int)
    for _, label in split_pairs:
        counts[label] += 1
    return dict(counts)


print(f"Train: {len(train_pairs)} imagenes -> {count_by_label(train_pairs)}")
print(f"Val:   {len(val_pairs)} imagenes -> {count_by_label(val_pairs)}")
print(f"Test:  {len(test_pairs)} imagenes -> {count_by_label(test_pairs)}")

[markdown] ## Pipeline de datos con augmentation
Armamos `tf.data.Dataset` para cada split. El de train tiene augmentation
(flip horizontal, brillo, rotacion leve) ya que PKLot tiene variacion de
clima/luz; val y test no llevan augmentation.

[code] # Celda 7: Construccion de los tf.data.Dataset
def load_and_resize(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0
    return image, tf.cast(label, tf.float32)


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.keras.layers.RandomRotation(0.04)(tf.expand_dims(image, 0))[0]  # ~ +/-15 grados
    return image, label


def make_dataset(split_pairs, training):
    paths = [p for p, _ in split_pairs]
    labels = [l for _, l in split_pairs]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=len(split_pairs), seed=RANDOM_SEED)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(train_pairs, training=True)
val_ds = make_dataset(val_pairs, training=False)
test_ds = make_dataset(test_pairs, training=False)

[markdown] ## Modelo: transfer learning con MobileNetV2
Usamos MobileNetV2 preentrenado en ImageNet como extractor de caracteristicas
(congelado), con una cabeza de clasificacion binaria simple encima.

[code] # Celda 8: Definicion del modelo
base_model = MobileNetV2(input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet")
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
)

model.summary()

[markdown] ## Entrenamiento con callbacks
EarlyStopping evita sobreentrenar, ModelCheckpoint guarda el mejor modelo
visto durante el entrenamiento, y ReduceLROnPlateau baja el learning rate
cuando el val_loss se estanca.

[code] # Celda 9: Entrenamiento
checkpoint_path = os.path.join(OUTPUT_PATH, "best_checkpoint.h5")

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

[markdown] ## Evaluacion sobre el set de test
Calculamos accuracy, precision, recall, F1 y la matriz de confusion sobre
imagenes que el modelo nunca vio durante el entrenamiento.

[code] # Celda 10: Evaluacion en test
y_true = []
y_pred = []

for images, labels in test_ds:
    probabilities = model.predict(images, verbose=0).flatten()
    y_pred.extend((probabilities >= 0.5).astype(int).tolist())
    y_true.extend(labels.numpy().astype(int).tolist())

print(classification_report(y_true, y_pred, target_names=["Empty", "Occupied"]))

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Empty", "Occupied"]).plot(cmap="Blues")
plt.title("Matriz de confusion - Test set")
plt.show()

[markdown] ## Curvas de entrenamiento
Graficamos loss y accuracy de train vs val por epoch para ver si hubo
overfitting o underfitting.

[code] # Celda 11: Graficos de loss y accuracy
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["loss"], label="train")
axes[0].plot(history.history["val_loss"], label="val")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["accuracy"], label="train")
axes[1].plot(history.history["val_accuracy"], label="val")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.show()

[markdown] ## Exportar el modelo final a Google Drive
Guardamos el modelo entrenado en `OUTPUT_PATH` en formato `.h5`, listo para
descargar y usar localmente con `inference/detect.py`.

[code] # Celda 12: Exportar modelo final
final_model_path = os.path.join(OUTPUT_PATH, "modelo_pklot.h5")
model.save(final_model_path)
print(f"Modelo guardado en: {final_model_path}")
print("Descargalo desde Google Drive y usalo con --model en inference/detect.py")
```

Write this as a valid `.ipynb` (nbformat 4) JSON file: markdown blocks become `"cell_type": "markdown"` cells with the text (minus the leading `[markdown]`) split into a `source` list of lines; code blocks become `"cell_type": "code"` cells with the code (minus the leading `[code] # ...` comment line kept as the first source line) as `source`, `"outputs": []`, `"execution_count": null`. Use this minimal metadata:

```json
"metadata": {
  "kernelspec": {"name": "python3", "display_name": "Python 3"},
  "language_info": {"name": "python"}
},
"nbformat": 4,
"nbformat_minor": 5
```

- [ ] **Step 2: Validate the notebook is well-formed JSON and syntactically valid Python**

```bash
pip install --quiet nbformat
python3 -c "import nbformat; nb = nbformat.read('/Users/agussoul/projects/ua/va/tps/tpfinal/training/PKLot_training.ipynb', as_version=4); nbformat.validate(nb); print('OK:', len(nb.cells), 'celdas')"
jupyter nbconvert --to script --stdout /Users/agussoul/projects/ua/va/tps/tpfinal/training/PKLot_training.ipynb > /tmp/pklot_check.py
python3 -m py_compile /tmp/pklot_check.py
echo "Sintaxis OK"
```

Expected: `OK: <N> celdas` printed, no `nbformat.ValidationError`; `py_compile` exits with no output (success); final `echo` prints `Sintaxis OK`. Note: this only checks the notebook is structurally valid and the Python is syntactically correct — it does NOT execute the training (no dataset/GPU available here); actual execution happens manually by the user in Colab.

- [ ] **Step 3: Remove the placeholder and commit**

```bash
rm /Users/agussoul/projects/ua/va/tps/tpfinal/training/.gitkeep
git -C /Users/agussoul/projects/ua/va/tps/tpfinal add training/PKLot_training.ipynb
git -C /Users/agussoul/projects/ua/va/tps/tpfinal rm training/.gitkeep
git -C /Users/agussoul/projects/ua/va/tps/tpfinal commit -m "Add PKLot training notebook for Google Colab"
```

---

### Task 8: Documentación (`README.md` raíz, `training/README.md`, `inference/README.md`)

**Files:**
- Create: `README.md`
- Create: `training/README.md`
- Create: `inference/README.md`
- Delete: `inference/.gitkeep`

**Interfaces:**
- Produces: human-facing documentation only; no code interfaces.

- [ ] **Step 1: Write `training/README.md`**

```markdown
# Entrenamiento (Google Colab)

## 1. Subir el dataset PKLot a Google Drive

Subi la carpeta del dataset PKLot a tu Google Drive (por ejemplo a
`MyDrive/PKLot`). No importa la estructura interna exacta (PKLot suele venir
como `PKLot/PKLotSegmented/<LOTE>/<CLIMA>/<FECHA>/Empty|Occupied/...`); el
notebook explora la estructura real antes de entrenar.

## 2. Abrir y correr el notebook

1. Abri `PKLot_training.ipynb` en Google Colab (botón derecho sobre el
   archivo en Drive > "Abrir con" > Google Colaboratory, o subilo directo a
   [colab.research.google.com](https://colab.research.google.com)).
2. Activa GPU: Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU.
3. Corré la celda de montaje de Drive y autorizá el acceso cuando lo pida.
4. En la celda de configuración, ajustá `DATASET_PATH` a la ruta real donde
   subiste el dataset, y `OUTPUT_PATH` a la carpeta de Drive donde querés que
   se guarde el modelo entrenado.
5. Corré la celda de exploración de estructura y confirmá que encuentra
   imágenes `Empty`/`Occupied` antes de seguir.
6. Corré el resto de las celdas en orden (carga/split, entrenamiento,
   evaluación, gráficos, export).

## 3. Descargar el modelo entrenado

Al final del notebook se guarda `modelo_pklot.h5` dentro de `OUTPUT_PATH` en
tu Drive. Descargalo a tu máquina local para usarlo con `/inference`.
```

- [ ] **Step 2: Write `inference/README.md`**

```markdown
# Inferencia local

Corre 100% en tu máquina, sin GPU (usa `tensorflow-cpu`).

## 1. Instalar dependencias

```bash
cd inference
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Calibrar los espacios (una sola vez por cámara/layout)

```bash
python calibrate.py --image referencia.jpg --output spots.json
```

Se abre una ventana con la imagen de referencia. Hacé click en las 4 esquinas
de cada espacio (en cualquier orden) para cerrarlo; repetí para cada espacio.

Teclas:
- `u`: deshacer el último punto o el último espacio cerrado.
- `s`: guardar `spots.json` y salir.
- `q` / `ESC`: salir sin guardar.

Mirá `spots.json.example` para ver el formato esperado.

## 3. Clasificar una imagen nueva

```bash
python detect.py --image estacionamiento.jpg --spots spots.json --model modelo_pklot.h5
```

Esto recorta cada espacio según `spots.json`, lo clasifica con el modelo
descargado de Colab, y muestra/guarda la imagen con cada espacio en verde
(libre) o rojo (ocupado), junto con el conteo total (ej. `"12/40 espacios
libres"`). La imagen resultado se guarda como `<imagen>_resultado.jpg` salvo
que se pase `--output`.
```

- [ ] **Step 3: Write the root `README.md`**

```markdown
# PKLot: Detección de espacios de estacionamiento libres/ocupados

Proyecto de visión artificial con dos partes que corren en entornos distintos:

- [`/training`](training/README.md): notebook de Google Colab que entrena un
  clasificador binario (libre/ocupado) con transfer learning sobre MobileNetV2
  usando el dataset PKLot, y exporta el modelo a Google Drive.
- [`/inference`](inference/README.md): scripts Python para correr localmente
  (sin GPU) que calibran los espacios de un layout de cámara fija una sola
  vez, y luego clasifican y dibujan el resultado sobre fotos nuevas del
  estacionamiento.

Ver el diseño completo en
[`docs/superpowers/specs/2026-06-25-pklot-parking-detection-design.md`](docs/superpowers/specs/2026-06-25-pklot-parking-detection-design.md).
```

- [ ] **Step 4: Remove the remaining placeholder and commit**

```bash
rm /Users/agussoul/projects/ua/va/tps/tpfinal/inference/.gitkeep
git -C /Users/agussoul/projects/ua/va/tps/tpfinal add README.md training/README.md inference/README.md
git -C /Users/agussoul/projects/ua/va/tps/tpfinal rm inference/.gitkeep
git -C /Users/agussoul/projects/ua/va/tps/tpfinal commit -m "Add documentation for training and inference"
```

---

## Final Verification

- [ ] **Run the full local test suite**

```bash
cd /Users/agussoul/projects/ua/va/tps/tpfinal/inference
python -m venv /tmp/pklot_check_venv
source /tmp/pklot_check_venv/bin/activate
pip install -r requirements-dev.txt opencv-python-headless numpy
python -m pytest tests/ -v
deactivate
rm -rf /tmp/pklot_check_venv
```

Expected: all 21 tests across `test_calibrate.py` (12) and `test_detect.py` (9) pass. (`opencv-python-headless` is installed instead of `opencv-python` only for this throwaway check venv, to avoid GUI library requirements in a headless verification environment — `requirements.txt` itself keeps `opencv-python`.)

- [ ] **Confirm final tree**

```bash
git -C /Users/agussoul/projects/ua/va/tps/tpfinal log --oneline -10
find /Users/agussoul/projects/ua/va/tps/tpfinal -maxdepth 2 -not -path '*/.git*'
```

Expected: 8 new commits from Tasks 1–8, and the tree matches the structure approved at design time (`README.md`, `.gitignore`, `training/`, `inference/`, `docs/`).
