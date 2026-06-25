import json

import numpy as np
import pytest

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
    assert probability == pytest.approx(0.2)


def test_classify_spot_above_threshold_is_occupied():
    crop = np.zeros((224, 224, 3), dtype=np.uint8)

    is_occupied, probability = classify_spot(FakeModel(0.9), crop)

    assert is_occupied is True
    assert probability == pytest.approx(0.9)


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
