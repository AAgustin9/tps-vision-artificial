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
