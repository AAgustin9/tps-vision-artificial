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
