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
