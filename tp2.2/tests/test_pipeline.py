from __future__ import annotations

from pathlib import Path

import numpy as np

from src.tp22.detector import detect_shapes
from src.tp22.synthetic import LABELS, generate_shape_image, generate_synthetic_records
from src.tp22.training import load_model_bundle, train_from_csv
from src.tp22.descriptors import write_dataset_csv


def test_synthetic_dataset_contains_expected_labels() -> None:
    records = generate_synthetic_records(samples_per_class=10, seed=7)

    labels = {record.label for record in records}
    assert labels == set(LABELS)
    assert all(record.features.shape == (7,) for record in records)


def test_training_persists_model_and_metrics(tmp_path: Path) -> None:
    records = generate_synthetic_records(samples_per_class=18, seed=11)
    dataset_path = write_dataset_csv(records, tmp_path / "hu.csv")
    model_path = tmp_path / "shape_classifier.joblib"
    metrics_path = tmp_path / "metrics.json"

    metrics = train_from_csv(dataset_path, model_path, metrics_path, random_state=11)

    assert model_path.exists()
    assert metrics_path.exists()
    assert metrics["accuracy"] >= 0.75


def test_detector_classifies_known_shapes(tmp_path: Path) -> None:
    records = generate_synthetic_records(samples_per_class=25, seed=13)
    dataset_path = write_dataset_csv(records, tmp_path / "hu.csv")
    model_path = tmp_path / "shape_classifier.joblib"
    train_from_csv(dataset_path, model_path, tmp_path / "metrics.json", random_state=13)
    bundle = load_model_bundle(model_path)

    for label in LABELS:
        image = generate_shape_image(label, rng=np.random.default_rng(99))
        detections, _ = detect_shapes(image, bundle, min_area=1000)
        detected_labels = [detection.label for detection in detections]
        assert label in detected_labels
