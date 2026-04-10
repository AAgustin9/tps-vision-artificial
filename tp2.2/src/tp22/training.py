from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .descriptors import read_dataset_csv


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> tuple[DecisionTreeClassifier, dict[str, Any]]:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )
    evaluation_model = DecisionTreeClassifier(max_depth=8, random_state=random_state)
    evaluation_model.fit(x_train, y_train)

    predictions = evaluation_model.predict(x_test)
    labels = sorted(set(y.tolist()))
    metrics: dict[str, Any] = {
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }
    final_model = DecisionTreeClassifier(max_depth=8, random_state=random_state)
    final_model.fit(x, y)
    return final_model, metrics


def train_from_csv(
    dataset_csv: Path,
    model_path: Path,
    metrics_path: Path | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    x, y = read_dataset_csv(dataset_csv)
    model, metrics = train_model(x, y, random_state=random_state)
    bundle = {
        "model": model,
        "labels": sorted(set(y.tolist())),
        "feature_names": [f"hu_{index}" for index in range(1, 8)],
        "metrics": metrics,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    bundle = joblib.load(model_path)
    if "model" not in bundle:
        raise ValueError(f"El archivo no contiene un modelo valido: {model_path}")
    return bundle
