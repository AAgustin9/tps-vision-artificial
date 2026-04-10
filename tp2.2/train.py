"""
Entrenador (tp2.2).

Lee el dataset desde data/hu_moments.csv, entrena un
DecisionTreeClassifier con scikit-learn, guarda el modelo en
models/shape_classifier.joblib y exporta una imagen del arbol de
decision.

Ejecucion:
    python train.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from joblib import dump
from sklearn import tree
from sklearn.model_selection import cross_val_score

from labels import LABELS


DATA_PATH = Path("data/hu_moments.csv")
MODEL_PATH = Path("models/shape_classifier.joblib")
TREE_IMG_PATH = Path("models/decision_tree.png")


def load_dataset(path: Path) -> tuple[list[list[float]], list[int]]:
    # Reads hu_moments.csv and returns X (list of 7-element Hu moment vectors)
    # and Y (list of integer labels), one entry per row.
    X: list[list[float]] = []
    Y: list[int] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row[f"hu{i}"]) for i in range(1, 8)])
            Y.append(int(row["label"]))
    return X, Y


def main() -> None:
    # Loads the dataset, evaluates accuracy with 5-fold cross-validation, trains the final
    # DecisionTreeClassifier on all samples, saves the model to models/shape_classifier.joblib,
    # and exports a decision tree visualization image.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {DATA_PATH}\n"
            "Ejecuta primero: python create_dataset.py"
        )

    print(f"Cargando dataset desde {DATA_PATH}...")
    X, Y = load_dataset(DATA_PATH)
    print(f"  {len(X)} muestras cargadas.")

    label_names = [LABELS[k] for k in sorted(LABELS)]

    clf = tree.DecisionTreeClassifier(random_state=42)

    # Validacion cruzada antes de entrenar con todo el dataset
    scores = cross_val_score(clf, X, Y, cv=5)
    print(f"Precision con validacion cruzada (5-fold): "
          f"{scores.mean():.3f} +/- {scores.std():.3f}")

    # Entrenamiento final sobre todo el dataset
    clf.fit(X, Y)
    print("Modelo entrenado con el dataset completo.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(clf, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")

    # Exportar visualizacion del arbol
    fig, ax = plt.subplots(figsize=(20, 10))
    tree.plot_tree(
        clf,
        feature_names=[f"hu{i}" for i in range(1, 8)],
        class_names=label_names,
        filled=True,
        rounded=True,
        ax=ax,
    )
    plt.title("Arbol de decision - clasificacion de formas")
    plt.tight_layout()
    plt.savefig(TREE_IMG_PATH, dpi=120)
    print(f"Arbol de decision guardado en {TREE_IMG_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
