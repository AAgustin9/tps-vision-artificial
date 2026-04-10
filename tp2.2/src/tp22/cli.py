from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .descriptors import collect_records_from_labeled_images, write_dataset_csv
from .detector import is_supported_image, iter_images, process_image
from .synthetic import LABELS, generate_synthetic_records
from .training import load_model_bundle, train_from_csv


DEFAULT_DATASET = Path("data/hu_moments.csv")
DEFAULT_MODEL = Path("models/shape_classifier.joblib")
DEFAULT_METRICS = Path("output/training_metrics.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TP2.2 - clasificacion de formas con machine learning."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser(
        "generate-dataset",
        help="Genera un CSV de momentos invariantes de Hu.",
    )
    dataset_parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("train"),
        help="Carpeta con subcarpetas por clase: circle, rectangle_outline y star.",
    )
    dataset_parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_DATASET,
        help="Ruta del CSV de salida.",
    )
    dataset_parser.add_argument(
        "--synthetic-samples-per-class",
        type=int,
        default=120,
        help="Cantidad de muestras sinteticas por clase para reforzar el dataset.",
    )
    dataset_parser.add_argument(
        "--min-area",
        type=int,
        default=50,
        help="Area minima para extraer contornos de las imagenes de entrenamiento.",
    )
    dataset_parser.add_argument("--seed", type=int, default=42)

    train_parser = subparsers.add_parser(
        "train",
        help="Entrena y guarda el modelo de clasificacion.",
    )
    train_parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    train_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    train_parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS)
    train_parser.add_argument("--seed", type=int, default=42)

    classify_parser = subparsers.add_parser(
        "classify",
        help="Clasifica las formas detectadas en una imagen o carpeta.",
    )
    classify_parser.add_argument("--image", type=Path, help="Ruta a una sola imagen.")
    classify_parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input"),
        help="Carpeta con imagenes de entrada.",
    )
    classify_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Carpeta donde guardar resultados.",
    )
    classify_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    classify_parser.add_argument("--min-area", type=int, default=50)
    classify_parser.add_argument("--save-masks", action="store_true")

    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        help="Genera dataset sintetico, entrena el modelo y guarda metricas.",
    )
    bootstrap_parser.add_argument("--train-dir", type=Path, default=Path("train"))
    bootstrap_parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET)
    bootstrap_parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    bootstrap_parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS)
    bootstrap_parser.add_argument(
        "--synthetic-samples-per-class",
        type=int,
        default=120,
    )
    bootstrap_parser.add_argument("--min-area", type=int, default=50)
    bootstrap_parser.add_argument("--seed", type=int, default=42)

    return parser


def resolve_inputs(image: Path | None, input_dir: Path) -> list[Path]:
    if image is not None:
        if not image.exists():
            raise FileNotFoundError(f"La imagen no existe: {image}")
        if not is_supported_image(image):
            raise ValueError(f"Formato no soportado: {image.suffix}")
        return [image]

    if not input_dir.exists():
        raise FileNotFoundError(f"La carpeta de entrada no existe: {input_dir}")

    images = list(iter_images(input_dir))
    if not images:
        raise FileNotFoundError(
            f"No se encontraron imagenes soportadas en: {input_dir}"
        )
    return images


def generate_dataset(
    train_dir: Path,
    output_csv: Path,
    synthetic_samples_per_class: int,
    min_area: int,
    seed: int,
) -> Path:
    real_records = collect_records_from_labeled_images(train_dir, LABELS, min_area=min_area)
    synthetic_records = generate_synthetic_records(
        samples_per_class=synthetic_samples_per_class,
        seed=seed,
        min_area=min_area,
    )
    records = [*real_records, *synthetic_records]
    if not records:
        raise ValueError("No se pudo construir el dataset de entrenamiento.")
    write_dataset_csv(records, output_csv)
    print(
        f"Dataset generado en {output_csv} con {len(records)} filas "
        f"({len(real_records)} reales, {len(synthetic_records)} sinteticas)."
    )
    return output_csv


def run_train(dataset_csv: Path, model_path: Path, metrics_path: Path, seed: int) -> int:
    metrics = train_from_csv(
        dataset_csv=dataset_csv,
        model_path=model_path,
        metrics_path=metrics_path,
        random_state=seed,
    )
    print(
        f"Modelo guardado en {model_path}. Accuracy de validacion: "
        f"{metrics['accuracy']:.4f}"
    )
    return 0


def run_classify(
    image: Path | None,
    input_dir: Path,
    output_dir: Path,
    model_path: Path,
    min_area: int,
    save_masks: bool,
) -> int:
    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe el modelo entrenado: {model_path}. Ejecuta primero 'bootstrap' o 'train'."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_bundle = load_model_bundle(model_path)
    images = resolve_inputs(image, input_dir)
    results = []

    for image_path in images:
        processed = process_image(image_path, model_bundle=model_bundle, min_area=min_area)
        stem = Path(processed.image_name).stem
        annotated_path = output_dir / f"{stem}_annotated.png"
        cv2.imwrite(str(annotated_path), processed.annotated_image)

        if save_masks:
            mask_path = output_dir / f"{stem}_mask.png"
            cv2.imwrite(str(mask_path), processed.binary_mask)

        results.append(processed.to_dict())

    json_path = output_dir / "detections.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Procesadas {len(images)} imagen(es). Resultados en: {output_dir}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate-dataset":
        generate_dataset(
            train_dir=args.train_dir,
            output_csv=args.output_csv,
            synthetic_samples_per_class=args.synthetic_samples_per_class,
            min_area=args.min_area,
            seed=args.seed,
        )
        return 0

    if args.command == "train":
        return run_train(
            dataset_csv=args.dataset_csv,
            model_path=args.model_path,
            metrics_path=args.metrics_path,
            seed=args.seed,
        )

    if args.command == "classify":
        return run_classify(
            image=args.image,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_path=args.model_path,
            min_area=args.min_area,
            save_masks=args.save_masks,
        )

    if args.command == "bootstrap":
        dataset_csv = generate_dataset(
            train_dir=args.train_dir,
            output_csv=args.dataset_csv,
            synthetic_samples_per_class=args.synthetic_samples_per_class,
            min_area=args.min_area,
            seed=args.seed,
        )
        return run_train(
            dataset_csv=dataset_csv,
            model_path=args.model_path,
            metrics_path=args.metrics_path,
            seed=args.seed,
        )

    parser.error(f"Comando no soportado: {args.command}")
    return 2
