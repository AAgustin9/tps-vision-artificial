from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.tp22.synthetic import LABELS, generate_shape_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera imagenes de ejemplo en la carpeta input/."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("input"),
        help="Carpeta donde guardar las imagenes generadas.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=2,
        help="Cantidad de imagenes a generar por clase.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para generar resultados reproducibles.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    generated = 0

    for label in LABELS:
        extension = ".png" if label == "star" else ".jpeg"
        for index in range(1, args.samples_per_class + 1):
            image = generate_shape_image(label, rng=rng)
            output_path = output_dir / f"{label}_{index:02d}{extension}"
            cv2.imwrite(str(output_path), image)
            generated += 1

    print(f"Generadas {generated} imagen(es) en: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
