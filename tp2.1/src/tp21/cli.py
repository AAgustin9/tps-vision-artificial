from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .detector import (
    DetectionParams,
    iter_images,
    load_reference_shapes,
    process_image,
)


CONTROL_WINDOW = "Controles"
ANNOTATED_WINDOW = "Webcam anotada"
MASK_WINDOW = "Mascara binaria"


def build_parser() -> argparse.ArgumentParser:
    # Builds and returns the CLI argument parser with all supported flags.
    parser = argparse.ArgumentParser(
        description="Deteccion y clasificacion de formas con webcam en tiempo real."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input"),
        help="Carpeta con las imagenes de referencia.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Carpeta para guardar resultados en modo imagen.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Procesa una sola imagen en lugar de abrir la webcam.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Indice de la webcam a usar.",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Guarda mascaras en modo imagen.",
    )
    return parser


def build_default_params() -> DetectionParams:
    # Returns a DetectionParams instance with sensible defaults for webcam/image mode.
    return DetectionParams(
        threshold_value=200,
        min_area=50,
        morph_kernel_size=3,
        match_threshold=0.18,
    )


def resolve_image_inputs(image: Path | None, input_dir: Path) -> list[Path]:
    # If a single image path was provided returns it as a one-element list,
    # otherwise returns all supported images found in input_dir.
    if image is not None:
        if not image.exists():
            raise FileNotFoundError(f"La imagen no existe: {image}")
        return [image]

    return list(iter_images(input_dir))


def run_image_mode(
    image: Path | None,
    input_dir: Path,
    output_dir: Path,
    save_masks: bool,
) -> int:
    # Processes images from disk (single file or full input_dir), saves annotated results
    # and a detections.json to output_dir. Reference images are excluded from processing.
    references = load_reference_shapes(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = resolve_image_inputs(image, input_dir)

    if image is None:
        images = [path for path in images if path.name not in {ref.source_path.name for ref in references}]

    if not images:
        raise FileNotFoundError("No hay imagenes para procesar en modo archivo.")

    params = build_default_params()
    results = []
    for image_path in images:
        processed = process_image(image_path, references, params)
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


def create_trackbars(params: DetectionParams) -> None:
    # Creates the OpenCV control window with trackbars for threshold, kernel size,
    # minimum area, and match threshold, initialized to the given params values.
    cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW, 520, 240)
    cv2.createTrackbar("Threshold", CONTROL_WINDOW, params.threshold_value, 255, lambda _: None)
    cv2.createTrackbar("Kernel", CONTROL_WINDOW, params.morph_kernel_size, 25, lambda _: None)
    cv2.createTrackbar("Min area", CONTROL_WINDOW, params.min_area, 10000, lambda _: None)
    cv2.createTrackbar(
        "Match x1000",
        CONTROL_WINDOW,
        int(params.match_threshold * 1000),
        1000,
        lambda _: None,
    )


def read_params_from_trackbars() -> DetectionParams:
    # Reads current trackbar positions and returns a DetectionParams with those values.
    # Called every frame so parameter changes take effect in real time.
    threshold_value = cv2.getTrackbarPos("Threshold", CONTROL_WINDOW)
    kernel = cv2.getTrackbarPos("Kernel", CONTROL_WINDOW)
    min_area = cv2.getTrackbarPos("Min area", CONTROL_WINDOW)
    match_x1000 = cv2.getTrackbarPos("Match x1000", CONTROL_WINDOW)
    return DetectionParams(
        threshold_value=max(0, threshold_value),
        min_area=max(1, min_area),
        morph_kernel_size=max(1, kernel),
        match_threshold=max(0.001, match_x1000 / 1000),
    )


def run_webcam_mode(input_dir: Path, camera_index: int) -> int:
    # Opens the webcam, runs the detection+classification loop in real time,
    # and displays the annotated frame and binary mask. Press q to quit, s to save a frame.
    references = load_reference_shapes(input_dir)
    params = build_default_params()
    create_trackbars(params)

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la webcam con indice {camera_index}. "
            "Verifica permisos del sistema o prueba otro indice."
        )

    cv2.namedWindow(ANNOTATED_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)

    print("Webcam activa. Teclas: q para salir, s para guardar un frame anotado.")

    saved_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("No se pudo leer un frame desde la webcam.")

            params = read_params_from_trackbars()
            from .detector import detect_shapes
            from .drawing import draw_detections

            detections, binary = detect_shapes(frame, references, params)
            annotated = draw_detections(frame, detections)

            cv2.putText(
                annotated,
                f"Refs: {', '.join(reference.label for reference in references)}",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                (
                    f"thr={params.threshold_value} kernel={params.morph_kernel_size} "
                    f"min_area={params.min_area} match<{params.match_threshold:.3f}"
                ),
                (16, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(ANNOTATED_WINDOW, annotated)
            cv2.imshow(MASK_WINDOW, binary)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                output_dir = Path("output")
                output_dir.mkdir(parents=True, exist_ok=True)
                frame_path = output_dir / f"webcam_frame_{saved_frames:03d}.png"
                mask_path = output_dir / f"webcam_frame_{saved_frames:03d}_mask.png"
                cv2.imwrite(str(frame_path), annotated)
                cv2.imwrite(str(mask_path), binary)
                saved_frames += 1
                print(f"Frame guardado en {frame_path}")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    # Entry point: parses CLI arguments and dispatches to image mode or webcam mode.
    parser = build_parser()
    args = parser.parse_args()

    if args.image is not None:
        return run_image_mode(args.image, args.input_dir, args.output_dir, args.save_masks)

    return run_webcam_mode(args.input_dir, args.camera_index)
