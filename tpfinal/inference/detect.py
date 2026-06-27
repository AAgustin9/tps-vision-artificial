"""Script principal de inferencia: clasifica cada espacio de estacionamiento
de una imagen nueva como libre/ocupado usando el modelo entrenado en Colab,
y dibuja el resultado sobre la imagen."""
import json

import cv2
import numpy as np


def load_spots(spots_path):
    """Carga la lista de espacios calibrados desde spots.json."""
    with open(spots_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["spots"]


def order_points(points):
    """Ordena 4 puntos arbitrarios como top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = pts[:, 0] - pts[:, 1]

    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmax(diffs)]
    bottom_left = pts[np.argmin(diffs)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def crop_spot(image, points, output_size=(224, 224)):
    """Recorta y endereza el cuadrilatero de un espacio via warpPerspective."""
    width, height = output_size
    src = order_points(points)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, transform, (width, height))


def preprocess_crop(crop):
    """Convierte un crop BGR uint8 en el batch float32 normalizado que espera el modelo."""
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def classify_spot(model, crop, threshold=0.98):
    """Clasifica un crop ya recortado: devuelve (is_occupied, probabilidad)."""
    batch = preprocess_crop(crop)
    probability = float(model.predict(batch, verbose=0)[0][0])
    return probability >= threshold, probability


def build_summary_text(results):
    """Construye el texto resumen, ej. '12/40 espacios libres'."""
    total = len(results)
    free = sum(1 for is_occupied in results if not is_occupied)
    return f"{free}/{total} espacios libres"


def draw_results(image, spots, results):
    """Dibuja cada espacio (verde=libre, rojo=ocupado) y el resumen sobre una copia de la imagen."""
    annotated = image.copy()
    for spot, is_occupied in zip(spots, results):
        points = np.array(spot["points"], dtype=np.int32)
        color = (0, 0, 255) if is_occupied else (0, 255, 0)
        cv2.polylines(annotated, [points], isClosed=True, color=color, thickness=2)
        cv2.putText(
            annotated, spot["id"], tuple(points[0]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )

    summary = build_summary_text(results)
    cv2.putText(
        annotated, summary, (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
    )
    return annotated


import argparse
import os
import sys


def load_model_from_path(model_path):
    """Carga el modelo Keras forzando ejecucion en CPU (sin GPU disponible)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from tensorflow import keras

    return keras.models.load_model(model_path)


def run(image_path, spots_path, model_path, output_path=None, model_loader=load_model_from_path):
    """Orquesta: lee imagen, carga spots y modelo, clasifica cada espacio y dibuja el resultado.

    Devuelve (imagen_anotada, texto_resumen). Si output_path es provisto, guarda la imagen.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: no se pudo leer la imagen '{image_path}'", file=sys.stderr)
        sys.exit(1)

    spots = load_spots(spots_path)
    model = model_loader(model_path)

    # Leer el tamaño de entrada esperado por el modelo (puede ser 96 o 224 segun como fue entrenado)
    _, h, w, _ = model.input_shape
    input_size = (w, h)

    results = []
    for spot in spots:
        crop = crop_spot(image, spot["points"], output_size=input_size)
        is_occupied, _ = classify_spot(model, crop)
        results.append(is_occupied)

    annotated = draw_results(image, spots, results)
    summary = build_summary_text(results)

    if output_path:
        cv2.imwrite(output_path, annotated)

    return annotated, summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Clasifica cada espacio de estacionamiento de una imagen como libre/ocupado."
    )
    parser.add_argument("--image", required=True, help="Ruta a la imagen del estacionamiento a analizar")
    parser.add_argument("--spots", required=True, help="Ruta al archivo spots.json calibrado")
    parser.add_argument("--model", required=True, help="Ruta al modelo .h5 entrenado")
    parser.add_argument("--output", default=None, help="Ruta de la imagen resultado (default: <imagen>_resultado.jpg)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_path = args.output or f"{os.path.splitext(args.image)[0]}_resultado.jpg"

    annotated, summary = run(args.image, args.spots, args.model, output_path=output_path, model_loader=load_model_from_path)

    print(summary)
    cv2.imshow("Resultado", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
