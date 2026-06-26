"""
Deteccion de espacios de estacionamiento en video.
Procesa un .mp4 (o camara en vivo) frame a frame y muestra el resultado en tiempo real.

Uso:
    python detect_video.py --model modelo_pklot.h5 --video parking.mp4 --spots spots.json
    python detect_video.py --model modelo_pklot.h5 --camera 0 --spots spots.json
"""
import argparse
import sys

import cv2
import numpy as np

from detect import (
    load_spots,
    load_model_from_path,
    crop_spot,
    build_summary_text,
    draw_results,
)

WINDOW = "Deteccion de estacionamiento"


def predict_batch(model, crops):
    """Clasifica todos los crops de un frame en una sola llamada al modelo."""
    batch = np.stack([
        cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for c in crops
    ], axis=0)
    probs = model.predict(batch, verbose=0).flatten()
    return probs >= 0.5, probs


def process_video(video_source, spots, model, output_path=None, skip_frames=2):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir '{video_source}'", file=sys.stderr)
        sys.exit(1)

    _, h, w, _ = model.input_shape
    input_size = (w, h)

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_video   = total > 0  # False para camara en vivo

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Crear ventana y trackbar (solo para video grabado)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    if is_video:
        cv2.createTrackbar("Tiempo", WINDOW, 0, max(total - 1, 1), lambda x: None)

    results     = [False] * len(spots)
    last_frame  = None
    frame_idx   = 0
    paused      = False

    print("Controles: 'q' salir | 'espacio' pausar/reanudar | 'r' replay | slider para buscar en el video")

    while True:
        if is_video:
            slider_pos = cv2.getTrackbarPos("Tiempo", WINDOW)
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Si el usuario movio el slider, buscar ese frame
            if abs(slider_pos - current_pos) > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, slider_pos)
                frame_idx = slider_pos

        if not paused:
            ret, frame = cap.read()

            if not ret:
                # Video terminado: pausar en el ultimo frame en vez de salir
                if last_frame is not None:
                    paused = True
                    frame  = last_frame
                else:
                    break
            else:
                last_frame = frame.copy()

                # Actualizar slider con la posicion actual
                if is_video:
                    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    cv2.setTrackbarPos("Tiempo", WINDOW, min(pos, total - 1))

                # Clasificar solo cada N frames para ganar velocidad
                if frame_idx % (skip_frames + 1) == 0:
                    crops   = [crop_spot(frame, s["points"], output_size=input_size) for s in spots]
                    results, _ = predict_batch(model, crops)

                frame_idx += 1
        else:
            frame = last_frame

        if frame is None:
            continue

        display = draw_results(frame, spots, results)
        summary = build_summary_text(results)

        estado = "|| PAUSADO" if paused else "► "
        cv2.putText(display, f"{estado}  {summary}",
                    (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(WINDOW, display)

        if writer and not paused:
            writer.write(display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("r") and is_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cv2.setTrackbarPos("Tiempo", WINDOW, 0)
            frame_idx = 0
            paused = False

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if output_path:
        print(f"Video guardado en: {output_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Detecta espacios libres/ocupados en un video o camara en vivo."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",  type=str, help="Ruta al archivo de video (.mp4, .avi, etc.)")
    src.add_argument("--camera", type=int, help="Indice de camara (0 = webcam principal)")
    parser.add_argument("--model",  required=True, help="Ruta al modelo .h5 o .keras")
    parser.add_argument("--spots",  required=True, help="Ruta al archivo spots.json calibrado")
    parser.add_argument("--output", default=None,  help="Guardar video resultado (opcional)")
    parser.add_argument("--skip",   type=int, default=2,
                        help="Procesar 1 de cada (N+1) frames. Default=2. 0=todos los frames.")
    return parser.parse_args(argv)


def main(argv=None):
    args   = parse_args(argv)
    spots  = load_spots(args.spots)
    model  = load_model_from_path(args.model)
    source = args.video if args.video is not None else args.camera
    process_video(source, spots, model, output_path=args.output, skip_frames=args.skip)


if __name__ == "__main__":
    main()
