"""TP4 homography perspective project.

Modes:
- Visualization: draw a perspective grid on the webcam frame and show a rectified view.
- QR acquisition: press q, then any key to detect a QR code and compute homography.
- Manual acquisition: press h, click four square corners, or press any key to abort.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


WINDOW_CAMERA = "TP4 - camara / grilla"
WINDOW_WARPED = "TP4 - vista frontal"


@dataclass
class AppState:
    mode: str = "view"
    clicked_points: list[tuple[int, int]] = field(default_factory=list)
    homography: Optional[np.ndarray] = None  # image -> frontal square
    last_status: str = "Presiona 'q' para QR, 'h' para manual, ESC para salir."


def order_points(points: np.ndarray) -> np.ndarray:
    """Return points ordered as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    ordered[0] = pts[np.argmin(sums)]  # top-left has smallest x + y
    ordered[2] = pts[np.argmax(sums)]  # bottom-right has largest x + y
    ordered[1] = pts[np.argmin(diffs)]  # top-right has smallest y - x
    ordered[3] = pts[np.argmax(diffs)]  # bottom-left has largest y - x
    return ordered


def destination_square(size: int) -> np.ndarray:
    max_coord = float(size - 1)
    return np.array(
        [[0.0, 0.0], [max_coord, 0.0], [max_coord, max_coord], [0.0, max_coord]],
        dtype=np.float32,
    )


def compute_homography(image_points: np.ndarray, square_size: int) -> Optional[np.ndarray]:
    src = order_points(image_points)
    dst = destination_square(square_size)
    homography = cv2.getPerspectiveTransform(src, dst)
    if homography is None or not np.isfinite(homography).all():
        return None
    return homography


def detect_qr_homography(frame: np.ndarray, square_size: int) -> tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    detector = cv2.QRCodeDetector()
    decoded_text, points, _ = detector.detectAndDecode(frame)
    if points is None:
        return None, None, "No se detecto ningun QR. Se conserva la homografia anterior."

    qr_points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if qr_points.shape[0] != 4:
        return None, qr_points, "QR detectado, pero no tiene 4 vertices validos."

    homography = compute_homography(qr_points, square_size)
    if homography is None:
        return None, qr_points, "No se pudo computar la homografia del QR."

    label = f"QR detectado: {decoded_text}" if decoded_text else "QR detectado sin texto decodificado."
    return homography, order_points(qr_points), label


def draw_grid(frame: np.ndarray, homography: np.ndarray, square_size: int, cells: int) -> None:
    inv_homography = np.linalg.inv(homography)
    positions = np.linspace(0, square_size - 1, cells + 1, dtype=np.float32)

    for pos in positions:
        vertical = np.array([[[pos, 0.0]], [[pos, float(square_size - 1)]]], dtype=np.float32)
        horizontal = np.array([[[0.0, pos]], [[float(square_size - 1), pos]]], dtype=np.float32)
        for line in (vertical, horizontal):
            projected = cv2.perspectiveTransform(line, inv_homography).reshape(-1, 2)
            pts = np.round(projected).astype(np.int32)
            cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (0, 255, 255), 2, cv2.LINE_AA)


def draw_polygon(frame: np.ndarray, points: np.ndarray, color: tuple[int, int, int]) -> None:
    pts = np.round(order_points(points)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)


def overlay_text(frame: np.ndarray, lines: list[str]) -> None:
    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28


def handle_mouse(event: int, x: int, y: int, _flags: int, userdata: AppState) -> None:
    if event != cv2.EVENT_LBUTTONDOWN or userdata.mode != "manual":
        return
    if len(userdata.clicked_points) < 4:
        userdata.clicked_points.append((x, y))
        userdata.last_status = f"Punto {len(userdata.clicked_points)}/4 registrado."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TP4: perspectiva homografica con QR o 4 puntos manuales.")
    parser.add_argument("--camera", type=int, default=0, help="Indice de camara OpenCV. Default: 0.")
    parser.add_argument("--square-size", type=int, default=600, help="Tamano en pixeles de la vista frontal. Default: 600.")
    parser.add_argument("--grid-cells", type=int, default=3, help="Cantidad de celdas por lado para la grilla. Default: 3.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.square_size < 64:
        raise ValueError("--square-size debe ser al menos 64")
    if args.grid_cells < 1:
        raise ValueError("--grid-cells debe ser al menos 1")

    state = AppState()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"No se pudo abrir la camara {args.camera}.")
        return 1

    cv2.namedWindow(WINDOW_CAMERA)
    cv2.namedWindow(WINDOW_WARPED)
    cv2.setMouseCallback(WINDOW_CAMERA, handle_mouse, state)

    blank_warp = np.zeros((args.square_size, args.square_size, 3), dtype=np.uint8)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer un frame de la camara.")
            break

        display = frame.copy()
        warped = blank_warp.copy()

        if state.homography is not None:
            try:
                draw_grid(display, state.homography, args.square_size, args.grid_cells)
                warped = cv2.warpPerspective(frame, state.homography, (args.square_size, args.square_size))
            except np.linalg.LinAlgError:
                state.last_status = "Homografia singular. Volve a calibrar con QR o puntos manuales."
                state.homography = None

        if state.mode == "manual":
            for idx, point in enumerate(state.clicked_points, start=1):
                cv2.circle(display, point, 5, (0, 80, 255), -1, cv2.LINE_AA)
                cv2.putText(display, str(idx), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
            overlay_text(display, ["MODO MANUAL: clic en 4 vertices del cuadrado.", "Cualquier tecla aborta.", state.last_status])
            if len(state.clicked_points) == 4:
                homography = compute_homography(np.array(state.clicked_points, dtype=np.float32), args.square_size)
                if homography is not None:
                    state.homography = homography
                    state.last_status = "Homografia manual actualizada."
                else:
                    state.last_status = "No se pudo computar la homografia manual."
                state.clicked_points.clear()
                state.mode = "view"
        elif state.mode == "qr":
            overlay_text(display, ["MODO QR: presiona cualquier tecla para detectar el QR en este frame.", state.last_status])
        else:
            overlay_text(display, ["VISUALIZACION: q=QR, h=manual, ESC=salir.", state.last_status])

        cv2.imshow(WINDOW_CAMERA, display)
        cv2.imshow(WINDOW_WARPED, warped)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            break

        if state.mode == "qr" and key != 255:
            homography, qr_points, status = detect_qr_homography(frame, args.square_size)
            if qr_points is not None and qr_points.shape[0] == 4:
                draw_polygon(display, qr_points, (0, 255, 0))
            if homography is not None:
                state.homography = homography
            state.last_status = status
            state.mode = "view"
            continue

        if state.mode == "manual" and key != 255:
            state.clicked_points.clear()
            state.last_status = "Modo manual abortado. Se conserva la homografia anterior."
            state.mode = "view"
            continue

        if state.mode == "view":
            if key == ord("q"):
                state.mode = "qr"
                state.last_status = "Alinea un QR cuadrado y presiona cualquier tecla."
            elif key == ord("h"):
                state.mode = "manual"
                state.clicked_points.clear()
                state.last_status = "Marca los 4 vertices del cuadrado."

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
