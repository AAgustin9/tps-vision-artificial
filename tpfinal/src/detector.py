"""Full detection pipeline: image -> annotated image + stats, via YOLOv8."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import cv2
from PIL import Image

from utils import MODEL_PATH, COLOR_FREE_BGR, COLOR_OCCUPIED_BGR

# Conservative cap for the OpenCV fallback.  The fallback is intentionally used
# only when YOLO returns no boxes, so it should prefer obvious painted stall
# grids over noisy detections.
MAX_FALLBACK_SPACES = 250

_model = None
_car_model = None


def get_model(model_path: Path = MODEL_PATH):
    """Lazily load and cache the YOLO model."""
    global _model
    if _model is None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Train it first: python src/convert_to_yolo.py && python src/train_yolo.py"
            )
        from ultralytics import YOLO
        _model = YOLO(str(model_path))
    return _model


def get_car_model():
    """Load a pretrained COCO YOLO model for vehicle detection."""
    global _car_model
    if _car_model is None:
        from ultralytics import YOLO
        _car_model = YOLO("yolov8n.pt")
    return _car_model


def _yolo_detect(image_rgb: np.ndarray, threshold: float, model_path: Path) -> list:
    """Run the trained parking-space YOLO model and normalize its output format."""
    if not model_path.exists():
        return []
    model = get_model(model_path)
    # Ultralytics numpy inputs are OpenCV-style BGR arrays. The app uses RGB
    # images from PIL/Streamlit, so convert before prediction.
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result = model.predict(image_bgr, conf=threshold, imgsz=640, verbose=False)[0]

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        raw_name = str(model.names.get(class_id, class_id)).lower()
        label = "Occupied" if "occ" in raw_name else "Empty"
        detections.append({
            "x": max(0, int(round(x1))), "y": max(0, int(round(y1))),
            "w": max(1, int(round(x2 - x1))), "h": max(1, int(round(y2 - y1))),
            "label": label,
            "confidence": confidence,
            "source": "yolo",
        })
    return detections


def _cluster_positions(values: list, tolerance: float) -> list:
    """Cluster 1D line coordinates, returning cluster means."""
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for value in values[1:]:
        if abs(value - np.mean(clusters[-1])) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [float(np.mean(c)) for c in clusters]


def _dedupe_boxes(boxes: list, iou_threshold: float = 0.35) -> list:
    """Simple non-maximum suppression for fallback boxes."""
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
        bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
        ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = a["w"] * a["h"] + b["w"] * b["h"] - inter
        return inter / union if union else 0

    kept = []
    for box in sorted(boxes, key=lambda d: d["confidence"], reverse=True):
        if all(iou(box, old) < iou_threshold for old in kept):
            kept.append(box)
        if len(kept) >= MAX_FALLBACK_SPACES:
            break
    return sorted(kept, key=lambda d: (d["y"], d["x"]))


def _detect_slots_from_separators(image_rgb: np.ndarray) -> list:
    """Find plausible parking slots between long painted divider lines."""
    h, w = image_rgb.shape[:2]
    if h < 80 or w < 80:
        return []

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([180, 95, 255]))
    yellow = cv2.inRange(hsv, np.array([14, 40, 90]), np.array([48, 255, 255]))
    paint = cv2.bitwise_or(white, yellow)
    paint = cv2.medianBlur(paint, 3)

    edges = cv2.Canny(paint, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=max(30, int(min(h, w) * 0.06)),
        minLineLength=max(50, int(h * 0.22)),
        maxLineGap=max(12, int(h * 0.05)),
    )
    if lines is None:
        return []

    raw_separators = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(float, line)
        length = np.hypot(x2 - x1, y2 - y1)
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        angle = angle if angle <= 90 else 180 - angle
        # This handles the common top-down/near-top-down case: parking spaces
        # are bounded by mostly vertical painted dividers.
        if angle >= 68 and length >= h * 0.20:
            raw_separators.append(((x1 + x2) / 2, min(y1, y2), max(y1, y2), length))

    if len(raw_separators) < 2:
        return []

    clusters = []
    for sep in sorted(raw_separators, key=lambda s: s[0]):
        if clusters and abs(sep[0] - np.mean([s[0] for s in clusters[-1]])) <= max(10, w * 0.018):
            clusters[-1].append(sep)
        else:
            clusters.append([sep])

    separators = []
    for cluster in clusters:
        separators.append({
            "x": float(np.mean([s[0] for s in cluster])),
            "y1": float(min(s[1] for s in cluster)),
            "y2": float(max(s[2] for s in cluster)),
        })

    boxes = []
    for left, right in zip(separators, separators[1:]):
        x1, x2 = int(round(left["x"])), int(round(right["x"]))
        bw = x2 - x1
        if not (w * 0.06 <= bw <= w * 0.32):
            continue

        y1 = max(0, int(round(min(left["y1"], right["y1"]) - bw * 0.10)))
        y2 = min(h, int(round(max(left["y2"], right["y2"]) + bw * 0.65)))
        if y2 - y1 < max(70, bw * 1.25):
            continue

        boxes.append({
            "x": x1, "y": y1,
            "w": bw, "h": y2 - y1,
            "label": "Empty",          # final label is assigned after car matching
            "confidence": 0.50,
            "source": "slot-lines",
        })

    return _dedupe_boxes(boxes, iou_threshold=0.20)


def _detect_slots_regular_grid(image_rgb: np.ndarray) -> list:
    """Infer a regular parking grid from painted lines, including multi-row lots."""
    h, w = image_rgb.shape[:2]
    if h < 80 or w < 80:
        return []

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 105, 255]))
    yellow = cv2.inRange(hsv, np.array([14, 35, 85]), np.array([50, 255, 255]))
    paint = cv2.medianBlur(cv2.bitwise_or(white, yellow), 3)
    edges = cv2.Canny(paint, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=28, minLineLength=max(35, int(min(h, w) * 0.10)), maxLineGap=max(12, int(min(h, w) * 0.06)))
    if lines is None:
        return []

    vertical, horizontal = [], []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(float, line)
        length = np.hypot(x2 - x1, y2 - y1)
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        angle = angle if angle <= 90 else 180 - angle
        if angle >= 68 and length >= h * 0.16:
            vertical.append({"x": (x1 + x2) / 2, "y1": min(y1, y2), "y2": max(y1, y2), "length": length})
        elif angle <= 18 and length >= w * 0.12:
            horizontal.append({"y": (y1 + y2) / 2, "x1": min(x1, x2), "x2": max(x1, x2), "length": length})

    if len(vertical) < 3:
        return []

    # Cluster duplicate edges from the same painted divider.
    clusters = []
    for item in sorted(vertical, key=lambda d: d["x"]):
        if clusters and abs(item["x"] - np.mean([v["x"] for v in clusters[-1]])) <= max(12, w * 0.018):
            clusters[-1].append(item)
        else:
            clusters.append([item])
    x_candidates = []
    for cluster in clusters:
        x_candidates.append({
            "x": float(np.mean([v["x"] for v in cluster])),
            "y1": float(min(v["y1"] for v in cluster)),
            "y2": float(max(v["y2"] for v in cluster)),
            "strength": float(sum(v["length"] for v in cluster)),
        })
    xs = [c["x"] for c in x_candidates if 0 <= c["x"] <= w]
    if len(xs) < 3:
        return []

    # Find the most plausible repeated stall width from candidate line spacing.
    best_start, best_step, best_score = None, None, -1
    sorted_x = sorted(xs)
    for i, start in enumerate(sorted_x[:-1]):
        for end in sorted_x[i + 1:]:
            step = end - start
            if not (w * 0.06 <= step <= w * 0.20):
                continue
            n = int(round((max(sorted_x) - start) / step)) + 1
            predicted = [start + k * step for k in range(n) if 0 <= start + k * step <= w]
            matches = sum(any(abs(p - x) <= step * 0.23 for x in sorted_x) for p in predicted)
            score = matches * 10 + len(predicted) - abs(step - np.median(np.diff(sorted_x))) * 0.03
            if score > best_score:
                best_start, best_step, best_score = start, step, score
    if best_start is None:
        return []

    # Extend the regular sequence across the visible row.
    x_lines = []
    x = best_start
    while x - best_step >= 0:
        x -= best_step
    while x <= w:
        if x >= 0:
            x_lines.append(float(x))
        x += best_step
    x_lines = [x for x in x_lines if 0 <= x <= w]
    if len(x_lines) < 3:
        return []

    top = max(0, int(np.percentile([v["y1"] for v in vertical], 10) - best_step * 0.10))
    # Use a robust percentile so non-parking vertical artifacts at the image
    # edges do not stretch a one-row parking area into the drive aisle.
    base_bottom = min(h, int(np.percentile([v["y2"] for v in vertical], 85) + best_step * 0.35))
    bottom = base_bottom

    h_clusters = []
    for item in sorted(horizontal, key=lambda d: d["y"]):
        if h_clusters and abs(item["y"] - np.mean([v["y"] for v in h_clusters[-1]])) <= max(10, h * 0.025):
            h_clusters[-1].append(item)
        else:
            h_clusters.append([item])
    strong_h = []
    for cluster in h_clusters:
        y = float(np.mean([v["y"] for v in cluster]))
        span = sum(v["length"] for v in cluster)
        # Row dividers can be broken by cars, so accept shorter horizontal
        # evidence once we are below the noisy top-car area.
        required_span = w * 0.15 if y > top + best_step * 1.45 else w * 0.35
        if span >= required_span:
            strong_h.append(y)
    if strong_h:
        bottom = min(h, int(max(bottom, max(strong_h) + best_step * 0.25)))

    y_internal = []
    for y in strong_h:
        # Ignore top artifacts from car roofs/text; real two-row separators are
        # usually near the middle/lower half of the visible parking grid.
        if max(top + best_step * 1.45, h * 0.42) < y < bottom - best_step * 0.65:
            y_internal.append(y)
    # For this geometry there should normally be at most one row separator.
    # Keeping the strongest/lower candidate avoids splitting cars into fake rows.
    valid_y_internal = []
    for y in y_internal:
        band = max(3, int(best_step * 0.08))
        coverages = []
        for x in x_lines:
            xi = int(round(x))
            sample = paint[int(round(y)):bottom, max(0, xi - band):min(w, xi + band)]
            coverages.append(cv2.countNonZero(sample) / max(1, sample.size))
        enough_lower_lines = sum(c > 0.02 for c in coverages) >= max(3, int(len(x_lines) * 0.70))
        strong_lower_lines = float(np.mean(coverages)) >= 0.35
        if enough_lower_lines and strong_lower_lines:
            valid_y_internal.append(y)

    if len(valid_y_internal) > 1:
        valid_y_internal = [max(valid_y_internal)]
    # The regular-grid extrapolator is only safe when a real second row is
    # confirmed. For one-row lots, use the direct separator detector instead;
    # otherwise car edges/logos can look like extra grid lines and duplicate or
    # hallucinate spaces into the drive aisle.
    if not valid_y_internal:
        return []
    y_lines = [float(top)] + sorted(valid_y_internal) + [float(bottom)]
    # Avoid tiny row slices caused by car edges/text.
    y_lines = [y_lines[0]] + [y for a, y in zip(y_lines, y_lines[1:]) if y - a >= best_step * 0.75]
    if y_lines[-1] != float(bottom) and bottom - y_lines[-1] >= best_step * 0.75:
        y_lines.append(float(bottom))
    if len(y_lines) < 2:
        y_lines = [float(top), float(bottom)]

    boxes = []
    for x1, x2 in zip(x_lines, x_lines[1:]):
        bw = x2 - x1
        if not (w * 0.055 <= bw <= w * 0.22):
            continue
        for y1, y2 in zip(y_lines, y_lines[1:]):
            bh = y2 - y1
            if bh < best_step * 0.75:
                continue

            xi1, xi2 = int(round(x1)), int(round(x2))
            yi1, yi2 = max(0, int(round(y1))), min(h, int(round(y2)))
            band = max(3, int(bw * 0.08))
            left_band = paint[yi1:yi2, max(0, xi1 - band):min(w, xi1 + band)]
            right_band = paint[yi1:yi2, max(0, xi2 - band):min(w, xi2 + band)]
            left_cov = cv2.countNonZero(left_band) / max(1, left_band.size)
            right_cov = cv2.countNonZero(right_band) / max(1, right_band.size)
            side_cov = max(left_cov, right_cov)
            mean_cov = (left_cov + right_cov) / 2

            # Do not create slots in blank drive aisles. Inferred grid lines are
            # allowed, but each proposed cell must still have real painted-line
            # evidence on at least one/both vertical borders inside that row.
            if side_cov < 0.025 or mean_cov < 0.010:
                continue

            boxes.append({
                "x": max(0, xi1), "y": yi1,
                "w": int(round(bw)), "h": yi2 - yi1,
                "label": "Empty",
                "confidence": 0.50,
                "source": "slot-lines",
            })
    return _dedupe_boxes(boxes, iou_threshold=0.15)


def _detect_cars(image_rgb: np.ndarray, threshold: float = 0.25) -> list:
    """Detect vehicles with a pretrained COCO YOLO model."""
    model = get_car_model()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result = model.predict(image_bgr, conf=threshold, imgsz=640, verbose=False)[0]
    vehicle_names = {"car", "truck", "bus", "motorcycle"}
    cars = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        name = str(model.names.get(class_id, class_id)).lower()
        if name not in vehicle_names:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cars.append({
            "x": max(0, int(round(x1))), "y": max(0, int(round(y1))),
            "w": max(1, int(round(x2 - x1))), "h": max(1, int(round(y2 - y1))),
            "confidence": float(box.conf[0]),
            "name": name,
        })
    return cars


def _intersection(a: dict, b: dict) -> int:
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    return max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))


def _slot_visual_occupancy_score(image_rgb: np.ndarray, slot: dict) -> float:
    """Backup occupancy score from texture/brightness when top-down cars are missed by COCO YOLO."""
    x1, y1 = slot["x"], slot["y"]
    x2, y2 = x1 + slot["w"], y1 + slot["h"]
    margin = max(3, int(slot["w"] * 0.15))
    roi = image_rgb[y1:y2, x1 + margin:max(x1 + margin + 1, x2 - margin)]
    if roi.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    edge_density = cv2.countNonZero(edges) / edges.size
    bright_ratio = ((roi[:, :, 0] > 160) & (roi[:, :, 1] > 160) & (roi[:, :, 2] > 160)).mean()
    color_std = float(roi.std())
    saturation = float(hsv[:, :, 1].mean())

    # Empty asphalt tends to be smooth, gray, low-contrast. Cars add strong
    # edges, bright body panels/windshields, and greater contrast.
    score = 0.0
    score += max(0.0, (edge_density - 0.035) / 0.08) * 0.45
    score += max(0.0, (bright_ratio - 0.10) / 0.30) * 0.30
    score += max(0.0, (color_std - 38) / 45) * 0.20
    score += max(0.0, (saturation - 55) / 80) * 0.05
    return float(min(1.0, score))


def _classify_slots_with_cars(image_rgb: np.ndarray, slots: list, cars: list) -> list:
    """Label line-detected slots using vehicle boxes plus visual occupancy cues."""
    detections = []
    for idx, slot in enumerate(slots, start=1):
        slot_area = max(1, slot["w"] * slot["h"])
        occupied_conf = 0.0
        for car in cars:
            inter = _intersection(slot, car)
            car_cx = car["x"] + car["w"] / 2
            car_cy = car["y"] + car["h"] / 2
            center_inside = slot["x"] <= car_cx <= slot["x"] + slot["w"] and slot["y"] <= car_cy <= slot["y"] + slot["h"]
            overlap_ratio = inter / slot_area
            if center_inside or overlap_ratio >= 0.08:
                occupied_conf = max(occupied_conf, car["confidence"])

        visual_score = _slot_visual_occupancy_score(image_rgb, slot)
        det = slot.copy()
        visual_threshold = 0.10 if slot["y"] < image_rgb.shape[0] * 0.25 and slot["h"] < image_rgb.shape[0] * 0.75 else 0.35
        if occupied_conf > 0:
            det.update({"label": "Occupied", "confidence": float(max(0.50, occupied_conf)), "source": "slot+car"})
        elif visual_score >= visual_threshold:
            det.update({"label": "Occupied", "confidence": float(max(0.50, min(0.82, visual_score))), "source": "slot+visual"})
        else:
            det.update({"label": "Empty", "confidence": float(max(0.55, 0.85 - visual_score)), "source": "slot+car"})
        detections.append(det)
    return detections


def _detect_empty_spaces_from_grid(image_rgb: np.ndarray) -> list:
    """
    Fallback detector for empty lots when the trained model does not fire.

    It looks for bright/yellow painted stall lines, clusters near-horizontal and
    near-vertical line positions, then proposes cells bounded by those lines.
    These are marked as Empty because this fallback is only reliable for open
    spaces with visible markings; occupied-space classification remains YOLO's
    job.
    """
    h, w = image_rgb.shape[:2]
    if h < 80 or w < 80:
        return []

    scale = min(1.0, 1280.0 / max(h, w))
    small = cv2.resize(image_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image_rgb
    sh, sw = small.shape[:2]

    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    # White paint: low saturation, high value. Yellow paint: common in lots too.
    white = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 90, 255]))
    yellow = cv2.inRange(hsv, np.array([15, 45, 100]), np.array([45, 255, 255]))
    mask = cv2.bitwise_or(white, yellow)
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    edges = cv2.Canny(mask, 50, 150)
    min_len = max(25, int(min(sw, sh) * 0.08))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=min_len, maxLineGap=max(8, int(min_len * 0.35)))
    if lines is None:
        return []

    xs, ys = [], []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(float, line)
        length = np.hypot(x2 - x1, y2 - y1)
        if length < min_len:
            continue
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        angle = angle if angle <= 90 else 180 - angle
        if angle <= 18:
            ys.append((y1 + y2) / 2)
        elif angle >= 72:
            xs.append((x1 + x2) / 2)

    x_lines = _cluster_positions(xs, max(8, sw * 0.018))
    y_lines = _cluster_positions(ys, max(8, sh * 0.018))
    if len(x_lines) < 2 or len(y_lines) < 2:
        return []

    boxes = []
    min_bw, max_bw = sw * 0.035, sw * 0.35
    min_bh, max_bh = sh * 0.035, sh * 0.35
    for x1, x2 in zip(x_lines, x_lines[1:]):
        bw = x2 - x1
        if not (min_bw <= bw <= max_bw):
            continue
        for y1, y2 in zip(y_lines, y_lines[1:]):
            bh = y2 - y1
            if not (min_bh <= bh <= max_bh):
                continue
            xi1, yi1, xi2, yi2 = map(lambda v: int(round(v)), (x1, y1, x2, y2))
            pad = max(2, int(min(bw, bh) * 0.08))
            border = np.zeros_like(mask)
            cv2.rectangle(border, (xi1, yi1), (xi2, yi2), 255, pad)
            border_pixels = cv2.countNonZero(border)
            score = cv2.countNonZero(cv2.bitwise_and(mask, border)) / border_pixels if border_pixels else 0
            if score < 0.08:
                continue
            boxes.append({
                "x": int(round(xi1 / scale)), "y": int(round(yi1 / scale)),
                "w": int(round(bw / scale)), "h": int(round(bh / scale)),
                "label": "Empty",
                "confidence": float(min(0.60, 0.25 + score)),
                "source": "line-fallback",
            })

    return _dedupe_boxes(boxes)


def _fallback_detect(image_rgb: np.ndarray) -> list:
    """Last-resort empty-space detector for lots with complete painted grids."""
    return _detect_empty_spaces_from_grid(image_rgb)


def make_grid_slots(image_shape: tuple, x: int, y: int, width: int, height: int, cols: int, rows: int) -> list:
    """Create a regular user-calibrated grid of parking slots."""
    img_h, img_w = image_shape[:2]
    cols, rows = max(1, int(cols)), max(1, int(rows))
    x = int(np.clip(x, 0, max(0, img_w - 1)))
    y = int(np.clip(y, 0, max(0, img_h - 1)))
    width = int(np.clip(width, 1, img_w - x))
    height = int(np.clip(height, 1, img_h - y))
    slot_w = width / cols
    slot_h = height / rows

    slots = []
    for r in range(rows):
        for c in range(cols):
            x1 = int(round(x + c * slot_w))
            y1 = int(round(y + r * slot_h))
            x2 = int(round(x + (c + 1) * slot_w))
            y2 = int(round(y + (r + 1) * slot_h))
            slots.append({
                "x": x1, "y": y1,
                "w": max(1, x2 - x1), "h": max(1, y2 - y1),
                "label": "Empty",
                "confidence": 0.50,
                "source": "manual-grid",
            })
    return slots


def classify_slots(image_rgb: np.ndarray, slots: list) -> list:
    """Classify provided parking slots as Empty/Occupied."""
    cars = _detect_cars(image_rgb, threshold=0.25)
    classified = _classify_slots_with_cars(image_rgb, slots, cars)
    for det in classified:
        if det.get("source") in {"slot+car", "slot+visual"}:
            det["source"] = "manual+occupancy" if slots and slots[0].get("source") == "manual-grid" else det["source"]
    return classified


def detect(image_rgb: np.ndarray, threshold: float = 0.5, model_path: Path = MODEL_PATH, use_fallback: bool = True) -> list:
    """
    Detect parking occupancy.

    Preferred pipeline:
      1. Detect parking-slot geometry from painted divider lines.
      2. Detect vehicles with pretrained COCO YOLO.
      3. Mark each slot Occupied if a vehicle overlaps it, otherwise Empty.

    If slot geometry cannot be found, it falls back to the previously trained
    parking-space YOLO model and finally to empty-grid detection.
    """
    if use_fallback:
        slots = _detect_slots_regular_grid(image_rgb)
        if len(slots) < 3:
            slots = _detect_slots_from_separators(image_rgb)
        if slots:
            cars = _detect_cars(image_rgb, threshold=0.25)
            return _classify_slots_with_cars(image_rgb, slots, cars)

    detections = _yolo_detect(image_rgb, threshold, model_path)
    if detections:
        return detections

    low_threshold = min(threshold, 0.12)
    if low_threshold < threshold:
        detections = _yolo_detect(image_rgb, low_threshold, model_path)
        if detections:
            return detections

    return _fallback_detect(image_rgb) if use_fallback else []


def annotate_image(image: np.ndarray, detections: list, show_confidence: bool = True) -> np.ndarray:
    """Draw colored bounding boxes with optional confidence labels on image."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        draw = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        draw = image.copy()

    overlay = draw.copy()

    for idx, det in enumerate(detections, start=1):
        x, y, w, h = det["x"], det["y"], det["w"], det["h"]
        color = COLOR_FREE_BGR if det["label"] == "Empty" else COLOR_OCCUPIED_BGR

        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.3, draw, 0.7, 0, draw)
        overlay = draw.copy()

        cv2.rectangle(draw, (x, y), (x + w, y + h), color, 2)

        if show_confidence:
            text = f"#{idx} {det['confidence']:.0%}"
            text_y = max(y - 5, 15)
            cv2.putText(draw, text, (x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    return cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)


def run_detection(image_path: Path, threshold: float = 0.5, show_confidence: bool = True) -> tuple:
    """Full pipeline: image file -> annotated image (RGB) + stats dict."""
    img_rgb = np.array(Image.open(image_path).convert("RGB"))
    detections = detect(img_rgb, threshold)
    annotated = annotate_image(img_rgb, detections, show_confidence)

    free = sum(1 for d in detections if d["label"] == "Empty")
    occupied = len(detections) - free
    stats = {
        "total": len(detections),
        "free": free,
        "occupied": occupied,
        "occupancy_pct": (occupied / len(detections) * 100) if detections else 0.0
    }

    return annotated, stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run parking space detection on an image")
    parser.add_argument("--image", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="annotated.jpg")
    args = parser.parse_args()

    annotated, stats = run_detection(Path(args.image), args.threshold)
    Image.fromarray(annotated).save(args.output)
    print(f"Results: {stats}")
    print(f"Saved to {args.output}")
