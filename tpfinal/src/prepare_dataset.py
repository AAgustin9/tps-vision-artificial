"""
Prepares the PKLot dataset by extracting and sampling balanced space crops.
PKLot structure: data/pklot_raw/<LotName>/<Condition>/<Date>/<filename>.{jpg,xml}
Each XML contains <space> elements with bounding box info.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import random
import xml.etree.ElementTree as ET
from PIL import Image
from utils import (
    DATA_RAW, EMPTY_DIR, OCCUPIED_DIR, ensure_dirs, IMG_SIZE
)


def parse_xml(xml_path: Path) -> list:
    """Parse PKLot XML file, return list of space dicts with bounding boxes."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return []

    root = tree.getroot()
    spaces = []

    for space in root.findall("space"):
        occupied = space.get("occupied") == "1"

        bbox = None

        # Try contour (4 Point children)
        contour = space.find("contour")
        if contour is not None:
            points = contour.findall("Point")
            if len(points) >= 2:
                xs = [int(p.get("x", 0)) for p in points]
                ys = [int(p.get("y", 0)) for p in points]
                bbox = (min(xs), min(ys), max(xs), max(ys))

        # Fallback: rotatedRect
        if bbox is None:
            rr = space.find("rotatedRect")
            if rr is not None:
                center = rr.find("center")
                size = rr.find("size")
                if center is not None and size is not None:
                    cx = float(center.get("x", 0))
                    cy = float(center.get("y", 0))
                    w = float(size.get("w", 0))
                    h = float(size.get("h", 0))
                    bbox = (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))

        if bbox is None:
            continue

        xmin, ymin, xmax, ymax = bbox
        if (xmax - xmin) < 5 or (ymax - ymin) < 5:
            continue
        if xmin < 0 or ymin < 0:
            continue

        spaces.append({
            "occupied": occupied,
            "xmin": xmin, "ymin": ymin,
            "xmax": xmax, "ymax": ymax
        })

    return spaces


def collect_samples(raw_dir: Path) -> dict:
    """Walk PKLot directory, collect all annotated space references."""
    samples = {"empty": [], "occupied": []}

    for xml_path in raw_dir.rglob("*.xml"):
        jpg_path = xml_path.with_suffix(".jpg")
        if not jpg_path.exists():
            continue

        spaces = parse_xml(xml_path)
        for space in spaces:
            key = "occupied" if space["occupied"] else "empty"
            samples[key].append((jpg_path, space))

    return samples


def sample_balanced(samples: dict, n_per_class: int = 2000, seed: int = 42) -> dict:
    """Sample at most n_per_class items per class, uniformly."""
    rng = random.Random(seed)
    result = {}
    for key, items in samples.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        result[key] = shuffled[:n_per_class]
    return result


def crop_and_save(samples: dict) -> dict:
    """Crop each space from its source image and save to processed dirs."""
    output_dirs = {"empty": EMPTY_DIR, "occupied": OCCUPIED_DIR}
    counts = {"empty": 0, "occupied": 0}

    for key, items in samples.items():
        out_dir = output_dirs[key]
        for idx, (jpg_path, space) in enumerate(items):
            out_name = f"{jpg_path.stem}_s{idx}.jpg"
            out_path = out_dir / out_name

            if out_path.exists():
                counts[key] += 1
                continue

            try:
                img = Image.open(jpg_path).convert("RGB")
                w, h = img.size

                pad = 4
                xmin = max(0, space["xmin"] - pad)
                ymin = max(0, space["ymin"] - pad)
                xmax = min(w, space["xmax"] + pad)
                ymax = min(h, space["ymax"] + pad)

                crop = img.crop((xmin, ymin, xmax, ymax))
                crop = crop.resize(IMG_SIZE, Image.LANCZOS)
                crop.save(out_path, "JPEG", quality=90)
                counts[key] += 1
            except Exception as e:
                print(f"  Warning: could not process {jpg_path}: {e}")

    return counts


def main():
    ensure_dirs()

    # Idempotency check
    n_empty = len(list(EMPTY_DIR.glob("*.jpg")))
    n_occupied = len(list(OCCUPIED_DIR.glob("*.jpg")))
    if n_empty >= 1000 and n_occupied >= 1000:
        print(f"Dataset already prepared: {n_empty} empty, {n_occupied} occupied. Skipping.")
        return

    if not DATA_RAW.exists() or not any(DATA_RAW.iterdir()):
        print(f"ERROR: PKLot dataset not found at {DATA_RAW}")
        print("Please download PKLot and place it at data/pklot_raw/")
        print("  https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset")
        return

    print("Scanning PKLot dataset...")
    samples = collect_samples(DATA_RAW)
    print(f"  Found {len(samples['empty'])} empty, {len(samples['occupied'])} occupied space annotations")

    print("Sampling balanced subset (max 2000 per class)...")
    balanced = sample_balanced(samples, n_per_class=2000)

    print("Cropping and saving space images...")
    counts = crop_and_save(balanced)

    print(f"\nDataset preparation complete:")
    print(f"  Empty:    {counts['empty']} crops → data/processed/empty/")
    print(f"  Occupied: {counts['occupied']} crops → data/processed/occupied/")
    print(f"  Total:    {counts['empty'] + counts['occupied']}")


if __name__ == "__main__":
    main()
