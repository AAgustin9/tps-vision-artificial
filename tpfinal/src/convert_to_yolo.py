"""
Converts the Roboflow COCO export of PKLot into a YOLO-format dataset.

Expected input layout (data/pklot_raw/):
    {train,valid,test}/<images>.jpg
    {train,valid,test}/_annotations.coco.json
    Categories named like "space-empty" / "space-occupied".

Output layout (data/yolo/):
    images/{train,val,test}/<images>.jpg
    labels/{train,val,test}/<images>.txt   (YOLO format: class cx cy w h, normalized)
    data.yaml
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import DATA_RAW, DATA_YOLO, CLASS_NAMES, ensure_dirs

SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}


def classify_category(name: str):
    """Map a COCO category name to a YOLO class index, or None if not a space class."""
    lower = name.lower()
    if "occ" in lower:
        return CLASS_NAMES.index("occupied")
    if "empty" in lower or "free" in lower or "vacant" in lower:
        return CLASS_NAMES.index("empty")
    return None


def find_coco_json(split_dir: Path):
    for candidate in split_dir.glob("*.json"):
        try:
            with open(candidate) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if all(k in data for k in ("images", "annotations", "categories")):
            return data
    return None


def convert_split(raw_split_dir: Path, out_split: str) -> int:
    coco = find_coco_json(raw_split_dir)
    if coco is None:
        return 0

    cat_id_to_class = {
        cat["id"]: classify_category(cat.get("name", ""))
        for cat in coco["categories"]
    }

    anns_by_image = {}
    for ann in coco["annotations"]:
        class_id = cat_id_to_class.get(ann["category_id"])
        if class_id is None:
            continue
        anns_by_image.setdefault(ann["image_id"], []).append((class_id, ann["bbox"]))

    images_out = DATA_YOLO / "images" / out_split
    labels_out = DATA_YOLO / "labels" / out_split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for img in coco["images"]:
        src_img = raw_split_dir / img["file_name"]
        if not src_img.exists():
            continue

        anns = anns_by_image.get(img["id"], [])
        if not anns:
            continue

        img_w, img_h = img["width"], img["height"]
        lines = []
        for class_id, (x, y, w, h) in anns:
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        dest_img = images_out / src_img.name
        shutil.copy2(src_img, dest_img)
        (labels_out / f"{src_img.stem}.txt").write_text("\n".join(lines) + "\n")
        count += 1

    return count


def write_data_yaml():
    yaml_path = DATA_YOLO / "data.yaml"
    yaml_path.write_text(
        f"path: {DATA_YOLO}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES))
    )
    return yaml_path


def main():
    ensure_dirs()

    if not DATA_RAW.exists() or not any(DATA_RAW.iterdir()):
        print(f"ERROR: PKLot dataset not found at {DATA_RAW}")
        print("Extract the Roboflow PKLot COCO export there first.")
        return

    print("Converting COCO annotations to YOLO format...")
    total = 0
    for raw_split, out_split in SPLIT_MAP.items():
        raw_split_dir = DATA_RAW / raw_split
        if not raw_split_dir.exists():
            continue
        n = convert_split(raw_split_dir, out_split)
        print(f"  {raw_split} -> {out_split}: {n} images")
        total += n

    if total == 0:
        print("ERROR: No annotated images converted. Check the dataset structure.")
        return

    yaml_path = write_data_yaml()
    print(f"\nDone. {total} images converted.")
    print(f"Dataset config written to {yaml_path}")


if __name__ == "__main__":
    main()
