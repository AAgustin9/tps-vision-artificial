"""Interactive OpenCV tool for defining parking space ROIs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import cv2
import numpy as np
from utils import LOT_CONFIG_PATH, DEFAULT_CONFIGS


def load_config(config_path: Path = LOT_CONFIG_PATH) -> dict:
    """Load lot configs, merging with defaults."""
    config = dict(DEFAULT_CONFIGS)
    if config_path.exists():
        with open(config_path) as f:
            loaded = json.load(f)
        config.update(loaded)
    return config


def save_config(config: dict, config_path: Path = LOT_CONFIG_PATH) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


class SpaceSelector:
    """Interactive OpenCV-based parking space ROI selector."""

    def __init__(self, image_path: Path):
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        self.spaces = []
        self._drawing = False
        self._start = None
        self._current = None

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start = (x, y)
            self._current = None
        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            if self._start:
                x1, y1 = self._start
                rx = min(x1, x)
                ry = min(y1, y)
                rw = abs(x - x1)
                rh = abs(y - y1)
                if rw > 5 and rh > 5:
                    self.spaces.append({
                        "id": len(self.spaces) + 1,
                        "x": rx, "y": ry, "w": rw, "h": rh
                    })
            self._start = None
            self._current = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.spaces:
                self.spaces.pop()

    def run(self, lot_name: str) -> list:
        """Run interactive selection loop. Returns list of space dicts."""
        win = f"Space Selector — {lot_name}"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self._mouse_callback)

        print(f"\nDefining spaces for lot: {lot_name}")
        print("  Left-click drag: draw a space")
        print("  Right-click: undo last space")
        print("  Q: quit and save | R: reset all | S: save intermediate")

        while True:
            frame = self.image.copy()

            for space in self.spaces:
                cv2.rectangle(frame,
                              (space["x"], space["y"]),
                              (space["x"] + space["w"], space["y"] + space["h"]),
                              (0, 200, 0), 2)
                cv2.putText(frame, str(space["id"]),
                            (space["x"] + 3, space["y"] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

            if self._drawing and self._start and self._current:
                cv2.rectangle(frame, self._start, self._current, (0, 200, 200), 1)

            cv2.putText(frame, f"Spaces: {len(self.spaces)} | Q=save  R=reset  RClick=undo",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow(win, frame)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                self.spaces = []
            elif key == ord("s"):
                print(f"  Intermediate save: {len(self.spaces)} spaces")

        cv2.destroyAllWindows()
        return self.spaces


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Define parking space ROIs interactively")
    parser.add_argument("--image", required=True, help="Path to reference parking lot image")
    parser.add_argument("--lot-name", required=True, help="Name for this lot configuration")
    parser.add_argument("--output-config", default=str(LOT_CONFIG_PATH))
    args = parser.parse_args()

    selector = SpaceSelector(Path(args.image))
    spaces = selector.run(args.lot_name)

    config_path = Path(args.output_config)
    config = load_config(config_path)
    config[args.lot_name] = {
        "description": f"User-defined lot: {args.lot_name}",
        "spaces": spaces
    }
    save_config(config, config_path)
    print(f"\nSaved {len(spaces)} spaces for '{args.lot_name}' to {config_path}")
