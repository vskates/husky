from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def _is_bg_class(class_name: str) -> bool:
    name = (class_name or "").strip().lower()
    return name.startswith("bg")


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO segmentation on one RGB image and save a binary mask.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_bgr = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(args.input)

    model = YOLO(str(args.weights))
    names = getattr(model, "names", {})
    class_names = dict(names) if hasattr(names, "items") else dict(enumerate(names))
    prediction = model.predict(
        source=image_bgr,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )[0]

    height, width = image_bgr.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    if prediction.masks is not None and prediction.masks.data is not None and len(prediction.masks.data) > 0:
        masks = prediction.masks.data
        valid_ix = list(range(len(masks)))
        if prediction.boxes is not None and len(prediction.boxes.cls) == len(masks) and class_names:
            class_ids = prediction.boxes.cls.cpu().numpy().astype(int)
            valid_ix = [
                idx for idx in valid_ix
                if not _is_bg_class(class_names.get(class_ids[idx], str(class_ids[idx])))
            ]
        if valid_ix:
            masks = masks[valid_ix]
            mask = (masks > 0.5).any(dim=0).to("cpu").numpy().astype(np.uint8) * 255
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), mask):
        raise RuntimeError(f"Failed to write mask to {args.output}")


if __name__ == "__main__":
    main()
