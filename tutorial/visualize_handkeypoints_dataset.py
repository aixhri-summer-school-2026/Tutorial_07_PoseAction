"""Part 1 - Look at the hand-keypoints dataset before training.

This downloads (if needed) the Ultralytics "hand-keypoints" dataset and shows a
few training images with their bounding box and 21 keypoints drawn on top.

Dataset reference:
    https://docs.ultralytics.com/datasets/pose/hand-keypoints

Controls:
    - any key : next image
    - q       : quit

Run (inside the container shell, needs the display forwarded like the tests):
    python tutorial/visualize_handkeypoints_dataset.py
"""

import argparse
import os
import random

import cv2
import numpy as np
from ultralytics.data.utils import check_det_dataset


def find_label_file(image_path):
    """Given an image path, return the matching YOLO label .txt path.

    YOLO stores labels next to images but in an ``images`` -> ``labels`` folder
    and with a .txt extension.
    """
    label_path = image_path.replace(os.sep + "images" + os.sep,
                                    os.sep + "labels" + os.sep)
    base, _ = os.path.splitext(label_path)
    return base + ".txt"


def draw_annotation(image, label_path, num_kpts):
    """Draw every hand (box + keypoints) described in one label file."""
    height, width = image.shape[:2]

    if not os.path.exists(label_path):
        return

    with open(label_path, "r") as f:
        lines = f.read().strip().splitlines()

    for line in lines:
        values = [float(v) for v in line.split()]
        # Layout: class cx cy w h  then (x y v) per keypoint, all normalized.
        cx, cy, bw, bh = values[1:5]
        x1 = int((cx - bw / 2) * width)
        y1 = int((cy - bh / 2) * height)
        x2 = int((cx + bw / 2) * width)
        y2 = int((cy + bh / 2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        kpt_values = values[5:]
        step = len(kpt_values) // num_kpts  # 2 (x,y) or 3 (x,y,visible)
        for k in range(num_kpts):
            kx = kpt_values[k * step + 0] * width
            ky = kpt_values[k * step + 1] * height
            cv2.circle(image, (int(kx), int(ky)), 3, (0, 0, 255), -1)


def main():
    parser = argparse.ArgumentParser(description="Visualize hand-keypoints data.")
    parser.add_argument("--data", default="hand-keypoints.yaml",
                        help="Ultralytics dataset yaml name or path.")
    parser.add_argument("--num", type=int, default=20,
                        help="How many random images to show.")
    args = parser.parse_args()

    print("Loading dataset info (downloads it the first time)...")
    info = check_det_dataset(args.data)

    # kpt_shape looks like [21, 3] -> 21 keypoints, 3 numbers each.
    num_kpts = int(info["kpt_shape"][0])
    print(f"Dataset path: {info['path']}")
    print(f"Keypoints per hand: {num_kpts}")
    print(f"Classes: {info['names']}")

    # The train entry can be a folder or a list file; handle the common folder.
    train_images_dir = info["train"]
    if isinstance(train_images_dir, list):
        train_images_dir = train_images_dir[0]

    image_paths = []
    for root, _, files in os.walk(train_images_dir):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, name))

    if not image_paths:
        print(f"No images found under {train_images_dir}")
        return

    random.shuffle(image_paths)
    print(f"Found {len(image_paths)} images. Showing {args.num}. Press 'q' to quit.")

    for image_path in image_paths[:args.num]:
        image = cv2.imread(image_path)
        if image is None:
            continue
        draw_annotation(image, find_label_file(image_path), num_kpts)

        cv2.imshow("hand-keypoints dataset", image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
