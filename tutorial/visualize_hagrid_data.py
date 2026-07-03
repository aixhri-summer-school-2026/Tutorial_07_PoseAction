"""Part 2 - Look at the HaGRID keypoint dataset.

Shows random hands from the .npz files produced by prepare_hagrid_dataset.py.
There are no images here, just the 21 keypoints stored in normalized [0, 1]
coordinates, so each sample is drawn on a blank canvas.

Controls:
    - any key : next sample
    - q       : quit

Run (inside the container shell, display forwarded like the tests):
    python tutorial/visualize_hagrid_data.py
"""

import argparse
import json
import os
import random

import cv2
import numpy as np

from gesture_utils import draw_hand, draw_label

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "hagrid_keypoints")

CANVAS_W = 640
CANVAS_H = 480


def keypoints_to_pixels(keypoints, width, height):
    """Turn [0, 1] keypoints into pixel coords on a canvas."""
    kpts = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    kpts_px = kpts.copy()
    kpts_px[:, 0] *= width
    kpts_px[:, 1] *= height
    return kpts_px


def draw_sample(keypoints, gesture_name, width, height):
    """Draw one hand on a white canvas and return the image."""
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    kpts_px = keypoints_to_pixels(keypoints, width, height)
    draw_hand(frame, kpts_px)
    draw_label(frame, gesture_name, (10, 30))
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the HaGRID keypoint dataset.")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--split", choices=["train", "val", "test"],
                        default="train")
    parser.add_argument("--num", type=int, default=100,
                        help="How many random samples to show. use -1 to show all samples.")
    args = parser.parse_args()

    npz_path = os.path.join(args.data_dir, f"{args.split}.npz")
    labels_path = os.path.join(args.data_dir, "labels.json")

    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    with open(labels_path) as f:
        labels = json.load(f)

    n = len(X) if args.num == -1 else min(args.num, len(X))
    indices = random.sample(range(len(X)), n)
    random.shuffle(indices)

    print(f"Split: {args.split}  ({len(X)} samples)")
    print(f"Showing {n} samples. Press any key for next, 'q' to quit.")

    for index in indices:
        gesture = labels[str(int(y[index]))]
        frame = draw_sample(X[index], gesture, CANVAS_W, CANVAS_H)
        cv2.imshow("HaGRID keypoints", frame)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
