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

from keypoints_utils import draw_hand_edges, HAND21_EDGES_MEDIAPIPE
from handkeypoints_dataset import HandKeypointDataset

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "hagrid_keypoints")

CANVAS_W = 640
CANVAS_H = 480

NORM_CANVAS_W = 480
NORM_CANVAS_H = 480


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
    draw_hand_edges(frame, kpts_px, edges=HAND21_EDGES_MEDIAPIPE)
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the HaGRID keypoint dataset.")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--split", choices=["train", "val", "test"],
                        default="train")
    args = parser.parse_args()


    # Load raw data
    npz_path = os.path.join(args.data_dir, f"{args.split}.npz")
    labels_path = os.path.join(args.data_dir, "labels.json")

    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    with open(labels_path) as f:
        labels = json.load(f)

    # Load with dataset
    dataset = HandKeypointDataset(npz_path, augment=True)
    
    # Shuffle indices
    indices = np.arange(len(dataset))
    random.shuffle(indices)

    print(f"Split: {args.split}  ({len(dataset)} samples)")
    print(f"Showing all samples. Press any key for next, 'q' to quit.")

    for index in indices:
        
        # from raw data
        gesture = labels[str(int(y[index]))]
        raw_data_frame = draw_sample(X[index], gesture, CANVAS_W, CANVAS_H)
        
        # from dataset
        keypoints_dataset, label_dataset = dataset[index]
        gesture_dataset = labels[str(int(label_dataset))]
        # keypoints are in [-1.0, 1.0] we remap to [0, 1] (keypoints + 1.0)/2.0 then apply keypoints_to_pixels
        dataset_frame = draw_sample((keypoints_dataset + 1.0)/2.0, gesture_dataset, NORM_CANVAS_W, NORM_CANVAS_H)
        
        # concat frames
        black_line = np.full((NORM_CANVAS_H, 10, 3), 0, dtype=np.uint8)
        frame = np.concatenate((raw_data_frame, black_line, dataset_frame), axis=1)
        
        cv2.putText(frame, f"Index: {index} | Label: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("HaGRID keypoints", frame)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
