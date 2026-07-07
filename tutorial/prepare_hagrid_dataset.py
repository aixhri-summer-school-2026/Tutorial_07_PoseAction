"""Part 2 - PREP STEP (run this once before the summer school).

Goal: build a small, ready-to-train dataset of hand keypoints from the
HaGRIDv2 annotations that are already in this repository
(``HaGRIDv2_annotations/``).

Good news: the annotations already contain the 21 hand landmarks for every
hand, so we DO NOT need to download any images. We just pick a few hands per
gesture from each official split and save their keypoints to compact ``.npz``
files.

What it produces (in ``tutorial/data/hagrid_keypoints/``):
    train.npz, val.npz, test.npz   -> arrays X (N, 21, 2) and y (N,)
    labels.json                    -> id -> class name mapping

Run (inside the container shell):
    python tutorial/prepare_hagrid_dataset.py
"""

import argparse
import json
import os

import numpy as np

from keypoints_utils import GESTURES, NUM_KEYPOINTS_HAND

ANNOTATIONS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "HaGRIDv2_annotations",
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "hagrid_keypoints")

N_TRAIN = 400
N_VAL = 100
N_TEST = 100


def collect_hands_for_gesture(gesture, split, max_needed):
    """Read one annotation file and return up to ``max_needed`` hands.

    Each returned item is a (21, 2) numpy array of normalized [0, 1] keypoints.
    """
    json_path = os.path.join(ANNOTATIONS_ROOT, split, f"{gesture}.json")

    if not os.path.exists(json_path):
        print(f"  [WARN] annotation file not found: {json_path}")
        return []

    with open(json_path, "r") as f:
        annotations = json.load(f)

    hands = []
    for sample in annotations.values():
        labels = sample["labels"]
        landmarks_per_hand = sample["hand_landmarks"]

        for hand_index in range(len(labels)):
            if labels[hand_index] != gesture:
                continue
            landmarks = landmarks_per_hand[hand_index]
            if len(landmarks) != NUM_KEYPOINTS_HAND:
                continue

            hands.append(np.array(landmarks, dtype=np.float32))
            if len(hands) >= max_needed:
                return hands

    return hands


def save_split(name, samples_x, samples_y):
    path = os.path.join(OUTPUT_DIR, f"{name}.npz")
    x = np.stack(samples_x).astype(np.float32)   # (N, 21, 2)
    y = np.array(samples_y, dtype=np.int64)       # (N,)
    np.savez(path, X=x, y=y)
    print(f"  saved {name}.npz: X={x.shape}, y={y.shape}")


def main():
    parser = argparse.ArgumentParser(description="Subsample HaGRID keypoints.")
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-val", type=int, default=N_VAL)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    split_sizes = {
        "train": args.n_train,
        "val": args.n_val,
        "test": args.n_test,
    }
    split_data = {name: ([], []) for name in split_sizes}

    for label_index, gesture in enumerate(GESTURES):
        print(f"Gesture '{gesture}' (label {label_index}):")
        for split, n_wanted in split_sizes.items():
            hands = collect_hands_for_gesture(gesture, split, n_wanted)
            print(f"  {split}: collected {len(hands)} hands")
            if len(hands) < n_wanted:
                raise ValueError(f"Wanted {n_wanted} in {split} but only found {len(hands)}")

            x_list, y_list = split_data[split]
            for hand in hands:
                x_list.append(hand)
                y_list.append(label_index)

    print("\nSaving splits...")
    for split in split_sizes:
        x_list, y_list = split_data[split]
        save_split(split, x_list, y_list)

    label_map = {str(i): name for i, name in enumerate(GESTURES)}
    with open(os.path.join(OUTPUT_DIR, "labels.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nDone. Dataset written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
