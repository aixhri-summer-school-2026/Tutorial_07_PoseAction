"""Part 2 - PREP STEP (run this once before the summer school).

Goal: build a small, ready-to-train dataset of hand keypoints from the
HaGRIDv2 annotations that are already in this repository
(``HaGRIDv2_annotations/``).

Good news: the annotations already contain the 21 hand landmarks for every
hand, so we DO NOT need to download any images. We just pick a few hundred
hands per gesture and save their keypoints to compact ``.npz`` files.

What it produces (in ``tutorial/data/hagrid_keypoints/``):
    train.npz, val.npz, test.npz   -> arrays X (N, 21, 2) and y (N,)
    labels.json                    -> the class names (index = label)

Note about the splits: the official HaGRID val/test folders do not contain the
heart class, so to keep all 8 gestures in every split we take all of our
samples from the ``train`` annotation folder and slice it ourselves.

Run (inside the container shell):
    python tutorial/prepare_hagrid_dataset.py
"""

import argparse
import json
import os

import numpy as np

from gesture_utils import GESTURES, HAGRID_LABEL_MAP, NUM_KEYPOINTS

# Folder that holds the per-gesture json annotation files.
ANNOTATIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "HaGRIDv2_annotations",
    "train",
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "hagrid_keypoints")

# How many hands per gesture for each split.
N_TRAIN = 100
N_VAL = 50
N_TEST = 50


def collect_hands_for_gesture(gesture, max_needed):
    """Read one annotation file and return up to ``max_needed`` hands.

    Each returned item is a (21, 2) numpy array of normalized [0, 1] keypoints.
    """
    hagrid_label = HAGRID_LABEL_MAP[gesture]
    json_path = os.path.join(ANNOTATIONS_DIR, f"{hagrid_label}.json")

    if not os.path.exists(json_path):
        print(f"  [WARN] annotation file not found: {json_path}")
        return []

    with open(json_path, "r") as f:
        annotations = json.load(f)

    hands = []
    # Every entry can describe several hands. We keep only the hands whose
    # label is exactly the gesture we want, and that have all 21 landmarks.
    for sample in annotations.values():
        labels = sample["labels"]
        landmarks_per_hand = sample["hand_landmarks"]

        for hand_index in range(len(labels)):
            if labels[hand_index] != hagrid_label:
                continue
            landmarks = landmarks_per_hand[hand_index]
            if len(landmarks) != NUM_KEYPOINTS:
                continue  # skip hands without full landmarks

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

    total = args.n_train + args.n_val + args.n_test

    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []

    for label_index, gesture in enumerate(GESTURES):
        print(f"Gesture '{gesture}' (label {label_index}):")
        hands = collect_hands_for_gesture(gesture, total)
        print(f"  collected {len(hands)} hands")

        if len(hands) < total:
            print(f"  [WARN] wanted {total} but only found {len(hands)}; "
                  f"this gesture will have fewer samples.")

        # Slice the pool into the three splits.
        train_hands = hands[:args.n_train]
        val_hands = hands[args.n_train:args.n_train + args.n_val]
        test_hands = hands[args.n_train + args.n_val:total]

        for h in train_hands:
            train_x.append(h)
            train_y.append(label_index)
        for h in val_hands:
            val_x.append(h)
            val_y.append(label_index)
        for h in test_hands:
            test_x.append(h)
            test_y.append(label_index)

    print("\nSaving splits...")
    save_split("train", train_x, train_y)
    save_split("val", val_x, val_y)
    save_split("test", test_x, test_y)

    with open(os.path.join(OUTPUT_DIR, "labels.json"), "w") as f:
        json.dump(GESTURES, f, indent=2)

    print(f"\nDone. Dataset written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
