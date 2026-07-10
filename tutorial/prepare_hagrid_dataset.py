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

from keypoints_utils import GESTURE_SOURCES, GESTURES, NUM_KEYPOINTS_HAND

ANNOTATIONS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "HaGRIDv2_annotations",
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "hagrid_keypoints")

N_TRAIN = 1000
N_VAL = 200
N_TEST = 200

N_NO_GESTURE_TRAIN = 2000
N_NO_GESTURE_VAL = 200
N_NO_GESTURE_TEST = 200


def get_used_hagrid_sources():
    """Return HaGRID annotation classes already mapped to tutorial gestures."""
    used = set()
    for gesture in GESTURES:
        if gesture == "no_gesture":
            continue
        used.update(GESTURE_SOURCES.get(gesture, [gesture]))
    return used


def get_unused_hagrid_classes(split):
    """Return HaGRID classes not used by any tutorial gesture."""
    split_dir = os.path.join(ANNOTATIONS_ROOT, split)
    all_classes = {
        filename[:-5]
        for filename in os.listdir(split_dir)
        if filename.endswith(".json")
    }
    return sorted(all_classes - get_used_hagrid_sources() - {"no_gesture"})


def collect_mixed_from_sources(sources, split, max_needed):
    """Collect hands evenly mixed across several HaGRID source classes."""
    if max_needed == 0:
        return []

    hands = []
    base = max_needed // len(sources)
    remainder = max_needed % len(sources)
    for index, source in enumerate(sources):
        n_wanted = base + (1 if index < remainder else 0)
        if n_wanted == 0:
            continue
        collected = collect_hands_from_source(source, split, n_wanted)
        print(f"    from {source}: {len(collected)} hands")
        hands.extend(collected)

    if len(hands) < max_needed:
        for source in sources:
            if len(hands) >= max_needed:
                break
            still_needed = max_needed - len(hands)
            extra = collect_hands_from_source(source, split, still_needed)
            hands.extend(extra)

    return hands[:max_needed]


def collect_no_gesture_hands(split, n_wanted):
    """Build no_gesture samples from true negatives and other unused gestures."""
    n_direct = n_wanted // 2
    n_mixed = n_wanted - n_direct

    hands = collect_hands_from_source("no_gesture", split, n_direct)
    print(f"    from no_gesture: {len(hands)} hands")

    unused_classes = get_unused_hagrid_classes(split)
    if not unused_classes:
        raise ValueError(f"No unused HaGRID classes found for split '{split}'")

    mixed = collect_mixed_from_sources(unused_classes, split, n_mixed)
    hands.extend(mixed)
    return hands[:n_wanted]


def split_sizes_for_gesture(gesture, args):
    if gesture == "no_gesture":
        return {
            "train": args.n_no_gesture_train,
            "val": args.n_no_gesture_val,
            "test": args.n_no_gesture_test,
        }
    return {
        "train": args.n_train,
        "val": args.n_val,
        "test": args.n_test,
    }


def collect_hands_from_source(source_gesture, split, max_needed):
    """Read one HaGRID annotation file and return up to ``max_needed`` hands.

    Each returned item is a (21, 2) numpy array of normalized [0, 1] keypoints.
    """
    json_path = os.path.join(ANNOTATIONS_ROOT, split, f"{source_gesture}.json")

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
            if labels[hand_index] != source_gesture:
                continue
            landmarks = landmarks_per_hand[hand_index]
            if len(landmarks) != NUM_KEYPOINTS_HAND:
                continue

            hands.append(np.array(landmarks, dtype=np.float32))
            if len(hands) >= max_needed:
                return hands

    return hands


def collect_hands_for_gesture(gesture, split, max_needed):
    """Collect hands for a tutorial gesture, sampling all mapped HaGRID sources."""
    sources = GESTURE_SOURCES.get(gesture, [gesture])
    if len(sources) == 1:
        return collect_hands_from_source(sources[0], split, max_needed)

    hands = []
    base = max_needed // len(sources)
    remainder = max_needed % len(sources)
    for index, source in enumerate(sources):
        n_wanted = base + (1 if index < remainder else 0)
        collected = collect_hands_from_source(source, split, n_wanted)
        print(f"    from {source}: {len(collected)} hands")
        hands.extend(collected)

    if len(hands) < max_needed:
        for source in sources:
            if len(hands) >= max_needed:
                break
            still_needed = max_needed - len(hands)
            extra = collect_hands_from_source(source, split, still_needed)
            hands.extend(extra)

    return hands[:max_needed]


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
    parser.add_argument("--n-no-gesture-train", type=int, default=N_NO_GESTURE_TRAIN)
    parser.add_argument("--n-no-gesture-val", type=int, default=N_NO_GESTURE_VAL)
    parser.add_argument("--n-no-gesture-test", type=int, default=N_NO_GESTURE_TEST)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    split_names = ("train", "val", "test")
    split_data = {name: ([], []) for name in split_names}

    for label_index, gesture in enumerate(GESTURES):
        print(f"Gesture '{gesture}' (label {label_index}):")
        gesture_split_sizes = split_sizes_for_gesture(gesture, args)
        for split in split_names:
            n_wanted = gesture_split_sizes[split]
            if gesture == "no_gesture":
                hands = collect_no_gesture_hands(split, n_wanted)
            else:
                hands = collect_hands_for_gesture(gesture, split, n_wanted)
            print(f"  {split}: collected {len(hands)} hands")
            if len(hands) < n_wanted:
                raise ValueError(f"Wanted {n_wanted} in {split} but only found {len(hands)}")

            x_list, y_list = split_data[split]
            for hand in hands:
                x_list.append(hand)
                y_list.append(label_index)

    print("\nSaving splits...")
    for split in split_names:
        x_list, y_list = split_data[split]
        save_split(split, x_list, y_list)

    label_map = {str(i): name for i, name in enumerate(GESTURES)}
    with open(os.path.join(OUTPUT_DIR, "labels.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nDone. Dataset written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
