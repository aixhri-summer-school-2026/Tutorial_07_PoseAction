"""Part 1 - Train a hand detector + keypoints estimator with Ultralytics.

We fine-tune a small pretrained pose model (yolo11n-pose) on the
"hand-keypoints" dataset. The result is a model that, given an image, returns
hand bounding boxes AND 21 keypoints per hand.

This is a "prepare before" step: training takes a while, so do it before the
session (or use a low number of epochs for a quick demo). The best weights are
copied to ``tutorial/weights/hand_detector.pt`` for the live scripts to use.

Reference:
    https://docs.ultralytics.com/datasets/pose/hand-keypoints

Run (inside the container shell):
    python tutorial/train_hand_detector.py --epochs 20
"""

import argparse
import os
import shutil

from ultralytics import YOLO

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
DETECTOR_PATH = os.path.join(WEIGHTS_DIR, "hand_detector.pt")


def main():
    parser = argparse.ArgumentParser(description="Train the hand keypoint model.")
    parser.add_argument("--data", default="hand-keypoints.yaml")
    parser.add_argument("--model", default="yolo11n-pose.pt",
                        help="Pretrained pose model to start from.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None,
                        help="e.g. 0 for first GPU, or 'cpu'. Default: auto.")
    args = parser.parse_args()

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print(f"Starting from pretrained model: {args.model}")
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=os.path.join(WEIGHTS_DIR, "runs"),
        name="hand_detector",
        exist_ok=True,
    )

    # Ultralytics saves the best weights inside the run folder. Copy them out
    # to a stable, easy-to-find location for the rest of the tutorial.
    best = os.path.join(results.save_dir, "weights", "best.pt")
    if os.path.exists(best):
        shutil.copy(best, DETECTOR_PATH)
        print(f"\nDone. Best weights copied to: {DETECTOR_PATH}")
    else:
        print(f"\n[WARN] could not find best.pt at {best}")


if __name__ == "__main__":
    main()
