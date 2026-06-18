"""Part 2 - Live gesture classification on the robot camera.

Pipeline for every frame:
    1. detect hands + 21 keypoints with the Part 1 model,
    2. feed each hand's keypoints into the Part 2 classifier,
    3. draw the predicted gesture next to the hand.

Controls:
    q : quit

Run (inside the container shell, display forwarded):
    python tutorial/visualize_pose_classification_live.py
"""

import argparse
import os

import cv2
import torch
from reachy_mini import ReachyMini
from ultralytics import YOLO

from gesture_utils import classify_hand, draw_hand, draw_label, load_classifier

HERE = os.path.dirname(os.path.abspath(__file__))
# Pretrained hand detector by default (use hand_detector.pt if you trained one).
DEFAULT_DETECTOR = os.path.join(HERE, "weights", "yolo26s-pose-hands.pt")
DEFAULT_CLASSIFIER = os.path.join(HERE, "weights", "gesture_classifier.pt")


def main():
    parser = argparse.ArgumentParser(description="Live gesture classification.")
    parser.add_argument("--detector", default=DEFAULT_DETECTOR)
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER)
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    for path in (args.detector, args.classifier):
        if not os.path.exists(path):
            print(f"[ERROR] missing file: {path}")
            print("Make sure you trained both Part 1 and Part 2 models.")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    detector = YOLO(args.detector)
    classifier, labels = load_classifier(args.classifier, device)
    print(f"Classes: {labels}")

    # Create the window before connecting (avoids the viewer blocking).
    window = "Part 2 - live gesture classification"
    cv2.namedWindow(window)

    print("Connecting to Reachy Mini camera... press 'q' to quit.")
    with ReachyMini() as mini:
        while True:
            frame = mini.media.get_frame()
            if frame is None:
                continue
            # The camera frame is read-only; copy it so we can draw on it.
            frame = frame.copy()

            results = detector(frame, conf=args.conf, verbose=False)
            result = results[0]

            if result.keypoints is not None and len(result.keypoints) > 0:
                kpts_pixels = result.keypoints.xy.cpu().numpy()    # for drawing
                kpts_norm = result.keypoints.xyn.cpu().numpy()     # for the model

                for hand_pixels, hand_norm in zip(kpts_pixels, kpts_norm):
                    gesture, confidence = classify_hand(
                        classifier, labels, hand_norm, device)

                    draw_hand(frame, hand_pixels)
                    wrist = hand_pixels[0]
                    draw_label(frame, f"{gesture} {confidence:.2f}",
                               (wrist[0], wrist[1] - 10))

            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
