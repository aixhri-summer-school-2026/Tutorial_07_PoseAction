"""Part 2 - Live gesture classification on the robot camera.

Pipeline for every frame:
    1. detect hands + 21 keypoints with the MediaPipe detector (Part 1),
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

from gesture_utils import classify_hand, draw_hand, draw_label, load_classifier
from hand_detector import HandDetector

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CLASSIFIER = os.path.join(HERE, "weights", "gesture_classifier.pt")


def main():
    parser = argparse.ArgumentParser(description="Live gesture classification.")
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER)
    parser.add_argument("--conf", type=float, default=0.8)
    parser.add_argument("--class_conf", type=float, default=0.8)
    args = parser.parse_args()

    if not os.path.exists(args.classifier):
        print(f"[ERROR] missing file: {args.classifier}")
        print("Train it first: python tutorial/train_hand_pose_classification.py")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    detector = HandDetector(hand_conf=args.conf)
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

            for hand in detector.detect(frame):
                gesture, confidence = classify_hand(
                    classifier, labels, hand["norm"], device)

                draw_hand(frame, hand["pixels"])
                if confidence > args.class_conf:
                    wrist = hand["pixels"][0]
                    draw_label(frame, f"{gesture} {confidence:.2f}",
                            (wrist[0], wrist[1] - 10))

            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
