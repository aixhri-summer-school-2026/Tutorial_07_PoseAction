"""Part 1 - Run the hand detector live on the robot camera.

Opens the Reachy Mini camera, runs the hand keypoint model on each frame, and
draws the detected hands. This is the "does my detector actually work?" check
before we add gesture classification in Part 2.

Controls:
    q : quit

Run (inside the container shell, display forwarded like the tests):
    python tutorial/visualize_hands_live.py
"""

import argparse
import os
import time
import sys
import cv2
from reachy_mini import ReachyMini
from ultralytics import YOLO

from gesture_utils import draw_hand

# Pretrained hand keypoint detector (yolo26s-pose fine-tuned on hands).
# If you trained your own in Part 1, point --model at tutorial/weights/hand_detector.pt
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "weights", "yolo26s-pose-hands.pt")


def main():
    print("Starting hand detection live...")
    cv2.namedWindow("Part 1 - live hands")
    print("Live stream active. Press 'q' on the video window to quit.")

    parser = argparse.ArgumentParser(description="Live hand detection.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Path to the trained hand_detector.pt")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Detection confidence threshold.")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] model not found: {args.model}")
        print("Train it first with: python tutorial/train_hand_detector.py")
        return

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    print("Connecting to Reachy Mini camera... press 'q' to quit.")
    with ReachyMini() as mini:
        start_time = time.time()
        frame = mini.media.get_frame()
        
        while frame is None:
            if time.time() - start_time > 10:
                print("[FAIL] Camera timeout. No frames received.")
                sys.exit(1)
            print("Waiting for camera frames...")
            time.sleep(0.5)
            frame = mini.media.get_frame()

        print(f"[PASS] Camera active. Resolution: {frame.shape}")

        try:
            while True:
                frame = mini.media.get_frame()
                if frame is None:
                    continue
                # The camera frame is read-only; copy it so we can draw on it.
                frame = frame.copy()

                # Run the detector. verbose=False keeps the console quiet.
                results = model(frame, conf=args.conf, verbose=False)
                result = results[0]

                if result.keypoints is not None and len(result.keypoints) > 0:
                    # .xy gives pixel coordinates, shape (num_hands, 21, 2)
                    all_hands = result.keypoints.xy.cpu().numpy()
                    for hand_kpts in all_hands:
                        draw_hand(frame, hand_kpts)

                cv2.imshow("Part 1 - live hands", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                
        except KeyboardInterrupt:
            print("Interrupted. Closing viewer...")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
