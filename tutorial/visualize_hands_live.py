"""Part 1 - Run the hand detector live on the robot camera.

Opens the Reachy Mini camera, runs the MediaPipe hand keypoint estimator on
each frame, and draws the detected hands. This is the "does my detector
actually work?" check before we add gesture classification in Part 2.

Controls:
    q : quit

Run (inside the container shell, display forwarded like the tests):
    python tutorial/visualize_hands_live.py
"""

import argparse
import sys
import time

import cv2
from reachy_mini import ReachyMini

from gesture_utils import draw_hand, draw_label
from hand_detector import HandDetector


def main():
    parser = argparse.ArgumentParser(description="Live hand detection.")
    parser.add_argument("--conf", type=float, default=0.8,
                        help="Hand confidence threshold.")
    args = parser.parse_args()

    # Create the window before connecting (avoids the viewer blocking).
    cv2.namedWindow("Part 1 - live hands")
    print("Loading the MediaPipe hand detector...")
    detector = HandDetector(hand_conf=args.conf)

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

                for hand in detector.detect(frame):
                    draw_hand(frame, hand["pixels"])
                    wrist = hand["pixels"][0]
                    draw_label(frame, hand["handedness"], (wrist[0], wrist[1] - 10))

                cv2.imshow("Part 1 - live hands", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            print("Interrupted. Closing viewer...")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
