"""Part 1 - Run the whole-body detector live on the robot camera.

Opens the Reachy Mini camera, runs the RTMLib whole-body pose tracker on each
frame, and draws a simplified skeleton for each detected person. This is the
"does my detector actually work?" check before adding higher-level logic.

Controls:
    q : quit

Run (inside the container shell, display forwarded like the tests):
    python tutorial/visualize_wholebody_live.py
"""

import sys
import time
import argparse
import cv2
import numpy as np
import onnxruntime as ort
from reachy_mini import ReachyMini
from rtmlib import PoseTracker, Wholebody

from keypoints_utils import (
    LEFT_HAND_IDS,
    LEFT_WRIST_ID,
    RIGHT_HAND_IDS,
    RIGHT_WRIST_ID,
    draw_bbox,
    draw_label,
    draw_skeleton,
    get_hand_bbox,
    get_face_bbox,
)


def build_pose_tracker():
    """Create the whole-body tracker with the same setup as the demo script."""
    device = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    print(f"Loading the RTMLib whole-body tracker on {device}...")

    solution_kwargs = {
        "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",  # noqa: E501
        "det_input_size": (640, 640),
        "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip",  # noqa: E501
        "pose_input_size": (288, 384),
    }

    tracker = PoseTracker(
        Wholebody,
        det_frequency=1,
        backend="onnxruntime",
        device=device,
        to_openpose=False,
        solution_kwargs=solution_kwargs,
    )
    return tracker


def wait_for_first_frame(mini, timeout_s=10.0):
    """Wait until the camera produces a frame or fail after a timeout."""
    start_time = time.time()
    frame = mini.media.get_frame()
    while frame is None:
        if time.time() - start_time > timeout_s:
            print("[FAIL] Camera timeout. No frames received.")
            sys.exit(1)
        print("Waiting for camera frames...")
        time.sleep(0.5)
        frame = mini.media.get_frame()
    return frame


def main(args):
    cv2.namedWindow("Part 1 - live whole body")
    pose_tracker = build_pose_tracker()

    print("Connecting to Reachy Mini camera... press 'q' to quit.")
    with ReachyMini() as mini:
        frame = wait_for_first_frame(mini)
        print(f"[PASS] Camera active. Resolution: {frame.shape}")

        try:
            while True:
                frame = mini.media.get_frame()
                if frame is None:
                    continue

                frame = frame.copy()
                keypoints, scores = pose_tracker(frame)
                for det in range(scores.shape[0]):
                    if scores[det, LEFT_WRIST_ID] < args.score_threshold:
                        scores[det, LEFT_HAND_IDS] = 0.0
                    if scores[det, RIGHT_WRIST_ID] < args.score_threshold:
                        scores[det, RIGHT_HAND_IDS] = 0.0

                for person_kpts, person_scores in zip(keypoints, scores):
                    draw_skeleton(
                        frame,
                        person_kpts,
                        scores=person_scores,
                        kpt_thr=args.score_threshold,
                        hands_style=args.hands_style,
                    )
                    left_bbox = get_hand_bbox(
                        person_kpts,
                        scores=person_scores,
                        kpt_thr=args.score_threshold,
                        side="left",
                    )
                    right_bbox = get_hand_bbox(
                        person_kpts,
                        scores=person_scores,
                        kpt_thr=args.score_threshold,
                        side="right",
                    )
                    draw_bbox(frame, left_bbox, color=(100, 100, 255))
                    draw_bbox(frame, right_bbox, color=(100, 255, 100))
                    face_bbox = get_face_bbox(
                        person_kpts,
                        scores=person_scores,
                        kpt_thr=args.score_threshold,
                    )
                    draw_bbox(frame, face_bbox, color=(255, 100, 100))

                draw_label(
                    frame,
                    f"people count: {keypoints.shape[0]}",
                    (10, 30),
                    color=(255, 255, 0),
                )

                cv2.imshow("Part 1 - live whole body", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        except KeyboardInterrupt:
            print("Interrupted. Closing viewer...")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands_style", type=str, default="mediapipe", choices=["mediapipe", "coco133"])
    parser.add_argument("--score_threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
