"""Part 2 - Live gesture classification on the robot camera.

Pipeline for every frame:
    1. run the whole-body pose tracker (Part 1),
    2. extract each hand's 21 keypoints,
    3. classify the gesture,
    4. draw the skeleton and label.

Controls:
    q : quit

Run (inside the container shell, display forwarded):
    python tutorial/visualize_pose_classification_live.py
"""

import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort
import torch
from reachy_mini import ReachyMini
from rtmlib import PoseTracker, Wholebody

from handkeypoints_infer import classify_hand, load_classifier
from keypoints_utils import (
    LEFT_HAND_IDS,
    LEFT_WRIST_ID,
    RIGHT_HAND_IDS,
    RIGHT_WRIST_ID,
    draw_label,
    draw_skeleton,
)

# Utils to handle the input source (webcam or reachy_mini)
class FakeCamera:
    def __init__(self):
        self.frame = None
    def isOpened(self):
        return True

class FakeContext:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass



HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CLASSIFIER = os.path.join(HERE, "weights", "gesture_classifier.pt")

def build_pose_tracker():
    device = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    print(f"Loading the RTMLib whole-body tracker on {device}...")

    solution_kwargs = {
        "det": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",  # noqa: E501
        "det_input_size": (640, 640),
        "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip",  # noqa: E501
        "pose_input_size": (288, 384),
    }

    return PoseTracker(
        Wholebody,
        det_frequency=1,
        backend="onnxruntime",
        device=device,
        to_openpose=False,
        solution_kwargs=solution_kwargs,
    )


def get_hands(person_kpts, person_scores, frame_shape, kpt_thr):
    """Return visible hands as dicts with pixel and normalized keypoints."""
    h, w = frame_shape[:2]
    scale = np.array([w, h], dtype=np.float32)
    hands = []

    for hand_ids, wrist_id in (
        (LEFT_HAND_IDS, LEFT_WRIST_ID),
        (RIGHT_HAND_IDS, RIGHT_WRIST_ID),
    ):
        if person_scores[wrist_id] < kpt_thr:
            continue
        hand_scores = person_scores[hand_ids]
        if (hand_scores >= kpt_thr).sum() < 5:
            continue

        hand_px = person_kpts[hand_ids].astype(np.float32)
        hands.append({
            "pixels": hand_px,
            "norm": hand_px / scale,
        })

    return hands


def main(args):

    # Handle the input source (webcam or reachy_mini)
    if args.use_input == "webcam":
        cap = cv2.VideoCapture(0)
        ctx = FakeContext
    elif args.use_input == "reachy_mini":
        ctx = ReachyMini
        cap = FakeCamera()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pose_tracker = build_pose_tracker()
    classifier, labels = load_classifier(args.classifier, device)
    print(f"Classes: {labels}")

    window = "Part 2 - live gesture classification"
    cv2.namedWindow(window)

    print("Connecting to Reachy Mini camera... press 'q' to quit.")
    with ctx() as mini:
        while cap.isOpened():
        
            # Get the frame from the input source
            if args.use_input == "reachy_mini":
                frame = mini.media.get_frame()
            elif args.use_input == "webcam":
                success, frame = cap.read()
                if not success:
                    break
                
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
                    include_face=False,
                    hands_style=args.hands_style,
                )

                for hand in get_hands(person_kpts, person_scores, frame.shape,
                                      args.score_threshold):
                    gesture, confidence = classify_hand(
                        classifier, labels, hand["norm"], device)

                    if confidence > args.class_conf:
                        wrist = hand["pixels"][0]
                        draw_label(
                            frame,
                            f"{gesture} {confidence:.2f}",
                            (wrist[0], wrist[1] - 10),
                        )

            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live gesture classification.")
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER)
    parser.add_argument("--class_conf", type=float, default=0.8, help="Class confidence threshold (for gesture classification)")
    parser.add_argument("--score_threshold", type=float, default=0.5, help="Score threshold (for hand keypoints detection)")
    parser.add_argument("--hands_style", type=str, default="mediapipe",
                        choices=["mediapipe", "coco133"])
    args = parser.parse_args()

    if not os.path.exists(args.classifier):
        print(f"[ERROR] missing file: {args.classifier}")
        print("Train it first: python tutorial/handkeypoints_train.py")
        raise FileNotFoundError(f"missing file: {args.classifier}")

    main(args)
