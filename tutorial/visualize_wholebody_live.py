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



def build_pose_tracker():
    """Create the whole-body tracker with the same setup as the demo script."""
    device = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    print(f"Loading the RTMLib whole-body tracker on {device}...")

    # You can replace those path by URLs like : "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
    # and the checkpoints will be downloaded automatically
    det_onnx_model = "/app/downloads/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx"
    pose_onnx_model = "/app/downloads/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.onnx"
    
    solution_kwargs = {
        "det": det_onnx_model,
        "det_input_size": (640, 640),
        "pose": pose_onnx_model,
        "pose_input_size": (288, 384),
        "biggest_n_boxes_only": 3,
    }

    tracker = PoseTracker(
        Wholebody,
        det_frequency=1,
        backend="onnxruntime",
        device=device,
        to_openpose=False,
        biggest_n_boxes_only=3, # will perfom pose estimation on the 3 biggest boxes only (to speed up the process)
        solution_kwargs=solution_kwargs,
    )
    return tracker


def main(args):
    
    # Handle the input source (webcam or reachy_mini)
    if args.use_input == "webcam":
        cap = cv2.VideoCapture(0)
        ctx = FakeContext
    elif args.use_input == "reachy_mini":
        ctx = ReachyMini
        cap = FakeCamera()

    cv2.namedWindow("Part 1 - live whole body")
    pose_tracker = build_pose_tracker()

    print(f"Connecting to {args.use_input} camera... press 'q' to quit.")
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
            
            print(f"Got keypoints and scores of shape {keypoints.shape} and {scores.shape} !")
            
            # Uncomment this block of code and fill the blanks here and in keypoints_utils.py
            """
            # Tip : the model may misplace the left hand on top the right one (or vice versa), so we need to check if the wrist 
            # are present in the frame, otherwise we set the hand scores to 0
            for det in range(scores.shape[0]):
                if scores[det, LEFT_WRIST_ID] < args.score_threshold:
                    scores[det, LEFT_HAND_IDS] = 0.0
                if scores[det, RIGHT_WRIST_ID] < args.score_threshold:
                    scores[det, RIGHT_HAND_IDS] = 0.0

            for person_kpts, person_scores in zip(keypoints, scores):
                draw_skeleton(
                    # place your args
                )

                # Use the keypoints to extract bounding boxes for the left and right hand and the face
                left_bbox = get_hand_bbox(
                    # place your args
                )
                right_bbox = get_hand_bbox(
                    # place your args
                )
                draw_bbox(frame, left_bbox, color=(100, 100, 255))
                draw_bbox(frame, right_bbox, color=(100, 255, 100))
                face_bbox = get_face_bbox(
                    # place your args
                )
                draw_bbox(frame, face_bbox, color=(255, 100, 100))

            cv2.putText(frame, 
                        # str() # place the people count str here 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            """

            cv2.imshow("Part 1 - live whole body", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands_style", type=str, default="mediapipe", choices=["mediapipe", "coco133"])
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--use_input", type=str, default="reachy_mini", choices=["webcam", "reachy_mini"])
    args = parser.parse_args()
    main(args)
