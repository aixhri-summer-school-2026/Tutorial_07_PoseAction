"""Robust hand keypoint estimator (MediaPipe, via OpenCV DNN).

This wraps the two-stage MediaPipe hand pipeline from the OpenCV Zoo:

    1. a palm detector finds where the hands are,
    2. a hand-landmark model estimates 21 keypoints for each detected palm.

The two ONNX models and the wrapper classes (``mp_palmdet.py`` /
``mp_handpose.py``) come from:
    https://huggingface.co/opencv/handpose_estimation_mediapipe
    https://huggingface.co/opencv/palm_detection_mediapipe

We use this instead of the YOLO pose model because it is much more robust, and
because its 21 landmarks follow the same MediaPipe convention as the HaGRID
data our gesture classifier was trained on.

Everything runs on OpenCV's DNN module, so there are no extra pip dependencies.
"""

import os
import sys

import cv2
import numpy as np

# The downloaded wrapper classes live in the "mediapipe" sub-folder.
_MP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mediapipe")
sys.path.insert(0, _MP_DIR)
from mp_palmdet import MPPalmDet      # noqa: E402
from mp_handpose import MPHandPose    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PALM_MODEL = os.path.join(HERE, "weights",
                                  "palm_detection_mediapipe_2023feb.onnx")
DEFAULT_HAND_MODEL = os.path.join(HERE, "weights",
                                  "handpose_estimation_mediapipe_2023feb.onnx")


class HandDetector:
    """Detect hands and their 21 keypoints in a BGR OpenCV frame."""

    def __init__(self, palm_model=DEFAULT_PALM_MODEL, hand_model=DEFAULT_HAND_MODEL,
                 palm_score=0.6, hand_conf=0.8):
        self.palm_detector = MPPalmDet(
            modelPath=palm_model,
            nmsThreshold=0.3,
            scoreThreshold=palm_score,
        )
        self.hand_estimator = MPHandPose(
            modelPath=hand_model,
            confThreshold=hand_conf,
        )

    def detect(self, frame):
        """Return a list of hands, one dict per detected hand.

        Each dict has:
            pixels     - (21, 2) keypoints in pixel coordinates
            norm       - (21, 2) keypoints normalized to [0, 1] by frame size
            handedness - 'Left' or 'Right' (from the model's point of view)
            confidence - detection confidence
            bbox       - (x1, y1, x2, y2) hand bounding box in pixels
            mean_x     - average normalized x (handy to pick the right-most hand)
        """
        height, width = frame.shape[:2]

        palms = self.palm_detector.infer(frame)
        hands = []
        for palm in palms:
            result = self.hand_estimator.infer(frame, palm)
            if result is None:
                continue

            # See mp_handpose.py for the layout of the 132-value result vector.
            bbox = result[0:4]
            landmarks = result[4:67].reshape(21, 3)  # x, y, z (z = depth)
            handedness = "Right" if result[130] > 0.5 else "Left"
            confidence = float(result[131])

            pixels = landmarks[:, :2].astype(np.float32)
            norm = pixels.copy()
            norm[:, 0] /= width
            norm[:, 1] /= height

            hands.append({
                "pixels": pixels,
                "norm": norm,
                "handedness": handedness,
                "confidence": confidence,
                "bbox": bbox.astype(int),
                "mean_x": float(norm[:, 0].mean()),
            })
        return hands
