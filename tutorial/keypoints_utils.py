"""Shared helpers for the pose/action tutorial.

This file is imported by several scripts in Parts 1-4. It keeps the things
that would otherwise be copy/pasted everywhere:

- the list of gestures we work with,
- how we turn 21 hand keypoints into a fixed-size vector for the classifier,
- two small neural networks (an MLP and a tiny GCN),
- a couple of drawing helpers for the live demos.

The code is intentionally written in a simple, explicit style so that it is
easy to read for people with an intermediate Python level.
"""

import numpy as np
import torch
import torch.nn as nn
import cv2

# from rtmlib.visualization.skeleton.coco17 import coco17
# from rtmlib.visualization.skeleton.coco133 import coco133


# ---------------------------------------------------------------------------
# Gestures
# ---------------------------------------------------------------------------
# HaGRIDv2 class names used in this tutorial. The ORDER matters: the index
# in this list is the integer label used everywhere (dataset, training, live).
GESTURES = [
    "no_gesture",
    "fist",
    "peace",
    "rock",
    "point",
    "stop",
    "mute",
    "hand_heart",
]


# Keypoints references
NUM_KEYPOINTS_HAND = 21 # Number of keypoints in the hand skeleton

LEFT_WRIST_ID = 9   # left wrist in the COCO body skeleton (keypoint index 9)
RIGHT_WRIST_ID = 10 # right wrist in the COCO body skeleton (keypoint index 10)
LEFT_HAND_IDS = np.arange(91, 112)   # 21 left-hand keypoints in the 133-keypoint layout
RIGHT_HAND_IDS = np.arange(112, 133) # 21 right-hand keypoints
FACE_IDS = np.arange(23, 91)         # face landmarks
BODY_IDS = np.arange(17)             # COCO body keypoints

# Kinematic structure of the hand skeleton with COCO-Wholebody format
HAND21_EDGES_COCO133 = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]

# Kinematic structure of the hand skeleton with MediaPipe format
HAND21_EDGES_MEDIAPIPE = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17),                                # palm_base
]


# ---------------------------------------------------------------------------
# Drawing helpers (for the live demos)
# ---------------------------------------------------------------------------
def draw_hand_edges(frame, keypoints_px, edges, scores=None, kpt_thr=0.5,
                     color=(0, 255, 0), point_color=(0, 0, 255)):
    
    """Draw one 21-keypoint hand using a provided edge list."""
    
    kpts = np.asarray(keypoints_px, dtype=np.float32).reshape(NUM_KEYPOINTS_HAND, 2)
   
    if scores is None:
        visible = np.ones(NUM_KEYPOINTS_HAND, dtype=bool)
    else:
        visible = np.asarray(scores, dtype=np.float32).reshape(NUM_KEYPOINTS_HAND) >= kpt_thr

    # YOUR CODE HERE — draw edges between visible keypoint pairs (cv2.line)
    # for i, j in edges:
    #     ...

    # YOUR CODE HERE — draw a dot on each visible joint (cv2.circle)
    # for idx, (x, y) in enumerate(kpts):
    #     ...



def draw_visible_points(frame, person_kpts, visible, indices, point_color):
    for idx in indices:
        if visible[idx]:
            x, y = person_kpts[idx]
            cv2.circle(frame, (int(x), int(y)), 3, point_color, -1)

def draw_skeleton(frame,
                  keypoints_px,
                  scores=None,
                  kpt_thr=0.5,
                  include_hands=True,
                  hands_style="mediapipe",
                  include_face=True,
                  color=(0, 255, 0),
                  point_color=(0, 0, 255)):
    
    """Draw a simplified COCO133 whole-body skeleton for one person.

    The body always uses the first 17 keypoints. Hands and face can be toggled
    on/off to keep the display readable.
    """
    kpts = np.asarray(keypoints_px, dtype=np.float32)
    if kpts.shape != (133, 2):
        raise ValueError("keypoints_px must have shape (133, 2)")

    if scores is None:
        conf = np.ones(133, dtype=np.float32)
    else:
        conf = np.asarray(scores, dtype=np.float32)
        if conf.shape != (133,):
            raise ValueError("scores must have shape (133,)")

    visible = conf >= kpt_thr

    # draw the body keypoints
    draw_visible_points(frame, kpts, visible, BODY_IDS, point_color)

    if include_hands:
        # draw the left and right hand edges
        
        # choose edges depending on the hands_style
        # edges = ...
        draw_hand_edges(
            # place your args
        )
        draw_hand_edges(
            # place your args
        )

    if include_face:
        # draw the face keypoints
        draw_visible_points(frame, kpts, visible, FACE_IDS, point_color)


def draw_label(frame, text, position, color=(0, 255, 0)):
    """Draw a text label with a readable outline."""
    x, y = int(position[0]), int(position[1])
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color, 2, cv2.LINE_AA)


def get_hand_bbox(keypoints_px, scores, kpt_thr=0.5, padding=0.2, side="left"):
    """Compute a hand bounding box from COCO133 whole-body keypoints.

    Returns `(x1, y1, x2, y2)` in pixel coordinates, or `None` if too few hand
    keypoints are visible.
    """
    kpts = np.asarray(keypoints_px, dtype=np.float32)
    if kpts.shape != (133, 2):
        raise ValueError("keypoints_px must have shape (133, 2)")

    if side == "left":
        hand_ids = LEFT_HAND_IDS
    elif side == "right":
        hand_ids = RIGHT_HAND_IDS
    else:
        raise ValueError("side must be 'left' or 'right'")

    hand_kpts = kpts[hand_ids]
    visible = scores[hand_ids] >= kpt_thr

    visible_kpts = hand_kpts[visible]
    if len(visible_kpts) < 2:
        return None

    x1, y1 = visible_kpts.min(axis=0)
    x2, y2 = visible_kpts.max(axis=0)

    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    pad_x = width * padding
    pad_y = height * padding

    return (
        int(x1 - pad_x),
        int(y1 - pad_y),
        int(x2 + pad_x),
        int(y2 + pad_y),
    )


def get_face_bbox(keypoints_px, scores, kpt_thr=0.5,
                  padding={"top": 0.3, "bottom": 0.05, "left": 0.05, "right": 0.05}):
    """Compute a face bounding box from COCO133 whole-body keypoints.

    Returns `(x1, y1, x2, y2)` in pixel coordinates, or `None` if too few face
    keypoints are visible. Padding is controlled independently on each side.
    """
    kpts = np.asarray(keypoints_px, dtype=np.float32)
    if kpts.shape != (133, 2):
        raise ValueError("keypoints_px must have shape (133, 2)")

    conf = np.asarray(scores, dtype=np.float32)
    if conf.shape != (133,):
        raise ValueError("scores must have shape (133,)")

    face_kpts = kpts[FACE_IDS]
    visible = conf[FACE_IDS] >= kpt_thr

    visible_kpts = face_kpts[visible]
    if len(visible_kpts) < 2:
        return None

    x1, y1 = visible_kpts.min(axis=0)
    x2, y2 = visible_kpts.max(axis=0)

    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)

    return (
        int(x1 - width * padding["left"]),
        int(y1 - height * padding["top"]),
        int(x2 + width * padding["right"]),
        int(y2 + height * padding["bottom"]),
    )


def draw_bbox(frame, bbox, color=(0, 255, 0), thickness=2):
    """Draw a bounding box if it exists."""
    if bbox is None:
        return

    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
