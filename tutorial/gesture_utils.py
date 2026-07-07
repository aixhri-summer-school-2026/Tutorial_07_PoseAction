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


NUM_KEYPOINTS = 21

LEFT_WRIST_ID = 9
RIGHT_WRIST_ID = 10
LEFT_HAND_IDS = np.arange(91, 112)
RIGHT_HAND_IDS = np.arange(112, 133)
FACE_IDS = np.arange(23, 91)
BODY_IDS = np.arange(17)

HAND21_EDGES_COCO133 = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
HAND21_EDGES_MEDIAPIPE = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17),                                # palm_base
]

# ---------------------------------------------------------------------------
# Keypoint preprocessing
# ---------------------------------------------------------------------------
def normalize_keypoints(keypoints):
    """Turn raw (21, 2) keypoints into a position/scale invariant (21, 2) array.

    Why we need this: the same gesture can appear anywhere in the image and at
    any size. We remove that information so the classifier only sees the SHAPE
    of the hand:

    1. move the wrist (keypoint 0) to the origin,
    2. divide by the largest wrist-to-point distance so the hand fits in a
       unit circle.

    Important: we divide x and y by the SAME number, so the hand is not
    squished. Feed in coordinates that are already normalized by the image
    size (x / width, y / height) so that training data (HaGRID) and the live
    camera use the same convention.
    """
    kpts = np.asarray(keypoints, dtype=np.float32).reshape(NUM_KEYPOINTS, 2)

    wrist = kpts[0]
    centered = kpts - wrist

    distances = np.sqrt((centered ** 2).sum(axis=1))
    scale = distances.max()
    if scale < 1e-6:
        scale = 1.0  # avoid dividing by zero on a degenerate hand

    return centered / scale


def flip_keypoints(keypoints):
    """Mirror a normalized hand left/right (simple data augmentation)."""
    flipped = np.array(keypoints, dtype=np.float32)
    flipped[:, 0] = -flipped[:, 0]
    return flipped


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class HandMLP(nn.Module):
    """A plain multi-layer perceptron on the flattened 42 numbers (21 x 2)."""

    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_KEYPOINTS * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x comes in as (batch, 21, 2); flatten it to (batch, 42).
        x = x.reshape(x.shape[0], -1)
        return self.net(x)


def build_hand_adjacency():
    """Build the (21, 21) normalized adjacency matrix of the hand graph.

    This is used by the GCN. Each keypoint is a node; bones (from RTMLib's
    skeleton definition) are
    the connections. We add self-connections and do simple symmetric
    normalization (the standard GCN trick).
    """
    a = np.eye(NUM_KEYPOINTS, dtype=np.float32)
    for i, j in HAND21_EDGES_MEDIAPIPE:
        a[i, j] = 1.0
        a[j, i] = 1.0

    degree = a.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(degree)
    d_mat = np.diag(d_inv_sqrt)
    return d_mat @ a @ d_mat  # A_hat = D^-1/2 A D^-1/2


class HandGCN(nn.Module):
    """A tiny graph convolutional network on the 21-keypoint hand graph.

    One GCN "layer" is just: new_features = A_hat @ features @ W, then ReLU.
    We stack two of them, then average over the 21 nodes and classify.
    """

    def __init__(self, num_classes, hidden=64):
        super().__init__()
        adjacency = build_hand_adjacency()
        # register_buffer = a constant tensor that moves with .to(device)
        self.register_buffer("adjacency", torch.tensor(adjacency))

        self.linear1 = nn.Linear(2, hidden)        # input feature per node is (x, y)
        self.linear2 = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, num_classes)

    def graph_conv(self, features, linear):
        # features: (batch, 21, in) -> mix neighbours -> linear projection
        mixed = torch.matmul(self.adjacency, features)
        return torch.relu(linear(mixed))

    def forward(self, x):
        # x: (batch, 21, 2)
        h = self.graph_conv(x, self.linear1)
        h = self.graph_conv(h, self.linear2)
        h = h.mean(dim=1)  # pool over the 21 nodes -> (batch, hidden)
        return self.classifier(h)


def build_model(model_type, num_classes):
    """Create a model from a string ('mlp' or 'gcn')."""
    if model_type == "mlp":
        return HandMLP(num_classes)
    if model_type == "gcn":
        return HandGCN(num_classes)
    raise ValueError(f"Unknown model type: {model_type!r} (use 'mlp' or 'gcn')")


def load_classifier(path, device):
    """Load a gesture classifier saved by train_hand_pose_classification.py.

    Returns (model, labels).
    """
    checkpoint = torch.load(path, map_location=device)
    model = build_model(checkpoint["model_type"], len(checkpoint["labels"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["labels"]


def classify_hand(model, labels, keypoints_normalized_01, device):
    """Classify one hand.

    keypoints_normalized_01: (21, 2) keypoints in image-normalized [0, 1]
    coordinates (use YOLO's ``.xyn`` output).

    Returns (gesture_name, confidence).
    """
    features = normalize_keypoints(keypoints_normalized_01)
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)[0]
    best_index = int(probabilities.argmax())
    return labels[best_index], float(probabilities[best_index])


# ---------------------------------------------------------------------------
# Drawing helpers (for the live demos)
# ---------------------------------------------------------------------------
def draw_hand_edges(frame, keypoints_px, edges, scores=None, kpt_thr=0.5,
                     color=(0, 255, 0), point_color=(0, 0, 255)):
    """Draw one 21-keypoint hand using a provided edge list."""
    kpts = np.asarray(keypoints_px, dtype=np.float32).reshape(NUM_KEYPOINTS, 2)
    if scores is None:
        visible = np.ones(NUM_KEYPOINTS, dtype=bool)
    else:
        visible = np.asarray(scores, dtype=np.float32).reshape(NUM_KEYPOINTS) >= kpt_thr

    for i, j in edges:
        if visible[i] and visible[j]:
            p1 = (int(kpts[i, 0]), int(kpts[i, 1]))
            p2 = (int(kpts[j, 0]), int(kpts[j, 1]))
            cv2.line(frame, p1, p2, color, 2)
    for idx, (x, y) in enumerate(kpts):
        if visible[idx]:
            cv2.circle(frame, (int(x), int(y)), 3, point_color, -1)


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

    draw_visible_points(frame, kpts, visible, BODY_IDS, point_color)

    if include_hands:
        edges = HAND21_EDGES_MEDIAPIPE if hands_style == "mediapipe" else HAND21_EDGES_COCO133
        draw_hand_edges(frame, kpts[LEFT_HAND_IDS], edges=edges,
                        color=(100, 100, 255), point_color=point_color,
                        scores=conf[LEFT_HAND_IDS], kpt_thr=kpt_thr)
        draw_hand_edges(frame, kpts[RIGHT_HAND_IDS], edges=edges,
                        color=(100, 255, 100), point_color=point_color,
                        scores=conf[RIGHT_HAND_IDS], kpt_thr=kpt_thr)

    if include_face:
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
