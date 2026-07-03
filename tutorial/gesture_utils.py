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


# ---------------------------------------------------------------------------
# Hand skeleton (21 MediaPipe-style keypoints)
# ---------------------------------------------------------------------------
# Pairs of keypoint indices that are connected by a "bone". Used only for
# drawing the hand on screen. Both the Ultralytics hand-keypoints model and the
# HaGRID landmarks follow this same 21-point ordering.
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17),                                # palm base
]

NUM_KEYPOINTS = 21


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

    This is used by the GCN. Each keypoint is a node; bones (HAND_EDGES) are
    the connections. We add self-connections and do simple symmetric
    normalization (the standard GCN trick).
    """
    a = np.eye(NUM_KEYPOINTS, dtype=np.float32)
    for i, j in HAND_EDGES:
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
def draw_hand(frame, keypoints_px, color=(0, 255, 0)):
    """Draw the 21 keypoints and the bones on a BGR OpenCV frame.

    keypoints_px must be pixel coordinates with shape (21, 2).
    """
    import cv2

    kpts = np.asarray(keypoints_px).reshape(NUM_KEYPOINTS, 2)
    for i, j in HAND_EDGES:
        p1 = (int(kpts[i, 0]), int(kpts[i, 1]))
        p2 = (int(kpts[j, 0]), int(kpts[j, 1]))
        cv2.line(frame, p1, p2, color, 2)
    for (x, y) in kpts:
        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)


def draw_label(frame, text, position, color=(0, 255, 0)):
    """Draw a text label with a readable outline."""
    import cv2

    x, y = int(position[0]), int(position[1])
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color, 2, cv2.LINE_AA)
