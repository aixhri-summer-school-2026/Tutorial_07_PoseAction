import numpy as np
import torch
import torch.nn as nn
from keypoints_utils import HAND21_EDGES_MEDIAPIPE

NUM_KEYPOINTS = 21

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
