import numpy as np
import torch
import torch.nn as nn
from keypoints_utils import HAND21_EDGES_MEDIAPIPE

NUM_KEYPOINTS_HAND = 21

###############
# MLP model
###############
class HandMLP(nn.Module):
    """A plain multi-layer perceptron on the flattened 42 numbers (21 x 2)."""

    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(NUM_KEYPOINTS_HAND * 2, ...0),
            # ...
            # Implement the MLP architecture here with Linear layers (intermediate sizes 128 then 64 recommended) and activations (ReLU recommended)
        )

    def forward(self, x):
        # x comes in as (batch, 21, 2); flatten it to (batch, 42).
        # x = ...
        return self.net(x)


###############
# GCN model
###############

def build_hand_adjacency():
    """
    Build:
        A_hat = D^{-1/2} (A + I) D^{-1/2}

    following Kipf & Welling GCN normalization.

    See the rule in illustrations/gcn_ref.png
    https://arxiv.org/abs/1609.02907
    """

    # adjacency matrix
    # A = ...

    # for ... in ...:
        # ...

    # add self-connections: A_hat = A + I
    # A = A + ...

    # degree matrix: D_ii = sum_j A_ij
    # degree = ...
    
    # D = np.diag(degree)

    # D^{-1/2}
    # D_inv_sqrt = ...

    # normalized adjacency
    # A_hat = ...

    # return A_hat

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
        projected = linear(features)
        mixed = torch.matmul(
            #  fill the missing arguments
            # ...
        )
        return torch.relu(mixed)

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
