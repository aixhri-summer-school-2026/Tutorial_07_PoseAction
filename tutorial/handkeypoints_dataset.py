from torch.utils.data import Dataset
import numpy as np
import torch

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


class HandKeypointDataset(Dataset):
    """Loads one split (.npz) and normalizes each hand on the fly."""

    def __init__(self, npz_path, augment=False):
        data = np.load(npz_path)
        self.X = data["X"]  # (N, 21, 2) raw normalized [0,1] keypoints
        self.y = data["y"]  # (N,)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        keypoints = self.X[index]

        # Data augmentation: randomly mirror the hand left/right.
        if self.augment and np.random.rand() < 0.5:
            keypoints = normalize_keypoints(keypoints)
            keypoints = flip_keypoints(keypoints)
        else:
            keypoints = normalize_keypoints(keypoints)

        x = torch.tensor(keypoints, dtype=torch.float32)  # (21, 2)
        label = torch.tensor(self.y[index], dtype=torch.long)
        return x, label
