"""Part 2 - Train a gesture classifier on hand keypoints.

Input : the 21 hand keypoints of ONE hand (from prepare_hagrid_dataset.py).
Output: one of the 8 gestures in gesture_utils.GESTURES.

You can train either a plain MLP or a tiny GCN:
    python tutorial/train_hand_pose_classification.py --model mlp
    python tutorial/train_hand_pose_classification.py --model gcn

The trained model is saved to ``tutorial/weights/gesture_classifier.pt`` along
with the info needed to rebuild it (model type + class names).
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from gesture_utils import (GESTURES, build_model, flip_keypoints,
                           normalize_keypoints)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "hagrid_keypoints")
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
CLASSIFIER_PATH = os.path.join(WEIGHTS_DIR, "gesture_classifier.pt")


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


def evaluate(model, loader, device):
    """Return the accuracy of the model over a data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, label in loader:
            x, label = x.to(device), label.to(device)
            prediction = model(x).argmax(dim=1)
            correct += (prediction == label).sum().item()
            total += label.size(0)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train the gesture classifier.")
    parser.add_argument("--model", choices=["mlp", "gcn"], default="mlp")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_set = HandKeypointDataset(os.path.join(args.data_dir, "train.npz"),
                                    augment=True)
    val_set = HandKeypointDataset(os.path.join(args.data_dir, "val.npz"))
    test_set = HandKeypointDataset(os.path.join(args.data_dir, "test.npz"))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model = build_model(args.model, num_classes=len(GESTURES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, label in train_loader:
            x, label = x.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch:3d} | loss {epoch_loss / len(train_loader):.4f} "
              f"| val acc {val_acc:.3f}")

        # Keep the best model on the validation split.
        if val_acc >= best_val:
            best_val = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_type": args.model,
                    "labels": GESTURES,
                },
                CLASSIFIER_PATH,
            )

    test_acc = evaluate(model, test_loader, device)
    print(f"\nBest val acc: {best_val:.3f} | test acc: {test_acc:.3f}")
    print(f"Saved classifier to: {CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
