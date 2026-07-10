"""Part 2 - Train a gesture classifier on hand keypoints.

Input : the 21 hand keypoints of ONE hand (from prepare_hagrid_dataset.py).
Output: one of the gestures in keypoints_utils.GESTURES.

You can train either a plain MLP or a tiny GCN:
    python tutorial/train_hand_pose_classification.py --model mlp
    python tutorial/train_hand_pose_classification.py --model gcn

The trained model is saved to ``tutorial/weights/gesture_classifier.pt`` along
with the info needed to rebuild it (model type + class names).
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from keypoints_utils import GESTURES
from handkeypoints_models import build_model
from handkeypoints_dataset import HandKeypointDataset

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "hagrid_keypoints")
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
CLASSIFIER_PATH = os.path.join(WEIGHTS_DIR, "gesture_classifier.pt")

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
        correct = 0
        total = 0
        for x, label in train_loader:
            x, label = x.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            prediction = output.argmax(dim=1)
            correct += (prediction == label).sum().item()
            total += label.size(0)

        train_acc = correct / max(total, 1)
        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch:3d} | loss {epoch_loss / len(train_loader):.4f} "
              f"| train acc {train_acc:.3f} | val acc {val_acc:.3f}")

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

    # load best model and test it
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device)["state_dict"])
    test_acc = evaluate(model, test_loader, device)
    
    print(f"\nBest val acc: {best_val:.3f} | test acc : {test_acc:.3f}")
    print(f"Saved classifier to: {CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
