import torch
from handkeypoints_models import build_model
from handkeypoints_dataset import normalize_keypoints

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