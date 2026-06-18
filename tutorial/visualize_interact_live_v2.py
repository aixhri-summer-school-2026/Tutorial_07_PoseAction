"""Part 4 - New tracking behaviour: mirror the person's head.

Same idea as Part 3, but the "peace / stop" behaviour now MIRRORS the person's
head orientation instead of following a hand. We estimate the person's head
pose (pitch, yaw, roll) with the SixDRepNet model that ships in
``tutorial/SixDRepNet`` and send a mirrored pose to the robot head.

Gesture -> behaviour:
    point  -> start sounds          mute  -> stop sounds
    rock   -> open antennas         fist  -> close antennas
    peace  -> start head mirroring  stop  -> stop mirroring
    heart  -> (both hands) wave the head

Controls:
    q : quit

Run (inside the container shell, display forwarded):
    python tutorial/visualize_interact_live_v2.py
"""

import argparse
import os
import random
import sys
import threading
import time
from collections import deque, Counter

import cv2
import numpy as np
import torch
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from ultralytics import YOLO

from gesture_utils import classify_hand, draw_hand, draw_label, load_classifier

HERE = os.path.dirname(os.path.abspath(__file__))
# Pretrained hand detector by default (use hand_detector.pt if you trained one).
DEFAULT_DETECTOR = os.path.join(HERE, "weights", "yolo26s-pose-hands.pt")
DEFAULT_CLASSIFIER = os.path.join(HERE, "weights", "gesture_classifier.pt")

# SixDRepNet lives in a sub-folder with its own "nets" and "utils" packages.
# We add it to the import path so we can load its model and face detector.
SIXD_DIR = os.path.join(HERE, "SixDRepNet")
sys.path.insert(0, SIXD_DIR)
from utils import util  # noqa: E402  (import after sys.path tweak)

SIXD_MODEL = os.path.join(SIXD_DIR, "weights", "best.pt")
SIXD_DETECTOR = os.path.join(SIXD_DIR, "weights", "detection.onnx")

ANTENNAS_OPEN = [1.0, 1.0]
ANTENNAS_CLOSED = [0.0, 0.0]

# How strongly the robot copies the person's head angles, and the limits.
MIRROR_GAIN = 1.0
MAX_YAW = 30.0
MAX_PITCH = 20.0
MAX_ROLL = 20.0


class HeadPoseEstimator:
    """Detect a face and estimate its (pitch, yaw, roll) in degrees."""

    def __init__(self, model_path, detector_path, device):
        from torchvision import transforms

        self.device = device
        # Load the SixDRepNet head-pose model. It was saved as a full model
        # OBJECT (not just weights), so we need weights_only=False. This is safe
        # here because the checkpoint ships with the tutorial.
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model = checkpoint["model"].float().fuse().to(device)
        self.model.eval()

        # Build the ONNX face detector with whatever runtime is available
        # (GPU if present, otherwise CPU). We create the session ourselves so we
        # are not forced onto CUDA.
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        session = onnxruntime.InferenceSession(detector_path, providers=providers)
        self.detector = util.FaceDetector(session=session)

        self.input_size = 224
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size + 32),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def estimate(self, frame):
        """Return (pitch, yaw, roll, box) for the biggest face, or None."""
        from PIL import Image

        boxes = self.detector.detect(frame, (640, 640))
        if boxes is None or len(boxes) == 0:
            return None

        # Use the largest detected face.
        boxes = boxes.astype("int32")
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        box = boxes[int(np.argmax(areas))]

        x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
        box_w = abs(x_max - x_min)
        box_h = abs(y_max - y_min)
        x_min = max(0, x_min - int(0.2 * box_h))
        y_min = max(0, y_min - int(0.2 * box_w))
        x_max = x_max + int(0.2 * box_h)
        y_max = y_max + int(0.2 * box_w)

        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return None

        # frame is BGR (OpenCV); convert to RGB for the model.
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = self.transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(self.device)

        output = self.model(image)
        euler = util.compute_euler(output) * 180 / np.pi  # to degrees
        pitch = float(euler[0, 0].cpu())
        yaw = float(euler[0, 1].cpu())
        roll = float(euler[0, 2].cpu())
        return pitch, yaw, roll, (x_min, y_min, x_max, y_max)


class GestureSmoother:
    def __init__(self, window=7):
        self.history = deque(maxlen=window)

    def update(self, gesture):
        self.history.append(gesture)
        return Counter(self.history).most_common(1)[0][0]


class SoundPlayer:
    def __init__(self, mini):
        self.mini = mini
        self.samplerate = mini.media.get_output_audio_samplerate()
        self.active = False
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.mini.media.start_playing()
        self.thread.start()

    def set_active(self, active):
        self.active = active

    def stop(self):
        self.running = False

    def _run(self):
        notes = [262, 294, 330, 349, 392, 440, 494, 523]
        chunk_seconds = 0.02
        samples_per_chunk = int(self.samplerate * chunk_seconds)
        phase = 0.0
        frequency = 440.0
        next_note_time = 0.0
        while self.running:
            if not self.active:
                time.sleep(0.05)
                continue
            if time.time() > next_note_time:
                frequency = random.choice(notes)
                next_note_time = time.time() + random.uniform(0.2, 0.5)
            t = np.arange(samples_per_chunk, dtype=np.float32) / self.samplerate
            mono = (0.3 * np.sin(2.0 * np.pi * frequency * t + phase)).astype(np.float32)
            phase += 2.0 * np.pi * frequency * samples_per_chunk / self.samplerate
            self.mini.media.push_audio_sample(mono)
            time.sleep(chunk_seconds * 0.5)


def analyze_hands(result, classifier, labels, device, conf_threshold):
    hands = []
    if result.keypoints is None or len(result.keypoints) == 0:
        return hands
    kpts_pixels = result.keypoints.xy.cpu().numpy()
    kpts_norm = result.keypoints.xyn.cpu().numpy()
    for hand_pixels, hand_norm in zip(kpts_pixels, kpts_norm):
        gesture, confidence = classify_hand(classifier, labels, hand_norm, device)
        if confidence < conf_threshold:
            gesture = "no_gesture"
        hands.append({
            "gesture": gesture,
            "confidence": confidence,
            "pixels": hand_pixels,
            "mean_x": float(hand_norm[:, 0].mean()),
        })
    return hands


def clamp(value, limit):
    return max(-limit, min(limit, value))


def main():
    parser = argparse.ArgumentParser(description="Mirror-head robot behaviours.")
    parser.add_argument("--detector", default=DEFAULT_DETECTOR)
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER)
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    for path in (args.detector, args.classifier, SIXD_MODEL, SIXD_DETECTOR):
        if not os.path.exists(path):
            print(f"[ERROR] missing file: {path}")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hand_detector = YOLO(args.detector)
    classifier, labels = load_classifier(args.classifier, device)
    head_estimator = HeadPoseEstimator(SIXD_MODEL, SIXD_DETECTOR, device)

    sound_on = False
    antennas_open = False
    mirror_on = False

    smoother = GestureSmoother(window=7)
    smooth_yaw = smooth_pitch = smooth_roll = 0.0

    # Create the window before connecting (avoids the viewer blocking).
    window = "Part 4 - mirror head"
    cv2.namedWindow(window)

    print("Connecting to Reachy Mini... press 'q' to quit.")
    with ReachyMini() as mini:
        sound_player = SoundPlayer(mini)
        sound_player.start()
        mini.goto_target(create_head_pose(), antennas=ANTENNAS_CLOSED, duration=1.0)

        try:
            while True:
                frame = mini.media.get_frame()
                if frame is None:
                    continue
                # The camera frame is read-only; copy it so we can draw on it.
                frame = frame.copy()

                result = hand_detector(frame, conf=args.conf, verbose=False)[0]
                hands = analyze_hands(result, classifier, labels, device, args.conf)

                command_hand = max(hands, key=lambda h: h["mean_x"]) if hands else None
                both_hearts = len([h for h in hands if h["gesture"] == "heart"]) >= 2

                raw_gesture = command_hand["gesture"] if command_hand else "no_gesture"
                gesture = smoother.update(raw_gesture)

                if gesture == "point":
                    sound_on = True
                elif gesture == "mute":
                    sound_on = False
                elif gesture == "rock":
                    antennas_open = True
                elif gesture == "fist":
                    antennas_open = False
                elif gesture == "peace":
                    mirror_on = True
                elif gesture == "stop":
                    mirror_on = False

                sound_player.set_active(sound_on)
                antennas = ANTENNAS_OPEN if antennas_open else ANTENNAS_CLOSED

                # --- Decide where the head looks ---
                if both_hearts:
                    wave_yaw = 20.0 * np.sin(2.0 * np.pi * 1.0 * time.time())
                    head = create_head_pose(yaw=wave_yaw, degrees=True)
                elif mirror_on:
                    pose = head_estimator.estimate(frame)
                    if pose is not None:
                        pitch, yaw, roll, box = pose
                        # Mirror left/right (negate yaw and roll). Tune the
                        # signs/gain if the robot copies the wrong way.
                        target_yaw = clamp(-MIRROR_GAIN * yaw, MAX_YAW)
                        target_pitch = clamp(MIRROR_GAIN * pitch, MAX_PITCH)
                        target_roll = clamp(-MIRROR_GAIN * roll, MAX_ROLL)
                        smooth_yaw = 0.7 * smooth_yaw + 0.3 * target_yaw
                        smooth_pitch = 0.7 * smooth_pitch + 0.3 * target_pitch
                        smooth_roll = 0.7 * smooth_roll + 0.3 * target_roll
                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    head = create_head_pose(yaw=smooth_yaw, pitch=smooth_pitch,
                                            roll=smooth_roll, degrees=True)
                else:
                    smooth_yaw = smooth_pitch = smooth_roll = 0.0
                    head = create_head_pose()

                mini.set_target(head=head, antennas=antennas)

                for hand in hands:
                    draw_hand(frame, hand["pixels"])
                    wrist = hand["pixels"][0]
                    draw_label(frame, hand["gesture"], (wrist[0], wrist[1] - 10))

                status = (f"cmd:{gesture} | sound:{'ON' if sound_on else 'off'} "
                          f"| antennas:{'open' if antennas_open else 'closed'} "
                          f"| mirror:{'ON' if mirror_on else 'off'} "
                          f"| heart:{'YES' if both_hearts else 'no'}")
                draw_label(frame, status, (10, 30), color=(255, 255, 0))

                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            sound_player.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
