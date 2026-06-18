"""Part 3 - Make the robot react to gestures.

We reuse the Part 1 detector + Part 2 classifier to read the gesture of the
"command hand" (the hand on the right side of the image), and map gestures to
robot behaviours:

    point  -> start playing random sounds
    mute   -> stop the sounds
    rock   -> open the antennas
    fist   -> close the antennas
    peace  -> start tracking the hand with the head
    stop   -> stop tracking
    heart  -> ONLY when BOTH hands show heart: wave the head

Controls:
    q : quit

Run (inside the container shell, display forwarded):
    python tutorial/visualize_interact_live.py
"""

import argparse
import os
import random
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

# Antenna poses in radians [left, right]. Tune these to taste.
ANTENNAS_OPEN = [1.0, 1.0]
ANTENNAS_CLOSED = [0.0, 0.0]

# How far the head can turn while tracking (degrees).
MAX_YAW = 30.0
MAX_PITCH = 20.0


class GestureSmoother:
    """Returns the most common gesture over the last few frames.

    This removes the single-frame flicker that would otherwise make the robot
    react to noise.
    """

    def __init__(self, window=7):
        self.history = deque(maxlen=window)

    def update(self, gesture):
        self.history.append(gesture)
        most_common = Counter(self.history).most_common(1)[0][0]
        return most_common


class SoundPlayer:
    """Plays random musical notes in a background thread while active.

    Pushing audio has to happen continuously, so we do it in its own thread and
    just flip a boolean from the main loop.
    """

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
        notes = [262, 294, 330, 349, 392, 440, 494, 523]  # a C major scale
        chunk_seconds = 0.02
        samples_per_chunk = int(self.samplerate * chunk_seconds)
        phase = 0.0
        frequency = 440.0
        next_note_time = 0.0

        while self.running:
            if not self.active:
                time.sleep(0.05)
                continue

            # Pick a new random note every so often.
            if time.time() > next_note_time:
                frequency = random.choice(notes)
                next_note_time = time.time() + random.uniform(0.2, 0.5)

            t = np.arange(samples_per_chunk, dtype=np.float32) / self.samplerate
            mono = (0.3 * np.sin(2.0 * np.pi * frequency * t + phase)).astype(np.float32)
            phase += 2.0 * np.pi * frequency * samples_per_chunk / self.samplerate

            self.mini.media.push_audio_sample(mono)
            time.sleep(chunk_seconds * 0.5)


def analyze_hands(result, classifier, labels, device, conf_threshold):
    """Classify every detected hand.

    Returns a list of dicts, one per hand, with its gesture, confidence,
    pixel keypoints and normalized wrist position.
    """
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
            "wrist_norm": hand_norm[0],     # (x, y) in [0, 1]
            "mean_x": float(hand_norm[:, 0].mean()),
        })
    return hands


def main():
    parser = argparse.ArgumentParser(description="Gesture-driven robot behaviours.")
    parser.add_argument("--detector", default=DEFAULT_DETECTOR)
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER)
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Minimum classifier confidence to trust a gesture.")
    args = parser.parse_args()

    for path in (args.detector, args.classifier):
        if not os.path.exists(path):
            print(f"[ERROR] missing file: {path}")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YOLO(args.detector)
    classifier, labels = load_classifier(args.classifier, device)

    # Behaviour state.
    sound_on = False
    antennas_open = False
    tracking_on = False

    smoother = GestureSmoother(window=7)
    smooth_yaw = 0.0    # low-pass filtered head angles (degrees)
    smooth_pitch = 0.0

    # Create the window before connecting (avoids the viewer blocking).
    window = "Part 3 - interact"
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

                result = detector(frame, conf=args.conf, verbose=False)[0]
                hands = analyze_hands(result, classifier, labels, device, args.conf)

                # The "command hand" is the right-most hand in the image.
                command_hand = None
                if hands:
                    command_hand = max(hands, key=lambda h: h["mean_x"])

                # Both hands showing a heart?
                heart_hands = [h for h in hands if h["gesture"] == "heart"]
                both_hearts = len(heart_hands) >= 2

                # Smooth the command gesture over a few frames.
                raw_gesture = command_hand["gesture"] if command_hand else "no_gesture"
                gesture = smoother.update(raw_gesture)

                # --- Update behaviour state from the gesture ---
                if gesture == "point":
                    sound_on = True
                elif gesture == "mute":
                    sound_on = False
                elif gesture == "rock":
                    antennas_open = True
                elif gesture == "fist":
                    antennas_open = False
                elif gesture == "peace":
                    tracking_on = True
                elif gesture == "stop":
                    tracking_on = False

                sound_player.set_active(sound_on)
                antennas = ANTENNAS_OPEN if antennas_open else ANTENNAS_CLOSED

                # --- Decide where the head looks ---
                # Priority: heart wave > tracking > neutral.
                if both_hearts:
                    wave_yaw = 20.0 * np.sin(2.0 * np.pi * 1.0 * time.time())
                    head = create_head_pose(yaw=wave_yaw, degrees=True)
                elif tracking_on and command_hand is not None:
                    hx, hy = command_hand["wrist_norm"]
                    # Center the hand: turn head toward it. Flip a sign here if
                    # the head moves the wrong way on your robot.
                    target_yaw = (0.5 - hx) * 2.0 * MAX_YAW
                    target_pitch = (hy - 0.5) * 2.0 * MAX_PITCH
                    smooth_yaw = 0.8 * smooth_yaw + 0.2 * target_yaw
                    smooth_pitch = 0.8 * smooth_pitch + 0.2 * target_pitch
                    head = create_head_pose(yaw=smooth_yaw, pitch=smooth_pitch,
                                            degrees=True)
                else:
                    smooth_yaw, smooth_pitch = 0.0, 0.0
                    head = create_head_pose()

                mini.set_target(head=head, antennas=antennas)

                # --- Draw the current state on screen ---
                for hand in hands:
                    draw_hand(frame, hand["pixels"])
                    wrist = hand["pixels"][0]
                    draw_label(frame, hand["gesture"], (wrist[0], wrist[1] - 10))

                status = (f"cmd:{gesture} | sound:{'ON' if sound_on else 'off'} "
                          f"| antennas:{'open' if antennas_open else 'closed'} "
                          f"| track:{'ON' if tracking_on else 'off'} "
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
