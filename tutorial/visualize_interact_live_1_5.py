"""Part 3.5 - Track the person's head instead of their hand.

Same gesture-driven behaviours as Part 3, but when "peace" is shown the robot
follows the center of the largest detected head (ONNX face detector from
``tutorial/SixDRepNet``) rather than the command-hand wrist. We do not run the
SixDRepNet pose model — only its head detector.

Gesture -> behaviour:
    point  -> (reserved)           mute  -> stop sounds
    rock   -> open antennas        fist  -> close antennas
    peace  -> start head tracking  stop  -> stop tracking
    heart  -> (both hands) wave the head

Controls:
    q : quit

Run (inside the container shell, display forwarded):
    python tutorial/visualize_interact_live_1_5.py
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
import scipy
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

from gesture_utils import classify_hand, draw_hand, draw_label, load_classifier
from hand_detector import HandDetector

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CLASSIFIER = os.path.join(HERE, "weights", "gesture_classifier.pt")

SIXD_DIR = os.path.join(HERE, "SixDRepNet")
sys.path.insert(0, SIXD_DIR)
from utils import util  # noqa: E402  (import after sys.path tweak)

SIXD_DETECTOR = os.path.join(SIXD_DIR, "weights", "detection.onnx")

ANTENNAS_CLOSED = [-180, 180]
ANTENNAS_OPEN = [0, 0]

CAMERA_FOV_H_DEG = 60.0
CAMERA_FOV_V_DEG = 45.0


class HeadDetector:
    """Find the largest head in the frame using the SixDRepNet ONNX detector."""

    def __init__(self, detector_path):
        import onnxruntime

        providers = onnxruntime.get_available_providers()
        session = onnxruntime.InferenceSession(detector_path, providers=providers)
        self.detector = util.FaceDetector(session=session)

    def detect_largest(self, frame):
        """Return ((cx, cy) normalized, pixel box) for the biggest head, or None."""
        boxes = self.detector.detect(frame, (640, 640))
        if boxes is None or len(boxes) == 0:
            return None

        boxes = boxes.astype("int32")
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        box = boxes[int(np.argmax(areas))]
        x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]

        h, w = frame.shape[:2]
        cx = (x_min + x_max) / 2.0 / w
        cy = (y_min + y_max) / 2.0 / h
        return (cx, cy), (x_min, y_min, x_max, y_max)


class GestureSmoother:
    """Returns the most common gesture over the last few frames."""

    def __init__(self, window=7):
        self.history = deque(maxlen=window)

    def update(self, gesture):
        self.history.append(gesture)
        return Counter(self.history).most_common(1)[0][0]


class SoundPlayer:
    """Plays random musical notes in a background thread while active."""

    def __init__(self, mini):
        self.mini = mini
        self.samplerate = mini.media.get_output_audio_samplerate()
        self.active = False
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def set_active(self, active):
        if active != self.active:
            if active:
                self.mini.media.start_playing()
            else:
                self.mini.media.stop_playing()
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


def analyze_hands(detector, frame, classifier, labels, device, conf_threshold):
    """Detect and classify every hand in the frame."""
    hands = []
    for hand in detector.detect(frame):
        gesture, confidence = classify_hand(classifier, labels, hand["norm"], device)
        if confidence < conf_threshold:
            gesture = "no_gesture"
        hands.append({
            "gesture": gesture,
            "confidence": confidence,
            "pixels": hand["pixels"],
            "mean_x": hand["mean_x"],
        })
    return hands


def main():
    parser = argparse.ArgumentParser(
        description="Gesture-driven robot behaviours with head-center tracking."
    )
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER)
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Minimum classifier confidence to trust a gesture.")
    args = parser.parse_args()

    for path in (args.classifier, SIXD_DETECTOR):
        if not os.path.exists(path):
            print(f"[ERROR] missing file: {path}")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hand_detector = HandDetector()
    head_detector = HeadDetector(SIXD_DETECTOR)
    classifier, labels = load_classifier(args.classifier, device)

    sound_on = False
    antennas_open = False
    tracking_on = False

    smoother = GestureSmoother(window=7)

    smooth_antenna_left = ANTENNAS_CLOSED[0]
    smooth_antenna_right = ANTENNAS_CLOSED[1]

    smooth_yaw = 0.0
    smooth_pitch = -10.0
    target_yaw = 0.0
    target_pitch = -10.0

    track_lost_counter = 0
    head_box = None

    window = "Part 3.5 - head tracking"
    cv2.namedWindow(window)

    print("Connecting to Reachy Mini... press 'q' to quit.")
    with ReachyMini() as mini:
        sound_player = SoundPlayer(mini)
        sound_player.start()
        mini.goto_target(create_head_pose(pitch=-10.0),
                         antennas=np.deg2rad(ANTENNAS_CLOSED), duration=1.0)
        mini.set_automatic_body_yaw(True)
        # mini.enable_wobbling() # too heavy wobbling, uncomment to enable it

        try:
            while True:
                frame = mini.media.get_frame()
                if frame is None:
                    continue
                frame = frame.copy()

                hands = analyze_hands(hand_detector, frame, classifier, labels,
                                      device, args.conf)
                head_result = head_detector.detect_largest(frame)

                command_hand = max(hands, key=lambda h: h["mean_x"]) if hands else None

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
                    tracking_on = True
                elif gesture == "stop":
                    tracking_on = False

                sound_player.set_active(sound_on)
                antennas = ANTENNAS_OPEN if antennas_open else ANTENNAS_CLOSED

                smooth_antenna_left = 0.8 * smooth_antenna_left + 0.2 * antennas[0]
                smooth_antenna_right = 0.8 * smooth_antenna_right + 0.2 * antennas[1]

                current_head_pos = mini.get_current_head_pose()
                current_yaw, current_pitch, _ = scipy.spatial.transform.Rotation.from_matrix(
                    current_head_pos[:3, :3]
                ).as_euler("zyx", degrees=True)

                head_box = None
                if tracking_on and head_result is not None:
                    track_lost_counter = 0
                    (hx, hy), head_box = head_result
                    _im_target_yaw = -(hx - 0.5) * (CAMERA_FOV_H_DEG * 1.3)
                    target_yaw = current_yaw + _im_target_yaw
                    _im_target_pitch = (hy - 0.5) * (CAMERA_FOV_V_DEG * 1.3)
                    target_pitch = current_pitch + _im_target_pitch
                elif tracking_on and head_result is None:
                    track_lost_counter += 1
                    if track_lost_counter > 10:
                        print("Head lost, stop tracking and reset.")
                        tracking_on = False
                        target_yaw = 0.0
                        target_pitch = -10.0
                        track_lost_counter = 0
                else:
                    target_yaw = 0.0
                    target_pitch = -10.0

                smooth_yaw = 0.8 * smooth_yaw + 0.2 * target_yaw
                smooth_pitch = 0.8 * smooth_pitch + 0.2 * target_pitch
                head = create_head_pose(yaw=smooth_yaw, pitch=smooth_pitch, degrees=True)

                mini.set_target(head=head,
                                antennas=np.deg2rad([smooth_antenna_left, smooth_antenna_right]))

                for hand in hands:
                    draw_hand(frame, hand["pixels"])
                    wrist = hand["pixels"][0]
                    draw_label(frame, hand["gesture"], (wrist[0], wrist[1] - 10))

                if head_box is not None:
                    x1, y1, x2, y2 = head_box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

                status = (f"cmd:{gesture} | sound:{'ON' if sound_on else 'off'} "
                          f"| antennas:{'open' if antennas_open else 'closed'} "
                          f"| track:{'ON' if tracking_on else 'off'} ({track_lost_counter} lost) "
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
