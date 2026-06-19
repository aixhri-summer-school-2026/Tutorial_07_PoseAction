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
import scipy
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

from gesture_utils import classify_hand, draw_hand, draw_label, load_classifier
from hand_detector import HandDetector

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CLASSIFIER = os.path.join(HERE, "weights", "gesture_classifier.pt")

# Antenna poses in degrees [left, right]. Tune these to taste.
ANTENNAS_CLOSED = [-180, 180]
ANTENNAS_OPEN = [0, 0]

# How far the head can turn while tracking (degrees).
# MAX_YAW = 30.0
# MAX_PITCH = 20.0


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
                # print("SoundPlayer: not active")
                time.sleep(0.05)
                continue
            
            # print("SoundPlayer: active")
            # Pick a new random note every so often.
            if time.time() > next_note_time:
                frequency = random.choice(notes)
                next_note_time = time.time() + random.uniform(0.2, 0.5)

            t = np.arange(samples_per_chunk, dtype=np.float32) / self.samplerate
            mono = (0.3 * np.sin(2.0 * np.pi * frequency * t + phase)).astype(np.float32)
            phase += 2.0 * np.pi * frequency * samples_per_chunk / self.samplerate

            self.mini.media.push_audio_sample(mono)
            time.sleep(chunk_seconds * 0.5)


def analyze_hands(detector, frame, classifier, labels, device, conf_threshold):
    """Detect and classify every hand in the frame.

    Returns a list of dicts, one per hand, with its gesture, confidence,
    pixel keypoints and normalized wrist position.
    """
    hands = []
    for hand in detector.detect(frame):
        gesture, confidence = classify_hand(classifier, labels, hand["norm"], device)
        if confidence < conf_threshold:
            gesture = "no_gesture"
        hands.append({
            "gesture": gesture,
            "confidence": confidence,
            "pixels": hand["pixels"],
            "wrist_norm": hand["norm"][0],     # (x, y) in [0, 1]
            "mean_x": hand["mean_x"],
        })
    return hands


def main():
    parser = argparse.ArgumentParser(description="Gesture-driven robot behaviours.")
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER)
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Minimum classifier confidence to trust a gesture.")
    args = parser.parse_args()

    if not os.path.exists(args.classifier):
        print(f"[ERROR] missing file: {args.classifier}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = HandDetector()
    classifier, labels = load_classifier(args.classifier, device)

    # Behaviour state.
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

    track_hand_lost_counter = 0
    
    # Create the window before connecting (avoids the viewer blocking).
    window = "Part 3 - interact"
    cv2.namedWindow(window)

    print("Connecting to Reachy Mini... press 'q' to quit.")
    with ReachyMini() as mini:
        sound_player = SoundPlayer(mini)
        sound_player.start()
        # mini.goto_target(create_head_pose(), antennas=ANTENNAS_CLOSED, duration=1.0)
        mini.goto_target(create_head_pose(pitch=-10.0), antennas=np.deg2rad(ANTENNAS_CLOSED), duration=1.0)
        mini.set_automatic_body_yaw(True)
        pos, antennas = mini.client.get_current_joints()

        # cv2.destroyAllWindows()
        # return
        lat = time.time()
        try:
            while True:
                time_since_start = time.time() - lat
                print(f"latency: {time_since_start:.2f} seconds")
                lat = time.time()
                
                frame = mini.media.get_frame()
                image_width = frame.shape[1]
                image_height = frame.shape[0]
                if frame is None:
                    continue
                # The camera frame is read-only; copy it so we can draw on it.
                frame = frame.copy()

                hands = analyze_hands(detector, frame, classifier, labels,
                                      device, args.conf)

                # The "command hand" is the right-most hand in the image.
                command_hand = None
                if hands:
                    track_hand_lost_counter = 0
                    command_hand = max(hands, key=lambda h: h["mean_x"])
                else:
                    command_hand = None
                    print("No hands detected")

                # Both hands showing a heart?
                heart_hands = [h for h in hands if h["gesture"] == "heart"]
                both_hearts = len(heart_hands) >= 2

                # Smooth the command gesture over a few frames.
                raw_gesture = command_hand["gesture"] if command_hand else "no_gesture"
                gesture = smoother.update(raw_gesture)

                # --- Update behaviour state from the gesture ---
                if gesture == "point":
                    # sound_on = True
                    pass
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
                
                # --- Decide where the head looks by yaw and pitch RELATIVE command ---
                # Priority: heart wave > tracking > neutral.
                current_head_pos = mini.get_current_head_pose()
                current_yaw, current_pitch, current_roll = scipy.spatial.transform.Rotation.from_matrix(current_head_pos[:3, :3]).as_euler('zyx', degrees=True)
                
                if both_hearts:
                    pass
                elif tracking_on and command_hand is not None:
                    hx, hy = command_hand["wrist_norm"]
                    CAMERA_FOV_H_DEG = 60.0
                    CAMERA_FOV_V_DEG = 45.0
                    _im_target_yaw = -(hx - 0.5) * CAMERA_FOV_H_DEG
                    target_yaw = current_yaw + _im_target_yaw
                    
                    _im_target_pitch = (hy - 0.5) * CAMERA_FOV_V_DEG
                    target_pitch = current_pitch + _im_target_pitch
                    
                elif tracking_on and command_hand is None:
                    # temporary lost, wait for 10 frames to confirm, keep the current command
                    track_hand_lost_counter += 1
                    if track_hand_lost_counter > 10:
                        print("lost, stop tracking and reset the command")
                        # lost, stop tracking and reset the command
                        tracking_on = False
                        target_yaw = 0.0
                        target_pitch = -10.0
                        track_hand_lost_counter = 0
                else:
                    target_yaw = 0.0
                    target_pitch = -10.0
                    
                smooth_yaw = 0.8 * smooth_yaw + 0.2 * target_yaw
                smooth_pitch = 0.8 * smooth_pitch + 0.2 * target_pitch
                head = create_head_pose(yaw=smooth_yaw, pitch=smooth_pitch, degrees=True)
                
                mini.set_target(head=head, antennas=np.deg2rad([smooth_antenna_left, smooth_antenna_right]))
                
                # --- Draw the current state on screen ---
                for hand in hands:
                    draw_hand(frame, hand["pixels"])
                    wrist = hand["pixels"][0]
                    draw_label(frame, hand["gesture"], (wrist[0], wrist[1] - 10))

                status = (f"cmd:{gesture} | sound:{'ON' if sound_on else 'off'} "
                          f"| antennas:{'open' if antennas_open else 'closed'} "
                          f"| track:{'ON' if tracking_on else 'off'} ({track_hand_lost_counter} lost) "
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
