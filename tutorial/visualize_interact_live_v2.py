"""Part 4 - Gesture-driven behaviours with head tracking or mirroring.

Same gesture-driven behaviours as Part 3.5. At launch, choose whether
``peace`` / ``stop`` start/stop **head-center tracking** (Part 3.5) or
**head-yaw mirroring** (SixDRepNet pose estimate, yaw only).

Gesture -> behaviour:
    point  -> (reserved)           mute  -> stop sounds
    rock   -> open antennas        fist  -> close antennas
    peace  -> start pose control   stop  -> stop pose control
    hand_heart -> (both hands) wave the head

Controls:
    q : quit

Run (inside the container shell, display forwarded):
    python tutorial/visualize_interact_live_v2.py --mode tracking
    python tutorial/visualize_interact_live_v2.py --mode mirroring
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
import onnxruntime as ort
import torch
import scipy
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from rtmlib import PoseTracker, Wholebody
from torchvision import transforms

from handkeypoints_infer import classify_hand, load_classifier
from keypoints_utils import (
    LEFT_HAND_IDS,
    LEFT_WRIST_ID,
    RIGHT_HAND_IDS,
    RIGHT_WRIST_ID,
    draw_bbox,
    draw_label,
    draw_skeleton,
    get_hand_bbox,
    get_face_bbox,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CLASSIFIER = os.path.join(HERE, "weights", "gesture_classifier.pt")

SIXD_DIR = os.path.join(HERE, "SixDRepNet")
sys.path.insert(0, SIXD_DIR)
from utils import util  # noqa: E402  (import after sys.path tweak)

SIXD_MODEL = os.path.join(SIXD_DIR, "weights", "best.pt")

ANTENNAS_CLOSED = [-180, 180]
ANTENNAS_OPEN = [0, 0]

CAMERA_FOV_H_DEG = 60.0
CAMERA_FOV_V_DEG = 45.0

MIRROR_GAIN = 1.0
MAX_YAW = 30.0
SCORE_THRESHOLD = 0.5


def build_pose_tracker():
    device = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    print(f"Loading the RTMLib whole-body tracker on {device}...")

    # You can replace those path by URLs like : "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
    # and the ehckpoints will be downloaded automatically
    det_onnx_model = "/app/downloads/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx"
    pose_onnx_model = "/app/downloads/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.onnx"
    
    solution_kwargs = {
        "det": det_onnx_model,
        "det_input_size": (640, 640),
        "pose": pose_onnx_model,
        "pose_input_size": (288, 384),
    }

    return PoseTracker(
        Wholebody,
        det_frequency=1,
        backend="onnxruntime",
        device=device,
        to_openpose=False,
        solution_kwargs=solution_kwargs,
    )


def get_largest_face_bbox(keypoints, scores, kpt_thr):
    """Return the largest face box found across all detected people."""
    best_bbox = None
    best_area = 0

    for person_kpts, person_scores in zip(keypoints, scores):
        bbox = get_face_bbox(person_kpts, person_scores, kpt_thr=kpt_thr)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_bbox = bbox

    return best_bbox


def face_bbox_center(bbox, frame_shape):
    """Return normalized face center and the pixel box."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    return (cx, cy), bbox


class HeadPoseEstimator:
    """Estimate (pitch, yaw, roll) from a face crop using SixDRepNet."""

    def __init__(self, model_path, device):
        self.device = device
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model = checkpoint["model"].float().fuse().to(device)
        self.model.eval()

        self.input_size = 224
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size + 32),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def estimate(self, frame, box):
        """Return (pitch, yaw, roll, box) for the given face box, or None."""
        from PIL import Image

        x_min, y_min, x_max, y_max = box
        box_w = abs(x_max - x_min)
        box_h = abs(y_max - y_min)
        x_min = max(0, x_min - int(0.2 * box_h))
        y_min = max(0, y_min - int(0.2 * box_w))
        x_max = x_max + int(0.2 * box_h)
        y_max = y_max + int(0.2 * box_w)

        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = self.transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(self.device)

        output = self.model(image)
        euler = util.compute_euler(output) * 180 / np.pi
        pitch = float(euler[0, 0].cpu())
        yaw = float(euler[0, 1].cpu())
        roll = float(euler[0, 2].cpu())
        return pitch, yaw, roll, (x_min, y_min, x_max, y_max)


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


def analyze_hands(keypoints, scores, frame_shape, classifier, labels, device,
                  kpt_thr, conf_threshold, track_ids):
    """Classify every visible hand from the whole-body pose."""
    h, w = frame_shape[:2]
    scale = np.array([w, h], dtype=np.float32)
    hands = []

    for det_idx, (person_kpts, person_scores, track_id) in enumerate(zip(keypoints, scores, track_ids)):
        for hand_ids, wrist_id, handedness in (
            (LEFT_HAND_IDS, LEFT_WRIST_ID, "left"),
            (RIGHT_HAND_IDS, RIGHT_WRIST_ID, "right"),
        ):
            if person_scores[wrist_id] < kpt_thr:
                continue
            hand_scores = person_scores[hand_ids]
            if (hand_scores >= kpt_thr).sum() < 5:
                continue

            hand_px = person_kpts[hand_ids].astype(np.float32)
            person_size = (person_kpts[:, 0].max() - person_kpts[:, 0].min()) * (person_kpts[:, 1].max() - person_kpts[:, 1].min())
            gesture, confidence = classify_hand(
                classifier, labels, hand_px / scale, device)
            if confidence < conf_threshold:
                gesture = "no_gesture"
                
            hand_bbox = get_hand_bbox(person_kpts, person_scores, side = handedness)

            hands.append({
                "det_idx": det_idx,
                "track_id": track_id,
                "gesture": gesture,
                "confidence": confidence,
                "pixels": hand_px,
                "handedness": handedness,
                "person_size": person_size,
                "hand_bbox": hand_bbox,
            })

    return hands


def clamp(value, limit):
    return max(-limit, min(limit, value))


def main():
    parser = argparse.ArgumentParser(
        description="Gesture-driven robot behaviours with head tracking or mirroring."
    )
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER)
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Minimum classifier confidence to trust a gesture.")
    parser.add_argument(
        "--mode",
        choices=["tracking", "mirroring"],
        required=True,
        help="Head control mode: follow head center (tracking) or mirror yaw (mirroring).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_tracker = build_pose_tracker()
    head_estimator = None
    if args.mode == "mirroring":
        head_estimator = HeadPoseEstimator(SIXD_MODEL, device)
    classifier, labels = load_classifier(args.classifier, device)

    sound_on = False
    antennas_open = False
    pose_active = False

    smoother = GestureSmoother(window=7)

    smooth_antenna_left = ANTENNAS_CLOSED[0]
    smooth_antenna_right = ANTENNAS_CLOSED[1]

    smooth_yaw = 0.0
    smooth_pitch = -10.0
    target_yaw = 0.0
    target_pitch = -10.0

    track_lost_counter = 0
    head_box = None
    head_pose_label = None

    window = f"Part 4 - {args.mode}"
    cv2.namedWindow(window)

    print(f"Connecting to Reachy Mini... press 'q' to quit.")
    with ReachyMini() as mini:
        sound_player = SoundPlayer(mini)
        sound_player.start()
        mini.goto_target(create_head_pose(pitch=-10.0),
                         antennas=np.deg2rad(ANTENNAS_CLOSED), duration=1.0)
        mini.set_automatic_body_yaw(True)

        try:
            while True:
                frame = mini.media.get_frame()
                if frame is None:
                    continue
                frame = frame.copy()

                keypoints, scores = pose_tracker(frame)
                track_ids = pose_tracker.track_ids_last_frame
                for det in range(scores.shape[0]):
                    if scores[det, LEFT_WRIST_ID] < SCORE_THRESHOLD:
                        scores[det, LEFT_HAND_IDS] = 0.0
                    if scores[det, RIGHT_WRIST_ID] < SCORE_THRESHOLD:
                        scores[det, RIGHT_HAND_IDS] = 0.0

                hands = analyze_hands(
                    keypoints, scores, frame.shape, classifier, labels,
                    device, SCORE_THRESHOLD, args.conf, track_ids)

                print(hands)
                
                # we only take left hand of the biggest person
                biggest_person_size = max(hands, key=lambda h: h["person_size"])["person_size"] if hands else None
                command_hand = [h for h in hands if (h["person_size"] == biggest_person_size and h["handedness"] == "left")] if biggest_person_size else None
                if command_hand and len(command_hand) > 0:
                    command_hand = command_hand[0]
                    draw_bbox(frame, command_hand["hand_bbox"], color=(0, 255, 0))
                
                # are there both hands of the biggest person doing the heart gesture?
                both_hearts = len([h for h in hands if (h["person_size"] == biggest_person_size and h["gesture"] == "hand_heart")]) >= 2

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
                    pose_active = True
                elif gesture == "stop":
                    pose_active = False

                sound_player.set_active(sound_on)
                antennas = ANTENNAS_OPEN if antennas_open else ANTENNAS_CLOSED

                smooth_antenna_left = 0.8 * smooth_antenna_left + 0.2 * antennas[0]
                smooth_antenna_right = 0.8 * smooth_antenna_right + 0.2 * antennas[1]

                head_box = None
                head_pose_label = None
                face_bbox = get_largest_face_bbox(keypoints, scores, SCORE_THRESHOLD)

                if both_hearts:
                    wave_yaw = 20.0 * np.sin(2.0 * np.pi * 1.0 * time.time())
                    target_yaw = wave_yaw
                    target_pitch = -10.0
                    
                elif pose_active and args.mode == "tracking":
                    current_head_pos = mini.get_current_head_pose()
                    current_yaw, current_pitch, _ = scipy.spatial.transform.Rotation.from_matrix(
                        current_head_pos[:3, :3]
                    ).as_euler("zyx", degrees=True)

                    if face_bbox is not None:
                        track_lost_counter = 0
                        (hx, hy), head_box = face_bbox_center(face_bbox, frame.shape)
                        _im_target_yaw = -(hx - 0.5) * (CAMERA_FOV_H_DEG * 1.3)
                        target_yaw = current_yaw + _im_target_yaw
                        _im_target_pitch = (hy - 0.5) * (CAMERA_FOV_V_DEG * 1.3)
                        target_pitch = current_pitch + _im_target_pitch
                    else:
                        track_lost_counter += 1
                        if track_lost_counter > 10:
                            print("Head lost, stop tracking and reset.")
                            pose_active = False
                            target_yaw = 0.0
                            target_pitch = -10.0
                            track_lost_counter = 0
                            
                elif pose_active and args.mode == "mirroring":
                    if face_bbox is not None:
                        pose = head_estimator.estimate(frame, face_bbox)
                        if pose is not None:
                            track_lost_counter = 0
                            pitch, yaw, roll, head_box = pose
                            target_yaw = clamp(MIRROR_GAIN * yaw, MAX_YAW)
                            target_pitch = -10.0
                            head_pose_label = (
                                f"yaw:{yaw:.1f} pitch:{pitch:.1f} roll:{roll:.1f}"
                            )
                        else:
                            track_lost_counter += 1
                    else:
                        track_lost_counter += 1

                    if track_lost_counter > 10:
                        print("Head lost, stop mirroring and reset.")
                        pose_active = False
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

                for person_kpts, person_scores in zip(keypoints, scores):
                    draw_skeleton(
                        frame,
                        person_kpts,
                        scores=person_scores,
                        kpt_thr=SCORE_THRESHOLD,
                        include_face=True,
                    )

                for hand in hands:
                    wrist = hand["pixels"][0]
                    draw_label(frame, hand["gesture"], (wrist[0], wrist[1] - 10))

                if head_box is not None:
                    draw_bbox(frame, head_box, color=(255, 0, 0))
                    x1, y1, x2, y2 = head_box
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
                    if head_pose_label is not None:
                        draw_label(frame, head_pose_label, (x1, y1 - 10))

                pose_status = "ON" if pose_active else "off"
                status = (f"cmd:{gesture} | sound:{'ON' if sound_on else 'off'} "
                          f"| antennas:{'open' if antennas_open else 'closed'} "
                          f"| {args.mode}:{pose_status} ({track_lost_counter} lost) "
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
