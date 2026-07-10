"""Part 4 - Gesture-driven behaviours with head tracking or mirroring.

``peace`` / ``stop`` start/stop head control and lock/unlock the commander
track ID. Use ``thumb_index`` on the command hand to toggle between
**head-center tracking** and **head-yaw mirroring** (SixDRepNet, yaw only).
While active, the robot follows the face of whoever issued ``peace``, not
necessarily the largest person in frame.

Gesture -> behaviour:
    call   -> play cartoon sound   mute  -> stop sounds
    rock   -> open antennas        fist  -> close antennas
    pinkie -> raise one antenna (left/right hand)
    peace  -> start pose control   stop  -> stop pose control
    thumb_index -> toggle tracking / mirroring mode
    hand_heart -> (both hands) wave the head

Controls:
    q : quit

Run (inside the container shell, display forwarded):
    python tutorial/visualize_interact_live_v2.py
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
from reachy_mini.media.gstreamer_utils import audio_duration_seconds
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
HEART_WAVE_DURATION = 3.0
HEART_WAVE_YAW_AMPLITUDE = 35.0
HEART_WAVE_PITCH_BASE = -10.0
HEART_WAVE_PITCH_AMPLITUDE = 12.0
HEART_WAVE_HEAD_FREQ = 1.0
HEART_WAVE_PITCH_FREQ = 1.6
HEART_WAVE_ANTENNA_AMPLITUDE = 28.0
HEART_WAVE_ANTENNA_FREQ = 3.0
HEART_WAVE_ANTENNA_JITTER_FREQ = 7.0


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
        "biggest_n_boxes_only": 10,
    }

    return PoseTracker(
        Wholebody,
        det_frequency=1,
        backend="onnxruntime",
        device=device,
        to_openpose=False,
        biggest_n_boxes_only=10,
        solution_kwargs=solution_kwargs,
    )


def get_person_bbox(person_kpts, person_scores, kpt_thr):
    """Return a full-body bbox from visible keypoints, or None."""
    visible = person_scores >= kpt_thr
    if not np.any(visible):
        return None

    visible_kpts = person_kpts[visible]
    x1, y1 = visible_kpts.min(axis=0)
    x2, y2 = visible_kpts.max(axis=0)
    return (int(x1), int(y1), int(x2), int(y2))


def find_largest_person(keypoints, scores, kpt_thr):
    """Return (det_idx, person_bbox, area) for the largest detected person."""
    best_idx = None
    best_bbox = None
    best_area = 0

    for det_idx, (person_kpts, person_scores) in enumerate(zip(keypoints, scores)):
        person_bbox = get_person_bbox(person_kpts, person_scores, kpt_thr)
        if person_bbox is None:
            continue

        x1, y1, x2, y2 = person_bbox
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_idx = det_idx
            best_bbox = person_bbox

    return best_idx, best_bbox, best_area


def find_person_by_track_id(keypoints, scores, track_ids, target_track_id,
                            kpt_thr):
    """Return (det_idx, face_bbox, person_bbox) for a tracked person id."""
    if target_track_id is None:
        return None, None, None

    for det_idx, track_id in enumerate(track_ids):
        if track_id != target_track_id:
            continue

        person_kpts = keypoints[det_idx]
        person_scores = scores[det_idx]
        face_bbox = get_face_bbox(person_kpts, person_scores, kpt_thr=kpt_thr)
        person_bbox = get_person_bbox(person_kpts, person_scores, kpt_thr)
        return det_idx, face_bbox, person_bbox

    return None, None, None


def get_person_hands(hands, track_id=None, det_idx=None):
    """Return left and right hand detections for one person."""
    if track_id is not None:
        person_hands = [h for h in hands if h["track_id"] == track_id]
    elif det_idx is not None:
        person_hands = [h for h in hands if h["det_idx"] == det_idx]
    else:
        return None, None

    left_hand = next((h for h in person_hands if h["handedness"] == "left"), None)
    right_hand = next((h for h in person_hands if h["handedness"] == "right"), None)
    return left_hand, right_hand


def bbox_iou(bbox_a, bbox_b):
    """Compute IoU between two xyxy bboxes."""
    if bbox_a is None or bbox_b is None:
        return 0.0

    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area == 0.0:
        return 0.0

    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


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


class BoolSmoother:
    """Returns True only after value stayed True for the full window."""

    def __init__(self, window=3):
        self.history = deque(maxlen=window)

    def update(self, value):
        self.history.append(bool(value))
        return len(self.history) == self.history.maxlen and all(self.history)

class FilePlayer:
    """Plays a sound file exactly once, on-demand, in a separate thread."""

    def __init__(self, mini, file_path: str):
        self.mini = mini
        self.file_path = os.path.abspath(file_path)
        self._thread = None

    def play(self):
        """Triggers file playback in a background thread if not already playing."""
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._play_once, daemon=True)
            self._thread.start()

    def _play_once(self):
        print(f"FilePlayer: Playing {self.file_path}...")
        self.mini.media.play_sound(self.file_path)
        time.sleep(audio_duration_seconds(self.file_path))
        print("FilePlayer: Playback finished.")

       
        

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
        default="tracking",
        help="Initial head control mode; thumb_index toggles between tracking and mirroring.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_tracker = build_pose_tracker()
    head_estimator = None
    control_mode = args.mode
    classifier, labels = load_classifier(args.classifier, device)

    sound_on = False
    antennas_open = False
    pose_active = False

    smoother = GestureSmoother(window=7)
    heart_smoother = BoolSmoother(window=3)
    prev_gesture = "no_gesture"

    smooth_antenna_left = ANTENNAS_CLOSED[0]
    smooth_antenna_right = ANTENNAS_CLOSED[1]

    smooth_yaw = 0.0
    smooth_pitch = -10.0
    target_yaw = 0.0
    target_pitch = -10.0

    track_lost_counter = 0
    commander_track_id = None
    heart_wave_until = 0.0
    prev_both_hearts = False
    commander_hand_overlap = 0.0
    head_box = None
    head_pose_label = None

    window = f"Part 4 - {control_mode}"
    cv2.namedWindow(window)

    print(f"Connecting to Reachy Mini... press 'q' to quit.")
    with ReachyMini() as mini:
        sound_player = SoundPlayer(mini)
        sound_player.start()
        
        fluttering_player = FilePlayer(mini, os.path.join(HERE, "sounds", "cartoon-fluttering.wav"))
        rebondissement_player = FilePlayer(mini, os.path.join(HERE, "sounds", "rebondissement.wav"))
        
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

                largest_det_idx, largest_person_bbox, _ = (
                    find_largest_person(keypoints, scores, SCORE_THRESHOLD)
                )

                # Command gestures come from the right hand of the biggest person.
                command_hand = None
                if largest_det_idx is not None:
                    if commander_track_id is not None:
                        # keep looking for the hand of the commander if it exists
                        matching = [
                            h for h in hands
                            if h["track_id"] == commander_track_id
                            and h["handedness"] == "right"
                        ]
                    else:
                        # if it does not exist, look for the right hand of the biggest person
                        matching = [
                            h for h in hands
                            if h["det_idx"] == largest_det_idx
                            and h["handedness"] == "right"
                        ]
                    if matching:
                        command_hand = matching[0]
                        draw_bbox(frame, command_hand["hand_bbox"], color=(0, 255, 0))
                
                # Both hearts: commander's hands must show heart gesture and overlap.
                left_hand, right_hand = get_person_hands(
                    hands,
                    track_id=commander_track_id,
                    det_idx=largest_det_idx if commander_track_id is None else None,
                )
                if (
                    left_hand is not None
                    and right_hand is not None
                    and left_hand["hand_bbox"] is not None
                    and right_hand["hand_bbox"] is not None
                ):
                    commander_hand_overlap = bbox_iou(
                        left_hand["hand_bbox"], right_hand["hand_bbox"])
                else:
                    commander_hand_overlap = 0.0

                both_hearts_raw = (
                    left_hand is not None
                    and right_hand is not None
                    and left_hand["gesture"] == "hand_heart"
                    and right_hand["gesture"] == "hand_heart"
                    and commander_hand_overlap > 0.0
                )
                both_hearts = heart_smoother.update(both_hearts_raw)

                if both_hearts:
                    heart_wave_until = max(
                        heart_wave_until, time.time() + HEART_WAVE_DURATION)
                    if not prev_both_hearts:
                        rebondissement_player.play()
                prev_both_hearts = both_hearts
                heart_wave_active = time.time() < heart_wave_until

                raw_gesture = command_hand["gesture"] if command_hand else "no_gesture"
                gesture = smoother.update(raw_gesture)

                if gesture == "thumb_index" and prev_gesture != "thumb_index":
                    control_mode = (
                        "mirroring" if control_mode == "tracking" else "tracking"
                    )
                    print(f"Switched head control mode to: {control_mode}")
                    track_lost_counter = 0
                    cv2.setWindowTitle(window, f"Part 4 - {control_mode}")
                prev_gesture = gesture

                if gesture == "call":
                    sound_on = True
                    fluttering_player.play() # single play
                elif gesture == "mute":
                    sound_on = False
                elif gesture == "rock":
                    antennas_open = True
                elif gesture == "fist":
                    antennas_open = False
                elif gesture == "peace":
                    if command_hand is not None:
                        pose_active = True
                        commander_track_id = command_hand["track_id"]
                        print(f"Head control target locked to track ID {commander_track_id}")
                elif gesture == "stop":
                    pose_active = False
                    commander_track_id = None

                # sound_player.set_active(sound_on)
                antenna_left_target = (
                    ANTENNAS_OPEN[0] if antennas_open else ANTENNAS_CLOSED[0]
                )
                antenna_right_target = (
                    ANTENNAS_OPEN[1] if antennas_open else ANTENNAS_CLOSED[1]
                )
                if largest_det_idx is not None:
                    for hand in hands:
                        if hand["det_idx"] != largest_det_idx:
                            continue
                        if hand["gesture"] != "pinkie":
                            continue
                        if hand["handedness"] == "left":
                            antenna_left_target = ANTENNAS_OPEN[0]
                        else:
                            antenna_right_target = ANTENNAS_OPEN[1]

                head_box = None
                head_pose_label = None
                _, face_bbox, commander_person_bbox = (
                    find_person_by_track_id(
                        keypoints, scores, track_ids,
                        commander_track_id, SCORE_THRESHOLD)
                )

                if heart_wave_active:
                    t = time.time()
                    target_yaw = (
                        HEART_WAVE_YAW_AMPLITUDE
                        * np.sin(2.0 * np.pi * HEART_WAVE_HEAD_FREQ * t)
                    )
                    target_pitch = (
                        HEART_WAVE_PITCH_BASE
                        + HEART_WAVE_PITCH_AMPLITUDE
                        * np.sin(2.0 * np.pi * HEART_WAVE_PITCH_FREQ * t + np.pi / 4)
                    )
                    antenna_wiggle = np.sin(2.0 * np.pi * HEART_WAVE_ANTENNA_FREQ * t)
                    antenna_jitter = 0.35 * np.sin(
                        2.0 * np.pi * HEART_WAVE_ANTENNA_JITTER_FREQ * t
                    )
                    antenna_left_target = HEART_WAVE_ANTENNA_AMPLITUDE * (
                        antenna_wiggle + antenna_jitter
                    )
                    antenna_right_target = HEART_WAVE_ANTENNA_AMPLITUDE * (
                        -antenna_wiggle + antenna_jitter
                    )
                    
                elif pose_active and control_mode == "tracking":
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
                            print("Commander lost, stop tracking and reset.")
                            pose_active = False
                            commander_track_id = None
                            target_yaw = 0.0
                            target_pitch = -10.0
                            track_lost_counter = 0
                            
                elif pose_active and control_mode == "mirroring":
                    if head_estimator is None:
                        print("Loading head pose estimator...")
                        head_estimator = HeadPoseEstimator(SIXD_MODEL, device)
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
                        print("Commander lost, stop mirroring and reset.")
                        pose_active = False
                        commander_track_id = None
                        target_yaw = 0.0
                        target_pitch = -10.0
                        track_lost_counter = 0
                else:
                    target_yaw = 0.0
                    target_pitch = -10.0

                antenna_smooth = 0.45 if heart_wave_active else 0.2

                smooth_yaw = 0.8 * smooth_yaw + 0.2 * target_yaw
                smooth_pitch = 0.8 * smooth_pitch + 0.2 * target_pitch
                smooth_antenna_left = (
                    (1.0 - antenna_smooth) * smooth_antenna_left
                    + antenna_smooth * antenna_left_target
                )
                smooth_antenna_right = (
                    (1.0 - antenna_smooth) * smooth_antenna_right
                    + antenna_smooth * antenna_right_target
                )
                head = create_head_pose(yaw=smooth_yaw, pitch=smooth_pitch, degrees=True)

                mini.set_target(head=head,
                                antennas=np.deg2rad([smooth_antenna_left, smooth_antenna_right]))

                for det_idx, (person_kpts, person_scores) in enumerate(
                        zip(keypoints, scores)):
                    track_id = (
                        track_ids[det_idx]
                        if det_idx < len(track_ids) else det_idx
                    )
                    draw_skeleton(
                        frame,
                        person_kpts,
                        scores=person_scores,
                        kpt_thr=SCORE_THRESHOLD,
                        include_face=True,
                    )
                    person_bbox = get_person_bbox(
                        person_kpts, person_scores, kpt_thr=SCORE_THRESHOLD)
                    if person_bbox is not None:
                        x1, y1, _, _ = person_bbox
                        draw_label(
                            frame,
                            f"ID {track_id}",
                            (x1, y1 - 10),
                            color=(255, 255, 0),
                        )

                show_face_bbox = (
                    pose_active
                    and not heart_wave_active
                    and control_mode in ("tracking", "mirroring")
                )

                if largest_person_bbox is not None:
                    draw_bbox(
                        frame,
                        largest_person_bbox,
                        color=(0, 255, 255),
                        thickness=3,
                    )

                if commander_person_bbox is not None:
                    draw_bbox(
                        frame,
                        commander_person_bbox,
                        color=(255, 128, 0),
                        thickness=2,
                    )

                if show_face_bbox and face_bbox is not None:
                    draw_bbox(frame, face_bbox, color=(255, 100, 100))

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
                commander_label = (
                    f"ID {commander_track_id}" if commander_track_id is not None else "none"
                )
                status = (f"cmd:{gesture} | sound:{'ON' if sound_on else 'off'} "
                          f"| antennas:{'open' if antennas_open else 'closed'} "
                          f"| {control_mode}:{pose_status} ({track_lost_counter} lost) "
                          f"| commander:{commander_label} "
                          f"| hand IoU:{commander_hand_overlap:.2f} "
                          f"| heart:{'YES' if heart_wave_active else 'no'}")
                draw_label(frame, status, (10, 30), color=(255, 255, 0))

                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            sound_player.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
