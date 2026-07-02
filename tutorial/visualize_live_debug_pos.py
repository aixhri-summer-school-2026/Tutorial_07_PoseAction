"""Debug head pose: keyboard target vs. actual robot pose.

Step the commanded head pose with the arrow keys and compare it to the pose
reported by ``get_current_head_pose``.

Controls:
    left / right : decrease / increase target yaw
    up / down    : increase / decrease target pitch
    r            : reset target to neutral
    q            : quit

Run (inside the container shell, display forwarded):
    python tutorial/visualize_live_debug_pos.py
"""

import argparse
import time

import cv2
import numpy as np
import scipy.spatial.transform
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

from gesture_utils import draw_label

# OpenCV arrow-key codes differ by platform; waitKeyEx preserves the full code.
ARROW_LEFT = {65361, 2424832}
ARROW_RIGHT = {65363, 2555904}
ARROW_UP = {65362, 2490368}
ARROW_DOWN = {65364, 2621440}

DEFAULT_YAW = 30.0
DEFAULT_PITCH = -10.0


def clamp(value, limit):
    return max(-limit, min(limit, value))


def euler_from_pose_matrix(pose_matrix):
    """Return (yaw, pitch, roll) in degrees from a 4x4 head pose matrix."""
    rotation = scipy.spatial.transform.Rotation.from_matrix(pose_matrix[:3, :3])
    yaw, pitch, roll = rotation.as_euler("zyx", degrees=True)
    return float(yaw), float(pitch), float(roll)


def handle_key(key, target_yaw, target_pitch, step_deg, max_yaw, max_pitch):
    """Apply one key press; return (continue, target_yaw, target_pitch)."""
    if key == ord("q"):
        return False, target_yaw, target_pitch
    if key == ord("r"):
        return True, DEFAULT_YAW, DEFAULT_PITCH
    if key in ARROW_LEFT:
        target_yaw -= step_deg
    elif key in ARROW_RIGHT:
        target_yaw += step_deg
    elif key in ARROW_UP:
        target_pitch += step_deg
    elif key in ARROW_DOWN:
        target_pitch -= step_deg

    # target_yaw = clamp(target_yaw, max_yaw)
    # target_pitch = clamp(target_pitch, max_pitch)
    return True, target_yaw, target_pitch


def main():
    parser = argparse.ArgumentParser(description="Debug head pose with arrow keys.")
    parser.add_argument("--step", type=float, default=2.0,
                        help="Yaw/pitch change per arrow key press (degrees).")
    parser.add_argument("--max-yaw", type=float, default=40.0,
                        help="Maximum absolute target yaw (degrees).")
    parser.add_argument("--max-pitch", type=float, default=30.0,
                        help="Maximum absolute target pitch (degrees).")
    args = parser.parse_args()

    window = "Head pose debug"
    cv2.namedWindow(window)

    print("Connecting to Reachy Mini...")
    print("Arrow keys: yaw/pitch target | r: reset | q: quit")
    with ReachyMini() as mini:
        target_yaw = DEFAULT_YAW
        target_pitch = DEFAULT_PITCH

        mini.set_automatic_body_yaw(True)

        head_pose = np.eye(4)
        head_pose[:3, :3] = scipy.spatial.transform.Rotation.from_euler('zyx', [target_yaw, target_pitch, 0.0], degrees=True).as_matrix()
        head_pose[3, 3] = 1.0
        mini.goto_target(head=head_pose, antennas=[0.0, 0.0], duration=3.0, method = "cartoon")

        # mini.goto_target(
        #     create_head_pose(yaw=target_yaw, pitch=target_pitch, degrees=True),
        #     antennas=[0.0, 0.0],
        #     duration=1.0,
        # )

        lat = time.time()
        try:
            while True:
                time_since_start = time.time() - lat
                
                print(f"latency: {time_since_start:.2f} seconds")
                
                lat = time.time()
                
                time.sleep(0.03)
                
                frame = mini.media.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                frame = frame.copy()

                # head = create_head_pose(
                #     yaw=target_yaw, pitch=target_pitch, degrees=True,
                # )
                head = np.eye(4)
                head[:3, :3] = scipy.spatial.transform.Rotation.from_euler('zyx', [target_yaw, target_pitch, 0.0], degrees=True).as_matrix()
                head[3, 3] = 1.0
                mini.set_target(head=head, antennas=[0.0, 0.0])
                # mini.goto_target(head=head_pose, antennas=[0.0, 0.0], duration=2.0, method = "cartoon")

                current_matrix = mini.get_current_head_pose()
                cur_yaw, cur_pitch, cur_roll = euler_from_pose_matrix(current_matrix)

                yaw_err = cur_yaw - target_yaw
                pitch_err = cur_pitch - target_pitch

                target_text = f"target  yaw:{target_yaw:+6.1f}  pitch:{target_pitch:+6.1f}"
                current_text = (
                    f"current yaw:{cur_yaw:+6.1f}  pitch:{cur_pitch:+6.1f}  "
                    f"roll:{cur_roll:+6.1f}"
                )
                error_text = f"error   yaw:{yaw_err:+6.1f}  pitch:{pitch_err:+6.1f}"

                draw_label(frame, target_text, (10, 30), color=(0, 255, 0))
                draw_label(frame, current_text, (10, 60), color=(255, 255, 0))
                draw_label(frame, error_text, (10, 90), color=(0, 200, 255))
                draw_label(
                    frame,
                    "arrows: yaw/pitch | r: reset | q: quit",
                    (10, frame.shape[0] - 20),
                    color=(200, 200, 200),
                )

                print(
                    f"{target_text} | {current_text} | {error_text}"
                )

                cv2.imshow(window, frame)
                key = cv2.waitKeyEx(1)
                if key != -1:
                    running, target_yaw, target_pitch = handle_key(
                        key, target_yaw, target_pitch,
                        args.step, args.max_yaw, args.max_pitch,
                    )
                    if not running:
                        break
        finally:
            print()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
