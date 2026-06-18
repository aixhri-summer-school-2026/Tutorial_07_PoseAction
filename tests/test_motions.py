import time
import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

def run_test():
    print("Testing Motor Kinematics (Minimal Demo)...")

    try:
        # media_backend="no_media" is perfect here since we are only testing motors
        with ReachyMini(media_backend="no_media") as mini:
            test_duration = 5.0  # Run the demo for 5 seconds

            print("Moving to neutral starting position...")
            # duration=1.0 ensures a smooth, 1-second transit to the center
            mini.goto_target(create_head_pose(), antennas=[0.0, 0.0], duration=1.0)

            print(f"Executing sine wave motion for {test_duration} seconds...")
            start_time = time.time()

            while time.time() - start_time < test_duration:
                t = time.time()

                # Calculate smooth sine wave offsets
                antennas_offset = np.deg2rad(20 * np.sin(2 * np.pi * 0.5 * t))
                pitch = np.deg2rad(10 * np.sin(2 * np.pi * 0.5 * t))

                head_pose = create_head_pose(
                    roll=0.0,
                    pitch=pitch,
                    yaw=0.0,
                    degrees=False,
                    mm=False,
                )

                # Stream the new coordinates to the daemon
                mini.set_target(head=head_pose, antennas=[antennas_offset, antennas_offset])

                # Sleep briefly to prevent maxing out the container's CPU loop
                time.sleep(0.01)

            print("[PASS] Motion test completed successfully. Did the head and antennas wobble smoothly?")

    except Exception as e:
        print(f"[FAIL] Motion test error: {e}")

if __name__ == "__main__":
    run_test()