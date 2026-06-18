import sys
import os
import time
import cv2
from reachy_mini import ReachyMini


def run_test():
    print("Testing Camera...")
    cv2.namedWindow("Reachy Camera Stream")
    print("Live stream active. Press 'q' on the video window to quit.")

    with ReachyMini() as mini:
        frame = mini.media.get_frame()
        start_time = time.time()

        while frame is None:
            if time.time() - start_time > 10:
                print("[FAIL] Camera timeout. No frames received.")
                sys.exit(1)
            print("Waiting for camera frames...")
            time.sleep(0.5)
            frame = mini.media.get_frame()

        print(f"[PASS] Camera active. Resolution: {frame.shape}")

        os.makedirs("tests/test_output", exist_ok=True)
        cv2.imwrite("tests/test_output/camera_test.jpg", frame)
        print("Snapshot saved to tests/test_output/camera_test.jpg")

        try:
            while True:
                frame = mini.media.get_frame()
                if frame is None:
                    continue

                cv2.imshow("Reachy Camera Stream", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Exiting stream...")
                    break

        except KeyboardInterrupt:
            print("Interrupted. Closing viewer...")
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run_test()
