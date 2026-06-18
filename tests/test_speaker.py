import time
import numpy as np
from reachy_mini import ReachyMini


def run_test():
    print("Testing Speaker Pipeline...")

    try:
        # Using default backend to test the daemon's WebRTC audio routing
        with ReachyMini() as mini:
            tone_hz = 440.0  # Standard A4 pitch
            test_duration = 3.0  # Play for 3 seconds

            sample_rate = mini.media.get_output_audio_samplerate()
            chunk_duration = 0.02  # 20 ms chunks
            samples_per_chunk = int(sample_rate * chunk_duration)
            phase = 0.0

            print(f"Playing {tone_hz} Hz tone for {test_duration} seconds...")
            mini.media.start_playing()

            start_time = time.time()
            while time.time() - start_time < test_duration:
                # Generate a raw sine wave chunk
                t = np.arange(samples_per_chunk, dtype=np.float32) / sample_rate
                mono = 0.5 * np.sin(2.0 * np.pi * tone_hz * t + phase).astype(
                    np.float32
                )
                phase += 2.0 * np.pi * tone_hz * samples_per_chunk / sample_rate

                # Push to the WebRTC audio pipeline
                mini.media.push_audio_sample(mono)
                time.sleep(0.01)

            mini.media.stop_playing()
            print(
                "[PASS] Audio pipeline successfully processed the stream. Did you hear the tone?"
            )

    except Exception as e:
        print(f"[FAIL] Speaker pipeline error: {e}")


if __name__ == "__main__":
    run_test()
