import time
import numpy as np
import scipy.signal
from reachy_mini import ReachyMini
from pocket_tts import TTSModel

def run_tts_stream(text: str, voice: str = "alba", wobble: bool = True):
    print("Loading Kyutai Pocket TTS model...")
    model = TTSModel.load_model(language="english")
    voice_state = model.get_state_for_audio_prompt(voice)
    model_sr = model.sample_rate

    print("Connecting to Reachy...")
    with ReachyMini() as mini:
        target_sr = mini.media.get_output_audio_samplerate()

        if wobble:
            print("Enabling audio-reactive wobbling...")
            mini.enable_wobbling()

        print(f"Streaming TTS: '{text}'")
        mini.media.start_playing()

        total_audio_duration = 0.0
        start_time = time.time()

        try:
            for chunk_tensor in model.generate_audio_stream(voice_state, text):
                audio_chunk = chunk_tensor.numpy().astype(np.float32)

                # 1. Fix the deep voice by resampling to Reachy's expected format
                if model_sr != target_sr:
                    num_samples = int(len(audio_chunk) * target_sr / model_sr)
                    audio_chunk = scipy.signal.resample(audio_chunk, num_samples).astype(np.float32)

                # Calculate the exact time this specific chunk takes to play
                chunk_duration = len(audio_chunk) / target_sr
                total_audio_duration += chunk_duration

                mini.media.push_audio_sample(audio_chunk)

                # 2. Pace the loop so we don't overflow the WebRTC buffer
                elapsed = time.time() - start_time
                ahead_by = total_audio_duration - elapsed
                if ahead_by > 0.1:  # If we are more than 100ms ahead of real-time
                    time.sleep(ahead_by - 0.05)

        except Exception as e:
            print(f"[FAIL] TTS Streaming error: {e}")
        finally:
            # 3. Wait for the exact remaining audio duration before cutting it off
            elapsed = time.time() - start_time
            remaining_time = total_audio_duration - elapsed
            if remaining_time > 0:
                time.sleep(remaining_time + 0.2)  # Add 200ms tail buffer to ensure it finishes

            mini.media.stop_playing()

            if wobble:
                mini.disable_wobbling()

            print("[PASS] Finished speaking.")

if __name__ == "__main__":
    test_text = "Hello! I am Reachy Mini, I am getting ready for the summer school!"
    run_tts_stream(test_text)