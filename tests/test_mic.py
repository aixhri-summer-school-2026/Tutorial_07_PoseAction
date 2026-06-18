import os
import time
import numpy as np
import torch
import scipy.signal
from reachy_mini import ReachyMini
from reachy_mini.media.audio_utils import save_audio_to_wav
from silero_vad import load_silero_vad

def run_voice_capture():
    print("Loading Silero VAD model...")
    vad_model = load_silero_vad()
    target_sr = 16000
    vad_window_samples = 512

    # Configurations for silence tolerance
    SILENCE_TOLERANCE_SECONDS = 1.0  # Time to wait before deciding the person finished speaking

    print("\nConnecting to Reachy...")
    try:
        with ReachyMini() as mini:
            mini.media.start_recording()
            print("Waiting for microphone stream...")

            # Wait for the stream to initialize
            start_time = time.time()
            while mini.media.get_audio_sample() is None:
                if time.time() - start_time > 2.0:
                    print("[FAIL] Microphone timeout. No data received.")
                    mini.media.stop_recording()
                    return
                time.sleep(0.01)

            samplerate = mini.media.get_input_audio_samplerate()
            print(f"Stream active at {samplerate} Hz.")
            print("\n🎤 State: WAITING - Start speaking whenever you are ready...")

            # State tracking variables
            is_speaking = False
            silence_start_time = None

            audio_samples_stream = []  # For processing chunk-by-chunk VAD
            recorded_speech_chunks = []  # To accumulate the final voice message

            while True:
                sample = mini.media.get_audio_sample()
                if sample is not None:
                    audio_samples_stream.append(sample)

                    # Accumulate for VAD window processing
                    current_buffer = np.concatenate(audio_samples_stream, axis=0)
                    samples_needed_raw = int(samplerate * (vad_window_samples / target_sr))

                    if len(current_buffer) >= samples_needed_raw:
                        chunk_to_process = current_buffer[:samples_needed_raw]
                        leftover = current_buffer[samples_needed_raw:]
                        audio_samples_stream = [leftover] if len(leftover) > 0 else []

                        if chunk_to_process.ndim > 1:
                            # Isolate just the first channel (true mono) instead of stretching it
                            chunk_to_process = chunk_to_process[:, 0]

                        # Resample to 16kHz for VAD
                        if samplerate != target_sr:
                            num_target_samples = int(len(chunk_to_process) * target_sr / samplerate)
                            chunk_16k = scipy.signal.resample(chunk_to_process, num_target_samples).astype(np.float32)
                        else:
                            chunk_16k = chunk_to_process

                        vad_input = chunk_16k[:vad_window_samples]

                        if len(vad_input) == vad_window_samples:
                            tensor = torch.from_numpy(vad_input).unsqueeze(0)
                            speech_prob = vad_model(tensor, target_sr).item()

                            if speech_prob > 0.4:  # Slightly lower threshold for stability while speaking
                                if not is_speaking:
                                    print("🗣️ State: SPEAKING - Recording voice...")
                                    is_speaking = True

                                # Reset silence timer because user is actively talking
                                silence_start_time = None
                            else:
                                # User is quiet. If they were speaking, check the tolerance window
                                if is_speaking:
                                    if silence_start_time == None:
                                        silence_start_time = time.time()

                                    elapsed_silence = time.time() - silence_start_time
                                    if elapsed_silence >= SILENCE_TOLERANCE_SECONDS:
                                        print("⏱️ State: SILENCE DETECTED - Finished phrase.")
                                        break  # Break loop to save file and exit

                            # If we are in the speaking state, save the raw sample block
                            if is_speaking:
                                recorded_speech_chunks.append(chunk_to_process)

                time.sleep(0.01)

            # --- POST-PROCESSING & SAVING ---
            mini.media.stop_recording()

            if recorded_speech_chunks:
                audio_data = np.concatenate(recorded_speech_chunks, axis=0)

                os.makedirs("tests/test_output", exist_ok=True)
                output_file = "tests/test_output/captured_speech.wav"

                # Note: We save using 'samplerate' because recorded_speech_chunks
                # stores the original raw chunks before 16kHz conversion
                save_audio_to_wav(audio_data, samplerate, output_file)
                print(f"[PASS] Speech captured and saved successfully to {output_file}.")
            else:
                print("[WARN] Recording loop ended but no valid voice data was stored.")

    except KeyboardInterrupt:
        print("\nStopping VAD capture.")
    except Exception as e:
        print(f"[FAIL] Microphone pipeline error: {e}")
    finally:
        try:
            mini.media.stop_recording()
        except:
            pass
        print("Microphone turned off. Script terminated.")

if __name__ == "__main__":
    run_voice_capture()