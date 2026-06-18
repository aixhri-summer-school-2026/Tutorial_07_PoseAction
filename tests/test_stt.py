import time
from faster_whisper import WhisperModel

def run_stt_test():
    print("Loading Faster-Whisper model...")
    # 'tiny.en' is extremely fast. 'base.en' is slightly slower but more accurate.
    model = WhisperModel("tiny.en", device="cuda", compute_type="float16")

    audio_file = "tests/test_output/captured_speech.wav"
    print(f"\nTranscribing: {audio_file}")

    start_time = time.time()
    segments, info = model.transcribe(audio_file, beam_size=5)

    print("\n--- Transcription ---")
    for segment in segments:
        print(segment.text.strip())

    print(f"---------------------\n(Completed in {time.time() - start_time:.2f} seconds)")

if __name__ == "__main__":
    run_stt_test()