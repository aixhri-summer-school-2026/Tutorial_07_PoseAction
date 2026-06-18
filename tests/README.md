# Tests for Reachy Mini Tutorial 02

This folder contains manual validation scripts for camera, microphone, speaker, motors, and recorded move demos.

## Where to run

Run tests from inside the container shell:

```bash
make run
make shell
```

From the container, execute commands from `/app`.

## Available tests

- `python tests/test_camera.py`
  - Opens a live camera window.
  - Press `q` in the window to quit.
  - Saves a snapshot to `tests/test_output/camera_test.jpg`.

- `python tests/test_mic.py`
  - Records microphone input for about 3 seconds.
  - Saves output to `tests/test_output/mic_test.wav`.

- `python tests/test_speaker.py`
  - Plays a 440 Hz tone for about 3 seconds.
  - Pass condition is hearing the tone.

- `python tests/test_motions.py`
  - Runs a short head + antenna sine-wave motion demo.
  - Pass condition is smooth visible movement.

- `python tests/test_demos.py --library dance`
- `python tests/test_demos.py --library emotions`
  - Plays recorded motion libraries continuously.
  - Stop with `Ctrl+C`.

You can also provide a custom dataset:

```bash
python tests/test_demos.py --dataset <local_path_or_hf_dataset_id>
```

## Outputs

Generated test artifacts are written to `tests/test_output/`.

## Notes

- Camera test needs host display forwarding (`DISPLAY`) and local X access.
- Audio tests need PulseAudio/PipeWire access from the container.
- If hardware is not detected, re-run `make install-rules` on the host and reconnect the robot.
