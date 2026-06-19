# Tutorial 07 - Pose & Action for HRI

Hands-on tutorial: detect hands, recognize gestures, and make the Reachy Mini
robot react. The robot is used both as the **camera** and as the **body**.

All commands below are run from **inside the container**:

```bash
make run     # start the daemon container (once)
make shell   # open a shell inside it (python = the right environment)
```

Live scripts open an OpenCV window, so you need the display forwarded (same
setup as the scripts in `tests/`).

## One-time setup (before the session)

```bash
# Extra package for Part 4 only (face detector). Already-present packages
# (ultralytics, torch, opencv, numpy) are not reinstalled.
uv pip install -r tutorial/requirements_extra.txt

# Build the Part 2 gesture dataset from the HaGRID annotations (no downloads,
# uses the keypoints already stored in HaGRIDv2_annotations/).
python tutorial/prepare_hagrid_dataset.py
```

The hand keypoint models (MediaPipe), the SixDRepNet head-pose weights, and an
optional YOLO detector are already in `tutorial/weights/` and
`tutorial/SixDRepNet/weights/`, so there is nothing else to download.

## Hand keypoint estimator (used by Parts 1-4)

For the live demos we use the **MediaPipe** hand pipeline (palm detector +
21-keypoint hand model) from the OpenCV Zoo, wrapped in `hand_detector.py`. It
is robust and its 21 keypoints follow the same convention as the HaGRID data
the classifier is trained on. It runs on OpenCV's DNN module, so there are no
extra dependencies. The model files live in `weights/` and the wrapper classes
(from [opencv_zoo](https://github.com/opencv/opencv_zoo), Apache-2.0) in
`mediapipe/`.

## Part 1 - Hand detection + keypoints

```bash
# Run the hand detector live on the robot camera
python tutorial/visualize_hands_live.py
```

Optional bonus (training your own YOLO detector, the original Part 1 approach):

```bash
python tutorial/visualize_handkeypoints_dataset.py   # look at the dataset
python tutorial/train_hand_detector.py --epochs 20   # slow; not used by the demos
```

## Part 2 - Gesture classification

```bash
# Train the classifier on hand keypoints (try mlp or gcn)
python tutorial/train_hand_pose_classification.py --model mlp

# Live: detector -> classifier -> label on screen
python tutorial/visualize_pose_classification_live.py
```

## Part 3 - Robot behaviours from gestures

```bash
python tutorial/visualize_interact_live.py
```

| Gesture (right hand) | Behaviour            |
| -------------------- | -------------------- |
| point                | start random sounds  |
| mute                 | stop sounds          |
| rock                 | open antennas        |
| fist                 | close antennas       |
| peace                | start head tracking  |
| stop                 | stop head tracking   |
| heart (BOTH hands)   | wave the head        |

## Part 4 - Mirror the person's head

```bash
python tutorial/visualize_interact_live_v2.py
```

Same as Part 3, but `peace` / `stop` now start/stop **mirroring the person's
head pose** (estimated with SixDRepNet) instead of following a hand.

## Files

| File                                   | What it is                                   |
| -------------------------------------- | -------------------------------------------- |
| `gesture_utils.py`                     | shared helpers (labels, models, drawing)     |
| `hand_detector.py`                     | MediaPipe hand keypoint estimator (Parts 1-4)|
| `prepare_hagrid_dataset.py`            | prep: build the keypoint dataset             |
| `visualize_hands_live.py`              | Part 1: live hands                           |
| `visualize_handkeypoints_dataset.py`   | Part 1 bonus: look at the YOLO dataset       |
| `train_hand_detector.py`               | Part 1 bonus: train a YOLO detector          |
| `train_hand_pose_classification.py`    | Part 2: train the classifier                 |
| `visualize_pose_classification_live.py`| Part 2: live gestures                        |
| `visualize_interact_live.py`           | Part 3: behaviours                           |
| `visualize_interact_live_v2.py`        | Part 4: mirror head                          |

## Notes / things to tune on the real robot

- "Right hand" is taken as the **right-most hand in the image**. The MediaPipe
  detector also reports a `handedness` (Left/Right) you could use instead, but
  note it assumes a mirrored/selfie view.
- The **sign** of head yaw/pitch/roll for tracking & mirroring, and the
  **antenna open/closed** angles, are reasonable guesses — flip/tune them in
  the scripts if the robot moves the wrong way.
- Gestures are **smoothed over several frames** to avoid reacting to noise.
