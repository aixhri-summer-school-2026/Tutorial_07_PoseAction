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

A pretrained hand detector (`weights/yolo26s-pose-hands.pt`) and the
SixDRepNet head-pose weights are already in `tutorial/weights/` and
`tutorial/SixDRepNet/weights/`, so you can skip the long detector training.

## Part 1 - Hand detection + keypoints

```bash
# Look at the dataset (downloads it the first time)
python tutorial/visualize_handkeypoints_dataset.py

# OPTIONAL: train your own detector (slow). A pretrained one is provided.
python tutorial/train_hand_detector.py --epochs 20

# Run the detector live on the robot camera
python tutorial/visualize_hands_live.py
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
| `prepare_hagrid_dataset.py`            | prep: build the keypoint dataset             |
| `visualize_handkeypoints_dataset.py`   | Part 1: look at the dataset                  |
| `train_hand_detector.py`               | Part 1: train the detector                   |
| `visualize_hands_live.py`              | Part 1: live hands                           |
| `train_hand_pose_classification.py`    | Part 2: train the classifier                 |
| `visualize_pose_classification_live.py`| Part 2: live gestures                        |
| `visualize_interact_live.py`           | Part 3: behaviours                           |
| `visualize_interact_live_v2.py`        | Part 4: mirror head                          |

## Notes / things to tune on the real robot

- "Right hand" is taken as the **right-most hand in the image** (the detector
  does not output handedness).
- The **sign** of head yaw/pitch/roll for tracking & mirroring, and the
  **antenna open/closed** angles, are reasonable guesses — flip/tune them in
  the scripts if the robot moves the wrong way.
- Gestures are **smoothed over several frames** to avoid reacting to noise.
