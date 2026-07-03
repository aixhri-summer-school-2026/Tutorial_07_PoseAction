# Tutorial 07 - Pose & Action for HRI

Hands-on tutorial: detect hands, recognize gestures, and make the Reachy Mini
robot react.

All commands below are run from **inside the container**:

```bash
make run     # start the daemon container (once)
make shell   # open a shell inside it (python = the right environment)
```

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

## Part 2 - Gesture classification

```bash
# Visualize the dataset
python tutorial/visualize_hagrid_data.py --num -1

# Train the classifier on hand keypoints (try mlp or gcn)
python tutorial/train_hand_pose_classification.py --model mlp

# Live: detector -> classifier -> label on screen
python tutorial/visualize_pose_classification_live.py
```

## Part 3 - Run SixDRepNet demo

```bash
python tutorial/SixDRepNet/main.py --demo
```

## Part 4 - Mirror the person's head

```bash
python tutorial/visualize_interact_live_v2.py
```

## OPTIONAL (for staff) : Download and prepare the data

Download the dataset (out of the docker)
```bash
wget https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip
# TODO : note down the alternative scp command from local network
unzip annotations.zip
mv annotations HaGRIDv2_annotations
```

Prepare keypoints (from the docker a compatible environment)
```bash
# Build the Part 2 gesture dataset from the HaGRID annotations (no downloads,
# uses the keypoints already stored in HaGRIDv2_annotations/).
python tutorial/prepare_hagrid_dataset.py
```

Then the processed data will be pushed on the github in `tutorial/data/`, 
the hand keypoint models (MediaPipe), the SixDRepNet head-pose weights, and an
optional YOLO detector are already in `tutorial/weights/` and
`tutorial/SixDRepNet/weights/`, so there is nothing else to download.