# Tutorial 07 - Pose & Action for HRI

Hands-on tutorial: detect a person with a whole-body pose estimator, recognize
hand gestures, and make the Reachy Mini robot react. 

For Part 1 and Part 2 you may use either the robot's camera or your own with the option `--use_input webcam`

```bash
make run     # start the daemon container (once)
make shell   # open a shell inside it (python = the right environment)
```

All commands below are run from **inside the container**:

## Whole-body pose estimator (used by Parts 1-4)

The live demos use the **RTMLib** whole-body pipeline (`tutorial/rtmlib/`).
It detects a person first, then predicts a pose with **133 keypoints** (body,
face, and both hands).

From that single prediction we extract:
- **hands** (21 keypoints per hand) for gesture classification,
- **face** (bounding box from face landmarks) for head tracking or mirroring.

Recent docker images ship the whole-body ONNX models in `/app/downloads/`.
If you use an older image, see **Troubleshooting** below.

Shared helpers live in:
- `keypoints_utils.py` — drawing, hand/face bounding boxes
- `handkeypoints_dataset.py` / `handkeypoints_models.py` / `handkeypoints_infer.py` — Part 2 classifier

## Part 1 - Whole-body detection + keypoints

```bash
# Run the whole-body detector live on the robot camera
python tutorial/visualize_wholebody_live.py

# Run the whole-body detector live on the USB camera / webcam
python tutorial/visualize_wholebody_live.py --use_input webcam

# Optional: try mediapipe-style or coco133-style hand drawing
python tutorial/visualize_wholebody_live.py --hands_style mediapipe
```

## Part 2 - Gesture classification

The classifier is trained on **MediaPipe-style 21-point hands** from HaGRID.
At runtime, hand keypoints are sliced from the whole-body prediction.

```bash
# Visualize the training dataset
python tutorial/visualize_hagrid_data.py

# Train the classifier on hand keypoints (try mlp or gcn)
python tutorial/handkeypoints_train.py --model mlp

# Live: whole-body pose -> hand keypoints -> classifier -> label on screen with the robot camera
python tutorial/visualize_pose_classification_live.py

# Alternatively with the USB camera / webcam
python tutorial/visualize_pose_classification_live.py --use_input webcam
```

## Part 3 - Run SixDRepNet demo

Standalone head-pose demo (not used directly by the live whole-body pipeline):

```bash
python tutorial/SixDRepNet/main.py --demo
```

## Part 4 - Gesture-driven robot behaviours

Uses the whole-body estimator for hands and face. Choose a head-control mode
at launch:

- **tracking** — `peace` / `stop` start/stop following the face center
- **mirroring** — `peace` / `stop` start/stop mirroring head yaw (SixDRepNet)

```bash
python tutorial/visualize_interact_live_v2.py --mode tracking
python tutorial/visualize_interact_live_v2.py --mode mirroring
```

Gesture → behaviour:
- `point` → (reserved) / `mute` → stop sounds
- `rock` → open antennas / `fist` → close antennas
- `peace` → start head control / `stop` → stop head control
- `hand_heart` (both hands) → wave the head

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

Then the processed data will be pushed on the github in `tutorial/data/`.
The trained gesture classifier can be saved to `tutorial/weights/gesture_classifier.pt`.
For mirroring mode, `tutorial/SixDRepNet/weights/best.pt` is also required.

## Troubleshooting

- If you pulled or built the docker **before 08/07/2026**, the whole-body ONNX models are not baked into the image. Either rebuild (`make build`) or do the following (faster):

```bash
# 1. On the host (outside the docker): download and extract the models into tutorial/
mkdir -p tutorial
cd tutorial
wget https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip
wget https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip
unzip yolox_m_8xb8-300e_humanart-c2c7a14a.zip -d yolox_det
unzip rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip -d rtmw_pose
mv "$(find yolox_det -name end2end.onnx)" yolox_m_8xb8-300e_humanart-c2c7a14a.onnx
mv "$(find rtmw_pose -name end2end.onnx)" rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.onnx
rm -rf *.zip yolox_det rtmw_pose
cd ..

# 2. Inside the docker: move them to /app/downloads/ (where the live scripts look)
make shell
mkdir -p /app/downloads
mv /app/tutorial/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx /app/downloads/
mv /app/tutorial/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.onnx /app/downloads/

# 3. Install onnxruntime-gpu and fix LD_LIBRARY_PATH
source tutorial/update.sh
```

Note: `/app/downloads/` is not bind-mounted, so if you recreate the container you must repeat step 2 (the `.onnx` files stay on the host in `tutorial/`).


- If you have issues with "permission denied" for video devices (integrated or usb webcam). Outside of the docker, do the following :
```
ls -la /dev/video*
sudo chmod 666 /dev/video2   # replace with your desired cam device
```
