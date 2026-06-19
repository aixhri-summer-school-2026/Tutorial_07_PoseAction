# Tutorial_07_PoseAction
Tutorial on person tracking, hand pose estimation, head pose estimation and action recognition for HRI
Updated version (v1).

# Part 1 - Use a hand detector and keypoints estimator 
## To prepare before

## Tutorial content
- Use the palm pose detector and hand pose estimator from Mediapipe
- Visualize detected hand

## Alternatives / TODOs
- Find a better / more modern approach, do not retrain, too long

# Part 2 - Action recognition
## To prepare before
- Use HaGRIDv2 dataset from https://github.com/hukenovs/hagrid - Prepare subsample :
    - Already downloaded in HaGRIDv2_annotations (annotations only with boxes hand keypoints).
    - Use at least heart, mute, peace, no_gesture, rock, point, stop, fist + add actions to leave students to work later
    - Use only 500 samples (training) + 50 samples (val) + 50 samples (test) per action for faster training

##  Tutorial content
- Visualize dataset of hand poses + actions associated `visualize_hagrid_dataset.py`
- Visualize augmentations (jittering, flipping) `visualize_data_augmentation.py`
- Train either a plain MLP or plain GCN for action classification (one hand). Simple PyTorch implementation, with also the dataloader, basic augmentation (jittering, flipping). `train_hand_pose_classification.py --...`
- Use the network trained on Part 1 for processing, feed into the classifier, and test it live with the robot feed on a visualizer `visualize_pose_classification_live.py`

# Part 3 - Head detection and pose estimation
## To prepare before

## Tutorial content
- Run the demo of 6DRepNet on a webcam
- Integrate it with the robot's camera `visualize_head_pose_estimation.py`

# Part 4 - Robot behaviors from action 
## To prepare before

##  Tutorial content
- Use the pipeline of Part 2 and the Reachy API create_head_pose, set_target, push_audio_sample, to implement behaviors (based on right hand action detection except for heart only if both hands are classified "heart")  `visualize_interact_live.py` (rename from v2)
    - sound (play random sounds) : start on "point", stop on "mute"
    - antennas (open or close antenas) : open on "rock", close on "fist"
    - track the person (move the head to track the right left hand position) : start on "peace", stop on "stop" or mirror the head
