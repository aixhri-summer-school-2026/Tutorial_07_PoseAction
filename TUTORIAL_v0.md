# Tutorial_07_PoseAction
Tutorial on person tracking, hand pose estimation, head pose estimation and action recognition for HRI

# Part 1 - Implement a hand detector and keypoints estimator 
## To prepare before

## Tutorial content
- Use from https://docs.ultralytics.com/datasets/pose/hand-keypoints#usage
    - Quickly load and visualize dataset `visualize_handkeypoints_dataset.py`
    - Launch training with ultralytics `train_hand_detector.py`

- Test it live with the robot feed on a visualizer `visualize_hands_live.py`

# Part 2 - Action recognition dataset
## To prepare before
- Use HaGRIDv2 dataset from https://github.com/hukenovs/hagrid
    - Already downloaded in HaGRIDv2_annotations (annotations only with boxes hand keypoints). TODO : prepare a subsampled dataset.
    - Use only actions : heart, mute, peace, no_gesture, rock, point, stop, fist
    - Use only 100 samples (training) + 50 samples (val) + 50 samples (test) per action for faster training

##  Tutorial content
- Train either a plain MLP or plain GCN for action classification (one hand). Simple PyTorch implementation, with also the dataloader, basic augmentation (flipping). `train_hand_pose_classification.py --...`
- Use the network trained on Part 1 for processing, feed into the classifier, and test it live with the robot feed on a visualizer `visualize_pose_classification_live.py`


# Part 3 - Robot behaviors from action 
## To prepare before

##  Tutorial content
- Use the pipeline of Part 2 and the Reachy API create_head_pose, set_target, push_audio_sample, to implement behaviors (based on right hand action detection except for heart only if both hands are classified "heart")  `visualize_interact_live.py`
    - sound (play random sounds) : start on "point", stop on "mute"
    - antennas (open or close antenas) : open on "rock", close on "fist"
    - track the person (move the head to track the right left hand position) : start on "peace", stop on "stop"
    - wave the head : one time then continuously while seing the "heart" from both hands


# Part 4 - New tracking behavior : mirror head
## To prepare before

##  Tutorial content
- Use the implementation in `SixDRepNet` of face detection + head pose estimation like in `demo.py`
- Update the "track the person" behavior to mirror the head pose instead of following a hand `visualize_interact_live_v2.py`