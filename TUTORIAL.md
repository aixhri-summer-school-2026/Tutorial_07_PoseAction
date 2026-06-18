# Tutorial_07_PoseAction
Tutorial on person tracking, pose estimation and action recognition for HRI


# Part 1 - Implement a pose estimator 
- simple ready-to-use SOTA detection + tracking + pose estimator like YOLO26 implementation on Ultralytics with coco 17 keypoints format
- *Optional* : Add another dataset with more keypoints + conversion to ultralytics format + `.train` with ultralytics (maybe too long and not very useful)
- Test on webcam with live visualization
- *Optional* : Add a 2D to 3D lifter

**Time** : ~ 30 minutes

# Part 2 - Action recognition dataset
- Intro + visualization + preprocessing action recognition data (take a simple dataset like [UTKinect-Action3D Dataset](https://cvrc.ece.utexas.edu/KinectDatasets/HOJ3D.html))
- Include 2D reprojection into the preprocessing (*optional* do it before and provide reprojected data)

**Time** : ~15 minutes

# Part 3
## Option A - Train an action recognition
- Different possible architectures, that can be trained locally (ST-GCN, 2s-AGCN, MotionBERT head...) or design your own
- Implement the dataloader and training loop
- Train and save the checkpoint model
- Visualize results on skeleton sequences + videos

## Option B - Hand-coded action recognition
- Only on simpler actions, appraoching, leaving etc...
- Visualize results on skeleton sequences + videos

**Time** : ~ 30 minutes

# Part 4 - Pipeline implementation with rosbag or webcam
- Have a rosbag of someone perfoming actions (*optional* everyone uses their own webcam)
- Implement the full pipeline of detection, tracking, pose estimation and action recognition
- *Optional* : If too long replace this by a implementing the pipeline on the dataset video

**Time** : ~ 30 minutes

# Part 5 - Robot behavior
- Add topic sending for actions on Shelfy (eye expression, sound playing, lights)
- Connect to shelfy and use the RGBD camera (or keep the same camera) and play the effects

**Time** : ~ 15 minutes (optional)
