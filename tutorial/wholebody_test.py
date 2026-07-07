from rtmlib import Body, Wholebody, Custom, PoseTracker, draw_skeleton
import cv2
import onnxruntime as ort
import numpy as np

cap = cv2.VideoCapture(0)  # for video file instead of webcam, use cap = cv2.VideoCapture('./demo.mp4')

device = 'cuda' if 'CUDAExecutionProvider' in ort.get_available_providers() else 'cpu'
print(f'Using device: {device}')
backend = 'onnxruntime'
openpose_skeleton = False


# default of the balanced mode
# solution_kwargs = {
#     'det': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',  # noqa
#     'det_input_size': (640, 640),
#     'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip',  # noqa
#     'pose_input_size': (192, 256),
# }

solution_kwargs = {
    'det': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',  # noqa
    'det_input_size': (640, 640),
    # 'pose': 'https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-s-wholebody.onnx',  # noqa
    # 'pose_input_size': (192, 256),
    # 'pose': 'https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/vitpose-s-coco.onnx',  # noqa
    # 'pose_input_size': (192, 256),
    'pose': "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip",
    'pose_input_size': (288, 384),
}



pose_tracker = PoseTracker(Wholebody,
                        # mode='balanced',
                        det_frequency=1,  # detect every 10 frames
                        backend=backend, 
                        device=device,
                        to_openpose=False,
                        solution_kwargs=solution_kwargs)

# # Or with a custom class
# from functools import partial
# custom = partial(Custom,
#                 det_class='YOLOX',
#                 det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
#                 det_input_size=(640, 640),
#                 pose_class='RTMPose',
#                 pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip',
#                 pose_input_size=(192, 256))
# pose_tracker = PoseTracker(custom,
#                         det_frequency=10,
#                         backend=backend, device=device,
#                         to_openpose=False)


left_wrist_id = 9
right_wrist_id = 10

left_hand_ids = np.arange(91, 111+1)
right_hand_ids = np.arange(112, 132+1)

threshold = 0.5

frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1
    if not success:
        break

    keypoints, scores = pose_tracker(frame)
    # print(keypoints.shape)
    
    # necessary filtering, if a wrist is not detected, the hand keypoints are not detected either
    for det in range(keypoints.shape[0]):
        if scores[det, left_wrist_id] < threshold:
            scores[det, left_hand_ids] = 0
        if scores[det, right_wrist_id] < threshold:
            scores[det, right_hand_ids] = 0
            
    print(f"Got {keypoints.shape[0]} detections with track ids : {pose_tracker.track_ids_last_frame}")

    img_show = frame.copy()
    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=threshold,
                             show_keypoints=True,
                             label_keypoint="conf")
    
    cv2.imshow('img', img_show)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break