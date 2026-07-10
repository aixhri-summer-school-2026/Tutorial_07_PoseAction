'''
Example:

import cv2
from functools import partial
from rtmlib import PoseTracker, Wholebody, Custom, draw_skeleton

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

cap = cv2.VideoCapture('./demo.mp4')

pose_tracker = PoseTracker(Wholebody,
                        det_frequency=10,  # detect every 10 frames
                        to_openpose=openpose_skeleton,
                        backend=backend, device=device)


# # Initialized slightly differently for Custom solution:
# custom = partial(Custom,
#                 to_openpose=openpose_skeleton,
#                 pose_class='RTMO',
#                 pose='https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip', # noqa
#                 pose_input_size=(640,640),
#                 backend=backend,
#                 device=device)
# # or
# custom = partial(
#             Custom,
#             to_openpose=openpose_skeleton,
#             det_class='YOLOX',
#             det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip', # noqa
#             det_input_size=(640, 640),
#             pose_class='RTMPose',
#             pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip', # noqa
#             pose_input_size=(192, 256),
#             backend=backend,
#             device=device)
# # then
# pose_tracker = PoseTracker(custom,
#                         det_frequency=10,
#                         to_openpose=openpose_skeleton,
#                         backend=backend, device=device)


frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break

    keypoints, scores = pose_tracker(frame)

    img_show = frame.copy()

    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.43)

    img_show = cv2.resize(img_show, (960, 540))
    cv2.imshow('img', img_show)
    cv2.waitKey(10)
'''
import warnings

import numpy as np


def compute_iou(bboxA, bboxB):
    """Compute the Intersection over Union (IoU) between two boxes .

    Args:
        bboxA (list): The first bbox info (left, top, right, bottom, score).
        bboxB (list): The second bbox info (left, top, right, bottom, score).

    Returns:
        float: The IoU value.
    """

    x1 = max(bboxA[0], bboxB[0])
    y1 = max(bboxA[1], bboxB[1])
    x2 = min(bboxA[2], bboxB[2])
    y2 = min(bboxA[3], bboxB[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    bboxA_area = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
    bboxB_area = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])
    union_area = float(bboxA_area + bboxB_area - inter_area)
    if union_area == 0:
        union_area = 1e-5
        warnings.warn('union_area=0 is unexpected')

    iou = inter_area / union_area

    return iou


def pose_to_bbox(keypoints: np.ndarray, expansion: float = 1.25) -> np.ndarray:
    """Get bounding box from keypoints.

    Args:
        keypoints (np.ndarray): Keypoints of person.
        expansion (float): Expansion ratio of bounding box.

    Returns:
        np.ndarray: Bounding box of person.
    """
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    bbox = np.array([x.min(), y.min(), x.max(), y.max()])
    center = np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]]) / 2
    bbox = np.concatenate([
        center - (center - bbox[:2]) * expansion,
        center + (bbox[2:] - center) * expansion
    ])
    return bbox


def keypoints_to_bbox(keypoints: np.ndarray,
                      scores: np.ndarray,
                      keypoint_ids: list,
                      kpt_thr: float = 0.3,
                      expansion: float = 1.25) -> np.ndarray | None:
    """Get a bounding box from a subset of keypoints.

    Args:
        keypoints (np.ndarray): Keypoints of one person, shape (K, 2).
        scores (np.ndarray): Confidence scores, shape (K,).
        keypoint_ids (list): Indices of keypoints used to build the bbox.
        kpt_thr (float): Minimum score for a keypoint to be used.
        expansion (float): Expansion ratio applied around the bbox center.

    Returns:
        np.ndarray | None: xyxy bbox, or None if too few keypoints are visible.
    """
    ids = np.asarray(keypoint_ids, dtype=int)
    subset_kpts = np.asarray(keypoints)[ids]
    subset_scores = np.asarray(scores)[ids]
    visible_kpts = subset_kpts[subset_scores >= kpt_thr]
    if len(visible_kpts) < 2:
        return None
    return pose_to_bbox(visible_kpts, expansion=expansion)


def det_bbox_xyxy(bbox) -> np.ndarray:
    """Normalize a detector bbox to xyxy (first four values)."""
    return np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]


class PoseTracker:
    """Pose tracker for pose estimation.

    Args:
        solution (type): rtmlib solutions, e.g. Wholebody, Body, Custom, etc.
        det_frequency (int): Frequency of object detection.
        tracking (bool): Whether to enable IoU-based person tracking.
        tracking_thr (float): IoU threshold for track matching.
        tracking_keypoint_ids (list, optional): If set, derive tracking bboxes
            from these keypoint indices (e.g. face keypoints) instead of
            detector boxes. More stable when tracking faces across frames.
        tracking_kpt_thr (float): Min keypoint score when building tracking bbox.
        tracking_bbox_expansion (float): Expansion ratio for keypoint tracking bbox.
        mode (str): 'performance', 'lightweight', or 'balanced'.
        to_openpose (bool): Whether to use openpose-style skeleton.
        backend (str): Backend of pose estimation model.
        device (str): Device of pose estimation model.
        biggest_n_boxes_only (int): Keep only the N largest detection boxes.
        solution_kwargs (dict): Extra kwargs passed to the solution class.
    """
    MIN_AREA = 1000
    MIN_KEYPOINT_TRACK_AREA = 100

    def __init__(self,
                 solution: type,
                 det_frequency: int = 1,
                 tracking: bool = True,
                 tracking_thr: float = 0.3,
                 mode: str = 'balanced',
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu',
                 biggest_n_boxes_only: int = 0,
                 tracking_keypoint_ids: list = None,
                 tracking_kpt_thr: float = 0.3,
                 tracking_bbox_expansion: float = 1.25,
                 solution_kwargs: dict = {}):

        model = solution(mode=mode,
                         to_openpose=to_openpose,
                         backend=backend,
                         device=device,
                         **solution_kwargs)

        try:
            self.det_model = model.det_model
            self.det_mode = self.det_model.mode
            if hasattr(model, 'det_categories') and model.det_categories:
                self.det_mode = 'multiclass'
                self.det_categories = model.det_categories
            else:
                self.det_categories = None
        except Exception as e:  # noqa
            print(f'Warning: {e}, pose tracker will not use detection results')
            self.det_model = None

        self.pose_model = model.pose_model
        
        self.biggest_n_boxes_only = biggest_n_boxes_only
        self.tracking_keypoint_ids = (
            list(tracking_keypoint_ids) if tracking_keypoint_ids else None
        )
        self.tracking_kpt_thr = tracking_kpt_thr
        self.tracking_bbox_expansion = tracking_bbox_expansion
        self.tracking_min_area = (
            self.MIN_KEYPOINT_TRACK_AREA
            if self.tracking_keypoint_ids else self.MIN_AREA
        )
        
        self.det_frequency = det_frequency
        self.tracking = tracking
        self.tracking_thr = tracking_thr
        self.reset()

        if self.tracking:
            if self.tracking_keypoint_ids:
                print('Tracking is on (keypoint bbox mode, '
                      f'{len(self.tracking_keypoint_ids)} keypoints).')
            else:
                print('Tracking is on, you can get higher FPS by turning it off:'
                      '`PoseTracker(tracking=False)`')

    def _tracking_bbox_from_person(self, keypoints, scores):
        """Build a tracking bbox for one person."""
        if self.tracking_keypoint_ids is not None:
            return keypoints_to_bbox(
                keypoints,
                scores,
                self.tracking_keypoint_ids,
                kpt_thr=self.tracking_kpt_thr,
                expansion=self.tracking_bbox_expansion,
            )
        return pose_to_bbox(keypoints)

    def _iter_tracking_bboxes(self, keypoints, scores, det_bboxes=None):
        """Yield (det_idx, bbox) pairs used for IoU tracking."""
        if self.tracking_keypoint_ids is not None:
            for det_idx, kpts in enumerate(keypoints):
                person_scores = scores[det_idx]
                bbox = self._tracking_bbox_from_person(kpts, person_scores)
                if bbox is not None:
                    yield det_idx, bbox
        elif det_bboxes is not None:
            for det_idx, bbox in enumerate(det_bboxes):
                yield det_idx, bbox
        else:
            for det_idx, kpts in enumerate(keypoints):
                yield det_idx, pose_to_bbox(kpts)

    def reset(self):
        """Reset pose tracker."""
        self.frame_cnt = 0
        self.next_id = 0
        self.bboxes_last_frame = []
        self.track_ids_last_frame = []
        self.det_bboxes_last_frame = []

    def __call__(self, image: np.ndarray):

        pose_model_name = type(self.pose_model).__name__
        track_bboxes = None
        empty_frame = False

        if self.det_model:  # top-down algorithm, e.g. rtmpose
            if self.frame_cnt % self.det_frequency == 0:
                try:
                    if self.det_categories or self.det_mode == 'multiclass':
                        if self.det_categories:
                            bboxes, classes = self.det_model(image)
                            bboxes = [
                                bbox for bbox, cls in zip(bboxes, classes)
                                if cls in self.det_categories
                            ]
                        else:
                            bboxes, _ = self.det_model(image)
                    else:
                        bboxes = self.det_model(image)
                except:  # noqa
                    return [], []
            else:
                bboxes = self.det_bboxes_last_frame
                
            if self.biggest_n_boxes_only > 0:
                bboxes_sizes = [bbox[2] * bbox[3] for bbox in bboxes]
                bboxes_sizes.sort(reverse=True)
                bboxes_sizes = bboxes_sizes[:self.biggest_n_boxes_only]
                bboxes = [bbox for bbox in bboxes if bbox[2] * bbox[3] in bboxes_sizes]

            if self.frame_cnt % self.det_frequency == 0:
                self.det_bboxes_last_frame = list(bboxes)

            track_bboxes = [det_bbox_xyxy(bbox) for bbox in bboxes]
            empty_frame = len(track_bboxes) == 0
            if self.tracking_keypoint_ids is not None:
                track_bboxes = None

            if pose_model_name == 'RTMPose3d':
                keypoints, scores, keypoints_simcc, keypoints2d = self.pose_model(
                    image, bboxes=bboxes)
            else:
                keypoints, scores = self.pose_model(image, bboxes=bboxes)

        else:  # one-stage algorithm, e.g. rtmo
            keypoints, scores = self.pose_model(image)
            empty_frame = len(keypoints) == 0

        if not self.tracking and self.det_frequency != 1:
            # without tracking
            bboxes_current_frame = []
            track_kpts = (
                keypoints2d if pose_model_name == 'RTMPose3d' else keypoints
            )
            for det_idx, kpts in enumerate(track_kpts):
                if self.tracking_keypoint_ids is not None:
                    bbox = self._tracking_bbox_from_person(kpts, scores[det_idx])
                else:
                    bbox = pose_to_bbox(kpts)
                if bbox is not None:
                    bboxes_current_frame.append(bbox)

            self.bboxes_last_frame = bboxes_current_frame

        else:
            # with tracking
            bboxes_prev = list(self.bboxes_last_frame)
            track_ids_prev = list(self.track_ids_last_frame)
            used_prev_indices = set()

            bboxes_current_frame = []
            track_ids_current_frame = []
            det_indices_current_frame = []
            track_kpts = (
                keypoints2d if pose_model_name == 'RTMPose3d' else keypoints
            )
            if not empty_frame:
                for det_idx, bbox in self._iter_tracking_bboxes(
                        track_kpts, scores, det_bboxes=track_bboxes):
                    track_id, _ = self.track_by_iou(
                        bbox, bboxes_prev, track_ids_prev, used_prev_indices)

                    if track_id > -1:
                        track_ids_current_frame.append(track_id)
                        bboxes_current_frame.append(bbox)
                        det_indices_current_frame.append(det_idx)

            self.bboxes_last_frame = bboxes_current_frame
            self.track_ids_last_frame = track_ids_current_frame

            if det_indices_current_frame:
                sort_order = np.argsort(track_ids_current_frame)
                det_indices = [
                    det_indices_current_frame[i] for i in sort_order
                ]
                keypoints = keypoints[det_indices]
                scores = scores[det_indices]
                self.track_ids_last_frame = [
                    track_ids_current_frame[i] for i in sort_order
                ]
        self.frame_cnt += 1

        if pose_model_name == 'RTMPose3d':
            return keypoints, scores, keypoints_simcc, keypoints2d

        return keypoints, scores

    def track_by_iou(self, bbox, bboxes_prev, track_ids_prev,
                     used_prev_indices):
        """Get track id using IoU tracking greedily.

        Args:
            bbox (list): The bbox info (left, top, right, bottom).
            bboxes_prev (list): Previous-frame bboxes to match against.
            track_ids_prev (list): Track ids aligned with bboxes_prev.
            used_prev_indices (set): Indices already matched this frame.

        Returns:
            track_id (int): The track id.
            match_result (list): The matched bbox, or None.
        """

        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        max_iou_score = -1
        max_index = -1
        match_result = None
        for index, each_bbox in enumerate(bboxes_prev):
            if index in used_prev_indices:
                continue

            iou_score = compute_iou(bbox, each_bbox)
            if iou_score > max_iou_score:
                max_iou_score = iou_score
                max_index = index

        if max_iou_score > self.tracking_thr:
            used_prev_indices.add(max_index)
            track_id = track_ids_prev[max_index]
            match_result = bboxes_prev[max_index]

        elif area >= self.tracking_min_area:
            track_id = self.next_id
            self.next_id += 1

        else:
            track_id = -1

        return track_id, match_result
