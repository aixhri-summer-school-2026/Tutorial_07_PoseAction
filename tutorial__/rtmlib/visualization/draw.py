import math

import cv2
import numpy as np

from .skeleton import *  # noqa


def _normalize_score(score):
    score = np.asarray(score).reshape(-1)[0]
    return float(np.clip(score, 0.0, 1.0))


def _confidence_color(score):
    score = _normalize_score(score)
    # Low confidence is red, high confidence is green.
    red = int(255 * (1.0 - score))
    green = int(255 * score)
    return (0, green, red)


def _format_keypoint_label(label_mode, keypoint_name, score):
    if not label_mode:
        return None

    label_mode = str(label_mode).lower()
    score_text = f'{_normalize_score(score):.2f}'

    if label_mode == 'name':
        return keypoint_name
    if label_mode in ['confidence', 'conf', 'score']:
        return score_text
    if label_mode == 'both':
        return f'{keypoint_name} {score_text}'

    raise ValueError(
        'label_keypoint must be one of None, "name", "confidence", '
        '"score", "conf", or "both"')


def _draw_keypoint_label(img, text, keypoint, color, radius):
    if not text:
        return img

    text_origin = (int(keypoint[0] + radius + 2), int(keypoint[1] - radius - 2))
    return cv2.putText(img, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                       color, 1, cv2.LINE_AA)


def draw_bbox(img, bboxes, color=(0, 255, 0)):
    for bbox in bboxes:
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])), color, 2)
    return img


def draw_skeleton(img,
                  keypoints,
                  scores,
                  openpose_skeleton=False,
                  kpt_thr=0.5,
                  radius=2,
                  line_width=2,
                  show_keypoints=True,
                  label_keypoint=None):

    if len(keypoints) == 0:
        return img

    num_keypoints = keypoints.shape[1]

    if openpose_skeleton:
        if num_keypoints == 17:
            skeleton = 'animal17'
        elif num_keypoints == 18:
            skeleton = 'openpose18'
        elif num_keypoints == 134:
            skeleton = 'openpose134'
        elif num_keypoints == 26:
            skeleton = 'halpe26'
        else:
            raise NotImplementedError
    else:
        if num_keypoints == 17:
            skeleton = 'coco17'
        elif num_keypoints == 133:
            skeleton = 'coco133'
        elif num_keypoints == 21:
            skeleton = 'hand21'
        elif num_keypoints == 25:
            skeleton = 'coco25'
        elif num_keypoints == 26:
            skeleton = 'halpe26'
        else:
            raise NotImplementedError

    skeleton_dict = eval(f'{skeleton}')
    keypoint_info = skeleton_dict['keypoint_info']
    skeleton_info = skeleton_dict['skeleton_info']

    if len(keypoints.shape) == 2:
        keypoints = keypoints[None, :, :]
        scores = scores[None, :, :]

    num_instance = keypoints.shape[0]
    if skeleton in ['coco17', 'coco133', 'hand21', 'coco25', 'halpe26']:
        for i in range(num_instance):
            img = draw_mmpose(img, keypoints[i], scores[i], keypoint_info,
                              skeleton_info, kpt_thr, radius, line_width,
                              show_keypoints, label_keypoint)
    elif skeleton in ['openpose18', 'openpose134', 'animal17']:
        for i in range(num_instance):
            img = draw_openpose(img,
                                keypoints[i],
                                scores[i],
                                keypoint_info,
                                skeleton_info,
                                kpt_thr,
                                radius * 2,
                                alpha=0.6,
                                line_width=line_width * 2,
                                show_keypoints=show_keypoints,
                                label_keypoint=label_keypoint)
    else:
        raise NotImplementedError
    return img


def draw_mmpose(img,
                keypoints,
                scores,
                keypoint_info,
                skeleton_info,
                kpt_thr=0.5,
                radius=2,
                line_width=2,
                show_keypoints=True,
                label_keypoint=None):
    assert len(keypoints.shape) == 2

    vis_kpt = [_normalize_score(s) >= kpt_thr for s in scores]

    link_dict = {}
    for i, kpt_info in keypoint_info.items():
        link_dict[kpt_info['name']] = kpt_info['id']

        kpt = keypoints[i]
        kpt_color = _confidence_color(scores[i])

        if show_keypoints and vis_kpt[i]:
            img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius),
                             kpt_color, -1)
        if vis_kpt[i]:
            label = _format_keypoint_label(label_keypoint, kpt_info['name'],
                                           scores[i])
            img = _draw_keypoint_label(img, label, kpt, kpt_color, radius)

    for i, ske_info in skeleton_info.items():
        link = ske_info['link']
        pt0, pt1 = link_dict[link[0]], link_dict[link[1]]

        if vis_kpt[pt0] and vis_kpt[pt1]:
            link_color = ske_info['color']
            kpt0 = keypoints[pt0]
            kpt1 = keypoints[pt1]

            img = cv2.line(img, (int(kpt0[0]), int(kpt0[1])),
                           (int(kpt1[0]), int(kpt1[1])),
                           link_color,
                           thickness=line_width)

    return img


def draw_openpose(img,
                  keypoints,
                  scores,
                  keypoint_info,
                  skeleton_info,
                  kpt_thr=0.4,
                  radius=4,
                  alpha=1.0,
                  line_width=2,
                  show_keypoints=True,
                  label_keypoint=None):
    h, w = img.shape[:2]

    link_dict = {}
    for i, kpt_info in keypoint_info.items():
        link_dict[kpt_info['name']] = kpt_info['id']

    for i, ske_info in skeleton_info.items():
        link = ske_info['link']
        pt0, pt1 = link_dict[link[0]], link_dict[link[1]]

        link_color = ske_info['color']
        kpt0, kpt1 = keypoints[pt0], keypoints[pt1]
        s0 = _normalize_score(scores[pt0])
        s1 = _normalize_score(scores[pt1])

        if (kpt0[0] <= 0 or kpt0[0] >= w or kpt0[1] <= 0 or kpt0[1] >= h
                or kpt1[0] <= 0 or kpt1[0] >= w or kpt1[1] <= 0 or kpt1[1] >= h
                or s0 < kpt_thr or s1 < kpt_thr or link_color is None):
            continue

        X = np.array([kpt0[0], kpt1[0]])
        Y = np.array([kpt0[1], kpt1[1]])

        if i <= 16:
            # body part
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
            transparency = 0.6
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            polygons = cv2.ellipse2Poly((int(mX), int(mY)),
                                        (int(length / 2), int(line_width)),
                                        int(angle), 0, 360, 1)
            img = draw_polygons(img,
                                polygons,
                                edge_colors=link_color,
                                alpha=transparency)
        else:
            img = cv2.line(img, (int(X[0]), int(Y[0])), (int(X[1]), int(Y[1])),
                           link_color,
                           thickness=2)

    for j, kpt_info in keypoint_info.items():
        kpt = keypoints[j]

        if _normalize_score(scores[j]) < kpt_thr:
            continue

        transparency = alpha
        if 24 <= j <= 91:
            j_radius = 3
        else:
            j_radius = 4
        # j_radius = radius // 2 if j > 17 else radius

        kpt_color = _confidence_color(scores[j])
        if show_keypoints:
            img = draw_circles(img,
                               kpt,
                               radius=j_radius,
                               face_colors=kpt_color,
                               alpha=transparency)
        label = _format_keypoint_label(label_keypoint, kpt_info['name'],
                                       scores[j])
        img = _draw_keypoint_label(img, label, kpt, kpt_color, j_radius)

    return img


def draw_polygons(img, polygons, edge_colors, alpha=1.0):
    if alpha == 1.0:
        img = cv2.fillConvexPoly(img, polygons, edge_colors)
    else:
        img = cv2.fillConvexPoly(img.copy(), polygons, edge_colors)
        img = cv2.addWeighted(img, 1 - alpha, img, alpha, 0)
    return img


def draw_circles(img, center, radius, face_colors, alpha=1.0):
    if alpha == 1.0:
        img = cv2.circle(img, (int(center[0]), int(center[1])), int(radius),
                         face_colors, -1)
    else:
        img = cv2.circle(img.copy(), (int(center[0]), int(center[1])),
                         int(radius), face_colors, -1)
        img = cv2.addWeighted(img, 1 - alpha, img, alpha, 0)
    return img
