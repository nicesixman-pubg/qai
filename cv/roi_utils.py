"""ROI 박스 계산 / 크롭. bbox 비율 fallback + (선택) MediaPipe 포즈.

ROI는 크롭 이미지 로컬 좌표(0,0 = 크롭 좌상단) 기준이다.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

import config

ROIBox = Tuple[int, int, int, int]  # 크롭 로컬 좌표


def bbox_ratio_rois(crop_w: int, crop_h: int) -> Dict[str, ROIBox]:
    """config.ROI_RATIOS 비율로 ROI 박스 생성."""
    out: Dict[str, ROIBox] = {}
    for name, (rx1, ry1, rx2, ry2) in config.ROI_RATIOS.items():
        out[name] = (
            int(rx1 * crop_w),
            int(ry1 * crop_h),
            int(rx2 * crop_w),
            int(ry2 * crop_h),
        )
    return out


def _box_around(cx: float, cy: float, w: float, h: float, crop_w: int, crop_h: int) -> ROIBox:
    x1 = int(max(0, cx - w / 2))
    y1 = int(max(0, cy - h / 2))
    x2 = int(min(crop_w, cx + w / 2))
    y2 = int(min(crop_h, cy + h / 2))
    return (x1, y1, x2, y2)


def pose_rois(crop_bgr: np.ndarray) -> Optional[Dict[str, ROIBox]]:
    """MediaPipe Pose로 관절 기반 ROI. 미설치/실패 시 None 반환(→ bbox 비율 fallback)."""
    try:
        import cv2
        import mediapipe as mp
    except Exception:
        return None

    h, w = crop_bgr.shape[:2]
    try:
        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True, model_complexity=1,
                          enable_segmentation=False, min_detection_confidence=0.4) as pose:
            res = pose.process(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks.landmark
        P = mp_pose.PoseLandmark

        def pt(landmark) -> Tuple[float, float]:
            return (landmark.x * w, landmark.y * h)

        lw, rw = pt(lm[P.LEFT_WRIST]), pt(lm[P.RIGHT_WRIST])
        la, ra = pt(lm[P.LEFT_ANKLE]), pt(lm[P.RIGHT_ANKLE])
        lhip, rhip = pt(lm[P.LEFT_HIP]), pt(lm[P.RIGHT_HIP])
        ls, rs = pt(lm[P.LEFT_SHOULDER]), pt(lm[P.RIGHT_SHOULDER])
        nose = pt(lm[P.NOSE])

        pelvis = ((lhip[0] + rhip[0]) / 2, (lhip[1] + rhip[1]) / 2)
        neck = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)

        bw = w * 0.30  # ROI 폭/높이 기본 스케일
        bh = h * 0.16
        return {
            "waist": _box_around(pelvis[0], pelvis[1], w * 0.55, h * 0.18, w, h),
            "left_wrist": _box_around(lw[0], lw[1], bw, bh, w, h),
            "right_wrist": _box_around(rw[0], rw[1], bw, bh, w, h),
            "left_ankle": _box_around(la[0], la[1], bw, bh, w, h),
            "right_ankle": _box_around(ra[0], ra[1], bw, bh, w, h),
            "neck": _box_around((neck[0] + nose[0]) / 2, (neck[1] + nose[1]) / 2,
                                w * 0.32, h * 0.16, w, h),
        }
    except Exception:
        return None


def compute_rois(crop_bgr: np.ndarray, method_priority: List[str]) -> List[dict]:
    """우선순위에 따라 ROI 계산. 반환: [{name, box, method, confidence}]"""
    h, w = crop_bgr.shape[:2]
    boxes: Optional[Dict[str, ROIBox]] = None
    method = "bbox_ratio"
    for m in method_priority:
        if m == "pose":
            boxes = pose_rois(crop_bgr)
            if boxes:
                method = "pose"
                break
        elif m == "bbox_ratio":
            boxes = bbox_ratio_rois(w, h)
            method = "bbox_ratio"
            break
    if boxes is None:
        boxes = bbox_ratio_rois(w, h)
        method = "bbox_ratio"

    conf = 0.8 if method == "pose" else 0.55
    return [
        {"name": name, "box": box, "method": method, "confidence": conf}
        for name, box in boxes.items()
    ]


def crop_box(image: np.ndarray, box: ROIBox) -> np.ndarray:
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]


def submask(mask: np.ndarray, box: ROIBox) -> np.ndarray:
    x1, y1, x2, y2 = box
    return mask[y1:y2, x1:x2]
