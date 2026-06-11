"""경계 파손(seam break) 점수 + 좌우 비대칭 점수.

투명화/클리핑 버그는 의상 seam에서 비정상 에지 파편, jagged hole, 급격한 단절을
만든다. ROI 마스크 경계를 따라 에지 연속성을 본다.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from . import mask_utils


def boundary_break_score(
    crop_bgr: np.ndarray,
    full_mask: np.ndarray,
    roi_box: Tuple[int, int, int, int],
) -> Tuple[float, np.ndarray]:
    """ROI 내 비정상 에지 파편 / 기대 경계 길이.

    반환: (score 0~1, 에지 시각화 마스크)
    """
    x1, y1, x2, y2 = roi_box
    roi_img = crop_bgr[y1:y2, x1:x2]
    roi_mask = mask_utils.to_binary(full_mask)[y1:y2, x1:x2]
    vis = np.zeros(roi_mask.shape, dtype=np.uint8)

    if roi_img.size == 0 or np.count_nonzero(roi_mask) == 0:
        return 0.0, vis

    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 기대 경계: 마스크 외곽선 부근(dilate - erode 링)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ring = cv2.subtract(cv2.dilate(roi_mask, k), cv2.erode(roi_mask, k))
    expected_len = int(np.count_nonzero(ring))
    if expected_len == 0:
        return 0.0, vis

    # 마스크 내부에 있지만 경계 링 밖에 있는 에지 = 비정상 파편 후보
    interior = cv2.erode(roi_mask, k)
    abnormal = cv2.bitwise_and(edges, interior)
    vis[abnormal > 0] = 255

    abnormal_len = int(np.count_nonzero(abnormal))
    score = abnormal_len / expected_len
    return float(round(min(1.0, score / 0.5), 4)), vis


def asymmetry_score(left_score: float, right_score: float) -> float:
    """좌우 ROI bug_score 비대칭. 한쪽만 튀면 의심 가중."""
    return float(round(min(1.0, abs(left_score - right_score)), 4))


def skin_exposure_score(crop_bgr: np.ndarray, full_mask: np.ndarray,
                        roi_box: Tuple[int, int, int, int]) -> Tuple[float, np.ndarray]:
    """보수적 스킨 노출 점수. 메타데이터가 없으므로 약하게만 사용(가중치 0.10).

    HSV/YCrCb 스킨 색 범위에 드는 픽셀 비율. 의도된 노출 의상이 많으므로 과신 금지.
    """
    x1, y1, x2, y2 = roi_box
    roi_img = crop_bgr[y1:y2, x1:x2]
    roi_mask = mask_utils.to_binary(full_mask)[y1:y2, x1:x2]
    vis = np.zeros(roi_mask.shape, dtype=np.uint8)
    avatar_px = int(np.count_nonzero(roi_mask))
    if roi_img.size == 0 or avatar_px == 0:
        return 0.0, vis

    ycrcb = cv2.cvtColor(roi_img, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    skin = cv2.inRange(ycrcb, lower, upper)
    inside = cv2.bitwise_and(skin, roi_mask)
    vis[inside > 0] = 255

    score = int(np.count_nonzero(inside)) / avatar_px
    # 0.4 이상이면 포화, 그래도 가중치 낮음
    return float(round(min(1.0, score / 0.4), 4)), vis
