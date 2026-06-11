"""내부 홀 탐지 — 실루엣 내부의 예기치 않은 구멍(투명화 핵심 신호 중 하나).

holes = filled_silhouette - mask. ROI 내부 홀 면적 / ROI 마스크 면적.
미세 홀은 노이즈로 무시.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

import config
from . import mask_utils


def hole_mask(full_mask: np.ndarray) -> np.ndarray:
    """전체 마스크에서 내부 홀만 추출한 0/255 마스크."""
    m = mask_utils.to_binary(full_mask)
    filled = mask_utils.filled_mask(m)
    holes = cv2.subtract(filled, m)
    # 미세 노이즈 제거
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, k)
    return holes


def internal_hole_score(full_mask: np.ndarray, roi_box: Tuple[int, int, int, int]) -> Tuple[float, np.ndarray]:
    """ROI 내부 홀 점수.

    반환: (score 0~1, roi 내 홀 마스크) — 히트맵용으로 홀 마스크도 돌려준다.
    """
    holes = hole_mask(full_mask)
    x1, y1, x2, y2 = roi_box
    roi_holes = holes[y1:y2, x1:x2]
    roi_mask = mask_utils.to_binary(full_mask)[y1:y2, x1:x2]

    roi_area = int(np.count_nonzero(roi_mask))
    if roi_area == 0:
        return 0.0, roi_holes

    # 미세 홀 무시: 개별 컴포넌트 면적이 ROI 마스크의 일정 비율 미만이면 제외
    min_area = max(8, int(config.MIN_HOLE_AREA_RATIO * roi_area))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(roi_holes, connectivity=8)
    significant = 0
    keep = np.zeros_like(roi_holes)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            significant += area
            keep[labels == i] = 255

    score = significant / roi_area
    # 0.08 이상이면 high로 포화
    return float(round(min(1.0, score / 0.08), 4)), keep
