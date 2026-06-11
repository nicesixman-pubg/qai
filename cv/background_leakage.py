"""배경 비침 점수 — 아바타 영역(ROI) 안에 '배경처럼 보이는' 픽셀 비율.

투명화 버그의 가장 강력한 신호. bbox 주변 배경 색 분포를 샘플링해 모델을 만들고,
ROI 내부에서 그 분포에 가까운 픽셀을 센다.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from . import mask_utils


def _sample_background(crop_bgr: np.ndarray, full_mask: np.ndarray) -> np.ndarray:
    """크롭 안에서 마스크 바깥(=배경) 픽셀을 HSV로 수집."""
    m = mask_utils.to_binary(full_mask)
    bg = m == 0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    samples = hsv[bg]
    if samples.shape[0] < 50:
        return np.empty((0, 3), dtype=np.float32)
    # 과대 표본 방지
    if samples.shape[0] > 20000:
        idx = np.linspace(0, samples.shape[0] - 1, 20000).astype(int)
        samples = samples[idx]
    return samples.astype(np.float32)


def background_leakage_score(
    crop_bgr: np.ndarray,
    full_mask: np.ndarray,
    roi_box: Tuple[int, int, int, int],
) -> Tuple[float, np.ndarray]:
    """반환: (score 0~1, ROI 내 배경유사 픽셀 마스크)"""
    bg_samples = _sample_background(crop_bgr, full_mask)
    x1, y1, x2, y2 = roi_box
    roi_img = crop_bgr[y1:y2, x1:x2]
    roi_mask = mask_utils.to_binary(full_mask)[y1:y2, x1:x2]
    avatar_px = int(np.count_nonzero(roi_mask))
    leak_vis = np.zeros(roi_mask.shape, dtype=np.uint8)

    if avatar_px == 0 or bg_samples.shape[0] == 0 or roi_img.size == 0:
        return 0.0, leak_vis

    # 배경 색 모델: HSV 평균/표준편차 (가벼운 가우시안 근사)
    mean = bg_samples.mean(axis=0)
    std = bg_samples.std(axis=0) + 1e-3

    roi_hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Hue는 원형이므로 거리 보정
    dh = np.abs(roi_hsv[..., 0] - mean[0])
    dh = np.minimum(dh, 180.0 - dh)
    ds = np.abs(roi_hsv[..., 1] - mean[1])
    dv = np.abs(roi_hsv[..., 2] - mean[2])
    # 정규화 마할라노비스 유사 거리
    dist = np.sqrt((dh / std[0]) ** 2 + (ds / std[1]) ** 2 + (dv / std[2]) ** 2)

    bg_like = dist < 2.5  # 2.5 표준편차 이내면 배경유사
    inside = (roi_mask > 0) & bg_like
    leak_vis[inside] = 255

    leak_px = int(np.count_nonzero(inside))
    score = leak_px / avatar_px
    # 0.15 이상이면 포화
    return float(round(min(1.0, score / 0.15), 4)), leak_vis
