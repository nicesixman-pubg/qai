"""마스크 정제 / 오버레이 / 누끼 / 크롭 유틸.

핵심 원칙: 배경을 너무 일찍 지우면 투명화 버그 신호(실루엣 내부 배경 비침)가
소실된다. 따라서 원본 크롭은 항상 보존하고, 마스크 정제 시 중·대형 내부 홀은
메우지 않고 남긴다.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2) — 절대 좌표


# ---------------------------------------------------------------------------
# 크롭
# ---------------------------------------------------------------------------
def crop_with_margin(image: np.ndarray, bbox: BBox, margin: float = 0.15) -> Tuple[np.ndarray, BBox]:
    """bbox에 margin(비율)을 더해 크롭. 실루엣 경계 증거 보존을 위해 타이트 크롭 금지.

    반환: (크롭 이미지, 프레임 절대좌표로 표현한 확장 bbox)
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin), int(bh * margin)
    nx1 = max(0, x1 - mx)
    ny1 = max(0, y1 - my)
    nx2 = min(w, x2 + mx)
    ny2 = min(h, y2 + my)
    return image[ny1:ny2, nx1:nx2].copy(), (nx1, ny1, nx2, ny2)


# ---------------------------------------------------------------------------
# 정제
# ---------------------------------------------------------------------------
def to_binary(mask: np.ndarray) -> np.ndarray:
    """임의 마스크 → 0/255 uint8."""
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    else:
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def refine_mask(mask: np.ndarray, open_ksize: int = 3, close_ksize: int = 5) -> np.ndarray:
    """작은 노이즈 제거(open) + 미세 갭 메움(close). 중·대형 내부 홀은 보존.

    주의: 전체 fill을 하지 않는다 — 내부 홀이 버그 신호이기 때문.
    """
    m = to_binary(mask)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
    # 가장 큰 외곽 컴포넌트만 아바타로 간주(외부 잡티 제거), 내부 홀은 유지
    return keep_largest_component(m)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    m = to_binary(mask)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m
    # 0은 배경
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def filled_mask(mask: np.ndarray) -> np.ndarray:
    """내부 홀을 전부 메운 '기대 실루엣'. (holes = filled - mask 계산용)"""
    m = to_binary(mask)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)
    return out


def mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(to_binary(mask)))


def mask_quality(raw_mask: np.ndarray, refined_mask: np.ndarray) -> float:
    """세그 신뢰도 프록시 0~1.

    raw와 refined의 IoU가 높을수록(정제로 거의 안 바뀜) 안정적이라고 본다.
    면적이 너무 작으면 패널티.
    """
    a = to_binary(raw_mask) > 0
    b = to_binary(refined_mask) > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    iou = inter / union if union else 0.0
    area_ratio = b.sum() / b.size if b.size else 0.0
    area_factor = min(1.0, area_ratio / 0.05)  # 5% 이상이면 만점
    return float(round(0.7 * iou + 0.3 * area_factor, 4))


# ---------------------------------------------------------------------------
# 시각화 / 누끼
# ---------------------------------------------------------------------------
def mask_overlay(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.45) -> np.ndarray:
    """마스크를 반투명 색으로 원본 위에 오버레이."""
    m = to_binary(mask)
    overlay = image.copy()
    colored = np.zeros_like(image)
    colored[m > 0] = color
    out = cv2.addWeighted(overlay, 1.0, colored, alpha, 0)
    # 외곽선 강조
    contours, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), 2)
    return out


def background_removed(image: np.ndarray, mask: np.ndarray, bg=(0, 0, 0)) -> np.ndarray:
    """배경 제거 크롭(참고용). 원본은 별도 보존되어야 함."""
    m = to_binary(mask)
    out = np.full_like(image, bg)
    out[m > 0] = image[m > 0]
    return out
