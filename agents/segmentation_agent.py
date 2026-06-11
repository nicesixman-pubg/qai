"""세그멘테이션 에이전트 — YOLO 마스크 + rembg 교차검증 → 정제.

원본 크롭 보존이 원칙. 5종 산출: original / mask(refined) / overlay / bg_removed / (heatmap은 이후).
중·대형 내부 홀은 정제 시 보존(버그 신호).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from cv import mask_utils

_rembg_session = None


def _rembg_mask(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    """rembg(U2Net)로 알파 마스크 추출. 미설치 시 None."""
    global _rembg_session
    try:
        import cv2
        from rembg import new_session, remove
    except Exception:
        return None
    try:
        if _rembg_session is None:
            _rembg_session = new_session("u2net")
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        out = remove(rgb, session=_rembg_session)  # RGBA
        if out.shape[-1] == 4:
            alpha = out[..., 3]
            return mask_utils.to_binary(alpha)
        return None
    except Exception as e:
        print(f"[segmentation] rembg 실패: {e}")
        return None


def run(crop_bgr: np.ndarray, yolo_mask_crop: Optional[np.ndarray] = None) -> dict:
    """crop 로컬 좌표 기준 마스크 산출.

    yolo_mask_crop: 탐지 단계에서 얻은 인스턴스 마스크를 crop 영역으로 자른 것(있으면).
    """
    candidates = []
    if yolo_mask_crop is not None and yolo_mask_crop.size > 0:
        candidates.append(mask_utils.to_binary(yolo_mask_crop))
    rembg = _rembg_mask(crop_bgr)
    if rembg is not None:
        candidates.append(rembg)

    if not candidates:
        # 최후 fallback: GrabCut 근사 (중앙 사각형 시드)
        raw = _grabcut_fallback(crop_bgr)
    elif len(candidates) == 1:
        raw = candidates[0]
    else:
        # 교차검증: 합집합(둘 중 하나라도 전경이면 전경) — recall 우선
        import cv2

        raw = cv2.bitwise_or(candidates[0], candidates[1])

    refined = mask_utils.refine_mask(raw)
    quality = mask_utils.mask_quality(raw, refined)

    return {
        "raw_mask": raw,
        "refined_mask": refined,
        "overlay": mask_utils.mask_overlay(crop_bgr, refined),
        "bg_removed": mask_utils.background_removed(crop_bgr, refined),
        "mask_confidence": quality,
        "mask_area_px": mask_utils.mask_area(refined),
    }


def _grabcut_fallback(crop_bgr: np.ndarray) -> np.ndarray:
    import cv2

    h, w = crop_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.1), int(h * 0.05), int(w * 0.8), int(h * 0.9))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop_bgr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return out
    except Exception:
        # 정말 최후: 사각형 마스크
        out = np.zeros((h, w), np.uint8)
        x, y, rw, rh = rect
        out[y:y + rh, x:x + rw] = 255
        return out
