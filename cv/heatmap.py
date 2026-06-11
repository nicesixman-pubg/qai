"""의심 픽셀 히트맵 렌더링 — VLM 증거보드 및 UI용.

각 ROI의 신호 시각화 마스크(홀/배경비침/에지)를 누적해 컬러 히트맵으로 합성하고
ROI별 점수를 라벨로 표시한다.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


def render(
    crop_bgr: np.ndarray,
    roi_results: List[dict],
) -> np.ndarray:
    """roi_results: [{name, box, bug_score, vis_masks: {signal: mask}}]

    반환: 히트맵이 오버레이된 BGR 이미지.
    """
    h, w = crop_bgr.shape[:2]
    accum = np.zeros((h, w), dtype=np.float32)

    for r in roi_results:
        x1, y1, x2, y2 = r["box"]
        for _signal, vis in r.get("vis_masks", {}).items():
            if vis is None or vis.size == 0:
                continue
            vh, vw = vis.shape[:2]
            # vis는 ROI 로컬 좌표 → 전체 좌표에 누적
            region = accum[y1:y1 + vh, x1:x1 + vw]
            region += (vis.astype(np.float32) / 255.0)

    if accum.max() > 0:
        accum = accum / accum.max()
    heat = (accum * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    out = cv2.addWeighted(crop_bgr, 0.6, heat_color, 0.4, 0)

    # ROI 박스 + 점수 라벨
    for r in roi_results:
        x1, y1, x2, y2 = r["box"]
        score = r.get("bug_score", 0.0)
        color = _score_color(score)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{r['name']}:{score:.2f}"
        cv2.putText(out, label, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def _score_color(score: float) -> Tuple[int, int, int]:
    if score >= 0.72:
        return (0, 0, 255)      # red — bug
    if score >= 0.40:
        return (0, 165, 255)    # orange — suspicious
    return (0, 200, 0)          # green — normal
