"""이상 점수 에이전트 — ROI별 6대(현재 5) 신호를 계산해 roi_bug_score로 융합.

roi_bug_score = Σ weight_i * signal_i   (config.SCORE_WEIGHTS)
avatar_bug_score = max(roi_bug_score)   — 평균 아니라 max (국소 버그 희석 방지)
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

import config
from cv import background_leakage, edge_artifact, hole_detection


def _score_one_roi(crop_bgr: np.ndarray, mask: np.ndarray, roi: dict) -> dict:
    box = roi["box"]
    bg_s, bg_vis = background_leakage.background_leakage_score(crop_bgr, mask, box)
    hole_s, hole_vis = hole_detection.internal_hole_score(mask, box)
    edge_s, edge_vis = edge_artifact.boundary_break_score(crop_bgr, mask, box)
    skin_s, skin_vis = edge_artifact.skin_exposure_score(crop_bgr, mask, box)

    signals = {
        "background_leakage": bg_s,
        "internal_hole": hole_s,
        "boundary_break": edge_s,
        "skin_exposure": skin_s,
        "asymmetry": 0.0,  # 쌍 계산 후 채움
    }
    return {
        "name": roi["name"],
        "box": box,
        "method": roi.get("method"),
        "roi_confidence": roi.get("confidence"),
        "signals": signals,
        "vis_masks": {
            "background_leakage": bg_vis,
            "internal_hole": hole_vis,
            "boundary_break": edge_vis,
            "skin_exposure": skin_vis,
        },
    }


def _weighted(signals: Dict[str, float]) -> float:
    return float(round(sum(config.SCORE_WEIGHTS[k] * signals.get(k, 0.0)
                           for k in config.SCORE_WEIGHTS), 4))


def run(crop_bgr: np.ndarray, mask: np.ndarray, rois: List[dict]) -> dict:
    by_name: Dict[str, dict] = {}
    results = []
    for roi in rois:
        r = _score_one_roi(crop_bgr, mask, roi)
        results.append(r)
        by_name[r["name"]] = r

    # 좌우 비대칭: 먼저 비대칭 제외 점수로 좌우 비교
    for left, right in config.ROI_SYMMETRY_PAIRS:
        if left in by_name and right in by_name:
            ls = _weighted({**by_name[left]["signals"], "asymmetry": 0.0})
            rs = _weighted({**by_name[right]["signals"], "asymmetry": 0.0})
            asym = edge_artifact.asymmetry_score(ls, rs)
            by_name[left]["signals"]["asymmetry"] = asym
            by_name[right]["signals"]["asymmetry"] = asym

    # 최종 roi_bug_score + top_signals
    for r in results:
        r["bug_score"] = _weighted(r["signals"])
        top = sorted(r["signals"].items(), key=lambda kv: kv[1], reverse=True)[:3]
        r["top_signals"] = {k: v for k, v in top}

    avatar_bug_score = max((r["bug_score"] for r in results), default=0.0)
    worst = max(results, key=lambda r: r["bug_score"], default=None)
    return {
        "roi_results": results,
        "avatar_bug_score": round(avatar_bug_score, 4),
        "worst_roi": worst["name"] if worst else None,
    }
