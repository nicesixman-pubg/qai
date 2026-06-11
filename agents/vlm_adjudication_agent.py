"""VLM 판정 에이전트 — 증거 보드 + 수치 점수를 provider에 전달해 구조화 JSON 수신.

VLM은 첫 판사가 아니라 마지막 추론·설명 레이어. 클린 이미지 한 장이 아니라
원본/오버레이/ROI 줌/히트맵/점수를 함께 준다.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from vlm import get_provider


def run(
    original_crop: np.ndarray,
    mask_overlay: np.ndarray,
    roi_crops: Dict[str, np.ndarray],
    heatmap: np.ndarray,
    numeric_scores: dict,
    cv_decision: dict,
) -> dict:
    images: List[dict] = [
        {"name": "original_crop", "image": original_crop},
        {"name": "mask_overlay", "image": mask_overlay},
        {"name": "heatmap", "image": heatmap},
    ]
    for name, img in roi_crops.items():
        if img is not None and img.size > 0:
            images.append({"name": f"roi_{name}", "image": img})

    payload = {
        "cv_band": cv_decision.get("cv_band"),
        "cv_label": cv_decision.get("cv_label"),
        "avatar_bug_score": numeric_scores.get("avatar_bug_score"),
        "worst_roi": numeric_scores.get("worst_roi"),
        "roi_scores": numeric_scores.get("roi_scores", {}),
    }

    provider = get_provider()
    return provider.adjudicate(images, payload)


def run_reference(
    reference_crop: np.ndarray,
    test_crop: np.ndarray,
    diff_overlay: np.ndarray,
    side_by_side: np.ndarray,
    candidate_crops: Dict[str, np.ndarray],
    diff_payload: dict,
) -> dict:
    images: List[dict] = [
        {"name": "reference_crop", "image": reference_crop},
        {"name": "test_crop", "image": test_crop},
        {"name": "diff_overlay", "image": diff_overlay},
        {"name": "side_by_side_reference_test_diff", "image": side_by_side},
    ]
    for name, img in candidate_crops.items():
        if img is not None and img.size > 0:
            images.append({"name": name, "image": img})

    provider = get_provider()
    return provider.adjudicate(images, diff_payload)
