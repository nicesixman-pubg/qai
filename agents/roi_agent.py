"""ROI 에이전트 — 고위험 영역(허리/손목/발목/목) 국소화.

우선순위: pose → bbox_ratio. cv.roi_utils 래핑.
"""
from __future__ import annotations

from typing import List

import numpy as np

from cv import roi_utils


def run(crop_bgr: np.ndarray, method_priority: List[str] = None) -> List[dict]:
    if method_priority is None:
        method_priority = ["pose", "bbox_ratio"]
    return roi_utils.compute_rois(crop_bgr, method_priority)
