"""프레임 품질 에이전트 — 비싼 처리 전에 나쁜 입력을 걸러낸다.

이미지 수준 체크(블러/저조도/해상도)를 담당. 아바타 크기·다중·잘림 같은
탐지 의존 체크는 detection 이후 pipeline에서 issues에 병합된다.
"""
from __future__ import annotations

import cv2
import numpy as np

import config


def run(image_bgr: np.ndarray) -> dict:
    issues = []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_var < config.BLUR_VAR_THRESHOLD:
        issues.append("motion_blur")

    mean_brightness = float(gray.mean())
    if mean_brightness < config.DARK_MEAN_THRESHOLD:
        issues.append("too_dark")

    h, w = image_bgr.shape[:2]
    if min(h, w) < 200:
        issues.append("low_resolution")

    # 등급 산정
    if "low_resolution" in issues or (blur_var < config.BLUR_VAR_THRESHOLD * 0.5):
        quality = "unusable"
    elif issues:
        quality = "weak"
    else:
        quality = "good"

    return {
        "frame_quality": quality,
        "issues": issues,
        "can_continue": quality != "unusable",
        "metrics": {"blur_var": round(blur_var, 2), "brightness": round(mean_brightness, 2)},
    }


def augment_with_detection(quality: dict, avatars: list, frame_shape) -> dict:
    """탐지 결과로 아바타 수준 이슈 병합(잘림/과소/다중)."""
    h, w = frame_shape[:2]
    issues = list(quality.get("issues", []))

    if len(avatars) == 0:
        issues.append("no_avatar_detected")
    if len(avatars) > 1:
        issues.append("multiple_avatars")

    for a in avatars:
        x1, y1, x2, y2 = a["bbox"]
        area_ratio = ((x2 - x1) * (y2 - y1)) / float(w * h)
        if area_ratio < config.MIN_AVATAR_AREA_RATIO:
            issues.append("avatar_too_small")
        m = config.EDGE_CUTOFF_MARGIN
        if x1 <= m or y1 <= m or x2 >= w - m or y2 >= h - m:
            issues.append("avatar_cut_off")

    issues = sorted(set(issues))
    quality = dict(quality)
    quality["issues"] = issues
    # 등급 하향 (단, unusable는 유지)
    if quality["frame_quality"] == "good" and issues:
        quality["frame_quality"] = "weak"
    return quality
