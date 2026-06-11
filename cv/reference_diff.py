"""Reference-based visual differencing.

This module deliberately avoids making the final bug decision. It produces
aligned visual evidence: diff maps, candidate boxes, and conservative numeric
scores that a VLM or reviewer can adjudicate against the known-good reference.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

import config
from . import mask_utils

BBox = Tuple[int, int, int, int]


def analyze(
    test_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    test_mask: np.ndarray | None = None,
    reference_mask: np.ndarray | None = None,
    max_candidates: int | None = None,
) -> dict:
    """Build a focused diff evidence board for a test/reference crop pair."""
    if test_bgr is None or reference_bgr is None or test_bgr.size == 0 or reference_bgr.size == 0:
        return _empty("empty_image")

    h, w = test_bgr.shape[:2]
    ref = cv2.resize(reference_bgr, (w, h), interpolation=cv2.INTER_AREA)

    tmask = _prepare_mask(test_mask, w, h)
    rmask = _prepare_mask(reference_mask, w, h)
    focus_mask = _focus_mask(tmask, rmask, w, h)

    lab_t = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_delta = np.linalg.norm(lab_t - lab_r, axis=2)
    lab_norm = np.clip(lab_delta / 60.0, 0.0, 1.0)

    gray_t = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    edge_t = cv2.Canny(gray_t, 50, 150)
    edge_r = cv2.Canny(gray_r, 50, 150)
    edge_delta = (cv2.absdiff(edge_t, edge_r).astype(np.float32) / 255.0)

    silhouette_delta = np.zeros((h, w), np.float32)
    if tmask is not None and rmask is not None:
        silhouette_delta = (cv2.bitwise_xor(tmask, rmask).astype(np.float32) / 255.0)

    score_map = (0.65 * lab_norm) + (0.25 * edge_delta) + (0.10 * silhouette_delta)
    score_map[focus_mask == 0] = 0.0

    diff_mask = _threshold_diff(score_map)
    candidates = _components(score_map, diff_mask, max_candidates or config.REFERENCE_DIFF_TOP_K)
    overlay = render_diff_overlay(test_bgr, candidates, score_map)
    side_by_side = _side_by_side(ref, test_bgr, overlay)

    overall = max((c["diff_score"] for c in candidates), default=0.0)
    coverage = float(np.count_nonzero(diff_mask)) / float(max(1, np.count_nonzero(focus_mask)))
    match_quality = _reference_match_quality(tmask, rmask, test_bgr, ref)

    return {
        "status": "ok",
        "reference_match_quality": round(match_quality, 4),
        "overall_diff_score": round(float(overall), 4),
        "diff_coverage": round(min(1.0, coverage), 4),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "diff_mask": (diff_mask * 255).astype(np.uint8),
        "score_map": score_map,
        "diff_overlay": overlay,
        "side_by_side": side_by_side,
    }


def crop_candidates(
    test_bgr: np.ndarray,
    reference_bgr: np.ndarray,
    candidates: List[dict],
    pad_ratio: float = 0.35,
) -> Dict[str, np.ndarray]:
    """Return compact reference/test/paired crops for VLM evidence."""
    out: Dict[str, np.ndarray] = {}
    h, w = test_bgr.shape[:2]
    ref = cv2.resize(reference_bgr, (w, h), interpolation=cv2.INTER_AREA)
    for c in candidates:
        x1, y1, x2, y2 = _pad_box(tuple(c["bbox"]), w, h, pad_ratio)
        ref_crop = ref[y1:y2, x1:x2]
        test_crop = test_bgr[y1:y2, x1:x2]
        if ref_crop.size == 0 or test_crop.size == 0:
            continue
        cid = c["candidate_id"]
        out[f"{cid}_reference"] = ref_crop
        out[f"{cid}_test"] = test_crop
        out[f"{cid}_pair"] = np.concatenate([ref_crop, test_crop], axis=1)
    return out


def render_diff_overlay(test_bgr: np.ndarray, candidates: List[dict], score_map: np.ndarray) -> np.ndarray:
    heat = np.clip(score_map * 255.0, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    out = cv2.addWeighted(test_bgr, 0.72, heat_color, 0.28, 0)
    for c in candidates:
        x1, y1, x2, y2 = c["bbox"]
        color = _score_color(c["diff_score"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{c['candidate_id']} {c['region_hint']}:{c['diff_score']:.2f}"
        cv2.putText(out, label, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def _prepare_mask(mask: np.ndarray | None, w: int, h: int) -> np.ndarray | None:
    if mask is None or mask.size == 0:
        return None
    m = mask_utils.to_binary(mask)
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.dilate(m, k, iterations=1)


def _focus_mask(tmask: np.ndarray | None, rmask: np.ndarray | None, w: int, h: int) -> np.ndarray:
    if tmask is None and rmask is None:
        return np.ones((h, w), np.uint8) * 255
    if tmask is None:
        return rmask
    if rmask is None:
        return tmask
    return cv2.bitwise_or(tmask, rmask)


def _threshold_diff(score_map: np.ndarray) -> np.ndarray:
    mask = (score_map >= config.REFERENCE_DIFF_PIXEL_THRESHOLD).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _components(score_map: np.ndarray, diff_mask: np.ndarray, max_candidates: int) -> List[dict]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(diff_mask, connectivity=8)
    h, w = diff_mask.shape[:2]
    min_area = max(12, int(config.REFERENCE_DIFF_MIN_AREA_RATIO * h * w))
    candidates = []
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        bw = int(stats[idx, cv2.CC_STAT_WIDTH])
        bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
        region = labels == idx
        vals = score_map[region]
        mean_score = float(vals.mean()) if vals.size else 0.0
        p90 = float(np.quantile(vals, 0.90)) if vals.size else 0.0
        area_factor = min(1.0, area / max(1.0, config.REFERENCE_DIFF_AREA_SATURATION * h * w))
        diff_score = min(1.0, 0.55 * p90 + 0.30 * mean_score + 0.15 * area_factor)
        bbox = (x, y, x + bw, y + bh)
        candidates.append({
            "candidate_id": f"cand_{len(candidates) + 1:03d}",
            "bbox": list(bbox),
            "area_px": area,
            "diff_score": round(float(diff_score), 4),
            "mean_delta": round(mean_score, 4),
            "p90_delta": round(p90, 4),
            "region_hint": _region_hint(bbox, w, h),
        })
    candidates.sort(key=lambda c: c["diff_score"], reverse=True)
    for idx, c in enumerate(candidates[:max_candidates], start=1):
        c["candidate_id"] = f"cand_{idx:03d}"
    return candidates[:max_candidates]


def _reference_match_quality(
    tmask: np.ndarray | None,
    rmask: np.ndarray | None,
    test_bgr: np.ndarray,
    ref_bgr: np.ndarray,
) -> float:
    quality = 0.75
    if tmask is not None and rmask is not None:
        t = tmask > 0
        r = rmask > 0
        inter = np.logical_and(t, r).sum()
        union = np.logical_or(t, r).sum()
        iou = inter / union if union else 0.0
        quality = 0.50 + 0.50 * iou
    bg_delta = abs(float(test_bgr.mean()) - float(ref_bgr.mean())) / 255.0
    quality -= min(0.25, bg_delta)
    return float(max(0.0, min(1.0, quality)))


def _region_hint(box: BBox, w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / max(1, w)
    cy = ((y1 + y2) / 2.0) / max(1, h)
    if cy < 0.24:
        return "neck"
    if cy < 0.43:
        return "torso"
    if cy < 0.64:
        if cx < 0.28:
            return "left_wrist"
        if cx > 0.72:
            return "right_wrist"
        return "waist"
    if cy > 0.76:
        return "left_ankle" if cx < 0.5 else "right_ankle"
    return "other"


def _pad_box(box: BBox, w: int, h: int, pad_ratio: float) -> BBox:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad = int(max(bw, bh) * pad_ratio)
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w, x2 + pad),
        min(h, y2 + pad),
    )


def _side_by_side(reference: np.ndarray, test: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    h, w = test.shape[:2]
    ref = cv2.resize(reference, (w, h), interpolation=cv2.INTER_AREA)
    return np.concatenate([ref, test, overlay], axis=1)


def _score_color(score: float) -> Tuple[int, int, int]:
    if score >= config.REFERENCE_BUG_DIFF_THRESHOLD:
        return (0, 0, 255)
    if score >= config.REFERENCE_SUSPICIOUS_DIFF_THRESHOLD:
        return (0, 165, 255)
    return (0, 200, 0)


def _empty(reason: str) -> dict:
    return {
        "status": reason,
        "reference_match_quality": 0.0,
        "overall_diff_score": 0.0,
        "diff_coverage": 0.0,
        "candidate_count": 0,
        "candidates": [],
        "diff_mask": None,
        "score_map": None,
        "diff_overlay": None,
        "side_by_side": None,
    }
