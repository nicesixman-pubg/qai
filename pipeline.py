"""qAI pipeline orchestration."""
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from agents import (anomaly_scoring_agent, avatar_detection_agent,
                    decision_fusion_agent, frame_quality_agent, report_agent,
                    roi_agent, segmentation_agent, vlm_adjudication_agent)
from cv import heatmap as heatmap_cv
from cv import mask_utils, reference_diff, roi_utils


def imread_unicode(path: str) -> Optional[np.ndarray]:
    """Read images safely from non-ASCII Windows paths."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return cv2.imread(path)


class _Stopwatch:
    def __init__(self):
        self.timings: dict = {}

    def time(self, name: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        self.timings[name] = round((time.perf_counter() - t0) * 1000, 1)
        return result


def _process_avatar(frame: np.ndarray, avatar: dict, quality: dict) -> dict:
    sw = _Stopwatch()
    crop, ext_bbox = mask_utils.crop_with_margin(frame, avatar["bbox"])

    yolo_mask_crop = None
    if avatar.get("mask") is not None:
        x1, y1, x2, y2 = ext_bbox
        yolo_mask_crop = avatar["mask"][y1:y2, x1:x2]

    seg = sw.time("segmentation_ms", segmentation_agent.run, crop, yolo_mask_crop)
    rois = sw.time("roi_ms", roi_agent.run, crop)
    scoring = sw.time("scoring_ms", anomaly_scoring_agent.run, crop, seg["refined_mask"], rois)
    roi_crops = {r["name"]: roi_utils.crop_box(crop, r["box"]) for r in scoring["roi_results"]}
    hm = sw.time("heatmap_ms", heatmap_cv.render, crop, scoring["roi_results"])

    pre = decision_fusion_agent.pre_vlm_decision(
        scoring["avatar_bug_score"], quality, seg["mask_confidence"])
    numeric = {
        "avatar_bug_score": scoring["avatar_bug_score"],
        "worst_roi": scoring["worst_roi"],
        "roi_scores": {r["name"]: r["bug_score"] for r in scoring["roi_results"]},
    }
    vlm = sw.time(
        "vlm_ms",
        vlm_adjudication_agent.run,
        original_crop=crop,
        mask_overlay=seg["overlay"],
        roi_crops=roi_crops,
        heatmap=hm,
        numeric_scores=numeric,
        cv_decision=pre,
    )
    final = decision_fusion_agent.final_decision(
        pre, vlm, scoring["avatar_bug_score"], quality,
        seg["mask_confidence"], scoring["worst_roi"])

    sw.timings["total_ms"] = round(sum(sw.timings.values()), 1)
    n_images = 3 + sum(1 for c in roi_crops.values() if c is not None and c.size > 0)
    return {
        "avatar": avatar,
        "crop": crop,
        "ext_bbox": ext_bbox,
        "seg": seg,
        "rois": rois,
        "scoring": scoring,
        "roi_crops": roi_crops,
        "heatmap": hm,
        "pre": pre,
        "vlm": vlm,
        "final": final,
        "timings": sw.timings,
        "vlm_image_count": n_images,
    }


def _primary_avatar(frame: np.ndarray, manual_bbox=None,
                    whole_frame_fallback: bool = False) -> Optional[dict]:
    avatars = avatar_detection_agent.run(frame)
    if not avatars and manual_bbox is not None:
        avatars = [avatar_detection_agent.manual_avatar(manual_bbox)]
    if not avatars and whole_frame_fallback:
        h, w = frame.shape[:2]
        avatars = [avatar_detection_agent.manual_avatar(
            (int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)))]
    return avatars[0] if avatars else None


def _seg_for_avatar(frame: np.ndarray, avatar: dict) -> tuple[np.ndarray, tuple, dict]:
    crop, ext_bbox = mask_utils.crop_with_margin(frame, avatar["bbox"])
    yolo_mask_crop = None
    if avatar.get("mask") is not None:
        x1, y1, x2, y2 = ext_bbox
        yolo_mask_crop = avatar["mask"][y1:y2, x1:x2]
    seg = segmentation_agent.run(crop, yolo_mask_crop)
    return crop, ext_bbox, seg


def _reference_payload(ref_status: dict, diff: dict) -> dict:
    candidates = [
        {
            "candidate_id": c.get("candidate_id"),
            "bbox": c.get("bbox"),
            "area_px": c.get("area_px"),
            "diff_score": c.get("diff_score"),
            "mean_delta": c.get("mean_delta"),
            "p90_delta": c.get("p90_delta"),
            "region_hint": c.get("region_hint"),
        }
        for c in diff.get("candidates", [])
    ]
    return {
        "workflow": "reference",
        "reference_status": ref_status.get("status", "ok"),
        "reference_reason": ref_status.get("reason", ""),
        "reference_match_quality": diff.get("reference_match_quality", 0.0),
        "overall_diff_score": diff.get("overall_diff_score", 0.0),
        "diff_coverage": diff.get("diff_coverage", 0.0),
        "candidate_count": diff.get("candidate_count", 0),
        "candidates": candidates,
    }


def _process_reference_avatar(frame: np.ndarray, reference_frame: np.ndarray, avatar: dict,
                              reference_avatar: dict, quality: dict, ref_status: dict) -> dict:
    sw = _Stopwatch()
    crop, ext_bbox, seg = sw.time("test_segmentation_ms", _seg_for_avatar, frame, avatar)
    ref_crop, ref_ext_bbox, ref_seg = sw.time(
        "reference_segmentation_ms", _seg_for_avatar, reference_frame, reference_avatar)
    diff = sw.time(
        "reference_diff_ms",
        reference_diff.analyze,
        crop,
        ref_crop,
        seg["refined_mask"],
        ref_seg["refined_mask"],
    )
    candidate_crops = reference_diff.crop_candidates(crop, ref_crop, diff.get("candidates", []))
    payload = _reference_payload(ref_status, diff)
    vlm = sw.time(
        "vlm_ms",
        vlm_adjudication_agent.run_reference,
        reference_crop=ref_crop,
        test_crop=crop,
        diff_overlay=diff["diff_overlay"],
        side_by_side=diff["side_by_side"],
        candidate_crops=candidate_crops,
        diff_payload=payload,
    )
    final = decision_fusion_agent.final_reference_decision(diff, vlm, quality)
    sw.timings["total_ms"] = round(sum(sw.timings.values()), 1)
    n_images = 4 + sum(1 for c in candidate_crops.values() if c is not None and c.size > 0)
    return {
        "avatar": avatar,
        "reference_avatar": reference_avatar,
        "crop": crop,
        "reference_crop": ref_crop,
        "ext_bbox": ext_bbox,
        "reference_ext_bbox": ref_ext_bbox,
        "seg": seg,
        "reference_seg": ref_seg,
        "reference_status": ref_status,
        "reference_analysis": diff,
        "candidate_crops": candidate_crops,
        "vlm": vlm,
        "final": final,
        "timings": sw.timings,
        "vlm_image_count": n_images,
    }


def run_qai_reference_case(image_path: str, reference_image_path: str,
                           manual_bbox: Optional[Tuple[int, int, int, int]] = None,
                           whole_frame_fallback: bool = False) -> dict:
    frame = imread_unicode(image_path)
    reference_frame = imread_unicode(reference_image_path)
    ref_status = {
        "status": "ok" if reference_frame is not None else "missing_reference",
        "reference_path": reference_image_path if reference_frame is not None else "",
        "reason": "explicit_reference",
    }
    if frame is None:
        return {
            "case_id": report_agent.new_case_id(),
            "input": image_path,
            "reference": reference_image_path,
            "final_label": "suspicious",
            "overall_score": 0.0,
            "decision_reason": "image load failed",
            "avatars": [],
            "needs_human_review": True,
            "safe_to_autopass": False,
        }

    quality = frame_quality_agent.run(frame)
    if not quality["can_continue"]:
        rep = report_agent.unusable_case(image_path, quality)
        rep["reference"] = reference_image_path
        return rep
    if reference_frame is None:
        rep = report_agent.unusable_case(image_path, quality)
        rep["reference"] = reference_image_path
        rep["decision_reason"] = "missing reference image; human review required"
        return rep

    avatar = _primary_avatar(frame, manual_bbox, whole_frame_fallback)
    ref_avatar = _primary_avatar(reference_frame, manual_bbox, whole_frame_fallback)
    quality = frame_quality_agent.augment_with_detection(
        quality, [avatar] if avatar else [], frame.shape)
    if avatar is None or ref_avatar is None:
        rep = report_agent.unusable_case(image_path, quality)
        rep["reference"] = reference_image_path
        rep["decision_reason"] = "avatar missing in test or reference; human review required"
        return rep

    avatar_case = _process_reference_avatar(frame, reference_frame, avatar, ref_avatar, quality, ref_status)
    return report_agent.generate(image_path, quality, [avatar_case], reference_path=reference_image_path)


def run_qai_case(image_path: str, manual_bbox: Optional[Tuple[int, int, int, int]] = None,
                 whole_frame_fallback: bool = False, reference_image_path: Optional[str] = None,
                 mode: Optional[str] = None, allow_standalone_fallback: bool = True) -> dict:
    mode = mode or ("reference" if reference_image_path else "standalone")
    if mode == "reference":
        if reference_image_path:
            return run_qai_reference_case(image_path, reference_image_path, manual_bbox, whole_frame_fallback)
        if not allow_standalone_fallback:
            return {
                "case_id": report_agent.new_case_id(),
                "input": image_path,
                "reference": None,
                "final_label": "suspicious",
                "overall_score": 0.0,
                "decision_reason": "reference mode requested but no reference image was provided",
                "avatars": [],
                "needs_human_review": True,
                "safe_to_autopass": False,
            }

    frame = imread_unicode(image_path)
    if frame is None:
        return {
            "case_id": report_agent.new_case_id(),
            "input": image_path,
            "reference": None,
            "final_label": "suspicious",
            "overall_score": 0.0,
            "decision_reason": "image load failed",
            "avatars": [],
            "needs_human_review": True,
            "safe_to_autopass": False,
        }

    quality = frame_quality_agent.run(frame)
    if not quality["can_continue"]:
        return report_agent.unusable_case(image_path, quality)

    avatars = avatar_detection_agent.run(frame)
    if not avatars and manual_bbox is not None:
        avatars = [avatar_detection_agent.manual_avatar(manual_bbox)]
    if not avatars and whole_frame_fallback:
        h, w = frame.shape[:2]
        avatars = [avatar_detection_agent.manual_avatar(
            (int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)))]

    quality = frame_quality_agent.augment_with_detection(quality, avatars, frame.shape)
    if not avatars:
        rep = report_agent.unusable_case(image_path, quality)
        rep["decision_reason"] = "avatar not detected; human review required"
        return rep

    avatar_cases = [_process_avatar(frame, a, quality) for a in avatars]
    return report_agent.generate(image_path, quality, avatar_cases)
