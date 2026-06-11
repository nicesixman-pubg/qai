"""Structured report generation and evidence image persistence."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

import config


def _save(img: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return str(path)


def new_case_id() -> str:
    return time.strftime("qai_%Y_%m_%d_") + str(int(time.time()) % 100000).zfill(5)


def unusable_case(image_path: str, quality: dict) -> dict:
    return {
        "case_id": new_case_id(),
        "input": image_path,
        "reference": None,
        "final_label": "suspicious",
        "overall_score": 0.0,
        "decision_reason": "frame unusable or required evidence missing; human review required",
        "frame_quality": quality,
        "avatars": [],
        "needs_human_review": True,
    }


def generate(image_path: str, frame_quality: dict, avatar_cases: List[dict],
             reference_path: str | None = None) -> dict:
    case_id = new_case_id()
    out_dir = config.OUTPUT_DIR / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    avatars_report = []
    overall = 0.0
    any_bug = any(c["final"]["final_label"] == "bug" for c in avatar_cases)
    any_susp = any(c["final"]["final_label"] == "suspicious" for c in avatar_cases)

    for c in avatar_cases:
        aid = c["avatar"]["avatar_id"]
        a_dir = out_dir / aid
        ref_analysis = c.get("reference_analysis") or {}

        paths = {
            "original_crop": _save(c["crop"], a_dir / "original_crop.png"),
            "mask": _save(c["seg"]["refined_mask"], a_dir / "mask.png"),
            "overlay": _save(c["seg"]["overlay"], a_dir / "overlay.png"),
            "bg_removed": _save(c["seg"]["bg_removed"], a_dir / "bg_removed.png"),
        }
        if c.get("heatmap") is not None:
            paths["heatmap"] = _save(c["heatmap"], a_dir / "heatmap.png")
        if c.get("reference_crop") is not None:
            paths["reference_crop"] = _save(c["reference_crop"], a_dir / "reference_crop.png")
        if ref_analysis.get("diff_mask") is not None:
            paths["diff_mask"] = _save(ref_analysis["diff_mask"], a_dir / "diff_mask.png")
        if ref_analysis.get("diff_overlay") is not None:
            paths["diff_overlay"] = _save(ref_analysis["diff_overlay"], a_dir / "diff_overlay.png")
        if ref_analysis.get("side_by_side") is not None:
            paths["side_by_side"] = _save(ref_analysis["side_by_side"], a_dir / "side_by_side.png")

        roi_report = []
        if c.get("scoring"):
            for r in c["scoring"]["roi_results"]:
                crop_img = c["roi_crops"].get(r["name"])
                ev_path = None
                if crop_img is not None and crop_img.size > 0:
                    ev_path = _save(crop_img, a_dir / f"roi_{r['name']}.png")
                roi_report.append({
                    "name": r["name"],
                    "bug_score": r["bug_score"],
                    "method": r.get("method"),
                    "top_signals": r.get("top_signals", {}),
                    "evidence_crop": ev_path,
                })

        candidates_report = []
        for cand in ref_analysis.get("candidates", []):
            cid = cand.get("candidate_id")
            crop_paths = {}
            for name, img in (c.get("candidate_crops") or {}).items():
                if cid and name.startswith(cid) and img is not None and img.size > 0:
                    crop_paths[name] = _save(img, a_dir / f"{name}.png")
            candidates_report.append({**cand, "evidence_crops": crop_paths})

        overall = max(overall, c["final"]["avatar_bug_score"])
        avatars_report.append({
            "avatar_id": aid,
            "bbox": list(c["avatar"]["bbox"]),
            "detector_confidence": c["avatar"].get("detector_confidence"),
            "label": c["final"]["final_label"],
            "mask_confidence": c["seg"]["mask_confidence"],
            "rois": roi_report,
            "reference": c.get("reference_status"),
            "reference_match_quality": ref_analysis.get("reference_match_quality"),
            "reference_diff_score": ref_analysis.get("overall_diff_score"),
            "reference_diff_coverage": ref_analysis.get("diff_coverage"),
            "reference_candidates": candidates_report,
            "vlm": c.get("vlm"),
            "needs_human_review": c["final"]["needs_human_review"],
            "safe_to_autopass": c["final"].get("safe_to_autopass", False),
            "decision_reason": c["final"]["decision_reason"],
            "evidence": paths,
            "timings": c.get("timings", {}),
            "vlm_image_count": c.get("vlm_image_count"),
        })

    final_label = "bug" if any_bug else ("suspicious" if any_susp else "normal")
    report = {
        "case_id": case_id,
        "input": image_path,
        "reference": reference_path,
        "final_label": final_label,
        "overall_score": round(overall, 4),
        "frame_quality": frame_quality,
        "decision_reason": _summarize(avatars_report),
        "avatars": avatars_report,
        "needs_human_review": any(a["needs_human_review"] for a in avatars_report) or any_susp,
        "safe_to_autopass": bool(avatars_report) and all(a.get("safe_to_autopass") for a in avatars_report),
        "output_dir": str(out_dir),
    }

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def _summarize(avatars_report: List[dict]) -> str:
    if not avatars_report:
        return "no avatar detected"
    return " | ".join(
        f"{a['avatar_id']}={a['label']}({a['decision_reason']})"
        for a in avatars_report
    )
