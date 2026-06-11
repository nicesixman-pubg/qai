"""Deterministic no-API provider for smoke tests and local plumbing checks."""
from __future__ import annotations

from typing import Dict, List

import config
from .base import VLMProvider, validate


class MockProvider(VLMProvider):
    def adjudicate(self, images: List[Dict], payload: dict) -> dict:
        if payload.get("workflow") == "reference":
            return validate(_reference_mock(payload))
        return validate(_standalone_mock(payload))


def _reference_mock(payload: dict) -> dict:
    status = payload.get("reference_status", "ok")
    match_quality = float(payload.get("reference_match_quality") or 0.0)
    diff_score = float(payload.get("overall_diff_score") or 0.0)
    candidates = payload.get("candidates") or []
    reviewed = [c.get("candidate_id", "") for c in candidates if c.get("candidate_id")]

    if status != "ok" or match_quality < config.REFERENCE_MATCH_MIN:
        label = "suspicious"
        safe = False
        reason = f"[MOCK] reference not reliable: status={status}, match={match_quality:.2f}"
    elif diff_score >= config.REFERENCE_BUG_DIFF_THRESHOLD:
        label = "bug"
        safe = False
        reason = f"[MOCK] reference diff above bug threshold: {diff_score:.2f}"
    elif diff_score >= config.REFERENCE_SUSPICIOUS_DIFF_THRESHOLD:
        label = "suspicious"
        safe = False
        reason = f"[MOCK] reference diff requires review: {diff_score:.2f}"
    else:
        label = "normal"
        safe = True
        reason = f"[MOCK] reference diff below auto-pass threshold: {diff_score:.2f}"

    defects = []
    if label == "bug" and candidates:
        c = candidates[0]
        defects.append({
            "candidate_id": c.get("candidate_id", "cand_001"),
            "bbox": c.get("bbox", [0, 0, 0, 0]),
            "region_hint": c.get("region_hint", "other"),
            "bug_type": "missing_mesh",
            "severity": "high",
            "evidence": "Top reference diff candidate exceeds bug threshold.",
        })

    return {
        "final_label": label,
        "safe_to_autopass": safe,
        "confidence": 0.8 if label != "suspicious" else 0.55,
        "reference_match_quality": match_quality,
        "candidate_ids_reviewed": reviewed,
        "defects": defects,
        "affected_regions": [d["region_hint"] for d in defects],
        "bug_types": [d["bug_type"] for d in defects],
        "visual_evidence": [
            {
                "region": d["region_hint"],
                "description": d["evidence"],
                "severity": d["severity"],
            }
            for d in defects
        ],
        "intentional_design_possible": False,
        "needs_human_review": label == "suspicious",
        "reason": reason,
    }


def _standalone_mock(payload: dict) -> dict:
    band = payload.get("cv_band", "low")
    worst = payload.get("worst_roi")
    label = {"high": "bug", "medium": "suspicious", "low": "normal"}.get(band, "suspicious")
    safe = label == "normal"
    return {
        "final_label": label,
        "safe_to_autopass": safe,
        "confidence": {"high": 0.8, "medium": 0.55, "low": 0.6}.get(band, 0.5),
        "reference_match_quality": 0.0,
        "candidate_ids_reviewed": [],
        "defects": [],
        "affected_regions": [worst] if (worst and band != "low") else [],
        "bug_types": ["background_leakage"] if band == "high" else [],
        "visual_evidence": (
            [{"region": worst or "other", "description": "CV anomaly detected (mock)", "severity": band}]
            if band != "low" else []
        ),
        "intentional_design_possible": band == "medium",
        "needs_human_review": band != "low",
        "reason": f"[MOCK] standalone CV band={band}; not accuracy-valid.",
    }
