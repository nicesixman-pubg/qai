"""결정 융합 에이전트 — CV와 VLM을 보수적으로 결합.

핵심 규칙: VLM이 CV가 강하게 잡은 버그를 normal로 내릴 수 없다.
불확실/불일치/저품질이면 suspicious.
"""
from __future__ import annotations

import config


def pre_vlm_decision(avatar_bug_score: float, frame_quality: dict, mask_quality: float) -> dict:
    """VLM 호출 전 CV 단독 사전판정."""
    band = config.cv_band(avatar_bug_score)
    label = {"high": "bug", "medium": "suspicious", "low": "normal"}[band]
    notes = []
    if frame_quality.get("frame_quality") != "good":
        notes.append("frame_quality_degraded")
    if mask_quality < config.MASK_QUALITY_MIN:
        notes.append("weak_segmentation")
    return {"cv_band": band, "cv_label": label, "notes": notes}


# (cv_band, vlm_label) → 최종 라벨
_FUSION = {
    ("high", "bug"): "bug",
    ("high", "suspicious"): "bug",
    ("high", "normal"): "suspicious",       # VLM이 강한 CV 버그를 못 내림
    ("medium", "bug"): "suspicious",        # ROI에 따라 bug 승급 가능(아래 보정)
    ("medium", "suspicious"): "suspicious",
    ("medium", "normal"): "suspicious",
    ("low", "bug"): "suspicious",
    ("low", "suspicious"): "suspicious",
    ("low", "normal"): "normal",
}


def final_decision(pre: dict, vlm_result: dict, avatar_bug_score: float,
                   frame_quality: dict, mask_quality: float, worst_roi: str = None) -> dict:
    band = pre["cv_band"]
    vlm_label = (vlm_result or {}).get("final_label", "suspicious")
    label = _FUSION.get((band, vlm_label), "suspicious")

    reasons = []

    # 프레임 품질 나쁨 → 절대 normal 자동통과 금지
    if frame_quality.get("frame_quality") == "unusable":
        label = "suspicious"
        reasons.append("프레임 품질 부족 — 더 나은 이미지 필요")
    elif frame_quality.get("frame_quality") == "weak" and label == "normal":
        label = "suspicious"
        reasons.append("프레임 품질 약함 → 자동통과 보류")

    # 세그 품질 약한데 이상 존재 → suspicious 하한
    if mask_quality < config.MASK_QUALITY_MIN and avatar_bug_score >= config.SUSPICIOUS_THRESHOLD:
        if label == "normal":
            label = "suspicious"
        reasons.append("세그멘테이션 신뢰 낮음")

    # medium + VLM bug 이고 핵심 ROI(허리/발목 등 배경비침 잘 나는 곳)면 bug 승급
    if band == "medium" and vlm_label == "bug" and worst_roi in {"waist", "right_ankle", "left_ankle"}:
        label = "bug"
        reasons.append(f"핵심 ROI({worst_roi})에서 VLM이 버그 확정")

    needs_review = label == "suspicious" or (vlm_result or {}).get("needs_human_review", False)

    if not reasons:
        reasons.append((vlm_result or {}).get("reason", "CV·VLM 합의"))

    return {
        "final_label": label,
        "cv_band": band,
        "vlm_label": vlm_label,
        "avatar_bug_score": round(avatar_bug_score, 4),
        "needs_human_review": needs_review,
        "decision_reason": " / ".join(reasons),
    }


def final_reference_decision(reference_analysis: dict, vlm_result: dict, frame_quality: dict) -> dict:
    """Conservative final decision for reference-based comparison."""
    status = reference_analysis.get("status", "unknown")
    diff_score = float(reference_analysis.get("overall_diff_score") or 0.0)
    match_quality = float(reference_analysis.get("reference_match_quality") or 0.0)
    vlm_label = (vlm_result or {}).get("final_label", "suspicious")
    safe_to_autopass = bool((vlm_result or {}).get("safe_to_autopass", False))
    reasons = []

    if frame_quality.get("frame_quality") == "unusable":
        return {
            "final_label": "suspicious",
            "cv_band": "reference",
            "vlm_label": vlm_label,
            "avatar_bug_score": round(diff_score, 4),
            "needs_human_review": True,
            "safe_to_autopass": False,
            "decision_reason": "frame quality unusable; human review required",
        }

    if status != "ok":
        reasons.append(f"reference_status={status}")
        label = "suspicious"
    elif match_quality < config.REFERENCE_MATCH_MIN:
        reasons.append(f"weak reference match ({match_quality:.2f})")
        label = "suspicious"
    elif vlm_label == "bug":
        label = "bug"
        reasons.append((vlm_result or {}).get("reason", "VLM confirmed reference difference as bug"))
    elif diff_score >= config.REFERENCE_BUG_DIFF_THRESHOLD:
        label = "suspicious"
        reasons.append(f"high reference diff ({diff_score:.2f}) without VLM bug confirmation")
    elif vlm_label == "suspicious":
        label = "suspicious"
        reasons.append((vlm_result or {}).get("reason", "VLM requested review"))
    elif diff_score <= config.REFERENCE_AUTO_PASS_DIFF_THRESHOLD and safe_to_autopass:
        label = "normal"
        reasons.append("reference diff clean and VLM marked safe_to_autopass")
    else:
        label = "suspicious"
        reasons.append(f"diff_score={diff_score:.2f}, safe_to_autopass={safe_to_autopass}")

    if frame_quality.get("frame_quality") == "weak" and label == "normal":
        label = "suspicious"
        reasons.append("weak frame quality blocks auto-pass")

    needs_review = label == "suspicious" or (vlm_result or {}).get("needs_human_review", False)
    return {
        "final_label": label,
        "cv_band": "reference",
        "vlm_label": vlm_label,
        "avatar_bug_score": round(diff_score, 4),
        "needs_human_review": needs_review,
        "safe_to_autopass": label == "normal" and safe_to_autopass,
        "decision_reason": " / ".join(reasons),
    }
