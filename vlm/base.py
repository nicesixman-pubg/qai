"""Common VLM provider interface, image encoding, and schema validation."""
from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from typing import Dict, List

import cv2
import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator

import config

PROMPTS_DIR = config.ROOT / "prompts"


class VisualEvidence(BaseModel):
    region: str
    description: str
    severity: str = "medium"


class DefectEvidence(BaseModel):
    candidate_id: str = ""
    bbox: List[int] = Field(default_factory=lambda: [0, 0, 0, 0])
    region_hint: str = "other"
    bug_type: str = "segmentation_uncertain"
    severity: str = "medium"
    evidence: str = ""


class VLMResult(BaseModel):
    final_label: str
    safe_to_autopass: bool = False
    confidence: float = 0.5
    reference_match_quality: float = 0.0
    candidate_ids_reviewed: List[str] = Field(default_factory=list)
    defects: List[DefectEvidence] = Field(default_factory=list)
    affected_regions: List[str] = Field(default_factory=list)
    bug_types: List[str] = Field(default_factory=list)
    visual_evidence: List[VisualEvidence] = Field(default_factory=list)
    intentional_design_possible: bool = False
    needs_human_review: bool = True
    reason: str = ""

    @field_validator("final_label")
    @classmethod
    def _check_label(cls, v):
        v = (v or "").lower().strip()
        return v if v in {"normal", "suspicious", "bug"} else "suspicious"


def load_prompt() -> str:
    return (PROMPTS_DIR / "vlm_bug_adjudication.txt").read_text(encoding="utf-8")


def load_schema() -> dict:
    return json.loads((PROMPTS_DIR / "vlm_schema.json").read_text(encoding="utf-8"))


def encode_png_b64(image_bgr: np.ndarray, max_side: int = 768) -> str:
    """Encode a BGR image as resized PNG base64."""
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_user_text(payload: dict) -> str:
    workflow = payload.get("workflow", "standalone")
    if workflow == "reference":
        intro = (
            "Reference-based CV evidence follows. Compare TEST against KNOWN-GOOD "
            "REFERENCE. Review candidate boxes and decide whether the differences "
            "are real avatar defects.\n"
        )
    else:
        intro = "Standalone CV anomaly evidence follows. Use this only as fallback evidence.\n"
    return intro + json.dumps(payload, ensure_ascii=False, indent=2)


def validate(raw: dict) -> dict:
    """Return a validated result dict; invalid model output becomes suspicious."""
    try:
        return VLMResult(**raw).model_dump()
    except (ValidationError, TypeError):
        return VLMResult(
            final_label="suspicious",
            safe_to_autopass=False,
            confidence=0.3,
            needs_human_review=True,
            reason="VLM output schema validation failed; conservative suspicious fallback",
        ).model_dump()


class VLMProvider(ABC):
    @abstractmethod
    def adjudicate(self, images: List[Dict], payload: dict) -> dict:
        """images: [{name, image(BGR ndarray)}], payload: JSON-serializable evidence."""
        raise NotImplementedError
