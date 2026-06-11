"""qAI 전역 설정 — 임계값, 가중치, 경로, VLM provider 상수.

모든 매직넘버는 여기서 관리한다. 16~20h 튜닝 단계에서 이 파일만 손대면 된다.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv 미설치여도 동작
    pass


# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
REFERENCE_DIR = DATA_DIR / "reference"
LABELS_DIR = DATA_DIR / "labels"

for _d in (INPUT_DIR, OUTPUT_DIR, REFERENCE_DIR, LABELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 탐지 / 세그멘테이션
# ---------------------------------------------------------------------------
YOLO_WEIGHTS = os.getenv("QAI_YOLO_WEIGHTS", "yolo11n-seg.pt")  # 최초 1회 자동 다운로드
YOLO_PERSON_CLASS = 0           # COCO person
DETECTOR_CONF_THRESHOLD = 0.25
CROP_MARGIN = 0.15              # 10~18% — 실루엣 경계 증거 보존 (타이트 크롭 금지)
MIN_AVATAR_AREA_RATIO = 0.01   # 프레임 대비 아바타 최소 크기


# ---------------------------------------------------------------------------
# 프레임 품질
# ---------------------------------------------------------------------------
BLUR_VAR_THRESHOLD = 60.0      # Laplacian 분산이 이보다 낮으면 블러
DARK_MEAN_THRESHOLD = 40.0     # 평균 밝기(0~255)
EDGE_CUTOFF_MARGIN = 3         # 아바타 bbox가 프레임 경계에 이만큼 붙으면 잘림


# ---------------------------------------------------------------------------
# 이상 점수 가중치 (MVP — 레퍼런스/멀티뷰 없음)
#   roi_bug_score = Σ weight_i * signal_i
# ---------------------------------------------------------------------------
SCORE_WEIGHTS = {
    "background_leakage": 0.35,
    "internal_hole": 0.30,
    "boundary_break": 0.20,
    "skin_exposure": 0.10,
    "asymmetry": 0.05,
}

# 미세 홀 무시: ROI 마스크 면적 대비 이 비율 미만의 홀은 노이즈로 간주
MIN_HOLE_AREA_RATIO = 0.004


# ---------------------------------------------------------------------------
# 판정 임계값
# ---------------------------------------------------------------------------
BUG_THRESHOLD = 0.72
SUSPICIOUS_THRESHOLD = 0.40
MASK_QUALITY_MIN = 0.55        # 이보다 낮으면 세그 신뢰 약함 → 보수적 처리


def cv_band(score: float) -> str:
    """roi_bug_score → CV 밴드(high/medium/low)."""
    if score >= BUG_THRESHOLD:
        return "high"
    if score >= SUSPICIOUS_THRESHOLD:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Reference-first workflow
# ---------------------------------------------------------------------------
QAI_WORKFLOW_MODE = os.getenv("QAI_WORKFLOW_MODE", "reference")  # reference | standalone

# Diff pixels below this score are ignored before component grouping.
REFERENCE_DIFF_PIXEL_THRESHOLD = float(os.getenv("QAI_REFERENCE_DIFF_PIXEL_THRESHOLD", "0.22"))
REFERENCE_DIFF_MIN_AREA_RATIO = float(os.getenv("QAI_REFERENCE_DIFF_MIN_AREA_RATIO", "0.0008"))
REFERENCE_DIFF_AREA_SATURATION = float(os.getenv("QAI_REFERENCE_DIFF_AREA_SATURATION", "0.035"))
REFERENCE_DIFF_TOP_K = int(os.getenv("QAI_REFERENCE_DIFF_TOP_K", "4"))

# Case-level diff bands. The final decision also considers VLM output; these
# numbers are evidence bands, not standalone truth labels.
REFERENCE_AUTO_PASS_DIFF_THRESHOLD = float(os.getenv("QAI_REFERENCE_AUTO_PASS_DIFF_THRESHOLD", "0.08"))
REFERENCE_SUSPICIOUS_DIFF_THRESHOLD = float(os.getenv("QAI_REFERENCE_SUSPICIOUS_DIFF_THRESHOLD", "0.18"))
REFERENCE_BUG_DIFF_THRESHOLD = float(os.getenv("QAI_REFERENCE_BUG_DIFF_THRESHOLD", "0.42"))
REFERENCE_MATCH_MIN = float(os.getenv("QAI_REFERENCE_MATCH_MIN", "0.45"))


# ---------------------------------------------------------------------------
# VLM provider
# ---------------------------------------------------------------------------
VLM_PROVIDER = os.getenv("QAI_VLM_PROVIDER", "claude")   # claude | openai | mock
VLM_MODEL = os.getenv("QAI_VLM_MODEL", "claude-fable-5")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "") or None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "") or None
OPENAI_MODEL = os.getenv("QAI_OPENAI_MODEL", "gpt-5.5")

VLM_MAX_RETRIES = 3
VLM_MAX_TOKENS = 1024


# ---------------------------------------------------------------------------
# ROI 정의 (bbox 비율 fallback) — (x1%, y1%, x2%, y2%)
# ---------------------------------------------------------------------------
ROI_RATIOS = {
    "waist":        (0.20, 0.42, 0.80, 0.60),
    "left_wrist":   (0.00, 0.42, 0.35, 0.72),
    "right_wrist":  (0.65, 0.42, 1.00, 0.72),
    "left_ankle":   (0.18, 0.78, 0.50, 0.98),
    "right_ankle":  (0.50, 0.78, 0.82, 0.98),
    "neck":         (0.35, 0.12, 0.65, 0.28),
}

# 좌우 대칭쌍 (asymmetry 계산용)
ROI_SYMMETRY_PAIRS = [("left_wrist", "right_wrist"), ("left_ankle", "right_ankle")]
