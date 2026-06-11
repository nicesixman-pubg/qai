"""스모크 테스트 — 무거운 ML 의존성 없이 핵심 로직 검증.

필요 패키지: numpy, opencv-python, pydantic (+ python-dotenv 선택).
YOLO/rembg/VLM API 없이도 동작하도록 manual_bbox + mock provider를 쓴다.

실행: python eval/smoke_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QAI_VLM_PROVIDER", "mock")  # config import 전에 설정
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import config  # noqa: E402


def synth_avatar(with_hole: bool) -> "np.ndarray":
    """텍스처 배경 위에 단색 인물 실루엣. with_hole이면 허리에 배경이 비치는 구멍."""
    import cv2

    h, w = 600, 320
    # 배경: 청색 노이즈 텍스처
    rng = np.random.RandomState(0)
    bg = rng.randint(120, 180, (h, w, 3), dtype=np.uint8)
    bg[..., 0] = np.clip(bg[..., 0] + 40, 0, 255)  # 푸른 톤
    img = bg.copy()

    # 인물: 갈색 몸통 + 머리
    body_color = (60, 90, 140)
    cv2.rectangle(img, (110, 150), (210, 480), body_color, -1)   # 몸통
    cv2.circle(img, (160, 110), 45, body_color, -1)              # 머리
    cv2.rectangle(img, (120, 480), (155, 580), body_color, -1)   # 다리L
    cv2.rectangle(img, (165, 480), (200, 580), body_color, -1)   # 다리R

    if with_hole:
        # 허리(y ~0.45*H)에 배경을 그대로 노출하는 구멍 → hole + bg_leakage 신호
        img[300:340, 140:185] = bg[300:340, 140:185]
    return img


def crafted_mask_test():
    """cv 스코어 함수 직접 검증 — 손으로 만든 마스크/이미지.

    두 신호는 상보적이다:
      - internal_hole: 마스크 자체에 구멍(세그가 배경으로 판정) → hole 점수
      - background_leakage: 마스크는 전경인데 색이 배경 → leakage 점수
    """
    from cv import background_leakage, hole_detection

    h, w = 200, 200
    rng = np.random.RandomState(1)
    bg = rng.randint(100, 160, (h, w, 3), dtype=np.uint8)
    roi = (50, 70, 150, 150)

    # (1) 홀 케이스: 마스크에 구멍
    mask_hole = np.zeros((h, w), np.uint8)
    mask_hole[40:160, 60:140] = 255
    mask_hole[90:120, 90:120] = 0
    hole_s, _ = hole_detection.internal_hole_score(mask_hole, roi)
    print(f"[crafted] internal_hole_score={hole_s}")
    assert hole_s > 0, "내부 홀 점수가 0 — hole_detection 이상"

    # (2) 누수 케이스: 마스크는 솔리드(전경)인데 영역 색이 배경
    img = bg.copy()
    img[40:160, 60:140] = (50, 80, 130)          # 전경 몸통 색
    mask_solid = np.zeros((h, w), np.uint8)
    mask_solid[40:160, 60:140] = 255             # 구멍 없음
    img[95:125, 95:125] = bg[95:125, 95:125]     # 전경 안인데 배경색 비침
    bg_s, _ = background_leakage.background_leakage_score(img, mask_solid, roi)
    print(f"[crafted] background_leakage_score={bg_s}")
    assert bg_s > 0, "배경 비침 점수가 0 — background_leakage 이상"
    print("[crafted] PASS")


def fusion_test():
    """보수적 융합 규칙: CV high + VLM normal → suspicious (정상 자동통과 금지)."""
    from agents import decision_fusion_agent as df

    quality = {"frame_quality": "good", "issues": []}
    pre = df.pre_vlm_decision(0.9, quality, mask_quality=0.9)   # high
    assert pre["cv_band"] == "high", pre
    final = df.final_decision(pre, {"final_label": "normal"}, 0.9, quality, 0.9, "waist")
    print(f"[fusion] CV=high + VLM=normal → {final['final_label']}")
    assert final["final_label"] == "suspicious", "VLM이 강한 CV 버그를 normal로 내림 — 규칙 위반!"

    # CV low + VLM normal → normal
    pre2 = df.pre_vlm_decision(0.1, quality, 0.9)
    final2 = df.final_decision(pre2, {"final_label": "normal"}, 0.1, quality, 0.9, "waist")
    assert final2["final_label"] == "normal", final2
    print("[fusion] CV=low + VLM=normal → normal  PASS")


def pipeline_test():
    """end-to-end (manual_bbox + grabcut fallback + mock VLM)."""
    import cv2

    from pipeline import run_qai_case

    config.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    normal_p = str(config.INPUT_DIR / "_smoke_normal.png")
    bug_p = str(config.INPUT_DIR / "_smoke_bug.png")
    cv2.imwrite(normal_p, synth_avatar(False))
    cv2.imwrite(bug_p, synth_avatar(True))

    bbox = (90, 60, 230, 590)
    for name, path in [("normal", normal_p), ("bug", bug_p)]:
        rep = run_qai_case(path, manual_bbox=bbox)
        a = rep["avatars"][0]
        print(f"[pipeline:{name}] label={rep['final_label']} score={rep['overall_score']} "
              f"worst_roi_scores={ {r['name']: r['bug_score'] for r in a['rois']} }")
        assert rep["avatars"], "아바타 케이스 없음"
        assert "report.json" in os.listdir(rep["output_dir"]), "리포트 저장 실패"
    print("[pipeline] PASS (리포트/증거 생성 확인)")


def reference_pipeline_test():
    """end-to-end reference-mode smoke test."""
    import cv2

    from pipeline import run_qai_case

    config.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    normal_p = str(config.INPUT_DIR / "_smoke_ref_normal.png")
    bug_p = str(config.INPUT_DIR / "_smoke_ref_bug.png")
    cv2.imwrite(normal_p, synth_avatar(False))
    cv2.imwrite(bug_p, synth_avatar(True))

    bbox = (90, 60, 230, 590)
    normal_rep = run_qai_case(normal_p, manual_bbox=bbox, reference_image_path=normal_p, mode="reference")
    bug_rep = run_qai_case(bug_p, manual_bbox=bbox, reference_image_path=normal_p, mode="reference")
    print(f"[reference:normal] label={normal_rep['final_label']} score={normal_rep['overall_score']}")
    print(f"[reference:bug] label={bug_rep['final_label']} score={bug_rep['overall_score']}")
    assert normal_rep["final_label"] == "normal", normal_rep
    assert bug_rep["final_label"] != "normal", bug_rep
    print("[reference] PASS")


if __name__ == "__main__":
    print("== qAI 스모크 테스트 ==")
    crafted_mask_test()
    fusion_test()
    pipeline_test()
    reference_pipeline_test()
    print("\n✅ 전체 통과")
