"""아바타 탐지 에이전트 — Ultralytics YOLO-seg(person)로 bbox + 인스턴스 마스크.

CPU에서 한 모델로 탐지+세그를 동시에 얻는다. 미설치/탐지실패 시 빈 결과를 반환하고
UI에서 수동 크롭으로 fallback 한다.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

import config

_model = None  # 지연 로드 캐시


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO  # 지연 import

        _model = YOLO(config.YOLO_WEIGHTS)
    return _model


def run(image_bgr: np.ndarray) -> List[dict]:
    """반환: [{avatar_id, bbox(x1,y1,x2,y2 절대), detector_confidence, mask(uint8 0/255 풀프레임 or None)}]"""
    try:
        model = _get_model()
    except Exception as e:  # ultralytics/torch 미설치 등
        print(f"[avatar_detection] YOLO 로드 실패 → 수동 크롭 fallback 필요: {e}")
        return []

    h, w = image_bgr.shape[:2]
    try:
        results = model.predict(
            source=image_bgr, classes=[config.YOLO_PERSON_CLASS],
            conf=config.DETECTOR_CONF_THRESHOLD, verbose=False,
        )
    except Exception as e:
        print(f"[avatar_detection] 추론 실패: {e}")
        return []

    avatars: List[dict] = []
    for res in results:
        boxes = getattr(res, "boxes", None)
        masks = getattr(res, "masks", None)
        if boxes is None:
            continue
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i].cpu().numpy())
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            mask_full = None
            if masks is not None and masks.data is not None and i < len(masks.data):
                m = masks.data[i].cpu().numpy()
                m = (m > 0.5).astype(np.uint8) * 255
                # YOLO 마스크는 입력 비율로 리사이즈 필요
                if m.shape[:2] != (h, w):
                    import cv2

                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_full = m
            avatars.append({
                "avatar_id": f"avatar_{len(avatars) + 1:03d}",
                "bbox": (max(0, x1), max(0, y1), min(w, x2), min(h, y2)),
                "detector_confidence": round(conf, 4),
                "mask": mask_full,
            })

    # 신뢰도 내림차순
    avatars.sort(key=lambda a: a["detector_confidence"], reverse=True)
    for idx, a in enumerate(avatars):
        a["avatar_id"] = f"avatar_{idx + 1:03d}"
    return avatars


def manual_avatar(bbox) -> dict:
    """UI 수동 크롭 fallback."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return {
        "avatar_id": "avatar_001",
        "bbox": (x1, y1, x2, y2),
        "detector_confidence": 0.0,
        "mask": None,
    }
