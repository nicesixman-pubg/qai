"""피드백 에이전트 (스텁) — 사람 검수 결과를 CSV로 누적.

액티브러닝은 업그레이드 경로. 여기서는 라벨만 append 한다.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import config

_FIELDS = ["timestamp", "case_id", "avatar_id", "qai_label", "human_label", "comment"]


def record(case_id: str, avatar_id: str, qai_label: str, human_label: str, comment: str = "") -> str:
    path = Path(config.LABELS_DIR) / "human_feedback.csv"
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "case_id": case_id,
            "avatar_id": avatar_id,
            "qai_label": qai_label,
            "human_label": human_label,
            "comment": comment,
        })
    return str(path)
