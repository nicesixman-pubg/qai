"""신호별 진단 — normal vs bug에서 각 CV 신호의 분포를 비교해 튜닝 방향을 찾는다.

VLM/리포트 저장 없이 detection→seg→roi→scoring만 돌려 빠르게(무료) 신호를 수집.
실행: ./.venv/Scripts/python.exe eval/diagnose.py --dir data/eval_set
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from agents import (anomaly_scoring_agent, avatar_detection_agent,  # noqa: E402
                    roi_agent, segmentation_agent)
from cv import mask_utils  # noqa: E402
from eval.batch_runner import build_items  # noqa: E402
from pipeline import imread_unicode  # noqa: E402

SIGNALS = ["background_leakage", "internal_hole", "boundary_break", "skin_exposure", "asymmetry"]


def score_image(path: str):
    frame = imread_unicode(path)
    if frame is None:
        return None
    avatars = avatar_detection_agent.run(frame)
    if not avatars:
        return {"detected": False}
    a = avatars[0]
    crop, ext = mask_utils.crop_with_margin(frame, a["bbox"])
    ym = a["mask"][ext[1]:ext[3], ext[0]:ext[2]] if a.get("mask") is not None else None
    seg = segmentation_agent.run(crop, ym)
    rois = roi_agent.run(crop)
    sc = anomaly_scoring_agent.run(crop, seg["refined_mask"], rois)
    # ROI 전체에서 신호별 최댓값(= avatar 수준 신호)
    sig_max = {s: 0.0 for s in SIGNALS}
    worst = {"name": None, "score": -1}
    for r in sc["roi_results"]:
        for s in SIGNALS:
            sig_max[s] = max(sig_max[s], r["signals"].get(s, 0.0))
        if r["bug_score"] > worst["score"]:
            worst = {"name": r["name"], "score": r["bug_score"]}
    return {"detected": True, "avatar_score": sc["avatar_bug_score"],
            "worst_roi": worst["name"], "mask_conf": seg["mask_confidence"],
            "method": rois[0]["method"] if rois else "", **sig_max}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=str(config.OUTPUT_DIR / "diagnose.csv"))
    args = ap.parse_args()

    items = build_items(Path(args.dir), {})
    rows = []
    agg = {"normal": defaultdict(list), "bug": defaultdict(list)}
    detected = 0
    roi_method = defaultdict(int)

    for i, it in enumerate(items):
        true = it["meta"]["true_label"]
        res = score_image(str(it["path"]))
        if res is None or not res.get("detected"):
            rows.append({**it["meta"], "detected": False})
            continue
        detected += 1
        roi_method[res["method"]] += 1
        row = {
            "rel_path": it["meta"]["rel_path"], "true": true, "detected": True,
            "avatar_score": res["avatar_score"], "worst_roi": res["worst_roi"],
            "mask_conf": res["mask_conf"], **{s: res[s] for s in SIGNALS},
        }
        rows.append(row)
        if true in agg:
            agg[true]["avatar_score"].append(res["avatar_score"])
            agg[true]["mask_conf"].append(res["mask_conf"])
            for s in SIGNALS:
                agg[true][s].append(res[s])
        if (i + 1) % 24 == 0:
            print(f"  ...{i+1}/{len(items)}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        cols = ["rel_path", "true", "detected", "avatar_score", "worst_roi", "mask_conf"] + SIGNALS
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    print(f"\n탐지: {detected}/{len(items)}  ROI method: {dict(roi_method)}\n")
    print(f"{'신호':<22}{'normal 평균':>12}{'bug 평균':>12}{'분리도(bug-normal)':>18}")
    for key in ["avatar_score", "mask_conf"] + SIGNALS:
        n, b = mean(agg['normal'][key]), mean(agg['bug'][key])
        print(f"{key:<22}{n:>12}{b:>12}{round(b-n,3):>18}")
    print(f"\nCSV: {args.out}")


if __name__ == "__main__":
    main()
