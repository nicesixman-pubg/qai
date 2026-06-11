"""Folder batch evaluation runner.

Default mode is reference-based: each image is paired with a known-good
reference through eval.reference_resolver before running the pipeline.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import vlm  # noqa: E402
from eval import metrics  # noqa: E402
from eval.reference_resolver import resolve_reference  # noqa: E402
from pipeline import run_qai_case  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
LABELS = {"normal", "suspicious", "bug"}
ANGLES = {"front", "back", "left", "right"}
META_COLS = ["image", "rel_path", "category", "jira", "item_combo", "gender_skin", "angle", "case_id"]


def load_truth(path: str) -> dict:
    truth = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            truth[Path(row["image"]).name] = row["true_label"].strip().lower()
    return truth


def discover_images(root: Path):
    return sorted((p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT),
                  key=lambda p: str(p).lower())


def parse_meta(path: Path, root: Path) -> dict:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(path.name)
    dir_parts = rel.parts[:-1]

    stem = path.stem
    head, true_label = stem, None
    if "," in stem:
        head, lbl = stem.rsplit(",", 1)
        lbl = lbl.strip().lower()
        if lbl in LABELS:
            true_label = lbl

    tok = head.rsplit("_", 1)[-1].lower() if "_" in head else head.lower()
    angle = tok if tok in ANGLES else ""

    def nth(i):
        return dir_parts[i] if i < len(dir_parts) else ""

    return {
        "image": path.name,
        "rel_path": str(rel).replace("\\", "/"),
        "category": nth(0),
        "jira": nth(1),
        "item_combo": nth(2),
        "gender_skin": nth(3),
        "angle": angle,
        "case_id": str(rel.parent).replace("\\", "/") if dir_parts else "",
        "true_label": true_label,
    }


def build_items(root: Path, truth: dict) -> list:
    items = []
    for p in discover_images(root):
        meta = parse_meta(p, root)
        if meta["true_label"] is None and truth:
            meta["true_label"] = truth.get(p.name)
        items.append({"path": p, "meta": meta})
    return items


def _agg_latency(report: dict) -> dict:
    total = vlm_ms = seg_ms = n_imgs = 0.0
    for a in report.get("avatars", []):
        t = a.get("timings", {})
        total += t.get("total_ms", 0)
        vlm_ms += t.get("vlm_ms", 0)
        seg_ms += t.get("segmentation_ms", 0) or t.get("test_segmentation_ms", 0)
        n_imgs += a.get("vlm_image_count", 0) or 0
    return {"total_ms": round(total, 1), "vlm_ms": round(vlm_ms, 1),
            "seg_ms": round(seg_ms, 1), "n_vlm_images": int(n_imgs)}


def _meta_cols(meta: dict) -> dict:
    return {k: meta.get(k, "") for k in META_COLS}


def _reference_fields(report: dict) -> dict:
    avatar = (report.get("avatars") or [{}])[0]
    ref = avatar.get("reference") or {}
    return {
        "reference_path": report.get("reference") or "",
        "reference_status": ref.get("status", ""),
        "reference_match_quality": avatar.get("reference_match_quality"),
        "reference_diff_score": avatar.get("reference_diff_score"),
        "reference_candidate_count": len(avatar.get("reference_candidates") or []),
        "safe_to_autopass": report.get("safe_to_autopass"),
    }


def _reference_for_item(it: dict, mode: str, eval_root: Path, reference_root: Path | None) -> dict:
    if mode != "reference":
        return {}
    return resolve_reference(it["path"], eval_root=eval_root, reference_root=reference_root)


def _run_case(it: dict, mode: str, eval_root: Path, reference_root: Path | None, whole_frame: bool) -> dict:
    ref = _reference_for_item(it, mode, eval_root, reference_root)
    return run_qai_case(
        str(it["path"]),
        whole_frame_fallback=whole_frame,
        reference_image_path=ref.get("reference_path") or None,
        mode=mode,
        allow_standalone_fallback=False,
    )


def _run_set(items, provider: str, whole_frame: bool, mode: str,
             eval_root: Path, reference_root: Path | None) -> dict:
    config.VLM_PROVIDER = provider
    vlm.reset()
    actual = type(vlm.get_provider()).__name__
    rows, pairs, img_counts = [], [], []

    for it in items:
        report = _run_case(it, mode, eval_root, reference_root, whole_frame)
        pred = report["final_label"]
        true = it["meta"]["true_label"]
        lat = _agg_latency(report)
        if true:
            pairs.append((true, pred))
        if lat["n_vlm_images"]:
            img_counts.append(lat["n_vlm_images"])
        rows.append({
            **_meta_cols(it["meta"]),
            "provider": provider,
            "actual_provider": actual,
            "mode": mode,
            "pred": pred,
            "true": true or "",
            "total_ms": lat["total_ms"],
            "vlm_ms": lat["vlm_ms"],
            "seg_ms": lat["seg_ms"],
            "n_vlm_images": lat["n_vlm_images"],
            "overall_score": report.get("overall_score"),
            **_reference_fields(report),
        })
    return {"provider": provider, "actual_provider": actual,
            "rows": rows, "pairs": pairs, "img_counts": img_counts}


def _avg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def _label_dist(rows):
    d = {"normal": 0, "suspicious": 0, "bug": 0}
    for r in rows:
        if r["pred"] in d:
            d[r["pred"]] += 1
    return d


def run_compare(items, out_path, whole_frame, mode, eval_root, reference_root):
    results = [_run_set(items, p, whole_frame, mode, eval_root, reference_root)
               for p in ("mock", "claude")]
    all_rows = [r for res in results for r in res["rows"]]
    _write_rows(out_path, all_rows)
    print(f"compare CSV: {out_path}\n")

    print(f"{'provider':<10}{'actual':<18}{'avg total ms':>13}{'avg vlm ms':>12}"
          f"{'avg seg ms':>12}   labels(N/S/B)")
    for res in results:
        d = _label_dist(res["rows"])
        print(f"{res['provider']:<10}{res['actual_provider']:<18}"
              f"{_avg(res['rows'],'total_ms'):>13}{_avg(res['rows'],'vlm_ms'):>12}"
              f"{_avg(res['rows'],'seg_ms'):>12}   {d['normal']}/{d['suspicious']}/{d['bug']}")

    for res in results:
        if res["pairs"]:
            print(f"\n[{res['provider']}] metrics\n" + metrics.pretty(metrics.summarize(res["pairs"])))


def run_single(items, out_path, whole_frame, mode, eval_root, reference_root):
    pairs, rows = [], []
    for it in items:
        t0 = time.time()
        report = _run_case(it, mode, eval_root, reference_root, whole_frame)
        dt = time.time() - t0
        pred = report["final_label"]
        true = it["meta"]["true_label"]
        lat = _agg_latency(report)
        if true:
            pairs.append((true, pred))
        row = {
            **_meta_cols(it["meta"]),
            "mode": mode,
            "pred": pred,
            "true": true or "",
            "overall_score": report.get("overall_score"),
            "needs_review": report.get("needs_human_review"),
            **_reference_fields(report),
            "total_ms": lat["total_ms"],
            "vlm_ms": lat["vlm_ms"],
            "wall_seconds": round(dt, 2),
            "qai_case_id": report.get("case_id"),
        }
        rows.append(row)
        print(f"{it['meta']['rel_path']:70s} pred={pred:10s} true={true or '-':10s} "
              f"score={report.get('overall_score')} ({dt:.1f}s)")

    _write_rows(out_path, rows)
    print(f"\nresults CSV: {out_path}")
    if pairs:
        print("\n" + metrics.pretty(metrics.summarize(pairs)))
    else:
        print("\nNo truth labels found; use filename labels or --labels.")


def _write_rows(out_path, rows):
    if not rows:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="image root")
    ap.add_argument("--labels", help="truth CSV with image,true_label")
    ap.add_argument("--compare", action="store_true", help="compare mock vs claude")
    ap.add_argument("--provider", choices=["mock", "claude", "openai"], default=None,
                    help="provider for non-compare runs; defaults to config/env")
    ap.add_argument("--whole-frame", action="store_true",
                    help="fallback to full-frame avatar when detection fails")
    ap.add_argument("--mode", choices=["reference", "standalone"], default="reference")
    ap.add_argument("--reference-root", default=None,
                    help="optional known-good reference mirror")
    ap.add_argument("--limit", type=int, default=0, help="maximum number of images to run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.dir)
    reference_root = Path(args.reference_root) if args.reference_root else None
    truth = load_truth(args.labels) if args.labels else {}
    items = build_items(root, truth)
    if not items:
        print(f"no images found: {root}")
        return
    if args.limit and args.limit > 0:
        items = items[:args.limit]
    labeled = sum(1 for it in items if it["meta"]["true_label"])
    print(f"found {len(items)} images ({labeled} labeled), mode={args.mode}, "
          f"whole_frame={args.whole_frame}\n")

    if args.compare:
        out = args.out or str(config.OUTPUT_DIR / "compare_results.csv")
        run_compare(items, out, args.whole_frame, args.mode, root, reference_root)
    else:
        if args.provider:
            config.VLM_PROVIDER = args.provider
            vlm.reset()
        out = args.out or str(config.OUTPUT_DIR / "batch_results.csv")
        run_single(items, out, args.whole_frame, args.mode, root, reference_root)


if __name__ == "__main__":
    main()
