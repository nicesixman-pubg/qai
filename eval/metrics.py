"""평가 지표 — QA용. 단순 정확도가 아니라 버그 recall / FN / 의심 포착률 중심.

라벨 공간: normal | suspicious | bug
"""
from __future__ import annotations

from typing import Dict, List, Tuple

LABELS = ["normal", "suspicious", "bug"]


def confusion_matrix(pairs: List[Tuple[str, str]]) -> Dict[str, Dict[str, int]]:
    """pairs: [(true, pred)] → {true: {pred: count}}"""
    cm = {t: {p: 0 for p in LABELS} for t in LABELS}
    for true, pred in pairs:
        if true in cm and pred in cm[true]:
            cm[true][pred] += 1
    return cm


def summarize(pairs: List[Tuple[str, str]]) -> dict:
    if not pairs:
        return {"n": 0}
    cm = confusion_matrix(pairs)
    n = len(pairs)

    # 버그 recall: 실제 bug 중 bug로 잡은 비율
    bug_total = sum(cm["bug"].values())
    bug_hit = cm["bug"]["bug"]
    bug_recall = bug_hit / bug_total if bug_total else None

    # false negative: 실제 bug인데 normal로 통과시킨 치명적 누락
    bug_as_normal = cm["bug"]["normal"]
    fn_rate = bug_as_normal / bug_total if bug_total else None

    # 의심 포착률: 실제 bug/suspicious 중 normal이 아닌(=잡아낸) 비율
    risky_total = sum(cm["bug"].values()) + sum(cm["suspicious"].values())
    risky_caught = risky_total - cm["bug"]["normal"] - cm["suspicious"]["normal"]
    capture_rate = risky_caught / risky_total if risky_total else None

    # false positive: 실제 normal인데 bug/suspicious로 올린 비율
    normal_total = sum(cm["normal"].values())
    normal_fp = cm["normal"]["bug"] + cm["normal"]["suspicious"]
    fp_rate = normal_fp / normal_total if normal_total else None

    # 정상 자동통과율: 전체 중 normal 예측 비율
    auto_pass = sum(1 for _, p in pairs if p == "normal") / n

    # 수동검수 절감 = 자동통과 비율(검수 불필요로 본 비율)
    review_reduction = auto_pass

    return {
        "n": n,
        "confusion_matrix": cm,
        "bug_recall": _r(bug_recall),
        "false_negative_rate": _r(fn_rate),
        "suspicious_capture_rate": _r(capture_rate),
        "false_positive_rate": _r(fp_rate),
        "normal_auto_pass_rate": _r(auto_pass),
        "human_review_reduction": _r(review_reduction),
    }


def _r(v):
    return None if v is None else round(v, 4)


# ---------------------------------------------------------------------------
# VLM 비용 추정 (per 1M tokens, 2026-06 기준)
# ---------------------------------------------------------------------------
PRICE_PER_MTOK = {
    "claude-haiku-4-5": {"in": 1.0, "out": 5.0},
    "claude-opus-4-8": {"in": 5.0, "out": 25.0},
    "claude-sonnet-4-6": {"in": 3.0, "out": 15.0},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.5, "out": 10.0},
}

# 증거보드 가정: 이미지 ≤768px → ~(W*H)/750 ≈ 이미지당 ~800토큰, 텍스트(프롬프트+스키마+점수) ~600, 출력 ~300
IMG_TOKENS = 800
TEXT_TOKENS = 600
OUTPUT_TOKENS = 300


def estimate_call_cost(n_images: int, model: str) -> dict:
    """VLM 1회 호출 비용 추정."""
    price = PRICE_PER_MTOK.get(model)
    in_tok = n_images * IMG_TOKENS + TEXT_TOKENS
    out_tok = OUTPUT_TOKENS
    usd = None
    if price:
        usd = round(in_tok / 1e6 * price["in"] + out_tok / 1e6 * price["out"], 6)
    return {"input_tokens": in_tok, "output_tokens": out_tok, "usd": usd}


def estimate_batch_cost(image_counts: list, model: str) -> dict:
    """이미지 수 리스트(아바타별 VLM 호출당 이미지 수)로 배치 총비용 추정."""
    calls = [estimate_call_cost(n, model) for n in image_counts if n]
    total_in = sum(c["input_tokens"] for c in calls)
    total_out = sum(c["output_tokens"] for c in calls)
    usds = [c["usd"] for c in calls if c["usd"] is not None]
    total_usd = round(sum(usds), 4) if usds else None
    return {
        "model": model,
        "n_calls": len(calls),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_usd": total_usd,
        "usd_per_call": round(total_usd / len(calls), 6) if (total_usd is not None and calls) else None,
    }


def pretty(summary: dict) -> str:
    if summary.get("n", 0) == 0:
        return "평가할 라벨 쌍 없음"
    lines = [f"n={summary['n']}",
             f"버그 recall          : {summary['bug_recall']}  (목표 ≥ 0.85)",
             f"false negative rate  : {summary['false_negative_rate']}  (낮을수록 좋음)",
             f"의심 포착률           : {summary['suspicious_capture_rate']}",
             f"false positive rate  : {summary['false_positive_rate']}",
             f"정상 자동통과율        : {summary['normal_auto_pass_rate']}  (목표 0.4~0.6)",
             f"수동검수 절감          : {summary['human_review_reduction']}  (목표 ≥ 0.5)",
             "", "혼동행렬 (행=정답, 열=예측):"]
    cm = summary["confusion_matrix"]
    header = "          " + "".join(f"{l:>12}" for l in LABELS)
    lines.append(header)
    for t in LABELS:
        lines.append(f"{t:>10}" + "".join(f"{cm[t][p]:>12}" for p in LABELS))
    return "\n".join(lines)
