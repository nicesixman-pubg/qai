"""qAI Streamlit UI — 로컬 웹앱.

실행: streamlit run app.py  → http://localhost:8501
모드:
  - 단일 스크린샷 업로드 → 1건 분석
  - 폴더 경로(일괄) → 재귀 스캔 후 일괄 분석(결과 표 + 행별 증거)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

import config
import vlm
from agents import feedback_agent
from eval import metrics
from eval.batch_runner import build_items
from eval.reference_resolver import resolve_reference
from pipeline import run_qai_case

st.set_page_config(page_title="qAI — 투명화 버그 QA", layout="wide")

_LABEL_BADGE = {
    "bug": ("🔴 BUG", "#ff4b4b"),
    "suspicious": ("🟠 SUSPICIOUS", "#ffa500"),
    "normal": ("🟢 NORMAL", "#21ba45"),
}


def badge(label: str) -> str:
    text, color = _LABEL_BADGE.get(label, (label, "#888"))
    return f"<span style='background:{color};color:white;padding:3px 10px;border-radius:6px;font-weight:600'>{text}</span>"


def _to_csv(rows: list) -> str:
    import csv
    import io
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def render_report(report: dict, key_prefix: str):
    """단일/일괄 공용 — 한 리포트의 결정·증거·VLM·검수버튼 렌더."""
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(f"### 케이스 결정: {badge(report['final_label'])}", unsafe_allow_html=True)
        st.write(report.get("decision_reason", ""))
    c2.metric("종합 점수", f"{report.get('overall_score', 0):.2f}")
    c3.metric("검수 필요", "예" if report.get("needs_human_review") else "아니오")

    fq = report.get("frame_quality", {})
    if fq.get("issues"):
        st.warning(f"프레임 품질 이슈: {', '.join(fq['issues'])}")

    for ai, a in enumerate(report.get("avatars", [])):
        pfx = f"{key_prefix}_{a['avatar_id']}_{ai}"
        with st.container(border=True):
            st.markdown(f"#### {a['avatar_id']} — {badge(a['label'])}", unsafe_allow_html=True)
            st.write(f"탐지 신뢰도 {a.get('detector_confidence')} · "
                     f"마스크 신뢰도 {a.get('mask_confidence')} · {a.get('decision_reason','')}")

            ev = a.get("evidence", {})
            ev_keys = ["original_crop", "reference_crop", "diff_overlay", "side_by_side"]
            ev_caps = ["test crop", "reference crop", "diff overlay", "reference / test / diff"]
            if not any(ev.get(k) for k in ev_keys):
                ev_keys = ["original_crop", "overlay", "heatmap", "bg_removed"]
                ev_caps = ["original crop", "mask overlay", "heatmap", "background removed"]
            cols = st.columns(min(4, len(ev_keys)) or 1)
            for col, key, cap in zip(cols, ev_keys, ev_caps):
                if ev.get(key) and Path(ev[key]).exists():
                    col.image(ev[key], caption=cap, use_container_width=True)

            if a.get("reference_candidates"):
                st.write("**Reference diff candidates**")
                st.dataframe([
                    {
                        "id": c.get("candidate_id"),
                        "region": c.get("region_hint"),
                        "score": c.get("diff_score"),
                        "bbox": c.get("bbox"),
                    }
                    for c in a.get("reference_candidates", [])
                ], use_container_width=True)

            st.write("**ROI 증거**")
            roi_cols = st.columns(len(a["rois"]) or 1)
            for col, r in zip(roi_cols, a["rois"]):
                if r.get("evidence_crop") and Path(r["evidence_crop"]).exists():
                    col.image(r["evidence_crop"], use_container_width=True)
                col.caption(f"{r['name']}: {r['bug_score']:.2f}")
                col.json(r.get("top_signals", {}), expanded=False)

            vlm_r = a.get("vlm", {})
            st.write("**VLM 판정**")
            st.write(f"`{vlm_r.get('final_label')}` (conf {vlm_r.get('confidence')}) — {vlm_r.get('reason','')}")
            if vlm_r.get("visual_evidence"):
                st.json(vlm_r["visual_evidence"], expanded=False)

            b1, b2, b3 = st.columns(3)
            if b1.button("✅ 정상 승인", key=f"ok_{pfx}"):
                feedback_agent.record(report["case_id"], a["avatar_id"], a["label"], "normal")
                st.success("정상으로 기록")
            if b2.button("🐞 버그 확정", key=f"bug_{pfx}"):
                feedback_agent.record(report["case_id"], a["avatar_id"], a["label"], "bug")
                st.success("버그로 기록")
            if b3.button("⚠️ 오탐 표시", key=f"fp_{pfx}"):
                feedback_agent.record(report["case_id"], a["avatar_id"], a["label"], "false_positive")
                st.success("오탐으로 기록")


# ---------------------------------------------------------------------------
# 헤더 / 사이드바
# ---------------------------------------------------------------------------
st.title("qAI — 아바타 투명화 버그 QA 트리아지")
st.caption(f"VLM provider: `{config.VLM_PROVIDER}` · model: `{config.VLM_MODEL}` "
           f"(키 없으면 자동 mock 폴백)")

with st.sidebar:
    st.header("설정")
    st.write("**판정 임계값**")
    st.write(f"- BUG ≥ {config.BUG_THRESHOLD}")
    st.write(f"- SUSPICIOUS ≥ {config.SUSPICIOUS_THRESHOLD}")
    st.write("**점수 가중치**")
    st.json(config.SCORE_WEIGHTS)

mode = st.radio("입력 방식", ["단일 스크린샷 업로드", "폴더 경로 (일괄)"], horizontal=True)


# ---------------------------------------------------------------------------
# 모드 1: 단일 업로드
# ---------------------------------------------------------------------------
if mode == "단일 스크린샷 업로드":
    uploaded = st.file_uploader("게임 스크린샷 업로드", type=["png", "jpg", "jpeg", "bmp", "webp"])
    reference_uploaded = st.file_uploader("Known-good reference upload (optional)", type=["png", "jpg", "jpeg", "bmp", "webp"])
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        ref_path = None
        if reference_uploaded is not None:
            ref_suffix = Path(reference_uploaded.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ref_suffix) as tmp_ref:
                tmp_ref.write(reference_uploaded.getbuffer())
                ref_path = tmp_ref.name
        st.image(uploaded, caption="원본 스크린샷", use_container_width=True)
        if reference_uploaded is not None:
            st.image(reference_uploaded, caption="known-good reference", use_container_width=True)
        if st.button("🔍 분석 실행", type="primary"):
            with st.spinner("파이프라인 실행 중..."):
                st.session_state["single_report"] = run_qai_case(
                    tmp_path,
                    reference_image_path=ref_path,
                    mode="reference" if ref_path else "standalone",
                )

    report = st.session_state.get("single_report")
    if report:
        st.divider()
        render_report(report, key_prefix="single")
        st.download_button(
            "📥 리포트 JSON 다운로드",
            data=json.dumps(report, ensure_ascii=False, indent=2),
            file_name=f"{report['case_id']}.json", mime="application/json",
        )


# ---------------------------------------------------------------------------
# 모드 2: 폴더 경로(일괄)
# ---------------------------------------------------------------------------
else:
    folder = st.text_input("폴더 경로 (재귀 스캔)", value="data/eval_set")
    col_a, col_b, col_c = st.columns([1, 1, 1])
    provider = col_a.radio("VLM provider", ["mock", "claude", "openai"], horizontal=True,
                           help="mock=plumbing only, claude/openai=real adjudication")
    workflow_mode = col_b.radio("Workflow", ["reference", "standalone"], horizontal=True)
    max_n = col_c.number_input("최대 처리 수 (0=전체)", min_value=0, value=0, step=8)

    if st.button("📂 폴더 스캔"):
        items = build_items(Path(folder), {})
        st.session_state["batch_items"] = [
            {"path": str(it["path"]), "meta": it["meta"]} for it in items
        ]
        st.session_state.pop("batch_reports", None)

    items = st.session_state.get("batch_items")
    if items is not None:
        labeled = sum(1 for it in items if it["meta"]["true_label"])
        from collections import Counter
        dist = Counter(it["meta"]["true_label"] or "미지정" for it in items)
        st.info(f"발견 {len(items)}장 · 라벨 보유 {labeled}장 · 분포 {dict(dist)}")

        n_run = len(items) if max_n == 0 else min(int(max_n), len(items))
        if provider == "claude":
            # 이미지당 증거보드 ~9장 가정으로 비용/시간 추정
            est = metrics.estimate_batch_cost([9] * n_run, config.VLM_MODEL)
            st.warning(f"⚠️ claude 실판정: {n_run}장 → 예상 ~{n_run*6//60}분 "
                       f"+ ${est['total_usd']} ({config.VLM_MODEL}). mock은 무료/즉시.")

        if st.button(f"🔍 일괄 분석 실행 ({n_run}장)", type="primary"):
            config.VLM_PROVIDER = provider
            vlm.reset()
            run_items = items[:n_run]
            reports, prog, status = [], st.progress(0.0), st.empty()
            for i, it in enumerate(run_items):
                status.write(f"처리 중 {i+1}/{n_run}: {it['meta']['rel_path']}")
                ref = resolve_reference(it["path"], eval_root=Path(folder)) if workflow_mode == "reference" else {}
                rep = run_qai_case(
                    it["path"],
                    reference_image_path=ref.get("reference_path") or None,
                    mode=workflow_mode,
                    allow_standalone_fallback=False,
                )
                reports.append({"meta": it["meta"], "report": rep})
                prog.progress((i + 1) / n_run)
            status.write("완료")
            st.session_state["batch_reports"] = reports

    reports = st.session_state.get("batch_reports")
    if reports:
        st.divider()
        # 결과 표
        table = [{
            "rel_path": r["meta"]["rel_path"],
            "pred": r["report"]["final_label"],
            "true": r["meta"]["true_label"] or "",
            "score": r["report"].get("overall_score"),
            "safe_to_autopass": r["report"].get("safe_to_autopass"),
            "reference": r["report"].get("reference") or "",
            "angle": r["meta"]["angle"],
            "gender_skin": r["meta"]["gender_skin"],
            "needs_review": r["report"].get("needs_human_review"),
        } for r in reports]
        st.dataframe(table, use_container_width=True, height=320)

        # 라벨 있으면 지표
        pairs = [(r["meta"]["true_label"], r["report"]["final_label"])
                 for r in reports if r["meta"]["true_label"]]
        if pairs:
            with st.expander("📊 지표 (recall / FN / 혼동행렬)", expanded=True):
                st.code(metrics.pretty(metrics.summarize(pairs)))

        # 다운로드
        d1, d2 = st.columns(2)
        d1.download_button("📥 결과 표 CSV", data=_to_csv(table),
                           file_name="batch_results.csv", mime="text/csv")
        d2.download_button("📥 전체 리포트 JSON",
                           data=json.dumps(reports, ensure_ascii=False, indent=2),
                           file_name="batch_reports.json", mime="application/json")

        # 행별 펼쳐 증거
        st.write("### 케이스별 증거")
        for idx, r in enumerate(reports):
            lbl = r["report"]["final_label"]
            mark = _LABEL_BADGE.get(lbl, (lbl, ""))[0]
            with st.expander(f"{mark}  {r['meta']['rel_path']}  (true={r['meta']['true_label'] or '-'})"):
                render_report(r["report"], key_prefix=f"batch{idx}")
