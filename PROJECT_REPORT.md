# qAI 프로젝트 상세 보고서

> **qAI — 아바타 투명화 버그 QA 트리아지 시스템**
> 작성 기준일: 2026-06-11

---

## 목차

1. [개요](#1-개요)
2. [기술 스택](#2-기술-스택)
3. [디렉터리 구조](#3-디렉터리-구조)
4. [파이프라인 아키텍처](#4-파이프라인-아키텍처)
5. [핵심 모듈 상세](#5-핵심-모듈-상세)
6. [설정 체계](#6-설정-체계)
7. [UI (Streamlit)](#7-ui-streamlit)
8. [실행 방법](#8-실행-방법)
9. [성능 / 비용](#9-성능--비용)
10. [현재 상태 및 향후 과제](#10-현재-상태-및-향후-과제)

---

## 1. 개요

### 1.1 목적

qAI는 PUBG 스타일 게임 아바타의 **투명화(transparency) / 메시 클리핑(mesh clipping) 버그**를 게임 스크린샷에서 자동으로 탐지·분류하는 **비주얼 QA 트리아지 도구**다. 의상 사이로 배경이 비치는 현상, 메시 누락으로 인한 구멍, 에셋 경계(허리띠·소매·신발 등)에서의 클리핑 아티팩트를 식별하여 QA 담당자의 수동 검수 부담을 줄이는 것이 목표다.

### 1.2 설계 철학

단일 VLM(Vision Language Model) 호출로 한 번에 판정하는 방식이 아니라, **다단계 증거 수집 파이프라인**을 채택했다.

```
탐지 → 세그멘테이션 → ROI 국소화 → 픽셀 단위 점수화 → VLM 검증 → 보수적 융합
```

- **CV(전통 컴퓨터 비전)가 1차 판정**을 내리고, VLM은 증거 이미지를 보고 보조 판정한다.
- **보수적 융합 원칙**: VLM이 CV가 강하게 잡은 버그를 normal로 강등할 수 없다. 불확실·불일치·저품질 케이스는 모두 `suspicious`로 분류되어 사람 검수로 넘어간다.
- 목표는 **실제 결함에 대한 높은 재현율(recall) 유지** — 놓치는 버그(FN)를 최소화하면서 명백한 정상 케이스만 자동 통과시킨다.

### 1.3 판정 라벨

| 라벨 | 의미 | 처리 |
|---|---|---|
| 🔴 `bug` | 명확한 결함 | 버그 리포트 대상 |
| 🟠 `suspicious` | 의심 — 증거 첨부 | 사람 검수 필요 |
| 🟢 `normal` | 정상 | 자동 통과 |

### 1.4 현재 상태 요약

- 전체 파이프라인 구현 완료, 스모크 테스트 통과 (~2,500 LOC Python)
- Mock VLM(무료·즉시)과 실제 Claude API 양쪽으로 엔드투엔드 동작 확인
- `data/output/`에 100건 이상의 테스트 케이스 리포트 누적

---

## 2. 기술 스택

**언어**: Python 3.11 / 3.12 (※ 3.14는 ML 라이브러리 휠 미지원으로 사용 불가)

| 분류 | 라이브러리 | 용도 |
|---|---|---|
| 수치/영상 | NumPy (≥1.24, <2.1), Pillow (≥10.0), OpenCV (≥4.8) | 행렬 연산, 이미지 I/O, 마스크·필터·모폴로지 |
| 탐지/세그 | Ultralytics (≥8.1) — YOLO11-seg | 아바타(person) 탐지 + 인스턴스 마스크 |
| | PyTorch (≥2.1) + TorchVision (≥0.16, CPU 휠) | YOLO 추론 백엔드 |
| | rembg (≥2.0) + ONNX Runtime (≥1.16) | U2Net 배경 제거 (세그 교차검증) |
| 포즈(선택) | MediaPipe (≥0.10, Python <3.13) | 인체 랜드마크 기반 ROI 국소화 (없으면 bbox 비율 폴백) |
| VLM | Anthropic SDK (≥0.40) | Claude Vision + tool-use (기본 프로바이더) |
| | OpenAI SDK (≥1.40) | GPT-4V 대안 (선택) |
| 스키마 | Pydantic (≥2.5) | VLM 구조화 출력 검증 |
| UI | Streamlit (≥1.30) | 웹 인터페이스 (단일/일괄 분석) |
| 설정 | python-dotenv (≥1.0) | `.env` 환경변수 관리 |

---

## 3. 디렉터리 구조

```
qai/
├── README.md                      # 전체 문서 (한국어)
├── requirements.txt               # 의존성
├── config.py                      # 중앙 설정: 임계값·가중치·경로·VLM (모든 매직넘버)
├── .env.example / .env            # API 키 등 환경변수
│
├── app.py                         # Streamlit 웹 UI (메인 진입점)
├── pipeline.py                    # 오케스트레이션: run_qai_case()
│
├── agents/                        # 파이프라인 단계별 에이전트 (모듈형)
│   ├── frame_quality_agent.py     #   블러/밝기/해상도 검사
│   ├── avatar_detection_agent.py  #   YOLO11-seg person 탐지 + bbox + 마스크
│   ├── segmentation_agent.py      #   YOLO 마스크 + rembg 교차검증 → refined_mask
│   ├── roi_agent.py               #   ROI(허리/손목/발목/목) 국소화
│   ├── anomaly_scoring_agent.py   #   5개 픽셀 신호 → ROI별 가중 버그 점수
│   ├── decision_fusion_agent.py   #   CV 사전판정 + VLM 후판정 → 최종 라벨
│   ├── vlm_adjudication_agent.py  #   VLM 증거보드 조립·호출
│   ├── report_agent.py            #   구조화 JSON 리포트 + 증거 이미지 저장
│   └── feedback_agent.py          #   사람 검수 결과 CSV 기록
│
├── cv/                            # 컴퓨터 비전 신호 추출
│   ├── mask_utils.py              #   마스크 정제·채움·마진 크롭
│   ├── roi_utils.py               #   ROI 박스 계산 (MediaPipe 또는 bbox 비율)
│   ├── background_leakage.py      #   HSV 배경 색 모델 → 비침(leak) 점수
│   ├── hole_detection.py          #   내부 홀 탐지 (filled − mask)
│   ├── edge_artifact.py           #   경계 파손 + 비대칭 + 피부 노출 점수
│   └── heatmap.py                 #   VLM 증거보드용 히트맵 렌더링
│
├── vlm/                           # VLM 프로바이더 추상화 (플러그형)
│   ├── base.py                    #   VLMProvider ABC, PNG base64 인코딩, 스키마 검증
│   ├── claude_provider.py         #   Anthropic Claude vision + tool-use (구조화 JSON)
│   ├── openai_provider.py         #   GPT-4V 대안 (스텁)
│   ├── mock_provider.py           #   더미 프로바이더 (무료·오프라인 데모)
│   └── __init__.py                #   프로바이더 팩토리 (키 없으면 mock 자동 폴백)
│
├── prompts/
│   ├── vlm_bug_adjudication.txt   # VLM 시스템 프롬프트
│   └── vlm_schema.json            # tool-use 출력 스키마
│
├── eval/                          # 평가·벤치마크
│   ├── batch_runner.py            #   폴더 일괄 평가, 메타데이터 파싱, 프로바이더 비교
│   ├── metrics.py                 #   혼동행렬, recall, FN율, 비용 추정 (PRICE_PER_MTOK)
│   └── smoke_test.py              #   경량 검증 (YOLO/rembg 불필요, GrabCut 폴백)
│
├── data/
│   ├── input/                     # 단일 업로드 입력
│   ├── output/                    # 케이스 리포트 (JSON + 증거 이미지)
│   ├── eval_set/                  # 벤치마크 데이터셋 (normal|bug/지라/아이템/성별피부/case_*.png)
│   ├── reference/                 # 정답 마스크 (선택)
│   └── labels/                    # 사람 피드백 CSV + 정답 라벨
│
├── analyze_rename.py              # eval_set 파일명 파싱·리네임 계획 출력
├── do_rename.py                   # 리네임 실행 (2단계 안전 방식)
└── yolo11n-seg.pt                 # YOLO11-nano 세그 가중치 (~50MB, 자동 다운로드)
```

---

## 4. 파이프라인 아키텍처

메인 오케스트레이터는 `pipeline.py`의 `run_qai_case(image_path)`다.

```
스크린샷
    │
    ▼
[frame_quality_agent]  블러(Laplacian 분산) · 밝기 · 해상도 검사
    │  good / weak → 계속,  unusable → 최종 suspicious 강제
    ▼
[avatar_detection_agent]  YOLO11-seg person 탐지
    │  → [{avatar_id, bbox, detector_confidence, mask}]  (신뢰도 내림차순)
    │  YOLO 미설치 시 → UI 수동 크롭 / 전체 프레임 폴백
    ▼
─── 아바타별 반복 (_process_avatar) ───────────────────────────
[crop_with_margin]  bbox 15% 확장 크롭 (경계 증거 보존, 타이트 크롭 금지)
    ▼
[segmentation_agent]  우선순위: YOLO 마스크 → rembg(U2Net) → GrabCut 폴백
    │  모폴로지 정제, 최대 컴포넌트 유지, ★내부 홀은 채우지 않음(버그 신호)
    │  → {refined_mask, overlay, mask_confidence}
    ▼
[roi_agent]  MediaPipe 포즈(가능 시) 또는 bbox 비율 → 6개 ROI
    │  waist / left·right_wrist / left·right_ankle / neck
    ▼
[anomaly_scoring_agent]  ROI별 5개 CV 신호 가중합 → roi_bug_score
    │  avatar_bug_score = max(roi_scores)  ← 최악 케이스 보존 (평균 아님)
    ▼
[decision_fusion_agent.pre_vlm_decision]  CV 단독 사전판정
    │  high(≥0.72)→bug / medium(≥0.40)→suspicious / low→normal
    ▼
[vlm_adjudication_agent]  증거보드 조립 → VLM 호출
    │  이미지: 원본 크롭 + 마스크 오버레이 + 히트맵 + ROI 줌(최대 6장)
    │  수치: cv_band, avatar_bug_score, worst_roi, roi_scores
    ▼
[decision_fusion_agent.final_decision]  CV × VLM 보수적 융합
─────────────────────────────────────────────────────────────
    ▼
[report_agent]  케이스 JSON + 증거 이미지 저장 (data/output/<case_id>/)
    │  overall_score = 아바타별 최댓값, needs_human_review 플래그
    ▼
최종 리포트
```

추가로 `pipeline.py`에는 Windows 비ASCII 경로를 안전하게 읽는 `imread_unicode()` (`np.fromfile` + `cv2.imdecode`)와 단계별 ms 측정용 `_Stopwatch` 클래스가 있다.

---

## 5. 핵심 모듈 상세

### 5.1 이상 점수 5개 신호 (`agents/anomaly_scoring_agent.py` + `cv/`)

`roi_bug_score = Σ (weight × signal)` — 가중치는 `config.SCORE_WEIGHTS`:

| 신호 | 가중치 | 방법 | 의미 |
|---|---|---|---|
| `background_leakage` | **0.35** | 배경 HSV 색 모델 + 거리 계산 | 아바타 내부가 배경과 닮은 정도 (최강 신호) |
| `internal_hole` | **0.30** | 채운 실루엣 − 마스크 차이 | 의상 내부의 예상 밖 구멍 |
| `boundary_break` | **0.20** | 내부 링 영역 Canny 엣지 vs 기대 둘레 | 들쭉날쭉한 솔기, 파편화된 경계 |
| `skin_exposure` | **0.10** | YCrCb 피부색 범위 탐지 | 의상이 있어야 할 곳에 신체 노출 (오탐 우려로 낮은 가중치) |
| `asymmetry` | **0.05** | 좌/우 대칭 ROI 점수 차 | 한쪽만 유난히 나쁜 경우 |

ROI 마스크 면적 대비 0.4% (`MIN_HOLE_AREA_RATIO = 0.004`) 미만의 미세 홀은 노이즈로 무시한다.

### 5.2 판정 임계값 (`config.py`)

| 항목 | 값 |
|---|---|
| BUG 임계값 | `avatar_bug_score ≥ 0.72` |
| SUSPICIOUS 임계값 | `0.40 ≤ score < 0.72` |
| NORMAL | `score < 0.40` |
| 마스크 품질 하한 | `mask_confidence < 0.55` → 세그 신뢰 약함, 보수적 처리 |

### 5.3 융합 규칙 매트릭스 (`agents/decision_fusion_agent.py`)

`(cv_band, vlm_label) → 최종 라벨`:

| CV band ＼ VLM | bug | suspicious | normal |
|---|---|---|---|
| **high** (≥0.72) | bug | bug | **suspicious** ← VLM이 강한 CV 버그를 못 내림 |
| **medium** (≥0.40) | suspicious * | suspicious | suspicious ← CV가 이상을 봤으면 normal 불신 |
| **low** (<0.40) | suspicious ← VLM 과탐 가능성 | suspicious | normal |

\* **승급 보정**: medium + VLM bug + 핵심 ROI(`waist`, `left_ankle`, `right_ankle` — 배경 비침이 잘 나는 곳)이면 `bug`로 승급.

**품질 가드**:
- 프레임 품질 `unusable` → 무조건 `suspicious` ("더 나은 이미지 필요")
- 프레임 품질 `weak` + 결과 normal → `suspicious` (자동통과 보류)
- 마스크 품질 < 0.55 + CV 점수 ≥ 0.40 → normal 금지
- `needs_human_review` = (최종 suspicious) 또는 (VLM이 검수 요청)

### 5.4 VLM 프로바이더 추상화 (`vlm/`)

| 프로바이더 | 특징 |
|---|---|
| `claude_provider.py` | Anthropic SDK, **tool-use 강제(`tool_choice`)로 구조화 JSON 보장**, 지수 백오프 재시도 최대 3회 |
| `openai_provider.py` | GPT-4V + JSON 모드 (스텁, 미완성) |
| `mock_provider.py` | API 호출 없음·즉시 반환. CV band를 따라감 (high→bug 0.8 / medium→suspicious 0.55 / low→normal 0.6). 무료 데모·오프라인 테스트용 |

프로바이더 선택은 `vlm/__init__.py:get_provider()` — `config.VLM_PROVIDER`와 API 키 존재 여부로 결정하며, **키가 없으면 mock으로 자동 폴백**한다.

VLM 출력 스키마 (`prompts/vlm_schema.json`, Pydantic 검증):

```json
{
  "final_label": "bug | suspicious | normal",
  "confidence": 0.9,
  "affected_regions": ["waist"],
  "bug_types": ["background_leakage"],
  "visual_evidence": "...",
  "intentional_design_possible": false,
  "needs_human_review": false,
  "reason": "..."
}
```

### 5.5 리포트 출력 (`agents/report_agent.py`)

`data/output/<case_id>/` 아래에 케이스 JSON과 아바타별 증거 이미지(원본 크롭·마스크·오버레이·히트맵·ROI 줌)를 저장한다.

```json
{
  "case_id": "qai_2026_06_11_65232",
  "final_label": "bug",
  "overall_score": 0.85,
  "frame_quality": { ... },
  "avatars": [{
    "avatar_id": "avatar_001",
    "label": "bug",
    "rois": [{ "name": "waist", "bug_score": 0.85, "top_signals": { ... } }],
    "vlm": { "final_label": "bug", "confidence": 0.9, "reason": "..." },
    "evidence": { "original_crop": "...", "heatmap": "..." },
    "timings": { "segmentation_ms": 220, "vlm_ms": 2100 }
  }],
  "needs_human_review": false
}
```

### 5.6 피드백 루프 (`agents/feedback_agent.py`)

UI의 검수 버튼(정상 승인 / 버그 확정 / 오탐 표시) 결과를 `data/labels/human_feedback.csv`에 누적한다. 형식: `timestamp, case_id, avatar_id, qai_label, human_label, comment`. 향후 active learning의 입력이 된다.

---

## 6. 설정 체계

### 6.1 `config.py` 주요 파라미터

> README: "16~20h 튜닝 단계에서 이 파일만 손대면 된다" — 모든 매직넘버가 이 파일에 집중되어 있다.

| 구분 | 파라미터 | 값 | 설명 |
|---|---|---|---|
| 탐지 | `YOLO_WEIGHTS` | `yolo11n-seg.pt` | 최초 1회 자동 다운로드 |
| | `DETECTOR_CONF_THRESHOLD` | 0.25 | YOLO 신뢰도 컷오프 |
| | `CROP_MARGIN` | 0.15 | bbox 15% 확장 (경계 증거 보존) |
| | `MIN_AVATAR_AREA_RATIO` | 0.01 | 프레임 대비 아바타 최소 크기 |
| 프레임 품질 | `BLUR_VAR_THRESHOLD` | 60.0 | Laplacian 분산 (블러 판정) |
| | `DARK_MEAN_THRESHOLD` | 40.0 | 평균 밝기 하한 (0~255) |
| | `EDGE_CUTOFF_MARGIN` | 3 | 프레임 경계 잘림 판정 픽셀 |
| 판정 | `BUG_THRESHOLD` | 0.72 | |
| | `SUSPICIOUS_THRESHOLD` | 0.40 | |
| | `MASK_QUALITY_MIN` | 0.55 | |
| VLM | `VLM_PROVIDER` | claude (기본) | claude \| openai \| mock |
| | `VLM_MODEL` | claude-haiku-4-5 (기본) | 정밀 판정 시 claude-opus-4-8 |
| | `VLM_MAX_RETRIES` / `VLM_MAX_TOKENS` | 3 / 1024 | |
| ROI | `ROI_RATIOS` | 6개 영역 비율 | MediaPipe 부재 시 폴백 |
| | `ROI_SYMMETRY_PAIRS` | 손목쌍, 발목쌍 | asymmetry 계산용 |

ROI bbox 비율 폴백 정의 (x1%, y1%, x2%, y2%):

| ROI | 비율 |
|---|---|
| waist | (0.20, 0.42, 0.80, 0.60) |
| left_wrist | (0.00, 0.42, 0.35, 0.72) |
| right_wrist | (0.65, 0.42, 1.00, 0.72) |
| left_ankle | (0.18, 0.78, 0.50, 0.98) |
| right_ankle | (0.50, 0.78, 0.82, 0.98) |
| neck | (0.35, 0.12, 0.65, 0.28) |

### 6.2 환경변수 (`.env`)

| 변수 | 용도 | 비고 |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude API 키 | 실제 VLM 사용 시 필수, 없으면 mock 폴백 |
| `ANTHROPIC_BASE_URL` | 커스텀 게이트웨이 | 공백 = 공식 엔드포인트 |
| `QAI_VLM_PROVIDER` | claude / openai / mock | |
| `QAI_VLM_MODEL` | 모델 ID | haiku(저비용·빠름) ↔ opus(정밀) |
| `QAI_YOLO_WEIGHTS` | YOLO 체크포인트 경로 | 기본 yolo11n-seg.pt |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `QAI_OPENAI_MODEL` | GPT-4V 대안 | 선택 (기본 gpt-4o-mini) |

---

## 7. UI (Streamlit)

진입점: `streamlit run app.py` → http://localhost:8501

**공통**: 헤더에 활성 VLM 프로바이더/모델 표시, 사이드바에 판정 임계값·신호 가중치 표시.

### 모드 1 — 단일 스크린샷 분석
1. 파일 업로드 (PNG/JPG/JPEG/BMP/WebP) → "🔍 분석 실행"
2. 결과 표시:
   - 케이스 판정 배지 (🔴 BUG / 🟠 SUSPICIOUS / 🟢 NORMAL) + 종합 점수 + 검수 필요 여부
   - 아바타별 확장 섹션: 원본 / 마스크 오버레이 / 히트맵 / 배경제거 4열 갤러리, ROI별 증거 크롭 + 버그 점수 + 상위 신호, VLM 판정(라벨·신뢰도·근거)
   - 검수 버튼: ✅ 정상 승인 / 🐞 버그 확정 / ⚠️ 오탐 표시 → 피드백 CSV 기록
   - 📥 리포트 JSON 다운로드

### 모드 2 — 폴더 일괄 분석
1. 폴더 경로 입력(재귀 스캔) + 프로바이더 선택(mock/claude) + 최대 처리 수 슬라이더
2. claude 선택 시 사전 비용·시간 추정 표시
3. 결과 테이블 (rel_path, pred, true, score, angle, gender_skin, needs_review)
4. 정답 라벨 존재 시 지표 패널: recall, FN율, FP율, 캡처율, 자동통과율
5. CSV / 전체 JSON 다운로드

---

## 8. 실행 방법

### 설치 (Windows PowerShell)

```powershell
# Python 3.11/3.12 가상환경 (3.14는 ML 휠 미지원 — 사용 금지)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1

# PyTorch CPU 휠 (별도 인덱스)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 나머지 의존성
pip install -r requirements.txt

# 환경변수 설정
Copy-Item .env.example .env   # ANTHROPIC_API_KEY 입력 (실제 Claude 사용 시)
```

### 실행

```powershell
# 웹 UI (인터랙티브)
streamlit run app.py

# 폴더 일괄 평가 (mock VLM — 무료·즉시)
python eval/batch_runner.py --dir data/input

# mock ↔ claude 지연·비용 실비교
python eval/batch_runner.py --dir data/input --compare

# 스모크 테스트 (YOLO/rembg 불필요 — GrabCut 폴백 + CV 점수 + 융합 로직 검증)
python eval/smoke_test.py
```

### 보조 스크립트

| 스크립트 | 용도 |
|---|---|
| `analyze_rename.py` | `data/eval_set/` 파일명 파싱, 리네임 계획 출력 |
| `do_rename.py` | 리네임 실행 (충돌 방지 2단계 방식) |

---

## 9. 성능 / 비용

리포트 JSON의 `avatars[].timings`에 단계별 지연이 기록된다 (`segmentation_ms / roi_ms / scoring_ms / heatmap_ms / vlm_ms / total_ms`).

### 처리 시간 (이미지 1장, 전형값)

| 단계 | 시간 |
|---|---|
| 세그멘테이션 | ~220ms (CPU GrabCut 기준; rembg ~400ms) — CPU에서 가장 무거운 단계 |
| ROI 추출 | ~5ms |
| 이상 점수화 | ~10ms |
| 히트맵 | ~5ms |
| VLM (mock) | ~0ms |
| **합계 (mock)** | **~240ms** |
| VLM (claude-haiku-4-5) | +1.5~4초 |
| VLM (claude-opus-4-8) | +4~12초 |

### VLM 비용 (호출당, 증거 이미지 5~8장 기준)

| 모델 | 호출당 비용 | 용도 |
|---|---|---|
| claude-haiku-4-5 | ~$0.009 | 반복·데모·대량 배치 |
| claude-opus-4-8 | ~$0.047 | 최종 정밀 판정 |

단가표는 `eval/metrics.py:PRICE_PER_MTOK`에서 관리하며, 배치 실행 전 UI/CLI에서 예상 비용을 미리 보여준다 (이미지 수 × ~800 토큰 + 텍스트 → 모델 단가 환산).

---

## 10. 현재 상태 및 향후 과제

### ✅ 완료

- 전체 파이프라인 (탐지 → 세그 → ROI → 점수 → VLM → 융합 → 리포트) 엔드투엔드 동작
- 스모크 테스트 통과 (YOLO/rembg 없이도 폴백으로 검증 가능)
- Mock VLM 프로바이더 (무료 데모) + Claude API 연동 (tool-use + vision)
- Streamlit UI 단일/일괄 분석 모드
- 메타데이터 파싱 기반 배치 평가 + 비용/지연 벤치마크
- 증거 이미지 자동 저장 — `data/output/`에 100건 이상 테스트 케이스 누적

### 🚧 미완성 / 스텁

- `vlm/openai_provider.py` — 골격만 존재, 미검증
- `agents/feedback_agent.py` — CSV 기록까지만 구현, **active learning 파이프라인은 미착수** (피드백 데이터 누적 대기)
- `data/eval_set/` 라벨링·리네임 작업 진행 중 (`analyze_rename.py` / `do_rename.py` 준비됨; 두 스크립트 간 각도 코드 불일치 — A90이 right vs left — 검토 필요 플래그 있음)

### ⚠️ 알려진 제약

| 제약 | 내용 |
|---|---|
| Python 3.14 미지원 | torch / rembg / ultralytics / mediapipe 휠 부재 → 3.11/3.12 필수 |
| MediaPipe Python <3.13 | 3.13+에서 포즈 랜드마크 비활성 → bbox 비율 폴백으로 동작 |
| rembg CPU 느림 | ~400ms/장; GrabCut 폴백은 ~220ms이나 정확도 낮음 |
| YOLO 최초 실행 | ~50MB 모델 자동 다운로드 — 인터넷 필요 |
| eval_set 정답 라벨 | 리네임/라벨링 미완료 시 지표 산출 제한 |

### 🔭 향후 방향

1. **임계값/가중치 튜닝** — eval_set 라벨링 완료 후 `config.py` 단일 파일 튜닝 (설계상 16~20h 작업으로 산정)
2. **Active learning** — `human_feedback.csv` 누적분으로 점수 가중치·임계값 재보정
3. **OpenAI 프로바이더 완성** — 프로바이더 간 정확도/비용 비교
4. **세그멘테이션 고도화** — GrabCut 의존 축소, YOLO 마스크 + rembg 교차검증 비중 확대

---

*본 보고서는 2026-06-11 기준 코드베이스를 직접 분석하여 작성되었으며, 수치(임계값·가중치·ROI 비율 등)는 `config.py` 및 `agents/decision_fusion_agent.py` 실제 값과 대조 검증되었다.*
