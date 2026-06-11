# qAI — 아바타 투명화 버그 QA 트리아지 (CPU MVP)

PUBG 스타일 아바타의 **투명화/메시누락/클리핑 버그**(허리·손목·발목·목 등 의상 경계)를
스크린샷에서 자동 트리아지하는 시각 QA 도구.

> 철학: VLM 한 방 분류가 아니라 **탐지 → 세그 → ROI 국소화 → 픽셀 이상 점수 → VLM 검증 →
> 증거 리포트 → 사람 검수** 파이프라인. 모든 디텍터가 일치할 때만 정상 자동통과,
> 불확실하면 의심으로 표시하고 증거를 보여준다(높은 recall 우선).

**현재 상태:** 전체 파이프라인(탐지→세그→ROI→이상점수→VLM 융합→리포트) 빌드 완료, 스모크 테스트 통과.
**키 없이도** `mock` provider로 끝까지 동작(판정은 더미) — `ANTHROPIC_API_KEY`를 넣으면 실제 Claude 판정으로 전환된다.

## ⚠️ Python 버전 주의
이 머신의 기본 Python은 **3.14**인데, `torch`/`ultralytics`/`rembg`/`mediapipe`는 아직
3.14 휠이 없을 가능성이 크다. **전체 ML 스택은 Python 3.11 또는 3.12 venv**에서 설치하라.
- 경량 코어(`numpy`, `opencv-python`, `pydantic`, `python-dotenv`)는 3.14에서도 동작 →
  YOLO/rembg 없이 GrabCut fallback + mock VLM으로 스모크 테스트는 가능(`python eval/smoke_test.py`).

## 설치 (Windows / PowerShell)
```powershell
# Python 3.11/3.12 권장 (예: py -3.12)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
Copy-Item .env.example .env   # 그리고 ANTHROPIC_API_KEY 채우기
```

## 실행
```powershell
# 로컬 웹앱 (브라우저 http://localhost:8501 자동 오픈)
streamlit run app.py

# 폴더 일괄 평가
python eval/batch_runner.py --dir data/input
```

VLM 키가 없으면 `QAI_VLM_PROVIDER=mock` 으로 두면 CV 단계까지 동작한다(판정은 더미).

## 키 설정 후 첫 실행 (Claude 실판정)
`ANTHROPIC_API_KEY`를 받은 직후, PowerShell 기준:
```powershell
# 1) 키 + provider/모델 설정 (.env에 기입해도 동일)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:QAI_VLM_PROVIDER  = "claude"
$env:QAI_VLM_MODEL     = "claude-haiku-4-5"   # 반복·데모용(저비용·빠름)
# $env:QAI_VLM_MODEL   = "claude-opus-4-8"    # 최종 판정용(정밀)

# 2) 웹앱으로 실판정 확인
streamlit run app.py

# 3) mock ↔ claude 지연·비용 실비교
python eval/batch_runner.py --dir data/input --compare
```
> 키가 비어 있거나 provider가 `claude`가 아니면 자동으로 **mock 폴백**(콘솔에 `claude API 키 없음 → mock 폴백` 출력). 즉 키 없이 먼저 흐름을 확인하고, 키를 넣은 뒤 같은 명령으로 실판정만 갈아끼우면 된다.

### 환경변수 (`config.py` / `.env.example` 기준)
| 변수 | 의미 | 기본값 |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude API 키 (실판정 필수) | (없음 → mock) |
| `ANTHROPIC_BASE_URL` | 사내 게이트웨이 등 엔드포인트 교체(옵션) | 기본 엔드포인트 |
| `QAI_VLM_PROVIDER` | `claude` \| `openai` \| `mock` | `claude` |
| `QAI_VLM_MODEL` | Claude 모델 ID | `claude-haiku-4-5` |
| `OPENAI_API_KEY` | OpenAI 어댑터용(옵션, 미발급) | (없음) |

## 성능 / 비용 벤치마크
```powershell
python eval/batch_runner.py --dir data/input --compare
```
같은 이미지셋을 **mock vs claude** provider로 각각 돌려 다음을 출력한다:
- **단계별·총 지연(ms)** — `segmentation_ms / roi_ms / scoring_ms / heatmap_ms / vlm_ms / total_ms` (report.json의 `avatars[].timings`)
- **라벨 분포**(N/S/B) provider별
- **VLM 배치 비용 추정** — haiku/opus 단가로 환산 (이미지 수 × ~800tok + 텍스트 → 모델 단가). 단가표는 `eval/metrics.py:PRICE_PER_MTOK`.

> 디텍터(YOLO) 미설치 환경에서도 CV/VLM 단계를 측정하도록 `--compare`는 whole-frame fallback을 쓴다(프레임 전체를 1 아바타로 가정). **실제 1·2·3 비교는 py3.12 풀스택 + `ANTHROPIC_API_KEY` 준비 후** 실행하면 grabcut↔YOLO+rembg 세그 지연 차이와 mock↔claude 판정 차이가 함께 잡힌다.

해석: `total_ms`는 이미지당 CPU 처리 + VLM 왕복 합. seg 단계가 CPU에서 가장 무겁고(현재 grabcut ~220ms, YOLO+rembg로 바뀜), `vlm_ms`는 provider/모델에 좌우된다(mock ≈0, haiku ~1.5~4s, opus ~4~12s). 비용은 호출당 haiku ~$0.009 / opus ~$0.047 수준(증거 이미지 5~8장 기준).

## 구조
- `pipeline.py` — `run_qai_case(image_path)` 오케스트레이션
- `agents/` — frame_quality / avatar_detection / segmentation / roi / anomaly_scoring / decision_fusion / vlm_adjudication / report / feedback
- `cv/` — mask/roi/hole/background_leakage/edge/heatmap 유틸
- `vlm/` — provider 추상화(claude/openai/mock)
- `config.py` — 임계값·가중치·경로 (튜닝은 여기서)
- `eval/` — batch_runner, metrics

자세한 설계는 플랜 문서 참조.
