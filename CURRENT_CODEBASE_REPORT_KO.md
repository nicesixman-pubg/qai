# QAI 현재 코드베이스 상세 보고서

작성 기준: 현재 로컬 코드베이스 점검 결과  
대상 독자: 개발자, QA 담당자, 평가 파이프라인 운영자

## 1. 프로젝트 개요

이 코드베이스는 캐릭터 또는 아바타 이미지에서 시각적 결함을 탐지하고, 최종 판정을 `normal`, `suspicious`, `bug` 중 하나로 산출하는 QAI(Quality AI) 평가 시스템이다.

현재 구현의 핵심 방향은 **정상 기준 이미지(reference image)** 와 테스트 이미지를 비교한 뒤, 컴퓨터 비전 기반 diff evidence와 GPT/Claude 계열 VLM 판정을 결합하는 것이다. 기준 이미지가 있으면 `reference` 워크플로우를 사용하고, 기준 이미지가 없으면 `standalone` 워크플로우로 동작한다.

시스템은 크게 다음 기능을 제공한다.

- Streamlit 기반 단일 이미지 분석 UI
- 폴더 단위 배치 평가
- 정상 기준 이미지 자동 매칭
- 아바타 탐지 및 세그멘테이션
- reference/test 이미지 차이 분석
- VLM 기반 결함 판정
- CV evidence와 VLM 결과를 결합한 최종 판정
- JSON 리포트, evidence 이미지, CSV 평가 결과 저장

## 2. 전체 실행 구조

주요 진입점은 두 가지다.

1. `app.py`
   - Streamlit UI를 제공한다.
   - 단일 이미지 업로드 분석과 폴더 배치 평가를 지원한다.
   - 단일 이미지 분석에서는 사용자가 정상 기준 이미지를 함께 업로드하면 `reference` 모드로 실행한다.
   - 기준 이미지가 없으면 `standalone` 모드로 실행한다.
   - 결과 화면에는 최종 판정, evidence 이미지, VLM 응답, reference diff 후보, JSON 리포트가 표시된다.

2. `eval/batch_runner.py`
   - 커맨드라인 배치 평가 도구다.
   - `data/eval_set` 같은 평가 폴더를 순회하며 이미지를 분석한다.
   - 기본 모드는 `reference`다.
   - `--provider mock`, `--provider claude`, `--provider openai`로 VLM provider를 선택할 수 있다.
   - 결과를 CSV로 저장하며, reference 해석 상태와 최종 판정 정보를 함께 기록한다.

핵심 분석 함수는 `pipeline.py`의 `run_qai_case(...)`다. 이 함수는 입력 이미지와 옵션을 받아 `reference` 또는 `standalone` 경로를 선택하고, 분석 결과 리스트를 반환한다.

## 3. 핵심 워크플로우

### 3.1 Reference 워크플로우

`reference` 워크플로우는 현재 코드베이스에서 정확도 개선을 위해 가장 중요한 경로다.

입력:

- 테스트 이미지
- 정상 기준 이미지
- 선택적 수동 bbox
- 선택적 whole-frame fallback 설정

출력:

- 최종 라벨: `normal`, `suspicious`, `bug`
- confidence
- reference diff score
- reference match quality
- VLM 판정
- 결함 후보 crop
- evidence 이미지
- JSON 리포트

처리 단계는 다음과 같다.

1. 테스트 이미지와 reference 이미지를 로드한다.
2. 이미지 품질을 검사한다.
3. 테스트 이미지와 reference 이미지에서 주요 아바타를 탐지한다.
4. 탐지된 아바타 영역을 세그멘테이션한다.
5. 테스트 crop과 reference crop을 같은 크기로 정렬한다.
6. foreground mask를 기반으로 비교 대상 영역을 만든다.
7. LAB 색상 차이, Canny edge 차이, silhouette 차이를 계산한다.
8. 여러 diff signal을 가중 합산해 score map을 만든다.
9. 연결 요소 분석으로 주요 diff 후보를 추출한다.
10. 각 후보의 crop, bbox, area, score, region hint를 생성한다.
11. VLM에 테스트 crop, reference crop, diff overlay, side-by-side, 후보 crop을 전달한다.
12. VLM은 strict schema에 맞춰 결함 여부를 판정한다.
13. decision fusion 단계에서 CV diff와 VLM 결과를 결합한다.
14. evidence 이미지와 JSON 리포트를 저장한다.

이 방식은 기준 이미지와 테스트 이미지의 실제 차이를 직접 보여주기 때문에, standalone 방식보다 오탐과 누락을 줄이기 좋은 구조다.

### 3.2 Standalone 워크플로우

`standalone` 워크플로우는 기준 이미지 없이 테스트 이미지만 분석한다.

처리 단계는 다음과 같다.

1. 테스트 이미지를 로드한다.
2. 이미지 품질을 검사한다.
3. 아바타를 탐지한다.
4. 세그멘테이션 mask를 만든다.
5. ROI와 anomaly score를 계산한다.
6. heatmap과 crop evidence를 생성한다.
7. VLM에 이미지 evidence를 전달한다.
8. decision fusion이 CV 점수와 VLM 판정을 결합한다.
9. 리포트를 저장한다.

이 경로는 기준 이미지가 없을 때 유용하지만, 의도된 디자인과 실제 결함을 구분하기 어렵다. 따라서 현재 정확도 개선의 중심은 reference workflow다.

## 4. 주요 모듈 설명

### 4.1 `pipeline.py`

분석 파이프라인의 중심 모듈이다.

주요 역할:

- `run_qai_case(...)`에서 workflow mode를 결정한다.
- reference 이미지가 있으면 `run_qai_reference_case(...)`를 실행한다.
- reference가 없거나 fallback이 허용되면 standalone 경로를 실행한다.
- 탐지, 세그멘테이션, diff 분석, VLM 판정, decision fusion, 리포트 저장을 연결한다.

중요한 동작:

- mode가 명시되지 않으면 reference 이미지 존재 여부에 따라 자동 선택한다.
- reference 모드에서 reference가 없고 fallback이 금지되어 있으면 보수적으로 처리한다.
- 각 avatar별 결과를 생성하고 리포트 저장 경로를 반환한다.

### 4.2 `cv/reference_diff.py`

reference/test 비교 evidence를 생성하는 모듈이다.

주요 역할:

- reference crop을 test crop 크기에 맞춘다.
- mask를 정리하고 비교 영역을 만든다.
- 색상 차이, edge 차이, silhouette 차이를 계산한다.
- score map과 diff mask를 만든다.
- 연결 요소 기반 diff 후보를 추출한다.
- VLM에 전달할 candidate crop과 overlay 이미지를 생성한다.

주요 산출물:

- `diff_mask`
- `score_map`
- `diff_overlay`
- `side_by_side`
- `reference_match_quality`
- `overall_diff_score`
- `diff_coverage`
- `candidates`

이 모듈은 VLM이 막연히 이미지를 보는 대신, 실제로 달라진 영역을 집중적으로 판단하도록 evidence를 만든다.

### 4.3 `eval/reference_resolver.py`

배치 평가에서 테스트 이미지에 대응되는 정상 기준 이미지를 찾는 모듈이다.

주요 동작:

- normal 이미지의 reference는 자기 자신으로 처리한다.
- bug 이미지는 경로의 `bug`를 `normal`로 바꾸고, 파일명의 `,bug`를 `,normal`로 바꿔 대응 이미지를 찾는다.
- 별도 `reference_root`가 지정되면 mirror 구조에서 reference를 찾는다.
- reference 상태를 `ok`, `missing_reference`, `ambiguous_reference` 등으로 기록한다.

이 모듈 덕분에 평가 폴더 구조가 정리되어 있으면 reference workflow를 자동으로 실행할 수 있다.

### 4.4 `eval/batch_runner.py`

평가 세트를 처리하는 CLI 도구다.

주요 역할:

- 평가 이미지 파일을 수집한다.
- 라벨 파일이 있으면 정답 라벨과 비교한다.
- reference 이미지를 자동으로 해석한다.
- `run_qai_case(...)`를 호출한다.
- 이미지별 결과를 CSV로 저장한다.

지원 옵션:

```powershell
python eval\batch_runner.py --dir data\eval_set --mode reference --provider mock --limit 8
python eval\batch_runner.py --dir data\eval_set --mode reference --provider claude
python eval\batch_runner.py --dir data\eval_set --mode reference --provider openai
python eval\batch_runner.py --dir data\eval_set --mode standalone --provider mock
```

### 4.5 `agents/decision_fusion_agent.py`

CV evidence와 VLM 판정을 결합해 최종 라벨을 결정하는 모듈이다.

reference 모드의 핵심 정책:

- VLM이 `bug`라고 판정하면 최종 `bug`가 될 수 있다.
- diff score가 높지만 VLM이 결함을 확인하지 못하면 `suspicious`로 보수 처리한다.
- reference match quality가 낮으면 `normal`로 쉽게 통과시키지 않는다.
- `normal`은 diff가 충분히 작고 VLM이 `safe_to_autopass=true`를 반환할 때만 허용한다.
- frame 품질이 약하거나 reference 상태가 불완전하면 `suspicious`로 남긴다.

standalone 모드의 정책:

- CV 점수가 높고 VLM이 normal이라고 해도 자동 통과하지 않는다.
- 품질이 낮거나 판단 근거가 약하면 `suspicious`로 처리한다.
- 확실한 결함 증거가 있을 때만 `bug`로 올린다.

전체적으로 이 모듈은 false normal을 줄이는 방향으로 설계되어 있다.

### 4.6 `agents/vlm_adjudication_agent.py`

VLM provider를 호출해 결함 여부를 판정하는 중간 계층이다.

주요 역할:

- reference mode용 evidence bundle을 구성한다.
- standalone mode용 evidence bundle을 구성한다.
- provider별 구현체에 이미지와 prompt를 전달한다.
- 응답을 schema에 맞게 검증한다.
- 실패 시 보수적으로 `suspicious`를 반환한다.

### 4.7 `agents/report_agent.py`

분석 결과와 evidence 파일을 저장하는 모듈이다.

저장되는 항목:

- 원본 crop
- mask
- overlay
- background removed 이미지
- heatmap
- reference crop
- diff mask
- diff overlay
- side-by-side 이미지
- candidate crop
- JSON 리포트

JSON 리포트에는 다음 정보가 포함된다.

- avatar id
- 최종 라벨
- confidence
- CV 점수
- reference diff score
- reference match quality
- VLM 결과
- `safe_to_autopass`
- 후보 결함 목록
- evidence 파일 경로
- 처리 시간

### 4.8 `vlm/*`

VLM provider 추상화 계층이다.

주요 구성:

- `vlm/base.py`
  - 공통 schema와 prompt 구성 로직을 가진다.
  - VLM 응답을 검증하고, 잘못된 응답은 `suspicious`로 처리한다.

- `vlm/openai_provider.py`
  - OpenAI API를 사용한다.
  - Responses API를 우선 사용하고, 필요 시 chat completions fallback을 사용한다.
  - strict JSON schema 기반 응답을 요구한다.

- `vlm/claude_provider.py`
  - Anthropic Claude API를 사용한다.
  - vision input과 tool use를 통해 구조화된 판정을 받는다.
  - 실패하면 보수적으로 `suspicious`를 반환한다.

- `vlm/mock_provider.py`
  - 로컬 smoke test와 빠른 평가 검증용 provider다.
  - 실제 모델 호출 없이 예측 가능한 결과를 반환한다.

- `vlm/__init__.py`
  - provider 선택을 담당한다.
  - provider가 명시적으로 `mock`일 때만 mock을 사용한다.
  - Claude/OpenAI 키가 없거나 초기화에 실패하면 `UnavailableProvider`를 사용하고, 결과는 보수적으로 `suspicious`가 된다.

## 5. Prompt와 Schema

### 5.1 `prompts/vlm_bug_adjudication.txt`

VLM에게 reference와 test 이미지를 비교하도록 지시하는 핵심 prompt다.

주요 정책:

- reference와 다른 명확한 시각 결함만 bug로 분류한다.
- 투명화, mesh 누락, skin exposure, clipping, layer overlap, broken seam 같은 결함 유형을 중점적으로 본다.
- 의도된 디자인 차이를 결함으로 오판하지 않도록 제한한다.
- reference가 없거나 맞지 않으면 확정 판정 대신 `suspicious`를 사용한다.
- 큰 diff가 있는데 VLM이 정상이라고 단정하지 않도록 한다.
- `safe_to_autopass`는 reference match가 좋고 명확히 정상일 때만 true가 될 수 있다.

### 5.2 `prompts/vlm_schema.json`

VLM 응답 형식을 강제하는 JSON schema다.

주요 필드:

- `final_label`
- `safe_to_autopass`
- `confidence`
- `reference_match_quality`
- `candidate_ids_reviewed`
- `defects`
- `affected_regions`
- `bug_types`
- `visual_evidence`
- `intentional_design_possible`
- `needs_human_review`
- `reason`

schema 검증이 실패하면 최종 시스템은 VLM 응답을 신뢰하지 않고 `suspicious`로 처리한다.

## 6. 설정과 환경 변수

설정의 중심은 `config.py`다.

주요 경로:

- `DATA_DIR`
- `INPUT_DIR`
- `OUTPUT_DIR`
- `REFERENCE_DIR`
- `LABELS_DIR`

탐지 관련 설정:

- YOLO segmentation weight: `yolo11n-seg.pt`
- person class 기반 탐지
- detection threshold
- segmentation threshold

reference workflow 관련 설정:

- `QAI_WORKFLOW_MODE`
- `REFERENCE_DIFF_PIXEL_THRESHOLD`
- `REFERENCE_DIFF_MIN_AREA_RATIO`
- `REFERENCE_DIFF_AREA_SATURATION`
- `REFERENCE_DIFF_TOP_K`
- `REFERENCE_AUTO_PASS_DIFF_THRESHOLD`
- `REFERENCE_SUSPICIOUS_DIFF_THRESHOLD`
- `REFERENCE_BUG_DIFF_THRESHOLD`
- `REFERENCE_MATCH_MIN`

VLM 관련 설정:

- `QAI_VLM_PROVIDER`
- `QAI_VLM_MODEL`
- `OPENAI_MODEL`
- Anthropic API key 변수
- OpenAI API key 변수

현재 로컬 `.env` 점검 기준으로는 Claude와 OpenAI API key가 모두 존재하는 상태로 확인되었다. 키 값 자체는 보안상 이 문서에 기록하지 않는다.

현재 확인된 effective 설정은 다음과 같다.

- provider: `claude`
- Claude model: `claude-opus-4-8`
- OpenAI model: `gpt-5.5`
- Claude key present: true
- OpenAI key present: true

단, 실제 Claude/OpenAI API 호출을 통한 전체 평가 검증은 별도로 수행해야 한다.

## 7. 실행 명령어

### 7.1 Streamlit UI 실행

```powershell
streamlit run app.py
```

브라우저에서 단일 이미지 분석과 폴더 배치 평가를 실행할 수 있다.

### 7.2 Smoke test 실행

```powershell
python eval\smoke_test.py
```

로컬 파이프라인이 기본적으로 동작하는지 확인한다.

### 7.3 Mock provider로 빠른 reference 평가

```powershell
python eval\batch_runner.py --dir data\eval_set --mode reference --provider mock --limit 8
```

실제 API 비용 없이 reference workflow의 구조를 점검한다.

### 7.4 Claude provider로 평가

```powershell
python eval\batch_runner.py --dir data\eval_set --mode reference --provider claude
```

현재 `.env` 기준으로 Claude key가 존재하므로 실행 조건은 갖춰져 있다. 실제 호출 비용과 rate limit은 별도로 고려해야 한다.

### 7.5 OpenAI provider로 평가

```powershell
python eval\batch_runner.py --dir data\eval_set --mode reference --provider openai
```

현재 `.env` 기준으로 OpenAI key가 존재하므로 실행 조건은 갖춰져 있다. 실제 호출 비용과 rate limit은 별도로 고려해야 한다.

## 8. 현재 확인된 검증 상태

이전 로컬 점검에서 다음 결과가 확인되었다.

- `python eval\smoke_test.py` 통과
- reference mode smoke test 통과
- mock provider 기준 bug 샘플 8개 평가에서 8개 모두 bug로 예측
- mock provider 기준 normal 샘플 8개 평가에서 8개 모두 normal로 예측
- `vlm.get_provider()`가 현재 설정에서 Claude provider로 초기화되는 것 확인

주의할 점:

- mock provider 결과는 실제 Claude/OpenAI 성능을 보장하지 않는다.
- normal 이미지를 자기 자신 reference로 사용하는 평가는 실제 운영보다 쉬운 조건이다.
- 실제 정확도 평가는 Claude 또는 OpenAI provider로 전체 eval set을 돌려 CSV 결과를 확인해야 한다.

## 9. 출력물 구조

분석 결과는 주로 `data/output` 아래에 저장된다.

예상 구조:

```text
data/output/
  <case_id>/
    <avatar_id>/
      report.json
      crop.png
      mask.png
      overlay.png
      bg_removed.png
      heatmap.png
      reference_crop.png
      diff_mask.png
      diff_overlay.png
      side_by_side.png
      candidates/
        candidate_*.png
```

배치 평가는 CSV 파일도 생성한다.

CSV에는 일반적으로 다음 정보가 포함된다.

- 이미지 경로
- 정답 라벨
- 예측 라벨
- confidence
- workflow mode
- provider
- reference path
- reference status
- reference diff score
- reference match quality
- VLM 결과 요약
- report path

## 10. 정확도 관점의 현재 설계 평가

현재 코드베이스는 기존 standalone 중심 구조보다 더 나은 정확도를 낼 수 있도록 reference 비교 중심으로 재구성되어 있다.

좋은 점:

- 정상 기준 이미지와 직접 비교하므로 의도된 디자인과 결함을 구분하기 쉽다.
- CV diff가 VLM에게 구체적인 후보 영역을 제공한다.
- VLM 응답이 strict schema로 제한되어 후처리가 안정적이다.
- 실패나 불확실성을 `normal`이 아니라 `suspicious`로 보내는 정책이 있다.
- 배치 평가에서 reference 자동 매칭이 가능하다.

주의할 점:

- reference 이미지가 잘못 매칭되면 결과가 급격히 나빠질 수 있다.
- reference match quality가 낮은 경우 자동 정상 통과가 제한된다.
- API provider 실패는 `suspicious` 증가로 이어진다.
- threshold는 데이터셋 전체 기준으로 추가 튜닝이 필요하다.
- 현재 mock 평가 결과만으로 실제 성능을 단정할 수 없다.

## 11. 현재 코드베이스의 한계

1. 실제 API 기반 전체 평가가 아직 필요하다.
   - Claude/OpenAI provider로 전체 eval set을 실행해야 실제 정확도, 비용, 속도를 판단할 수 있다.

2. reference 품질에 크게 의존한다.
   - 정상 기준 이미지가 없거나 잘못 매칭되면 `suspicious`가 늘어난다.

3. 일부 기존 문서와 UI 문자열이 현재 구현을 완전히 반영하지 않을 수 있다.
   - 특히 기존 README나 오래된 보고서는 현재 reference workflow 이전 내용을 포함할 수 있다.
   - 일부 터미널 출력에서 문자 인코딩 문제가 보일 수 있다.

4. threshold는 아직 운영 데이터 기반으로 확정된 값이 아니다.
   - `REFERENCE_AUTO_PASS_DIFF_THRESHOLD`
   - `REFERENCE_SUSPICIOUS_DIFF_THRESHOLD`
   - `REFERENCE_BUG_DIFF_THRESHOLD`
   - `REFERENCE_MATCH_MIN`

5. standalone 모드는 구조적으로 한계가 있다.
   - 기준 이미지 없이 결함 여부를 판단하므로 reference 모드보다 오탐/미탐 가능성이 높다.

## 12. 권장 운영 방식

현재 코드베이스를 사용할 때 권장되는 기본 방식은 다음과 같다.

1. reference workflow를 기본으로 사용한다.
2. 평가 폴더는 normal/bug 쌍이 자동 매칭되도록 구조화한다.
3. 먼저 mock provider로 pipeline 구조를 빠르게 검증한다.
4. 이후 Claude 또는 OpenAI provider로 전체 eval set을 실행한다.
5. CSV 결과에서 다음 항목을 중점적으로 확인한다.
   - false normal
   - false bug
   - suspicious 비율
   - reference missing 비율
   - reference match quality 분포
   - high diff but non-bug 케이스
6. threshold를 데이터셋 기준으로 조정한다.
7. 반복 평가 결과를 기준으로 prompt와 fusion 정책을 추가 조정한다.

## 13. 결론

현재 코드베이스는 단순 이미지 단독 판정 방식에서 벗어나, 정상 기준 이미지와 테스트 이미지를 비교하는 reference-first 평가 시스템으로 구성되어 있다. CV diff는 결함 후보를 좁히고, VLM은 해당 후보가 실제 결함인지 판단하며, decision fusion은 불확실한 결과를 보수적으로 `suspicious`로 처리한다.

따라서 성능 개선의 핵심은 다음 세 가지다.

- reference 이미지 매칭 정확도 확보
- 실제 Claude/OpenAI provider 기반 전체 평가 실행
- CSV 결과를 이용한 threshold 및 prompt/fusion 정책 튜닝

현재 상태에서는 mock smoke test와 작은 reference 평가가 통과했으므로, 다음 단계는 실제 API provider로 전체 평가를 실행해 정량 지표를 확보하는 것이다.
