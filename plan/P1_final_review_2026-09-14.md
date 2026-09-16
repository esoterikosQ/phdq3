# P1 최종검수 — 2026-09-14

> **사용자 설명 반영 및 문서 수정 기록 (2026-09-14)**
>
> 아래 본문은 수정 전 문서에 대한 검수 기록이며, 당시 줄 번호와 제안을 담는다.
> 현재 실행 정본은 수정된 `plan/P1.md`와 `plan/CLAUDE_P1_blt_hf.md`다.
> 2번은 계획 방향의 오류가 아니라 실행 순서를 잘못 전달한 표현 문제로 정정한다.
> 검증 완료를 기다리지 않고 파인튜닝을 본 학습까지 진행하고, 그동안 검증 규칙을
> 재검토한 뒤 검증한다. 동질성이 확인되지 않으면 현재 결과를 채택하지 않고
> 후속 계획을 새로 세워 다음 실험을 수행한다. 검수 당시 제안한 본 학습 선행
> parity 게이트는 채택하지 않았다.
> 1번 BOS/EOS 명시 삽입, 3번 자체 manifest 확장, 6번 경로·명령·무절단 표현은 반영했다.
> 3번은 HF 공식 클래스의 필드 누락이 아니라 P1 자체 manifest 설계의 누락이었다.
> 4번의 gold M2는 `data/Preprocessed/`와 `data/Raw/`에 존재한다. 없는 것은
> legacy wrapper가 참조하는 scorer 코드이며, 이를 구분해 평가 준비 작업에 명시했다.
> 제공 데이터와 제3자 validation/test는 그대로 사용한다. 5번의 중복 통계는
> 참고 기록으로 유지하되 재분할·중복 제거·평가 subset 대체 제안은 채택하지 않는다.

## 검수 JSON 생성 방법

`P1_final_review_data_2026-09-14.json`은 Python 표준 라이브러리
`pathlib`, `hashlib`, `json`으로 로컬 파일을 읽어 만든 집계다.
HF/BLT 모델, 공식 클래스, tokenizer를 실행하거나 attention mode를 측정한 결과가 아니다.

- `native`, `korean_learner`, `union`의 canonical train/val/test TSV 9개를 읽었다.
- 빈 줄을 제외하고 탭으로 source/target을 나눠 행 수와 형식을 집계했다.
- `len(text.encode("utf-8"))`로 길이를 계산했다. 학습열 길이는
  `1 + source_bytes + 15 + target_bytes + 1`, prompt는 `1 + source_bytes + 15`다.
  BOS/EOS 길이 1과 separator 15는 계획의 인코딩 계약에 근거한 계산이며
  실제 tokenizer 실행 결과가 아니다.
- 각 train에서 source 및 `(source, target)` 집합을 만들고, val/test의 각 행이
  그 집합에 있는지를 문자열 exact match로 세었다. 중복 제거로 데이터를 변경하지 않았다.
- val/test M2 6개의 `S ` 행을 추출해 문장 수와 TSV source의 `str.split()` 결과를
  비교했다. 공백 토큰화는 비교용이며 원본 파일을 수정하지 않았다.
- canonical TSV 원본 바이트의 SHA256을 기록하고 집계만 JSON으로 저장했다.

---


검수 대상: `plan/P1.md`, `plan/CLAUDE_P1_blt_hf.md`.

판정: **현 문서를 변경 없이 실행 지침으로 확정하는 것은 보류.** 재구축 방향을 폐기할 문제는 없지만, 학습 데이터 인코딩·검증 게이트·평가 재현성에 영향을 주는 수정 사항이 있다. 아래 P1 항목을 계획에 반영한 뒤 Phase A를 시작하는 것이 적절하다. GPU parity와 본 학습 성공을 이번 검수로 보증한 것은 아니다.

원본 계획 두 파일과 legacy 코드는 수정하지 않았다. 로컬 데이터는 읽기만 했으며, 재계산한 집계와 파일 SHA256은 [검수 데이터](/Users/esoterikos/Library/CloudStorage/Nextcloud-cloud.fruittravelers.net-monbug/projects_on/phdq3/plan/P1_final_review_data_2026-09-14.json)에 저장했다.

## 1. [P1] EOS 자동 삽입을 확정 사실로 전제하면 target EOS supervision이 빠질 수 있다

위치: [P1.md:709](/Users/esoterikos/Library/CloudStorage/Nextcloud-cloud.fruittravelers.net-monbug/projects_on/phdq3/plan/P1.md:709), [P1.md:1207](/Users/esoterikos/Library/CloudStorage/Nextcloud-cloud.fruittravelers.net-monbug/projects_on/phdq3/plan/P1.md:1207).

A-7-1 항목 6은 `add_bos_token=True, add_eos_token=True`를 실제 자동 삽입 동작으로 확정하고, B-1에서 이에 대응하라고 지시한다. 그러나 확인한 공식 변환기의 `create_tokenizer_json()`은 다음 post-processor를 만든다.

```python
single=f"{bos}:0 $A:0"
pair=f"{bos}:0 $A:0 $B:1"
special_tokens=[(bos, 1)]
```

EOS는 이 처리 규칙에 없다. `tokenizer_config.json`의 boolean 값만 보고 EOS가 붙는다고 판단할 수 없다. 이 지시를 따르면 target 마지막 EOS가 누락되어 종료 학습을 하지 않거나, prefix와 target을 따로 인코딩하면서 target에 불필요한 BOS를 삽입할 수 있다.

수정: “자동 삽입 확정”을 삭제하고 **실제 반환 token IDs**로 확인한다. 안전한 계약은 본문을 `add_special_tokens=False`로 인코딩하고 학습에서는 BOS/EOS를 각 한 번 명시적으로 붙이며, 생성 prompt에는 BOS만 붙이는 것이다. 학습열 `[BOS]+source+SEP+target+[EOS]`, prompt `[BOS]+source+SEP`, EOS loss 포함, 내부 label shift를 테스트로 고정한다. 현재 B-1 테스트에 있는 전체 인코딩 검사를 실제 게이트로 삼으면 발견 가능하지만, 서로 반대인 구현 지시를 먼저 없애야 한다.

근거: [공식 변환기](
https://github.com/huggingface/transformers/blob/main/src/transformers/models/blt/convert_blt_weights_to_hf.py). 검수 시점 main 소스에 대한 정적 확인이며, 착수 시 vendoring한 SHA에서도 같은 계약을 재검사해야 한다.

## 2. [P1] patcher 실패 시 학습 결과를 유지한다는 규칙이 본 학습 게이트와 충돌한다

위치: [P1.md:79](/Users/esoterikos/Library/CloudStorage/Nextcloud-cloud.fruittravelers.net-monbug/projects_on/phdq3/plan/P1.md:79), [P1.md:848](/Users/esoterikos/Library/CloudStorage/Nextcloud-cloud.fruittravelers.net-monbug/projects_on/phdq3/plan/P1.md:848), [CLAUDE_P1_blt_hf.md:33](/Users/esoterikos/Library/CloudStorage/Nextcloud-cloud.fruittravelers.net-monbug/projects_on/phdq3/plan/CLAUDE_P1_blt_hf.md:33).

P1 앞부분은 A-3 patcher 검증이 “파인튜닝 정당성과 무관”하고 실패해도 롤백하지 않는다고 한다. CLAUDE 규칙 3은 A-7-4와 A-5 두 조건만 본 학습 승격 조건으로 요약한다. 반면 뒤의 정식 승격 체크리스트는 A-3 완료와 B-1 전체 데이터 검사도 필수로 요구한다.

패치 경계는 학습 forward의 global 입력과 decoder가 참조하는 patch를 바꾼다. 학습에 사용한 patcher의 오류를 수정하면서 경계가 바뀌었다면, 앞서 만든 checkpoint를 그대로 “같은 OSC 모델의 정상 학습 결과”로 승격할 수 없다. 현재 HF 고정 버전도 `BltModel.forward()`에서 patcher 출력을 사용해 patch IDs와 cross-attention mask를 만든다.

수정:
- 하나의 게이트 정의를 정본으로 두고 두 문서가 이를 동일하게 참조하게 한다.
- 본 학습은 OSC mask 구현·padding 검증, B 기준 호환성, A-3 plain↔HF, A-7-4, A-5, B-1 통과 후 시작한다.
- A-3 실패 중 **실제 학습 patcher/설정/경계를 바꾸는 수정**은 기존 smoke를 폐기하거나 별도 variant로 격리한다.
- 참고용 legacy 비교만 실패하고 학습 경로가 바뀌지 않은 경우는 분리한다.
- A-3-0에서 legacy 참고 사용을 배제한 분기에서는 “HF vs legacy까지 일치”를 필수로 요구하지 않는다.

추가로 P1:860의 “A-7-4 층 1 전수(전체 입력 hidden state)”는 A-7-4에서 층 1을 정적 가중치 검사로 정의한 것과 모순된다. 삭제하거나 의도한 검사명으로 바꿔야 한다.

근거: [고정 버전 HF BLT 구현](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/blt/modeling_blt.py)의 `BltModel.forward()` 정적 확인.

## 3. [P1] aggregate fingerprint에 attention_mode와 데이터 식별자가 빠져 있다

위치: [P1.md:799](/Users/esoterikos/Library/CloudStorage/Nextcloud-cloud.fruittravelers.net-monbug/projects_on/phdq3/plan/P1.md:799), [P1.md:1117](/Users/esoterikos/Library/CloudStorage/Nextcloud-cloud.fruittravelers.net-monbug/projects_on/phdq3/plan/P1.md:1117).

A-7-4는 `attention_mode: osc|lre` 혼합을 aggregate에서 차단한다고 명시하지만, 실제 `FINGERPRINT_KEYS`에는 이 필드가 없다. 같은 checkpoint·code commit에서 실행 인자만 OSC/LRE로 바뀌면 현재 목록의 값은 모두 같을 수 있다. 이때 다른 모델 의미로 생성한 shard가 조용히 합쳐진다.

또한 데이터가 git에서 제외되고 노드 간 별도 동기화되는 구조인데 source/target/M2 파일 hash가 없다. 같은 길이·샘플 수를 유지한 데이터 변경은 현재 길이 검사와 shard 범위 검사로 검출되지 않는다.

수정: 최소한 `attention_mode`, 실제 모델 config hash, tokenizer artifact hash, dataset/split과 순서가 고정된 source/target/M2 hash를 fingerprint에 넣는다. 학습·평가에서 선택한 patched 모델 구현 및 mask 설정을 manifest로 검증한다. B-2 로딩 예제도 일반 `BltForCausalLM` 직접 호출 대신 검증된 OSC 경로를 선택하는 공통 loader를 사용하도록 명시하는 것이 안전하다. 잘못된 mode나 데이터 hash를 가진 shard를 섞는 음성 테스트가 필요하다.

## 4. [P1] 평가 wrapper만 이식하면 실제 GLEU/M2 scorer가 없어 실행되지 않는다

위치: [P1.md:1126](/Users/esoterikos/Library/CloudStorage/Nextcloud-cloud.fruittravelers.net-monbug/projects_on/phdq3/plan/P1.md:1126), [CLAUDE_P1_blt_hf.md:20](/Users/esoterikos/Library/CloudStorage/Nextcloud-cloud.fruittravelers.net-monbug/projects_on/phdq3/plan/CLAUDE_P1_blt_hf.md:20).

현재 저장소에는 `baseline/`이 없다. 그런데 이식 원본은 다음 의존성을 가진다.

- [blt_gec/metrics.py:10](/Users/esoterikos/Library/CloudStorage/Nextcloud-cloud.fruittravelers.net-monbug/projects_on/phdq3/blt_gec/metrics.py:10): `baseline.metric.gleumodule.run_gleu` import.
- [blt_gec/metrics.py:38](/Users/esoterikos/Library/CloudStorage/Nextcloud-cloud.fruittravelers.net-monbug/projects_on/phdq3/blt_gec/metrics.py:38): `baseline/metric/m2scorer/scripts/m2scorer.py` 실행.
- [blt_gec/m2_resumable.py:19](/Users/esoterikos/Library/CloudStorage/Nextcloud-cloud.fruittravelers.net-monbug/projects_on/phdq3/blt_gec/m2_resumable.py:19): 같은 scorer 디렉터리에서 `levenshtein` 로딩.

두 wrapper를 `blt_hf/`에 복사하고 import를 바꾸는 것만으로는 독립 평가가 완성되지 않는다. KoBART 비교표도 현재 저장소에 baseline 실행·artifact 회수 경로가 명시되어 있지 않다.

수정: 실제 GLEU/M2 scorer 본체와 필요한 부속 파일을 어디에서 어떤 commit으로 회수해 어디에 둘지, dependency와 사용 조건을 계획에 추가한다. legacy 환경에서 고정 hypothesis에 대한 기대 점수를 확보하고 새 환경의 독립 scorer와 비교한다. KoBART는 동일 split/source hash의 기존 hypothesis·점수를 회수할지 재실행할지도 명시한다. 이 확인은 학습 완료 후가 아니라 Phase A의 CPU 검사에서 끝낼 수 있다.

## 5. [P1: 연구 타당성] union 분할에 상당한 source 중복이 있는데 평가 해석 규칙이 없다

위치: [P1.md:892](/Users/esoterikos/Library/CloudStorage/Nextcloud-cloud.fruittravelers.net-monbug/projects_on/phdq3/plan/P1.md:892), [P1.md:1173](/Users/esoterikos/Library/CloudStorage/Nextcloud-cloud.fruittravelers.net-monbug/projects_on/phdq3/plan/P1.md:1173).

현재 canonical TSV를 직접 읽어, 각 데이터셋의 val/test **행**이 같은 데이터셋의 train에 존재하는지 문자열 exact match로 재계산했다.

| 데이터셋 / 평가 split | 평가 행 수 | train과 source 동일한 행 | train과 source-target 쌍까지 동일한 행 |
|---|---:|---:|---:|
| native / val | 2,634 | 0 | 0 |
| native / test | 2,634 | 3 | 0 |
| korean_learner / val | 4,264 | 36 | 14 |
| korean_learner / test | 4,265 | 36 | 15 |
| union / val | 23,332 | 4,613 | 109 |
| union / test | 23,333 | 4,608 | 115 |

union test의 **19.75%**는 학습에서 본 source다. 그중 3,632행은 5어절 이상이라서 짧은 상투 문장 중복으로만 설명할 수 없다. 다른 target이 붙은 source 중복은 완전히 같은 정답 쌍의 유출과는 구분해야 하지만, 미관측 문장에 대한 일반화 평가라고 해석할 수는 없다. 이는 이번 계획이 새로 만든 오류가 아니라 기존 split을 그대로 사용하는 계획이 이어받는 문제다.

수정: 데이터를 임의로 다시 나누기보다, 기존 split 재평가의 비교 가능성을 유지할지와 일반화 성능 평가를 어떻게 보완할지 문서에서 확정한다. 기존 benchmark를 유지한다면 중복률과 해석 제한을 보고하고, train-source 비중복 subset의 점수를 같은 KoBART/BLT hypothesis에서 함께 산출하는 방안이 가능하다. 이후 중복 제거 split을 새로 만든다면 별도 버전으로 고정하고 양 모델을 동일 조건으로 재학습·재평가해야 한다. 이번 검수에서는 어떤 데이터도 변경하지 않았다.

## 6. [P2] CLAUDE 배치 경로·상위 지침·명령이 현재 P1과 맞지 않는다

위치: [CLAUDE_P1_blt_hf.md:3](/Users/esoterikos/Library/CloudStorage/Nextcloud-cloud.fruittravelers.net-monbug/projects_on/phdq3/plan/CLAUDE_P1_blt_hf.md:3), [CLAUDE_P1_blt_hf.md:105](/Users/esoterikos/Library/CloudStorage/Nextcloud-cloud.fruittravelers.net-monbug/projects_on/phdq3/plan/CLAUDE_P1_blt_hf.md:105), [CLAUDE_P1_blt_hf.md:132](/Users/esoterikos/Library/CloudStorage/Nextcloud-cloud.fruittravelers.net-monbug/projects_on/phdq3/plan/CLAUDE_P1_blt_hf.md:132).

다음은 실행 전 함께 정리할 항목이다.

- 배치 위치와 상위 지침은 `PHDQ2`로 적혀 있으나 실제 대상은 PHDQ3다. 현재 루트에는 참조되는 `CLAUDE.md`가 없다. 상세 계획 파일명도 존재하지 않는 `P1_HF_BLT_GEC_재구축_계획.md`다. 실제 `plan/P1.md`와 노드 운영 지침의 회수/배치 경로를 지정해야 한다.
- 주요 명령의 `--model-id`는 P1의 bench 인터페이스 `--model-path`와 다르다.
- 변환 명령에 필수 `--model-revision`이 빠져 있다. vendored CLI가 이를 required로 처리하고 CLAUDE 예제에도 넣어야 한다.
- “source 절단은 UTF-8 문자 경계에 맞춘다”는 규칙은 P1의 “main run 절단 금지, 초과 시 실패”와 충돌한다. CLAUDE도 무절단을 기본으로 명시해야 한다.
- 환경 고정은 `torch==2.11.0+cu128`인데 설치 예제는 pin 없는 `pip install torch`다. 명령 자체에도 정확한 버전을 넣는다.
- CLAUDE 개요의 “각 Phase 종료 게이트 이후 다음 Phase”는 병렬 트랙 설명에 맞춰 정리한다.

## 확인되어 문제로 보지 않은 사항

- 9개 split 유효 행 합계 201,534, malformed TSV 0.
- 최대 학습 입력 1,380, 최대 생성 prompt 694, 최대 target 685.
- 1,024 초과 9건, 2,048 초과 0건. 현재 데이터에 대한 2,048 무절단 정책과 `max_new_bytes=768`의 길이 근거는 일치한다.
- union train/val/test 빈 줄은 각각 1개다.
- 6개 val/test M2의 문장 수가 대응 TSV와 일치하고, source의 공백 토큰화 결과도 전부 일치한다. 일부 원문 공백 표현 차이는 있었으나 M2 source 오정렬로 판정하지 않았다.
- [PyPI의 transformers 5.16.1](https://pypi.org/pypi/transformers/5.16.1/json)과 [PyTorch cu128 인덱스](https://download.pytorch.org/whl/cu128/torch/)에서 계획의 버전 존재를 확인했다. 이것만으로 실제 두 원격 노드의 driver/runtime 실행 성공이 증명되지는 않는다.
- 변환 provenance, B 기준 검증, legacy 동결, 무절단, 동일 GPU parity, cache 실패의 별도 후속 과제 분리는 유지할 만한 설계다.

## 검수 범위와 제한

두 문서 전량, 로컬 legacy 모델·어댑터·학습·평가 관련 코드, 참조된 이전 DDP/생성 계획, canonical TSV 9개와 val/test M2 6개를 확인했다. HF v5.16.1 BLT 구현과 공식 변환기·Meta 원본 마스크 코드도 정적으로 대조했다.

원격 itcerdo/neuron에 접속하거나 가중치를 다운로드해 GPU 테스트를 실행하지 않았다. 현재 로컬 Python에는 torch/transformers/tokenizers가 없어 토크나이저 및 모델 runtime 재현은 수행하지 않았으며, 해당 지적은 실제 소스의 처리 규칙에 근거한다. 아직 구현되지 않은 Phase A 검증이 미완료라는 이유 자체를 결함으로 세지는 않았다.

