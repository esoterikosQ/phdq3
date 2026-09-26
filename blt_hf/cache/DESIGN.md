# BLT 증분 생성: 단계 0 상태 수명 조사

상태: 2026-09-26 단계 0 실측 완료, 단계 1 민감도·시제품 조사 중. 실행 정본은
`plan/P3a_generation_cache_20260925.md`다. 이 문서는 캐시가 안전한 위치를
**증명하기 전의 가설**과 검사 결과를 구분한다.

## 실제 1B BF16 단계 0 결과

itcerdo의 converted BLT-1B, eager/OSC/BF16, RTX 5090에서 가중치 변경·학습
없이 조사했다. 원본 보고서:

- `blt_hf_checks/results/p3a_patch_causality_20260926_v2.json`:
  한국어 25, ASCII 16, 혼합 36, 512-window 56, EOS 26개 접두부를 검사했다.
  혼합 입력 5개 접두부에서 이전 patch 시작점이 바뀌었고 다른 fixture에서는
  경계 불일치가 없었다. 예를 들어 시작점 19를 결정하는 entropy가 전체
  입력에서는 1.3359375, 길이 22 접두부에서는 1.3515625였다. 유효 BF16
  threshold 1.3359375를 사이에 두므로 경계가 뒤집혔다.
- `blt_hf_checks/results/p3a_state_stability_repeat_20260926.json`:
  혼합·긴 문장 총 15개 접두부에서 닫힌 global patch와 decoder의 과거
  출력이 모두 바뀌었다. 이 비교의 최종 위치 argmax는 15/15 일치했지만
  전체 생성 출력이 일치한다는 뜻은 아니다. 같은 입력 길이를 다시 실행한
  entropy, encoder, global, decoder, logits는 15/15에서 비트 단위로
  동일했다. 입력 길이 변화가 수치 차이의 원인이다.
- `blt_hf_checks/results/p3a_forward_profile_20260926.json`:
  사전학습 가중치·반복 한글 합성 입력 64/128/256/512바이트의 full forward
  중앙값은 16.50/16.66/17.28/19.61ms, patcher는
  3.73/3.81/4.04/4.90ms, global은 7.25/7.27/7.29/7.29ms다.
  5090 한 대의 합성 입력 시간이며 A100의 실제 파인튜닝 모델 생성 속도나
  캐시 이득을 보장하지 않는다.

수학적으로 threshold patching이 인과적이어도, 현재 BF16 kernel의 텐서
형상 의존 연산 때문에 실제 entropy가 조금씩 변한다. 그러므로 최초 시제품은
매 step **전체 patcher를 다시 실행**해 현재 no-cache와 같은 경계를 사용한다.
`frontier.shared_closed_patch_count`는 경계가 같은 *구조적* 닫힌 구간만 찾으며
activation이 동일함을 보증하지 않는다. 닫힌 상태 재사용은 출력 ID 전수 비교
전에는 실험용으로만 취급한다.

## 단계 1 시제품의 현재 한계

`check_reuse_sensitivity.py`는 전체 global 계산 뒤 닫힌 patch 출력만 이전
step의 값으로 바꿨다. 사전학습 1B의 한국어·혼합·긴 입력에서 32 step씩 총
96개 다음 바이트 ID는 같았으나 logits는 최대 0.125 차이가 났다. 이 검사는
계산을 건너뛰지 않으므로 속도 결과가 아니다.

`cache/global_reuse.py`는 매 step **전체 entropy patcher와 local encoder**를
유지한다. 시작점이 그대로이고 patch가 둘 이상이면 decoder가 현재 열린
patch의 global 출력을 마지막 byte에서 참조하지 않는 특성을 이용해 global
forward를 건너뛴다. 새 시작점이 생기면 공통으로 닫힌 patch의 KV만 남기고
global tail을 재계산한다. 선택형 decoder KV 재사용은 경계가 그대로인 step에만
적용하고, 경계가 바뀌면 decoder 전체를 다시 계산한다. 이 구현은 batch 1,
빔 1, 연속 접두부, `eval()`과 `inference_mode()`로 제한된 **진단 시제품**이다.
module.forward를 한 실행 동안만 교체하므로 병렬 호출·학습·HF `generate`
기본 경로에 사용하지 않는다.

`p3a_global_skip_bench_20260926.json`에서 세 입력 각 32 step의 다음 ID
불일치는 0개였고 global pass를 건너뛴 step은 각각 27/27/21개였다.
같은 실행의 기준 대비 평균 forward 시간 개선은 1.49/1.45/1.35배였다.
`p3a_global_decoder_nocopy_20260926.json`의 global+decoder 시제품도
96/96 다음 ID 일치, decoder 재사용 27/27/21 step, 평균 개선
1.49/1.50/1.43배였다. 모두 **5090의 사전학습 가중치·합성 greedy
입력·짧은 forward** 검사다. 다른 시점의 5090 실행에서는 기준 시간이
약 16ms와 60ms 사이로 크게 달랐다. 따라서 2배 채택 목표 달성이나
A100 본 평가의 출력을 주장할 수 없다.

최종 EOS 재사용 차단·KV 복사 제거·반복 경계 검사 반영 뒤
`p3a_global_final_20260926.json`에서도 96/96 다음 ID 일치,
global/decoder 재사용 27/27/21 step, 평균 시간 개선 1.50/1.56/1.49배였다.
이 보고서의 `reuse_code_sha256`이 현재 시제품 소스를 식별한다.
`p3a_global_report_schema_20260926.json`은 최종 bench 코드의 8-step
보고서 필드(`probe_token_parity`, `peak_allocated_bytes`, 합산 속도)를
확인한 짧은 실행이며 32-step 속도 비교를 대신하지 않는다.

다음 판정은 실제 fine-tuned checkpoint와 native validation의 대표 문장을
A100에서 같은 job 내 기준 경로와 시제품으로 비교하는 것이다. 준비된
`scripts/bench_blt_cache.sh`는 사용자가 Neuron에서만 실행한다. mismatch가
나오거나 전체 validation 2배 개선이 없으면 기존 no-cache 평가 경로를
유지한다. 이 작은 probe 결과를 근거로 GLEU 선택 backend를 바꾸지 않는다.

## 기존 생성 경로

`blt_hf/generation.py`는 생성할 때마다 `[BOS] source SEP generated_prefix`를
`BltForCausalLM.generate(use_cache=False)`에 준다. `BltModel.forward()`는 매번
전체 바이트 hash embedding → entropy patcher → local encoder → global
transformer → local decoder를 실행한다. `attention.py:require_unpadded()`는
OSC에서 cache 입력을 거부하며 `BltModel.forward()`의 global transformer
호출은 `past_key_values`를 전달하지 않는다.

## 후보 재사용 경계

| 상태 | 현재 근거 | 확정 전 검사 |
|---|---|---|
| 과거 바이트 entropy | 같은 길이 재실행은 동일하지만 길이 변경 시 기존 값도 바뀜 | 초기 시제품은 전체 patcher 재실행 |
| patch 시작점 | 혼합 fixture 5/36에서 과거 시작점 변경, `include_next_token=True` | 매 step 새 경계를 비교하고 처음 바뀐 patch 전부터 무효화 |
| 마지막 열린 patch 표현 | pooling/cross-attention은 현재 patch의 바이트를 사용하므로 바뀔 수 있음 | 새 바이트 전후 hidden과 downstream 의존성 추적 |
| 확정 patch의 local encoder 출력 | 일부 닫힌 patch에서도 BF16 값 차이 관측 | output parity 없이는 재사용 금지 |
| 확정 patch의 global 출력 | 15/15 비교에서 닫힌 patch 값도 달라짐 | 재사용 시 다음 바이트 ID 민감도 측정 필수 |
| local decoder 과거 상태 | 15/15 비교에서 과거 byte 값도 달라짐 | 의존 구간과 output parity 측정 필수 |

`decoder_patch_ids = _patch_ids_from_lengths(patch_lengths[:, 1:], sequence_length)`
는 patch 한 칸 이동에 해당한다. 이것이 열린 patch의 변화가 과거 decoder
상태에 미치는 범위를 어디까지 제한하는지 실제 출력으로 검증한다. 안정성이
확인되지 않은 상태를 무조건 캐시하지 않는다.

## 단계 0의 검사 순서

1. 로컬 `tests/test_hf_cache_causality.py`에서 next-token 가상 슬롯의
   patch 길이 의미와 경계 비교기의 실패 감지를 검사한다.
2. BF16 GPU에서 `python -m blt_hf_checks.check_patch_causality --output <새 JSON>`를
   실행한다. 실제 1B entropy patcher에 짧은 한국어·영어·혼합 입력,
   512-window 경계, EOS 구간을 넣어 접두부 경계 안정성을 기록한다.
3. 실제 1B에서 local/global/decoder의 중간 출력 안정성과 동일 길이 반복
   오차를 측정했다. 반복 오차는 0이었으며 길이 변화 차이는 위에 기록했다.
4. `blt_hf_checks/bench_generation.py`로 동일 checkpoint의 현재 no-cache
   baseline을 A100에서 측정한다. 결과는 추후 증분 backend 성능 비교에 사용한다.

이 단계의 출력은 `check_patch_causality` 보고서, 구성요소별 상태 수명 보고서,
벤치마크 JSON이다. 이 증거가 나오기 전에는 증분 상태 구현을 시작하지 않는다.

## 소스와 운영 제약

- 모델 가중치·tokenizer·학습 코드는 변경하지 않는다. `blt_gec/`, 원본
  데이터, site-packages도 수정하지 않는다.
- Neuron에 에이전트가 접속하지 않는다. itcerdo는 사용자 제공 환경에서
  forward 검사만 수행하며 optimizer·학습 작업은 하지 않는다.
- 관련 소스: `blt_hf/patched/modeling_blt.py`, `blt_hf/patching.py`,
  `blt_hf/attention.py`, Meta의
  [threshold 패처](https://github.com/facebookresearch/blt/blob/main/bytelatent/data/patcher.py),
  [no-cache 생성](https://github.com/facebookresearch/blt/blob/main/bytelatent/generate_blt.py).
