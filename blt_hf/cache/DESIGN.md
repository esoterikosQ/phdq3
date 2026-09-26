# BLT 증분 생성: 단계 0 상태 수명 조사

상태: 2026-09-26 조사 중. 실행 정본은
`plan/P3a_generation_cache_20260925.md`다. 이 문서는 캐시가 안전한 위치를
**증명하기 전의 가설**과 검사 결과를 구분한다.

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
| 과거 바이트 entropy | entropy self-attention은 byte-causal, window 512, EOS 구간 마스크 | 전체 입력과 모든 검사 접두부에서 기존 위치의 entropy 및 경계 비교 |
| patch 시작점 | OSC는 고정 threshold를 각 위치 entropy에 적용, `include_next_token=True` | 뒤에 바이트가 붙어도 이전 시작점이 변하지 않는지 비교 |
| 마지막 열린 patch 표현 | pooling/cross-attention은 현재 patch의 바이트를 사용하므로 바뀔 수 있음 | 새 바이트 전후 hidden과 downstream 의존성 추적 |
| 확정 patch의 local encoder 출력 | self-attention은 과거 바이트만 봄; patch-level 출력은 patch 완료 뒤 안정될 것으로 예상 | 각 layer 출력 및 cross-attention query 확인 |
| 확정 patch의 global 출력 | patch-causal mask이므로 앞쪽 patch 출력은 안정될 것으로 예상 | prefix마다 동일 patch position의 hidden 비교 |
| local decoder 과거 상태 | byte self-attention은 인과적이나 patch 교차 입력·encoder 출력에 의존 | 과거 출력이 어느 patch의 잠정 상태를 참조했는지 확인 |

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
3. patch 경계가 안정적이면 작은 모델과 실제 1B에서 local/global/decoder의
   중간 출력 안정성 및 무효화 범위를 측정하는 probe를 추가한다. 출력의
   비교 기준은 동일 디바이스·동일 BF16 경로의 반복 오차를 먼저 측정한다.
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
