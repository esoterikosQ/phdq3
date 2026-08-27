# BLT-GEC Generation Renewal Plan

## 1. 목적

현재 BLT-GEC generation은 출력 품질 평가에는 사용할 수 있지만, 레퍼런스 BLT가 제공하는 prompt batching을 사용하지 않고 모든 문장과 beam을 순차 처리한다. 또한 매 바이트 생성 시 전체 prefix의 entropy patching과 BLT forward를 다시 수행한다. 이 때문에 현재 측정되는 긴 실행 시간은 BLT 아키텍처의 본질적인 비용뿐 아니라 generation 구현의 병목을 크게 포함한다.

이 문서의 목적은 다음 두 목표를 분리하여 달성하는 것이다.

1. 현재 모델과 체크포인트의 출력 의미를 보존하면서 GPU 내부 병렬성을 복원한다.
2. 이후 patch-aware incremental generation을 구현하여 BLT 논문이 가정하는 추론 구조에 접근한다.

현재 진행 중인 평가 shard에는 기존 generation 설정을 유지한다. 새로운 backend는 별도 출력 경로에서 처음부터 다시 평가하며, 서로 다른 backend의 shard를 하나의 aggregate 결과로 혼합하지 않는다.

---

## 2. 현재 구현 진단

### 2.1 현재 실행 구조

현재 `blt_gec/eval.py`는 예제를 순차적으로 호출한다.

```text
for example in examples:
    generate_correction(example.source)
```

`generate_correction()` 내부에서는 다음 중첩 반복이 발생한다.

```text
문장별 순차 실행
  └── 생성 바이트별 순차 실행
       └── beam별 순차 실행
            ├── 전체 prefix tensor 재생성
            ├── 전체 prefix entropy patching
            ├── 전체 BLT forward
            └── 다음 바이트 후보 선택
```

기본 `num_beams=4`에서는 첫 단계에 4개 후보를 만들고, 이후 최대 16개 후보를 비교한 뒤 4개 beam만 유지한다. 후보는 patch가 아니라 단일 바이트다.

### 2.2 레퍼런스 대비 손실된 기능

레퍼런스 `generate_blt.py::generate_nocache()`는 KV cache는 없지만 여러 prompt를 하나의 tensor로 처리한다. 현재 GEC 구현은 GEC separator, EOS 종료, beam search를 추가하는 과정에서 이 prompt batching을 보존하지 못했다.

GEC adaptation에 실제로 필요한 변경은 다음과 같다.

- `source + separator` 형태의 prompt 구성
- 생성 결과에서 prompt 제거
- EOS별 조기 종료
- GEC용 전체 길이 제한
- 필요한 경우 beam search 추가

이 요구사항은 단일 문장 API를 강제하지 않는다. FlashAttention/xFormers 호환성 우회 역시 모델 로딩 및 attention backend의 문제이며, prompt batching을 제거할 이유가 아니다.

### 2.3 캐시 관련 상태

레퍼런스 저장소에는 두 generation 경로가 있다.

| 경로 | 대상 | Prompt batching | KV cache |
|---|---|---:|---:|
| `generate_blt.py::generate_nocache()` | `ByteLatentTransformer` | 지원 | 미지원 |
| `generate.py::PackedCausalTransformerGenerator` | flat `LMTransformer` | 지원 | 지원 |

일반 KV-cache generator는 `tok_idx`, `mask`, `attn_impl` 인자를 받는 `LMTransformer.forward()`를 가정한다. BLT의 `forward(tokens, patch_lengths, ngram_ids)` 및 local encoder/global patch transformer/local decoder 상태를 관리하지 않으므로 그대로 사용할 수 없다.

### 2.4 현재 결과 해석

- 현재 생성 결과와 GLEU는 동일 backend와 설정 안에서는 유효하다.
- 현재 실행 시간은 BLT의 공정한 inference efficiency 지표로 사용하면 안 된다.
- 현재 shard들은 beam 4와 기존 scalar backend로 끝까지 완료해야 한다.
- backend나 beam 수를 중간에 변경한 shard는 기존 결과와 aggregate하면 안 된다.

---

## 3. 갱신 원칙

### 3.1 정확성 우선

성능 최적화 전에 다음 동등성을 확보한다.

1. `batch_size=1`, `num_beams=1`에서 레퍼런스 BLT greedy 출력과 일치
2. `batch_size=1`, `num_beams=4`에서 현재 scalar beam 출력과 일치
3. 동일한 token ID, EOS 위치 및 length-normalized score를 재현
4. UTF-8 decode와 빈 출력 처리 결과를 재현

부동소수점 backend 차이로 logits가 완전히 같지 않을 경우, token 선택과 최종 문자열의 일치를 우선 검증하고 logits 차이는 허용 오차와 함께 기록한다.

### 3.2 단계적 변경

다음 단계를 한 번에 구현하지 않는다.

1. batched greedy no-cache
2. batched beam no-cache
3. 평가 파이프라인 통합
4. entropy patching 증분화
5. BLT patch-aware KV cache

각 단계는 이전 단계와 출력 동등성 및 성능을 비교한 후 병합한다.

### 3.3 레퍼런스 코드 보존

`reference_code/blt`를 직접 수정하지 않는다. GEC adapter와 최적화 backend는 `blt_gec` 아래에 구현하고, 레퍼런스 함수는 동등성 검증용 기준으로 유지한다.

---

## 4. 1단계: Batched Greedy No-cache

### 4.1 목표

레퍼런스 `generate_nocache()`의 prompt batching을 GEC 입력 형식에 복원한다. 이 단계에서는 KV cache를 구현하지 않고, 모델 호출 횟수를 문장 수만큼 반복하는 문제부터 제거한다.

### 4.2 제안 API

```python
@dataclass(frozen=True)
class GenerationConfig:
    max_length: int
    max_gen_len: int
    num_beams: int = 1
    length_penalty: float = 1.0


def generate_corrections(
    model,
    tokenizer,
    patcher,
    sources: list[str],
    *,
    separator: str,
    config: GenerationConfig,
    device: torch.device,
) -> list[str]:
    ...
```

기존 `generate_correction()`은 하위 호환 wrapper로 남긴다.

```python
def generate_correction(..., source: str, ...) -> str:
    return generate_corrections(..., sources=[source], ...)[0]
```

### 4.3 Batched greedy 처리

각 batch에서 다음 상태를 관리한다.

- prompt token 목록
- 문장별 prompt 길이
- 문장별 현재 생성 길이
- EOS 도달 여부
- 생성 token 목록

각 생성 단계에서는 활성 문장을 한 tensor로 묶어 다음을 한 번씩 수행한다.

```python
patch_lengths = patcher.patch(active_tokens, include_next_token=True)[0]
logits = model(active_tokens, patch_lengths=patch_lengths)[:, -1]
next_tokens = logits.argmax(dim=-1)
```

길이가 다른 prompt는 다음 중 하나로 처리한다.

1. 비슷한 byte 길이끼리 bucket을 구성한다.
2. 레퍼런스 `generate_nocache()`처럼 긴 prompt의 기존 입력 위치를 유지하면서 짧은 prompt부터 생성을 시작한다.

초기 구현은 길이 bucket을 우선한다. 동적 patching에서 무효 padding이 patch 경계에 영향을 주지 않도록 실제 model 입력에는 embedding 범위를 벗어나는 `PAD_ID=-1`을 포함하지 않는다.

### 4.4 EOS 처리

문장별 EOS 상태를 별도로 관리한다. 한 문장이 EOS에 도달해도 다른 문장이 끝날 때까지 batch 전체를 종료하지 않는다.

완료된 문장을 계속 tensor에 남겨야 한다면 유효한 EOS ID로 채우되 결과에는 추가하지 않는다. 이후 활성 문장만 압축하여 새 batch로 만드는 방식과 성능을 비교한다.

### 4.5 입력 절단 수정

현재 코드는 긴 prompt의 뒷부분을 제거하여 separator가 잘릴 수 있다.

```python
prompt_ids = prompt_ids[: max_length - 1]
```

갱신 구현은 separator를 반드시 보존한다.

```text
허용 source 길이 = max_length - separator 길이 - 최소 생성 여유
source 앞부분을 절단하거나 명시적으로 오류 처리
최종 prompt는 항상 separator로 끝남
```

GEC에서는 문장 끝부분이 교정에 중요하므로 기본 정책은 source 앞부분 절단으로 한다. 절단 발생 횟수는 평가 메타데이터에 기록한다.

---

## 5. 2단계: Batched Beam Search

### 5.1 목표

현재 Python beam 반복을 tensor 연산으로 대체한다.

```text
현재:
for source:
    for byte:
        for beam:
            model.forward()

갱신:
for byte:
    model.forward(batch_size × beam_size)
```

### 5.2 Tensor 구조

```text
sequences: [batch, beam, sequence]
scores:    [batch, beam]
done:      [batch, beam]
```

모델 호출 직전에 다음처럼 평탄화한다.

```python
flat_sequences = sequences.reshape(batch_size * beam_size, sequence_length)
patch_lengths = build_patch_lengths(patcher, flat_sequences)
logits = model(flat_sequences, patch_lengths=patch_lengths)[:, -1]
```

후보 점수는 다음 shape로 계산한다.

```text
log_probs:        [batch, beam, vocab]
candidate_scores: [batch, beam × vocab]
```

각 문장에 대해 상위 `beam_size`를 한 번의 `topk`로 선택한다.

```python
top_scores, top_indices = candidate_scores.topk(beam_size, dim=-1)
parent_beam = top_indices // vocab_size
next_token = top_indices % vocab_size
```

선택된 `parent_beam`에 따라 sequence를 재정렬하고 다음 token을 붙인다.

### 5.3 점수 규칙

기존 결과와의 동등성을 위해 먼저 현재 규칙을 유지한다.

```text
score = 누적 log probability
normalized score = score / generated_length
```

이후 length penalty를 명시적 설정으로 분리한다.

```python
normalized = score / (generated_length ** length_penalty)
```

Beam 1과 beam 4는 별도 실험 조건으로 기록한다. BART의 beam 4는 BPE token 후보이고 BLT의 beam 4는 byte 후보이므로, 동일한 숫자가 동일한 탐색 비용이나 탐색 범위를 의미하지 않는다.

### 5.4 Beam별 종료 처리

- EOS를 생성한 beam은 완료 상태로 유지한다.
- 완료 beam은 추가 token을 score에 반영하지 않는다.
- 모든 beam이 끝난 문장은 active batch에서 제외할 수 있다.
- 최종 선택 시 완료 beam을 우선하되, 완료 beam이 없으면 최대 길이 beam 중 최고 점수를 선택한다.

---

## 6. 3단계: 평가 파이프라인 통합

### 6.1 CLI 인자

`blt_gec/eval.py`와 `scripts/eval_blt.sh`에 다음을 추가한다.

```text
GENERATION_BACKEND=scalar_legacy|batched_nocache|incremental
GENERATION_BATCH_SIZE=<int>
BLT_NUM_BEAMS=<int>
LENGTH_PENALTY=<float>
```

초기 전환 기간에는 `scalar_legacy`를 유지할 수 있게 하되, 검증 후 기본값을 `batched_nocache`로 변경한다.

### 6.2 결과 디렉토리 격리

서로 다른 generation 조건이 같은 디렉토리에 저장되지 않게 한다.

```text
outputs/blt_eval/
  <dataset>/<split>/<checkpoint>/
    scalar_legacy_beam4/
    batched_nocache_beam4/
    batched_nocache_greedy/
```

### 6.3 Shard manifest

각 shard의 `metrics_*.json`에 다음 정보를 기록한다.

```json
{
  "checkpoint": "outputs/blt_gec/union/best.ckpt",
  "checkpoint_hash": "...",
  "generation_backend": "batched_nocache",
  "generation_batch_size": 8,
  "num_beams": 4,
  "length_penalty": 1.0,
  "max_length": 2048,
  "max_gen_len": 256,
  "code_commit": "...",
  "start_index": 0,
  "end_index": 292
}
```

Aggregate는 모든 shard의 설정 fingerprint가 같을 때만 진행한다. 설정이 다른 shard가 하나라도 있으면 즉시 실패한다.

### 6.4 중단 복구

긴 shard가 wall-time에 걸려 전체 결과를 잃지 않도록 batch 단위 임시 파일을 사용한다.

```text
hypothesis_00000_00292.partial
reference_00000_00292.partial
source_00000_00292.partial
progress_00000_00292.json
```

정상 완료 시 atomic rename으로 최종 파일을 만든다. 재제출 시 progress를 읽고 완료된 example 다음부터 재개한다.

---

## 7. 4단계: Patch-aware Incremental Generation

### 7.1 목적

Batched no-cache는 GPU 활용률을 개선하지만 매 바이트 전체 prefix를 재계산하는 문제는 남는다. BLT 논문의 추론 구조에 접근하려면 완료된 patch와 attention 상태를 캐시해야 한다.

### 7.2 필요한 상태

BLT용 incremental state는 일반 Transformer KV cache보다 복잡하다.

```text
GenerationState
├── entropy model KV/state
├── 확정된 patch 경계
├── 현재 미완성 patch의 byte IDs
├── Local Encoder state
├── Global Transformer patch KV cache
├── Local Decoder state
└── byte ↔ patch cross-attention mapping
```

### 7.3 목표 실행 구조

```text
매 바이트:
  1. entropy model을 한 step 진행
  2. 다음 위치가 patch 경계인지 판단
  3. 경계가 아니면 작은 Local Decoder state만 갱신
  4. 경계면 현재 patch를 확정
  5. Global Transformer를 patch 한 step만 진행
  6. 새 patch용 Local Decoder state 초기화
  7. 다음 바이트 생성
```

큰 Global Transformer는 매 바이트가 아니라 새 patch가 만들어질 때만 한 step 전진해야 한다.

### 7.4 구현 순서

1. Greedy, batch size 1에서 incremental state 구현
2. 매 step full recomputation logits와 incremental logits 비교
3. 여러 prompt batching 추가
4. Beam parent 재정렬 시 cache도 함께 재정렬
5. beam별 cache copy 비용을 줄이기 위한 copy-on-write 검토

Beam search까지 동시에 구현하지 않는다. Greedy incremental parity를 확보한 후 확장한다.

### 7.5 동등성 검사

짧은 고정 prompt에 대해 각 step에서 다음을 비교한다.

```python
full_logits = full_recompute(prefix)
cached_logits = incremental_step(state, last_byte)
torch.testing.assert_close(full_logits, cached_logits, rtol=..., atol=...)
```

다음도 검사한다.

- entropy 값
- patch 시작 위치
- patch 길이
- Global Transformer step 수
- 최종 token IDs
- 최종 UTF-8 문자열

---

## 8. Attention Backend 및 Causal Mask 복구

### 8.1 현재 구현의 구조적 문제

현재 `blt_gec/model.py`는 메인 BLT의 모든 `attn_impl`을 `sdpa`로 변경한다. 레퍼런스 local encoder와 local decoder는 `local_block_causal`을 요청하지만, 레퍼런스 `create_causal_mask()`는 SDPA에서 이 마스크를 구현하지 않는다. 현재 shell script가 기본으로 설정하는 `BLT_SUPPRESS_ATTN_ERROR=1`은 오류를 중단하지 않고 일반 `"causal"`을 반환한다.

이 동작은 호환성 fallback일 뿐 레퍼런스와 동등한 구현이 아니다.

| 모듈 | 레퍼런스 마스크 | 현재 fallback | 영향 |
|---|---|---|---|
| Global Transformer | causal | causal | 의미상 동일 |
| Local Encoder | local block-causal | full causal | local window와 EOS block 격리 손실 |
| Local Decoder | local block-causal | full causal | local window와 EOS block 격리 손실 |
| Byte-patch cross-attention | FlexAttention BlockMask | SDPA dense mask | 대응 관계는 보존하되 희소 연산 손실 |

따라서 현재 체크포인트는 dynamic patching과 local/global/local 구조는 유지하지만, 레퍼런스 BLT-1B와 동일한 local attention 구조로 파인튜닝된 모델은 아니다. 기존 결과는 `SDPA full-causal compatibility fallback` 실험으로 분리해야 한다.

### 8.2 복구해야 하는 마스크 의미

query 위치를 \(q\), key 위치를 \(k\), local window를 \(W\), EOS 기준 segment ID를 \(s(i)\)라고 할 때 참조를 허용하는 조건은 다음과 같다.

\[
k \leq q,\qquad q-k < W,\qquad s(q)=s(k)
\]

즉, 다음 세 조건을 모두 만족해야 한다.

1. 미래 바이트를 참조하지 않는다.
2. 같은 segment에 있더라도 최근 \(W\)개 위치만 참조한다.
3. EOS로 구분된 다른 segment의 바이트는 참조하지 않는다.

일반 causal mask는 첫 번째 조건만 보존한다. 따라서 단순히 `is_causal=True`를 전달하는 것으로는 복구되지 않는다.

### 8.3 1차 구현: 의미가 정확한 SDPA dense mask

우선 성능보다 정확성을 검증하기 위해 `blt_gec/model.py`에 SDPA용 local block-causal additive mask 생성기를 구현한다. `reference_code/blt`는 수정하지 않고 adapter에서 local encoder와 local decoder에 정확한 tensor mask를 전달한다.

개념적인 구현은 다음과 같다.

```python
positions = torch.arange(seq_len, device=tokens.device)
q = positions[:, None]
k = positions[None, :]

causal = k <= q
local = (q - k) < sliding_window
same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
allowed = causal[None] & local[None] & same_segment

mask = torch.zeros(
    batch_size, 1, seq_len, seq_len,
    device=tokens.device,
    dtype=query_dtype,
)
mask.masked_fill_(~allowed[:, None], float("-inf"))
```

구현 시 다음을 지켜야 한다.

- `segment_ids`는 레퍼런스 `tokens_to_seqlen()`과 같은 EOS 경계 규칙을 사용한다.
- 마지막 위치의 virtual EOS 처리도 레퍼런스와 일치시킨다.
- padding key는 실제 token query가 참조하지 못하게 차단한다.
- padding query 행 전체를 `-inf`로 만들면 softmax가 `NaN`이 될 수 있으므로, 안전한 self 위치 하나를 허용한 뒤 해당 query 출력을 loss와 generation에서 무시한다.
- 마스크 dtype과 query dtype을 일치시킨다.
- Global Transformer의 일반 causal 경로는 변경하지 않는다.
- cross-attention용 byte-patch dense mask와 local self-attention mask를 구분한다.
- `BLT_SUPPRESS_ATTN_ERROR`에 의존해 마스크를 축소하지 않는다.

이 방식은 `[batch, 1, sequence, sequence]` 마스크를 생성하므로 메모리 비용이 \(O(BL^2)\)이다. 따라서 최종 성능 구현이 아니라 xFormers와의 의미 동등성을 확보하기 위한 기준 backend로 사용한다.

### 8.4 2차 구현: 희소성과 커널 효율 복구

정확한 dense mask로 결과를 검증한 뒤 다음 경로를 비교한다.

1. 레퍼런스 PyTorch nightly와 pinned xFormers 조합을 별도 환경에서 복원
2. 현재 PyTorch 버전에서 FlexAttention `BlockMask`로 local block-causal 구현
3. SDPA가 지원하는 범위에서 window mask를 최적화하고, 지원되지 않는 부분은 별도 커널로 처리

최적화 backend는 반드시 1차 dense mask와 같은 허용 위치 및 출력 token을 재현해야 한다. 단순히 실행 속도가 빠르다는 이유로 일반 causal fallback을 다시 허용하지 않는다.

### 8.5 검증 항목

마스크 단위 테스트에서는 작은 인공 token sequence를 사용해 다음을 직접 검사한다.

```text
- 미래 위치가 차단되는가
- window 밖의 과거 위치가 차단되는가
- EOS 이전 query가 EOS 이후 key를 참조하지 않는가
- EOS 이후 query가 EOS 이전 key를 참조하지 않는가
- batch별 segment 경계가 서로 독립적인가
```

레퍼런스 xFormers 환경을 사용할 수 있으면 동일한 weight, token, patch length에 대해 다음을 비교한다.

1. local encoder layer별 hidden state
2. global patch representation
3. local decoder logits
4. entropy 값과 patch 경계
5. greedy token과 최종 문자열

부동소수점 커널 차이로 logits가 비트 단위로 같지 않을 수 있으므로 `rtol`과 `atol`을 명시한다. 그러나 mask 허용 위치는 완전히 동일해야 한다.

### 8.6 기존 체크포인트 처리

- 기존 체크포인트 평가: 학습 때 사용한 full-causal fallback을 유지한다.
- 수정된 mask 평가: 별도 출력 디렉토리와 manifest를 사용한다.
- 서로 다른 attention mask로 생성한 shard는 aggregate하지 않는다.
- 기존 체크포인트에서 mask만 바꾸어 평가한 결과를 동일 모델 결과로 간주하지 않는다.
- 엄밀한 BLT 비교용 파인튜닝은 원본 `facebook/blt-1b` 체크포인트에서 다시 시작한다.
- 기존 15 epoch 체크포인트는 compatibility fallback ablation으로 보존한다.

Attention mask 복구는 generation 최적화와 독립적인 부가 작업이 아니라 모든 출력 동등성 검사의 선행 조건이다. 마스크 의미가 다른 상태에서 batching이나 cache의 출력을 비교하면 backend 차이와 generation 구현 차이를 분리할 수 없다.

---

## 9. 테스트 계획

### 9.1 단위 테스트

- GEC separator가 prompt 끝에 항상 존재하는지 검사
- 긴 source 절단 후에도 separator가 보존되는지 검사
- EOS별 조기 종료
- 빈 출력 및 즉시 EOS
- 유효하지 않은 UTF-8 byte 처리
- beam score와 length normalization
- parent beam 재정렬
- 모두 완료된 batch 종료

### 9.2 동등성 테스트

| 비교 | 기대 결과 |
|---|---|
| 레퍼런스 greedy vs batched greedy, batch=1 | 동일 token IDs |
| 기존 scalar beam vs batched beam, batch=1 | 동일 최종 token IDs와 score |
| batched beam, batch=1 vs batch>1 | 문장별 동일 출력 |
| full recompute vs incremental greedy | step별 logits 허용 오차 내 일치 |

### 9.3 품질 테스트

고정 validation subset에 대해 다음을 비교한다.

```text
scalar legacy beam 4
batched no-cache beam 4
batched no-cache beam 1
incremental beam 1
```

기록 지표:

- GLEU
- KAGAS/M2 precision, recall, F0.5
- 평균 생성 byte 수
- EOS 도달률
- 최대 길이 강제 종료율
- source 그대로 복사한 비율

### 9.4 성능 테스트

동일 checkpoint와 동일 validation subset으로 측정한다.

- examples/second
- generated bytes/second
- 문장당 평균 시간
- GPU utilization
- peak GPU memory
- entropy patcher 시간
- BLT forward 시간
- Global Transformer 호출 횟수
- 문장당 평균 patch 수

GPU 종류별 결과는 분리한다.

```text
V100
A100
H200
```

---

## 10. 파일별 변경 범위

### `blt_gec/generation.py`

- `GenerationConfig` 추가
- 기존 scalar backend 보존
- `generate_corrections()` batched API 추가
- batched greedy 구현
- batched beam 구현
- backend dispatch 추가

### `blt_gec/eval.py`

- example batch 구성
- generation backend 및 batch size 인자 추가
- shard manifest 생성
- partial 결과 저장 및 resume
- 처리량 통계 기록

### `blt_gec/generate.py`

- 단일 문장 CLI도 공용 batched API 사용
- backend와 beam 설정 출력

### `scripts/eval_blt.sh`

- `GENERATION_BACKEND`
- `GENERATION_BATCH_SIZE`
- `LENGTH_PENALTY`
- 출력 경로에 generation 조건 반영

### `blt_gec/model.py`

- SDPA local block-causal dense mask 생성
- local encoder와 local decoder에 정확한 self-attention mask 전달
- 일반 causal fallback 제거
- attention backend와 mask mode 식별자 제공
- incremental generation 단계에서 BLT cache hook 추가

### `tests/`

- `test_blt_local_block_causal_mask.py`
- `test_blt_attention_backend_parity.py`
- `test_blt_generation_equivalence.py`
- `test_blt_batched_beam.py`
- `test_blt_eval_manifest.py`
- `test_blt_incremental_generation.py`

---

## 11. 구현 우선순위

### Phase 0: 현재 결과 동결 및 명칭 분리

- 기존 scalar beam 4 shard 완료
- checkpoint, 코드 commit, beam 설정 기록
- 기존 결과 디렉토리를 `legacy_sdpa_full_causal` backend로 명시
- 기존 체크포인트를 compatibility fallback 결과로 분류

### Phase 1: Local block-causal 복구

- SDPA dense local block-causal mask 구현
- causal, local window, EOS block 조건 단위 테스트
- local encoder와 local decoder의 mask 적용 확인
- 가능한 경우 xFormers backend와 layer별 logits/hidden state 비교
- 원본 BLT-1B 체크포인트에서 짧은 smoke fine-tuning 수행

### Phase 2: Batched greedy

- 레퍼런스 출력 동등성 확보
- batch size별 메모리와 처리량 측정
- beam 1의 GLEU/F0.5 기준선 생성

### Phase 3: Batched beam

- 기존 scalar beam 4와 출력 동등성 확보
- batch 크기 자동 축소 또는 OOM 재시도 구현
- beam 1 대비 beam 4 품질 향상과 비용 비교

### Phase 4: 평가 운영 개선

- manifest 검증
- partial shard resume
- aggregate 설정 fingerprint 검사

### Phase 5: Incremental greedy

- entropy/local/global/decoder cache 설계
- full recomputation과 step별 logits 동등성 확보

### Phase 6: Incremental batched beam

- cache 재정렬 및 copy-on-write
- 최종 품질·속도·메모리 비교

---

## 12. 완료 기준

1. Local encoder와 local decoder가 causal, local window, EOS block 조건을 모두 보존한다.
2. `BLT_SUPPRESS_ATTN_ERROR`가 일반 causal fallback을 활성화하지 않는다.
3. Attention mask mode가 checkpoint 및 평가 manifest에 기록된다.
4. 새로운 batched backend가 `batch_size=1`에서 수정된 scalar 기준 출력과 일치한다.
5. 서로 다른 batch size에서 문장별 출력이 변하지 않는다.
6. Aggregate가 서로 다른 attention mask, backend, beam, checkpoint shard의 혼합을 차단한다.
7. 평가 중단 후 완료된 batch 다음부터 재개할 수 있다.
8. Beam 1과 beam 4의 품질 및 비용 차이가 정량적으로 보고된다.
9. Batched no-cache가 scalar legacy보다 높은 examples/second와 GPU utilization을 보인다.
10. Incremental backend는 full recomputation과 step별 logits 및 patch 경계가 일치한다.
11. BLT의 성능 보고에서 모델 FLOPs, 실제 wall-clock, GPU utilization을 구분한다.

---

## 13. 최종 판단

Generation 병목을 수정하기 전에 현재 일반 causal fallback을 정확한 local block-causal mask로 교체해야 한다. 이 문제는 성능 차이가 아니라 모델이 참조할 수 있는 문맥 범위의 차이이므로 prompt batching이나 KV cache보다 우선한다.

Attention 의미를 복구한 이후 generation에서 가장 먼저 해결해야 할 병목은 KV cache가 아니라 현재 코드가 제거한 prompt batching이다. Batched no-cache와 batched beam search는 모델 구조를 추가로 바꾸지 않고 적용할 수 있으며, 수정된 scalar backend와 출력 동등성도 검증하기 쉽다.

Patch-aware KV cache는 BLT 논문의 추론 효율을 실제 wall-clock 성능으로 연결하기 위해 필요하지만, local encoder, global patch transformer, local decoder 및 entropy patcher 상태를 함께 관리해야 하므로 별도 연구 단계로 취급한다.

따라서 즉시 구현 범위는 다음으로 제한한다.

```text
SDPA local block-causal parity
→ reference-equivalent batched greedy
→ vectorized batched beam
→ 평가 manifest와 resume
```

정확한 마스크로 원본 BLT-1B 체크포인트부터 다시 파인튜닝한 모델을 본 실험으로 사용하고, 기존 full-causal 체크포인트는 ablation으로 분리한다. 이후에만 incremental patch cache를 구현한다. 이 순서를 지켜야 attention 의미 변화, batching 효과와 cache 효과를 각각 분리하여 검증할 수 있다.
