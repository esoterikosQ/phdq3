# BLT-GEC: Generation 평가 분리 + Single-node DDP 학습

## Context

BLT-1B 학습이 GPU 노드 2시간 제한에 걸린다. 병목은 학습(52분/epoch)이 아니라 generation 평가(전체 validation 2633개 → ~48시간). 학습과 평가를 분리하고, 학습에 single-node multi-GPU DDP를 적용하면 2시간 안에 3 epochs 학습이 가능하다.

**Scope 밖**: multi-node DDP. single-node 8GPU가 안정적으로 돌고 checkpoint/eval이 검증된 뒤에만 확장 검토. NCCL rendezvous, HF cache 동기화 등 실패면이 큰 확장은 현 단계에서 불필요.

## 목표 시간 예산 (8 GPU, EVAL_GENERATION=0)

- epoch 학습: ~7분 × 3 = 21분
- validation loss: ~1.5분 × 3 = 4.5분
- 총 ~30분 (2시간 내 여유)

---

## Part 1: Generation 평가 독립 스크립트 (1순위)

### 새 파일: `blt_gec/metrics.py`

`train.py`에 흩어진 GLEU/M2 호출을 공용 유틸로 추출.

```python
from baseline.metric.gleumodule import run_gleu

def compute_gleu(reference: str, source: str, hypothesis: str) -> float: ...
def compute_m2(hypothesis_path, m2_source_gold_path) -> tuple[float|None, ...]: ...
```

`train.py`와 `eval.py`가 모두 `blt_gec.metrics`를 import.

### 새 파일: `blt_gec/eval.py`

`train.py`의 `evaluate_generation()` (L259-302) 로직을 독립 CLI로 추출.

#### 핵심 인자

- `--checkpoint` (필수): best.ckpt 경로
- `--split`: `val` / `test`
- `--start_index`, `--max_examples`: shard 분할용
- `--aggregate`: shard 결과를 합치고 GLEU 계산 (모델 로딩 불필요, GPU 불필요)
- 나머지: `generate.py`와 동일한 모델/generation 인자

#### 로직

1. `load_reference_blt_components()` + checkpoint 로드
2. `GecBltDataset` 로드 → shard 범위 계산: `end_index = min(start_index + max_examples, len(dataset.data))`
3. `generate_correction()` 순차 실행 → hypothesis/reference/source 파일 저장
4. shard 파일명: `hypothesis_{start:05d}_{end:05d}.txt` (정렬 용이)
5. `--aggregate` 모드: shard 파일들 합쳐서 `blt_gec.metrics` 호출

#### Shard 범위 안전 처리

```python
total = len(dataset.data)
end_index = min(start_index + max_examples, total) if max_examples > 0 else total
examples = dataset.data[start_index:end_index]
```

마지막 shard가 dataset 끝을 넘지 않도록 cap.

#### Aggregate 무결성 검증

파일명에서 `(start, end)` 범위를 파싱하여 gap/overlap 없이 `0..total`을 덮는지 검증:

```python
# shard 파일명: hypothesis_00000_00066.txt → (0, 66)
ranges = sorted(parse_range(f) for f in shard_files)

# 1. 범위 연속성 검사 (gap/overlap 없음)
for i in range(1, len(ranges)):
    if ranges[i][0] != ranges[i-1][1]:
        raise RuntimeError(f"Shard gap/overlap: {ranges[i-1]} -> {ranges[i]}")

# 2. 전체 커버리지 검사
if ranges[0][0] != 0 or ranges[-1][1] != expected_total:
    raise RuntimeError(f"Shard range mismatch: covers {ranges[0][0]}..{ranges[-1][1]}, expected 0..{expected_total}")
```

단순 line count만으로는 중복 shard를 놓칠 수 있으므로 범위 기반 검증이 필수.
누락/중복 shard가 있으면 즉시 실패. 부분 결과로 GLEU를 계산하지 않음.

#### DDP checkpoint 호환

`model.state_dict()` 키에서 `"module."` prefix 자동 제거:

```python
state = checkpoint["model"]
state = {k.removeprefix("module."): v for k, v in state.items()}
```

#### 출력 경로

```
outputs/blt_eval/<dataset>/<split>/<checkpoint_name>/
    hypothesis_00000_00066.txt   # shard 0
    reference_00000_00066.txt
    source_00000_00066.txt
    hypothesis_00066_00132.txt   # shard 1
    ...
    hypothesis.txt               # aggregate 결과 (전체)
    reference.txt
    source.txt
    metrics.json
```

checkpoint 이름은 파일명에서 자동 추출 (e.g., `best` from `best.ckpt`).

#### 재사용

- `generate.py`의 모델 로딩 패턴 (L35-48)
- `generation.py`의 `generate_correction()`
- `blt_gec/metrics.py` (신설)

### 새 파일: `scripts/eval_blt.sh`

`scripts/eval_bart.sh` 패턴 참조. GPU 1개, 1:55:00.

#### Shard 계산: 2시간 제한 기반

예제당 ~1.1분이므로 shard당 최대 ~100개가 2시간 안전 한계. 오버헤드(모델 로딩 ~5분, 느린 샘플) 감안하면 **shard당 ~66개 (40 shards)** 가 현실적.

```
48시간(전체) / 2시간(제한) = 최소 24 shards
오버헤드 감안 → 32~48 shards 권장 → 40 shards 기준
2633개 / 40 shards ≈ 66개/shard × ~1.1분 ≈ 73분/shard (2시간 내)
```

#### SLURM array 기반 shard 자동 계산

`TOTAL_EXAMPLES`를 하드코딩하지 않고 split 파일에서 동적으로 계산:

```bash
#SBATCH --array=0-39    # 기본 40 shards, 제출 시 override 가능

# split 파일에서 실제 example 수 계산
SPLIT="${SPLIT:-val}"
# ... (split 파일 경로 결정 로직) ...
TOTAL_EXAMPLES=$(wc -l < "$SPLIT_FILE")

# array가 아닌 단일 shard 테스트도 안전하게 처리
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-40}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SHARD_SIZE=$(( (TOTAL_EXAMPLES + NUM_SHARDS - 1) / NUM_SHARDS ))  # ceil division
START_INDEX=$(( TASK_ID * SHARD_SIZE ))
MAX_EXAMPLES="$SHARD_SIZE"
```

→ `native_val.txt` (2633줄), `native_test.txt` (2633줄), `korean_learner_val.txt` 등 어떤 split이든 자동 대응.

#### Aggregate 분기

`AGGREGATE=1`일 때 모델 로딩 없이 결과 합산만 수행. GPU 1개를 사용하지만 실질적으로 CPU 작업만 실행 (수 초 내 완료).

> GPU 파티션에서 `--gres=gpu:0`은 정책상 거부될 수 있으므로, aggregate도 동일한 `eval_blt.sh`에서 GPU 1개로 실행. GPU 낭비는 수 초로 무시 가능.

```bash
if [[ "${AGGREGATE:-0}" == "1" ]]; then
    srun "$PYTHON_BIN" -m blt_gec.eval --aggregate \
        --output_dir "$OUTPUT_DIR" \
        --data "$DATASET_TYPE" \
        --split "$SPLIT" \
        --checkpoint_name "$CKPT_NAME"
    exit 0
fi
```

#### 사용 예시

```bash
# 40 shards 병렬 실행 (한 줄)
CONDA_ENV=phdq_blt CKPT_PATH=outputs/blt_gec/native/best.ckpt \
  sbatch --array=0-39 scripts/eval_blt.sh

# 완료 후 합산
AGGREGATE=1 CKPT_PATH=outputs/blt_gec/native/best.ckpt \
  sbatch scripts/eval_blt.sh

# 다른 dataset/split
CONDA_ENV=phdq_blt DATASET_TYPE=learner SPLIT=test \
  CKPT_PATH=outputs/blt_gec/learner/best.ckpt \
  sbatch --array=0-39 scripts/eval_blt.sh
```

---

## Part 2: 학습 기본값 변경

### 수정: `scripts/train_blt.sh`

`EVAL_GENERATION` 기본값을 `0`으로 변경:

```bash
# 변경 전
EVAL_GENERATION="${EVAL_GENERATION:-1}"

# 변경 후
EVAL_GENERATION="${EVAL_GENERATION:-0}"
```

학습 중에는 validation loss만 계산. GLEU generation은 `eval.py`로 분리.
기존처럼 학습 중 generation이 필요하면 `EVAL_GENERATION=1`로 명시 가능 (하위 호환).

### 수정: `blt_gec/train.py` (metrics import 경로)

`run_gleu()` / `run_m2_scorer()` 호출을 `blt_gec.metrics`에서 import하도록 변경.

---

## Part 3: Single-node DDP 학습

### 수정: `blt_gec/train.py`

#### (a) 분산 헬퍼 함수 추가

```python
import os
import torch.distributed as dist
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()
```

#### (b) `main()` 수정 — 디바이스 + DDP 래핑

```python
distributed = "LOCAL_RANK" in os.environ
if distributed:
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드 후 (checkpoint load는 DDP 래핑 전에)
if distributed:
    model = DDP(model, device_ids=[local_rank])
raw_model = model.module if distributed else model
```

- `patcher`/`entropy_model`은 DDP 래핑하지 않음 (frozen, 각 rank에서 독립 실행)
- `torch.cuda.set_device(local_rank)` 후 reference BLT의 `to_device(entropy_model, "cuda")`는 default CUDA device를 따르므로 올바른 GPU에 배치됨

#### (c) DataLoader — DistributedSampler

```python
train_sampler = DistributedSampler(
    train_dataset, shuffle=True, seed=args.seed
) if distributed else None
train_loader = DataLoader(
    ..., shuffle=(train_sampler is None), sampler=train_sampler
)
```

- validation/test loader는 rank 0에서만 사용 → sampler 불필요
- 매 epoch 시작시 `train_sampler.set_epoch(epoch)`

#### (d) Gradient accumulation — `no_sync()`

```python
is_accum = (step_in_epoch % args.grad_accum_steps != 0
            and step_in_epoch != total_batches)
ctx = model.no_sync() if (distributed and is_accum) else nullcontext()
with ctx:
    with torch.autocast(...):
        logits = model(input_ids, patch_lengths=patch_lengths)
        loss = compute_loss(logits, labels) / args.grad_accum_steps
    loss.backward()
```

`model.no_sync()`는 DDP 전용 메서드. 비DDP 시 `nullcontext()`로 fallback.

#### (e) `grad_accum_steps` 자동 조정 + 로깅

DDP에서 effective batch size 유지: `grad_accum = max(grad_accum // world_size, 1)`

- 1 GPU: batch=1, accum=8 → effective=8
- 8 GPU: batch=1, accum=1 → effective=8

```python
if distributed:
    world_size = get_world_size()
    original_accum = args.grad_accum_steps
    args.grad_accum_steps = max(original_accum // world_size, 1)
    effective_batch = args.batch_size * args.grad_accum_steps * world_size
    if is_main_process():
        print(f"DDP: world_size={world_size}, "
              f"grad_accum {original_accum} -> {args.grad_accum_steps}, "
              f"effective_batch_size={effective_batch}")
```

실험 재현을 위해 requested/effective 값 모두 로그에 출력.

`build_scheduler()`는 `len(train_loader)` (DDP 시 dataset/world_size)와 조정된 `grad_accum`을 받으므로 `total_steps` 자동 정합:

- 1 GPU: steps_per_epoch = 12291/8 = 1536
- 8 GPU: steps_per_epoch = ceil(12291/8)/1 = 1537 (거의 동일)

#### (f) Rank-0 전용 작업 + non-rank barrier

```python
# 로깅
if global_step % args.log_every_steps == 0 and is_main_process():
    print(...)

# Epoch 끝 validation
if is_main_process():
    val_loss = evaluate(raw_model, patcher, val_loader, device, args.precision)
    save_checkpoint(...)
if distributed:
    dist.barrier()  # 모든 rank가 validation 완료까지 대기
```

- 로깅 (`print`): rank 0만
- checkpoint 저장: rank 0만, `raw_model.state_dict()` 사용
- validation loss 계산: rank 0만, `raw_model` 사용
- **`dist.barrier()`**: validation/checkpoint 후 삽입하여 non-rank-0이 다음 epoch으로 먼저 진입하는 것을 방지
- validation loss 1.5분이면 GPU idle 허용 가능. generation만 분리하면 충분

#### (g) Checkpoint 호환성

```python
# save: DDP 래퍼 벗기고 저장 (항상 raw state_dict)
model_to_save = model.module if hasattr(model, "module") else model
payload["model"] = model_to_save.state_dict()

# load: 저장 형식이 항상 raw이므로 DDP 래핑 전에 로드
model.load_state_dict(checkpoint["model"], strict=False)
```

→ 단일 GPU로 저장한 checkpoint를 DDP로 로드, 또는 그 반대 모두 가능.
→ `eval.py`에서도 동일한 checkpoint를 그대로 사용 가능.

#### (h) 종료 동기화

`STOP_REQUESTED` / 시간 초과 체크를 rank 0에서 broadcast:

```python
should_stop = STOP_REQUESTED or elapsed > max_seconds - 300
if distributed:
    stop_tensor = torch.tensor([int(should_stop)], device=device)
    dist.broadcast(stop_tensor, src=0)
    should_stop = bool(stop_tensor.item())
if should_stop:
    if is_main_process():
        save_checkpoint(...)
    if distributed:
        dist.barrier()
    return
```

### 수정: `scripts/train_blt.sh`

- `NUM_GPUS` 환경변수 추가 (기본값 1 — 하위 호환)
- `NUM_GPUS > 1`일 때:
  - `srun torchrun --standalone --nproc_per_node=$NUM_GPUS -m blt_gec.train ...`
  - `GRAD_ACCUM_STEPS` 자동 나누기
  - `--gres=gpu:$NUM_GPUS` (sbatch 커맨드라인 override)
  - `--cpus-per-task`: 16으로 시작 (`NUM_WORKERS=0`이므로 CPU 병목 없음, 필요시 증가)
- `NUM_GPUS == 1`일 때: 기존 `srun $PYTHON_BIN -m blt_gec.train ...` 유지
- `EVAL_GENERATION` 기본값 `0` (Part 2에서 변경)

#### GPU 할당 검증 (fail-fast)

```bash
if [[ "$NUM_GPUS" -gt 1 ]]; then
    ACTUAL_GPUS="${SLURM_GPUS_ON_NODE:-$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)}"
    if [[ "$ACTUAL_GPUS" -lt "$NUM_GPUS" ]]; then
        echo "Error: NUM_GPUS=$NUM_GPUS but only $ACTUAL_GPUS GPUs allocated."
        echo "Submit with: sbatch --gres=gpu:$NUM_GPUS scripts/train_blt.sh"
        exit 1
    fi
fi
```

`NUM_GPUS`와 실제 SLURM 할당 GPU 수가 불일치하면 즉시 실패. silent 오류 방지.

SBATCH에서 `#SBATCH --gres=gpu:1`은 기본값으로 유지. Multi-GPU 사용 시:

```bash
NUM_GPUS=8 sbatch --gres=gpu:8 --cpus-per-task=16 scripts/train_blt.sh
```

---

## SLURM 실행 전략 요약

### 학습

```bash
# single-node 8GPU DDP, generation 평가 없이
CONDA_ENV=phdq_blt NUM_GPUS=8 sbatch --gres=gpu:8 --cpus-per-task=16 scripts/train_blt.sh
```

### Generation 평가

```bash
# 40 shards 병렬 실행 (한 줄)
CONDA_ENV=phdq_blt CKPT_PATH=outputs/blt_gec/native/best.ckpt \
  sbatch --array=0-39 scripts/eval_blt.sh

# 완료 후 합산 (GPU 1개 사용, 수 초 내 완료)
AGGREGATE=1 CKPT_PATH=outputs/blt_gec/native/best.ckpt \
  sbatch scripts/eval_blt.sh
```

---

## 구현 순서

```
1. blt_gec/metrics.py 생성                  — GLEU/M2 공용 유틸
2. blt_gec/eval.py 생성                     — generation 평가 독립 CLI
3. scripts/eval_blt.sh 생성                  — SLURM array 기반 shard 실행
4. scripts/train_blt.sh EVAL_GENERATION=0    — 학습에서 generation 분리
5. blt_gec/train.py metrics import 변경      — blt_gec.metrics 사용
6. blt_gec/train.py DDP 수정                 — single-node 8GPU 지원
7. scripts/train_blt.sh NUM_GPUS 분기        — torchrun 런처 + GPU 검증
```

1-5는 학습 코드의 동작을 바꾸지 않고 독립 진행 가능.
6-7은 하위 호환 유지 (NUM_GPUS=1이면 기존과 동일).
multi-node DDP는 single-node가 검증된 후 필요할 때만 확장.

## 검증

1. `python -m blt_gec.eval --checkpoint <ckpt> --max_examples 5 --split val` → generation 결과 확인
2. `sbatch --array=0-1 scripts/eval_blt.sh` (2 shards) → shard 파일 생성 → `AGGREGATE=1` → GLEU 확인, 누락 shard 검출 확인
3. `NUM_GPUS=1 sbatch scripts/train_blt.sh` → 기존과 동일 동작 확인
4. `NUM_GPUS=2 sbatch --gres=gpu:2 scripts/train_blt.sh` → DDP 동작, loss 수렴, 로그에 world_size/effective_batch 출력 확인
5. DDP checkpoint → `eval.py` 로드 → key mismatch 없음 확인
6. `NUM_GPUS=8` 3 epochs가 2시간 내 완료되는지 확인
7. `NUM_GPUS=8`인데 `--gres=gpu:1`로 제출 시 fail-fast 확인

## 수정 대상 파일 요약

| 파일 | 작업 | 우선순위 |
|------|------|---------|
| `blt_gec/metrics.py` | 새로 생성 — GLEU/M2 공용 유틸 | 1 |
| `blt_gec/eval.py` | 새로 생성 — generation 평가 CLI | 1 |
| `scripts/eval_blt.sh` | 새로 생성 — SLURM array shard | 1 |
| `scripts/train_blt.sh` | `EVAL_GENERATION=0` + `NUM_GPUS` 분기 + GPU 검증 | 2-3 |
| `blt_gec/train.py` | metrics import 변경 + DDP 지원 | 2-3 |
