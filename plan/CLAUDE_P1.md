# CLAUDE.md — P1: HF Transformers 기반 BLT-GEC 재구축

> 배치 위치: `PHDQ2/blt_hf/CLAUDE.md`
> 루트 `PHDQ2/CLAUDE.md`(3-머신 워크플로, Neuron SLURM 파티션/시간제한/스토리지,
> 커밋 컨벤션)가 항상 함께 적용된다. 이 파일은 P1 고유 규칙만 담는다.
> 상세 계획: `P1_HF_BLT_GEC_재구축_계획.md` — 작업 착수 전 해당 Phase 절을 먼저 읽는다.

## 프로젝트 개요

`reference_code/blt`(xFormers 커밋 고정) 의존을 제거하고 Hugging Face
Transformers의 정식 BLT 구현(`BltForCausalLM`, `itazap/blt-1b-hf`) 위에
한국어 GEC 파인튜닝·생성·평가를 재구축한다. Phase A(검증) → B(파인튜닝) →
C(batched generation) → D(평가 재실행) 순서이며 각 Phase는 종료 게이트를
통과해야 다음으로 진행한다.

## 절대 규칙 (위반 금지)

1. **`blt_gec/`는 동결 상태다. 어떤 파일도 수정하지 않는다.** 기존 결과
   재현성 보존이 목적이다. 재사용이 필요한 코드는 `blt_hf/`로 복사해 온다
   (예외: `blt_gec/metrics.py`, `m2_resumable.py`는 import 재사용 허용 —
   중복 구현 금지 원칙이 우선).
2. **`blt_hf/` 패키지는 `reference_code/blt`를 import하지 않는다.** 유일한
   예외는 `blt_hf_checks/`의 교차 검증 스크립트이며, 이때도 legacy 환경
   (`phdq_blt`)에서 dump → HF 환경(`phdq_blt_hf`)에서 비교하는 2단계 방식만
   사용한다. 한 프로세스에서 두 스택을 동시에 로드하지 않는다.
3. **Phase 종료 게이트를 건너뛰지 않는다.** 특히 Phase A의 3대 검증
   (patcher 검증, attention mask 의미, tiny overfit)이 미완이면 Phase B
   코드를 작성하더라도 본 학습을 제출하지 않는다. patcher 검증의 기준은
   legacy 런타임이 아니라 **plain 기준 구현(`plain_patcher.py`, 패칭의
   수학적 정의)**이다 — legacy dump는 A-3-0에서 유효 판정된 경우에만
   참고용 3자 비교에 쓴다.
4. **site-packages의 transformers 코드를 직접 편집하지 않는다.** HF 구현의
   수정이 필요하면(예: A-6에서 얕은 cache 고장 판정 시) `modeling_blt.py`를
   `blt_hf/patched/`로 복사하거나 `BltForCausalLM`을 서브클래싱해 자기
   네임스페이스에서 명시적으로 관리한다. 조건 세 가지가 모두 충족되어야 한다:
   (a) 원본 대비 diff를 NOTES.md에 기록, (b) 무수정 경로와의 parity 테스트
   동반(예: cache 수정 시 no-cache 출력 완전 일치), (c) 재현 스크립트 확보 후
   upstream 이슈/PR 지향. 단 attention mask는 원본 논문 의미와 다르더라도
   수정하지 않는다 — 마스크 변경은 사전학습 가중치와의 정합을 깨므로
   HF self-consistency를 유지하고 논문에 차이를 명시한다.
5. **실행 조건이 다른 shard를 aggregate하지 않는다.** `blt_hf/manifest.py`의
   fingerprint(모델 ID/revision, transformers 버전, attn_impl, use_cache,
   beam, batch, 절단 정책, code commit) 완전 일치가 aggregate의 전제이며,
   불일치 시 즉시 실패하도록 구현되어 있다. 이 검사를 우회하는 코드를
   작성하지 않는다.
6. **legacy 결과와 신규 결과를 같은 표에서 직접 비교하지 않는다.** legacy는
   full-causal fallback ablation으로만 병기한다.

## 환경

- conda env: `phdq_blt_hf` (PyTorch stable **cu128 wheel** + transformers ≥ 4.56,
  xFormers 없음). cu128은 5090(sm_120)과 A100(sm_80)을 하나의 환경으로 커버한다.
- **노드 분업**: Phase A의 HF 쪽 검증은 5090 노드 기본(GPU 소요 작음, 큐 절감).
  legacy 스택(torch 2.6 nightly cu121 + xFormers)은 sm_120 미지원 —
  legacy dump는 반드시 A100/V100에서 실행한다. Phase B 이후 본 학습·평가는 A100.
- 5090↔A100 교차 비교는 정수 산출물(patch 경계, token ID) 완전 일치 +
  실수 산출물(entropy, logits) 허용 오차 원칙. 완전 일치를 실수 값에 요구하지
  않는다.
- 버전은 `blt_hf/requirements.lock.txt`로 고정. 환경 변경 시 lock 재생성 +
  manifest의 transformers_version 갱신.
- 모든 job 시작부에서 `python blt_hf_checks/check_env.py` 실행 (스크립트에 포함).
- attn_implementation은 Phase A-2에서 결정된 값으로 고정하고 이후 변경하지
  않는다 (현재 값은 NOTES.md 참조).

## 디렉토리 구조

```
blt_hf/            # 본 패키지: model, data_adapter, train, generation, eval,
                   #   metrics(위임), manifest
blt_hf_checks/     # Phase A 검증 + 벤치 스크립트. fixtures/와 results/ 포함
scripts/train_blt_hf.sh, scripts/eval_blt_hf.sh
tests/test_hf_*.py
outputs/blt_hf/<dataset>/                              # 학습 체크포인트
outputs/blt_hf_eval/<dataset>/<split>/<ckpt>/<cond>/   # 평가 (생성 조건별 격리)
```

`blt_hf_checks/results/*.json`(cache_probe, gen_bench, m2_profile)은 P3 착수
판정의 근거 자료이므로 삭제·덮어쓰기 전에 git commit한다.

## 주요 명령

```bash
# Phase A 검증 (인터랙티브 GPU 또는 짧은 sbatch)
srun -p amd_a100nv_8 --gres=gpu:1 --comment=pytorch --time=00:30:00 \
  python blt_hf_checks/check_patch_parity.py

# 학습 smoke → 본 학습
NUM_GPUS=1 MAX_STEPS=50 sbatch --gres=gpu:1 scripts/train_blt_hf.sh
NUM_GPUS=8 sbatch --gres=gpu:8 --cpus-per-task=16 scripts/train_blt_hf.sh
DATASET_TYPE=learner NUM_GPUS=8 sbatch --gres=gpu:8 scripts/train_blt_hf.sh

# 생성 벤치 (P3a 판정 자료)
sbatch --gres=gpu:1 --wrap "python blt_hf_checks/bench_generation.py"

# 평가 (shard 수는 bench 결과로 재산정) → 합산
CKPT_PATH=outputs/blt_hf/native/best.ckpt sbatch scripts/eval_blt_hf.sh
AGGREGATE=1 CKPT_PATH=... sbatch scripts/eval_blt_hf.sh

# 테스트 (Mac 또는 로그인 노드, GPU 불필요한 것 우선)
pytest tests/test_hf_data_adapter.py tests/test_hf_manifest.py -q
```

## 작업 방식

- **테스트 먼저**: data_adapter, generation, manifest는 대응 테스트 파일이
  존재해야 하며, 수정 시 테스트를 먼저 갱신한다. batch 불변성 테스트
  (동일 문장, batch=1 vs 혼합 batch → 동일 출력)는 generation 변경마다 실행.
- **UTF-8 안전 절단**: source 절단은 반드시 문자 경계 스냅. 한글 3바이트를
  쪼개는 절단은 버그로 취급한다. separator는 어떤 경우에도 보존.
- **padding 방향 주의**: 생성은 left-padding이 표준이지만 BLT patcher와의
  상호작용이 검증 대상이다. 관련 코드를 만지면
  `tests/test_hf_generation.py`의 batch 불변성 테스트를 반드시 재실행.
- **결정 기록**: 코드 리딩 결과(HF patcher 위치, patch_lengths 입력 경로,
  마스크 의미), 판정 분기 결과, 우회 사유는 전부 `blt_hf/NOTES.md`에 날짜와
  함께 기록. 실행 명령·복구 절차는 `blt_hf/COMMANDS.md`.
- **커밋**: `[P1][mac|slurm] 설명` 형식. Phase 게이트 통과 시 LOG.md 갱신.

## 알려진 함정

- HF BLT는 편입 1년 미만 — `patch_lengths` 학습 경로, left-padding, use_cache
  에 미발견 버그 가능. 이상 동작은 먼저 최소 재현을 만들고 버전을 의심한다.
- `use_cache=False`가 문서 기준 기본 전제. cache 미동작 시 A-6의 깊이 판정을
  따른다: 얕은 고장은 `blt_hf/patched/`로 P1 안에서 수정(규칙 4의 3조건 준수),
  증분 상태 관리의 근본 부재는 P3a 스코프 — 이 프로젝트에서 설계하지 않는다.
- legacy 스택은 5090에서 실행되지 않는다(sm_120 미지원). legacy 관련
  스크립트를 5090에 제출하는 실수 방지 — 스크립트 시작부에 capability 검사.
- bf16/fp16 비결정성으로 greedy 출력이 미세하게 흔들릴 수 있다. parity
  판정은 "출력 문자열 완전 일치"를 기본으로 하되, dtype 실험 결과를 NOTES에
  남긴다.
- V100(cas_v100*)은 bf16 미지원 — fp16 경로 검증 없이 V100에 제출하지 않는다.
