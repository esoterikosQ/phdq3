# Neuron 학습·성능 평가 실행

작업 위치는 `/scratch/r984a02/phdq3`이며, 접속·파일 이관·제출은 사용자가 수행한다.
에이전트는 neuron에 접속하지 않았다. 아래 스크립트는 기존 `phdq_blt_hf` 환경을
사용하며 패키지 설치나 CUDA module load를 하지 않는다.

## 준비

**코드 정본은 GitHub `main`이다.** 전달용
`artifacts/releases/p1-neuron-code-20260916.tar.gz`는 현재 `main`보다 오래됐으므로
최종 실행 코드로 사용하지 않는다. 이미 프로젝트 루트에 압축을 풀었다면 아래 명령으로
Git 추적 파일을 `main`으로 복구한다. `git reset --hard`는 추적 파일의 로컬 변경을
폐기하지만 Git에서 제외된 `data/`, `artifacts/`, `outputs/`, `ssh.md`는 지우지 않는다.
`git clean`은 실행하지 않는다.

기존 Git checkout인 경우:

```bash
cd /scratch/r984a02/phdq3
git fetch origin main
git reset --hard origin/main
git branch --set-upstream-to=origin/main main
git status --short
git log -1 --oneline
```

압축만 풀어서 `.git`이 없는 경우:

```bash
cd /scratch/r984a02/phdq3
git init
git remote add origin https://github.com/esoterikosQ/phdq3.git
git fetch origin main
git reset --hard FETCH_HEAD
git branch -M main
git branch --set-upstream-to=origin/main main
git status --short
git log -1 --oneline
```

압축에는 가중치·전체 데이터·인증 파일이 포함되지 않는다. 로컬 전용 M2 scorer 사본은
공개 Git에서도 제외되므로 기존 실험용 압축에서 남은 `blt_hf/vendor/m2/`는 보존한다.
그 존재 여부와 hash는 `THIRD_PARTY_NOTICES.md` 절차로 확인한다.

코드·`data/Preprocessed`의 제공된 9개 TSV/6개 val·test M2 및 직접 변환본 B가
프로젝트 아래에 있어야 한다. 기존 데이터는 수정하거나 재분할하지 않는다.
B의 경로는 `artifacts/converted/blt-1b-hf-own`이며 현재 itcerdo에 있는 B를 사용자가
이관해야 한다. model loader는 `conversion_B_20260915.json`의 출력 SHA256과 로컬
파일을 대조하므로 원본 소스 weights를 neuron에 중복 복사할 필요는 없다.
HF 토큰은 필요 없다. 모든 모델 로드는 offline/local-files-only다.

```bash
cd /scratch/r984a02/phdq3
conda activate phdq_blt_hf
showque
showappl
export FIELD=nlp
# batch shell에서 conda를 못 찾는 환경만 실제 conda.sh 경로를 지정한다.
# export CONDA_SH=/apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh
```

이 프로젝트에서 확인한 field는 `nlp`이며 제출 helper의 기본값도 `nlp`다. `showque`와
`showappl`은 로그인 shell에서 위와 같이 직접 확인한다. 사이트 helper는 비대화형
스크립트에서 비정상 종료 상태를 반환할 수 있어 제출 helper 내부에서는 다시 호출하지 않는다.
제출에는 `--comment="field=nlp;appl=pytorch"`를 지정한다. 잘못된 field는 scheduler가
거부한다. 임의 partition 접근·정책 우회·자동 연쇄 제출은 하지 않는다.

지원 GPU partition은 `amd_a100nv_8`(GPU당 CPU ≤8, active ≤4)와 `amd_a100_4`
(GPU당 CPU ≤16, active ≤2)다. GPU job에는 `--gres=gpu:N`을 지정한다. 기본은
`amd_a100nv_8`, 1 node/1 SLURM task, GPU당 CPU 8개이며 torchrun이 GPU별 rank를 만든다.
V100/H100/H200/GH200 경로는 현재 구현·검증 범위에 넣지 않았다.
job array는 사용하지 않으며 running limit은 scheduler가 적용한다.

CPU 채점은 `cpu` partition을 사용한다. GPU를 할당해 CPU 채점을 기다리지 않는다.
기본 제한 시간은 1:55, 10분 전 USR1 신호와 Python 실행 시간 제한으로 저장·중단한다.
반환 코드 75는 **재개 가능한 미완료**이며 완료로 해석하지 않는다. 자동 재제출하지 않는다.

## 1. 사용자 실행 학습 검사

```bash
# 1 GPU: 작은 모델의 실제 backward/optimizer 재개 검사 후, 실제 B의 긴 train 입력 2 update.
RUN_ID=native-smoke-01 DATASET_TYPE=native NUM_GPUS=1 \
  bash scripts/submit_blt_hf.sh smoke

# 위 작업 완료 및 메모리 확인 후 2 GPU DDP 검사. 같은 RUN_ID를 재사용하지 않는다.
RUN_ID=native-ddp-smoke-01 DATASET_TYPE=native NUM_GPUS=2 \
  bash scripts/submit_blt_hf.sh smoke

# tiny overfit: train 앞 4개에만 수행하는 별도 진단. 평가용 checkpoint로 사용하지 않는다.
RUN_ID=native-overfit-01 DATASET_TYPE=native NUM_GPUS=1 \
  LR=0.0001 OVERFIT_STEPS=200 bash scripts/submit_blt_hf.sh overfit
```

`smoke`는 `check_train_forward.py`에서 label shift, encoder/global/decoder gradient,
고정 entropy, optimizer/RNG 복원 후 연속 학습 결과를 확인한다. 이어 실제 B의 선택한
데이터셋에서 가장 긴 train 문장들을 사용해 optimizer 상태까지 할당한다.
이 진단이 실제 A100 학습 통과의 근거이며, 현재 에이전트가 실행한 상태는 아니다.
`overfit`의 최종 loss<0.05 여부는 `completed.json`에 기록한다.
`smoke/overfit` checkpoint는 본 평가 CLI가 거부한다.

검사 로그: `artifacts/logs/blt-hf-train-<jobid>.out`.
작업별 환경·9개 split 길이 보고서와 작은 모델의 training report는
`blt_hf_checks/results/neuron_<jobid>_<restart>_*.json`에 남는다.

## 2. 본 학습 및 재개

```bash
RUN_ID=native-main-01 DATASET_TYPE=native NUM_GPUS=8 EFFECTIVE_BATCH=32 \
  bash scripts/submit_blt_hf.sh train

# 시간 제한/중단 후, 동일 코드·설정·GPU 수·RUN_ID로 이어서 수행한다.
RUN_ID=native-main-01 DATASET_TYPE=native NUM_GPUS=8 EFFECTIVE_BATCH=32 \
  RESUME=outputs/blt_hf/native/native-main-01/latest.json \
  bash scripts/submit_blt_hf.sh train

# 별도 실험 ID, 각 원본 split 유지. native 결과 확인 후 순차 제출한다.
RUN_ID=learner-main-01 DATASET_TYPE=korean_learner NUM_GPUS=8 \
  bash scripts/submit_blt_hf.sh train
RUN_ID=union-main-01 DATASET_TYPE=union NUM_GPUS=8 \
  bash scripts/submit_blt_hf.sh train
```

학습 기본값은 3 epoch, LR 1e-5, warmup 2000 update(총 step 안으로 제한), cosine decay,
AdamW `(0.9,0.95)`, eps 1e-8, weight decay 0.1, grad clip 1.0이다.
main 모델 전체(해시 임베딩 포함)를 학습하며 entropy patcher는 고정한다.
파라미터·gradient·Adam 상태는 FP32, 연산은 bf16 autocast, gradient checkpointing을 쓴다.
이 선택은 작은 BF16 weight update를 손실하지 않기 위한 것이며 manifest에 명시한다.

microbatch는 rank당 **무패딩 1개**이며 `EFFECTIVE_BATCH`를 GPU 수에 맞춰 나누고
`no_sync`로 누적한다. 마지막 step도 원본 샘플을 버리거나 중복 가중하지 않는다.
남는 rank는 zero-loss 동기화 계산만 하며, loss는 실제 supervised token의 전역 평균이다.
DDP는 optimizer 상태 할당 전에 gradient bucket view를 준비하는 2회의 계산을 수행한다.
이때 파라미터는 업데이트하지 않고 RNG를 복구한다.

**메모리·저장공간**: 파라미터 수에는 큰 hash embedding이 포함되므로 “1B이니 작다”는
가정을 하지 않는다. FP32 Adam/DDP는 GPU당 큰 상태를 복제한다. 코드는 최소 메모리
추정치를 검사하지만 activation·통신·allocator까지 보장하지는 않으므로 긴 입력 smoke가
필요하다. OOM이면 full training을 강행하지 않고 sharding 계획을 추가한다. 현재 FSDP/ZeRO
자동 전환은 없다. checkpoint 하나는 모델 FP32와 Adam 상태를 함께 저장해 수십 GB다.
저장 전에 scratch 여유 공간/쿼터를 확인한다. 체크포인트는 자동 삭제하지 않는다.

- 경로: `outputs/blt_hf/<dataset>/<RUN_ID>/step-<update>-<id>/`
- `model.safetensors`, `training.pt`, `checkpoint.json`을 staging directory에 완성한 뒤 게시한다.
- `latest.json`/`best.json`은 작은 포인터다. **best는 전체 validation의 target-token loss**로 선택한다.
- 기본 `SAVE_EVERY=500` update, epoch 끝 및 중단 때 저장한다. `MAX_STEPS`는 이번 job의
  실행량만 제한하며 학습 schedule을 다시 만들지 않는다.
- optimizer·epoch 내 다음 배치·global step·rank별 RNG·실행 manifest를 복원한다.
  코드/데이터/모델/환경 설정/world size가 달라지면 같은 run의 resume를 거부한다.
- 진짜 epoch가 끝난 뒤 전체 validation을 평가하며, 중단된 validation 부분 점수로
  best를 갱신하지 않는다. `completed.json`이 있어야 모든 epoch 완료다.

## 3. 생성

학습 완료 후 `best.json`이 가리키는 **불변 step 디렉터리**를 확인해 `CKPT_PATH`로
지정한다. 포인터가 학습 중 움직이면 다른 checkpoint의 shard를 합칠 수 없도록 실패한다.
beam 1과 4, batch 설정별로 **서로 다른 EVAL_DIR**를 사용한다.

```bash
# 아래 STEP_DIRECTORY를 best.json의 실제 checkpoint 이름으로 바꾼다.
export CKPT_PATH=outputs/blt_hf/native/native-main-01/STEP_DIRECTORY
export EVAL_DIR=outputs/blt_hf_eval/native/test/native-main-01-beam1
DATASET_TYPE=native BLT_NUM_BEAMS=1 BATCH_SIZE=1 SHARD_COUNT=1 SHARD_ID=0 \
  bash scripts/submit_blt_hf.sh eval

# interrupted generation: 같은 인자와 EVAL_DIR로 다시 제출하면 완료 batch부터 재개한다.
# 여러 shard가 필요할 때만 SHARD_COUNT를 먼저 고정하고 SHARD_ID=0..N-1 각각 제출.
# active/running job 제한을 지키며 앞선 job 완료 후 다음 shard를 제출한다.
```

기본은 eager/bf16, beam1, no-cache, batch1, max_new_bytes768, context2048이다.
`BLT_NUM_BEAMS=4`도 지원한다. batch를 늘리면 prompt 길이가 정확히 같은 행만 묶고
원래 문장 순서로 복원한다. padding/문서 packing/cache는 사용하지 않는다.
실측 없이 batch8의 처리량을 가정하지 않으며 A100 batch 변경은 생성 fixture로 확인한다.
source/target 절단은 없다. BOS/SEP 보존·prompt+출력 budget 한도를 검사한다.
EOS 미도달·budget 소진·잘못된 UTF-8·잘못된 token ID를 각 예측에 기록한다.
UTF-8 오류는 replacement 문자로 드러내며 해당 문장을 평가에서 제외하지 않는다.
점수용 출력은 공백을 정규화하지만 raw text와 token IDs도 보존한다.

`shards/<id>/batches/*.json`은 완성 batch 단위 불변 기록이며 출력 SHA256을 완료 보고서에
기록한다. 여러 job이 같은 shard를 쓰면 잠금으로 실패한다. shard topology·모델·data·
code·config·scorer·생성 설정 fingerprint가 다르면 재개·합산을 거부한다.

## 4. 전체 성능 평가

```bash
# 모든 shard 완료 후 CPU job에서 GLEU + M2 P/R/F0.5를 계산한다.
DATASET_TYPE=native CPUS=8 M2_WORKERS=8 \
  bash scripts/submit_blt_hf.sh score
```

scorer는 `/Users/esoterikos/Nextcloud/QLab/phdq`에서 확인한 **기존 실험 구현**을 독립
이식했다. 원본 파일·commit·SHA256·namespace 변경 diff·라이선스는
`blt_hf/vendor/provenance.json` 및 vendor 디렉터리에 있다. `blt_gec`/이전 프로젝트를
runtime으로 import하지 않으며, 새 평가에는 NumPy/SciPy도 필요 없다.
기존 GLEU wrapper의 단일 reference·6자리 반올림·100배 척도를 유지한다.
M2는 기존 NUS scorer Python3 사본, beta=.5, max unchanged words=2, case/공백 무시 옵션 false다.

합산 시 모든 canonical 행이 정확히 1회 있어야 한다. GLEU는 corpus 충분통계를 합산하고
M2 annotator 선택도 원래 문장 순서로 수행한다. shard별 점수를 평균하지 않는다.
GLEU는 먼저 `scored/gleu.json`에 저장한다. M2는 문장별 통계를 저널에 저장하고
30초→120초→480초→1920초 timeout pass로 재시도한다. wall-time이 오면 중단 후 재개한다.
미완료 문장이 있으면 partial/75를 반환하고 전체 F0.5를 발표하지 않는다.
완료 시 `scored/metrics.json`에 GLEU(0–100), M2 P/R/F0.5(0–1), EOS/copy율,
UTF-8 오류·budget 소진 건수 및 전체 행 수를 기록한다.

## 검증 범위

최종 단위 검사: Mac 76개 중 56 통과/torch 관련 20 skip, itcerdo 76개 전부 통과.
증거: `blt_hf_checks/results/neuron_final_checks_20260916.json`.
Mac: 분할·누락·중복·hash·재개·지표·shell/CLI 테스트.
itcerdo: 준비된 HF 환경의 단위 테스트, 실제 B의 beam1/4 생성 및 혼합 정밀도 **forward만**.
Neuron: 실제 backward/optimizer/DDP/메모리/중단 재개는 위 사용자 smoke로 확인해야 한다.
원본 BLT 수치 동질성은 학습·성능 평가의 게이트가 아니다.

구현 참고: [PyTorch 2.11 DDP](https://docs.pytorch.org/docs/2.11/generated/torch.nn.parallel.DistributedDataParallel.html),
[AdamW](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html).
노드 정책의 정본은 프로젝트 `ssh.md`와 제출 시 `showque`/`showappl`이다.
