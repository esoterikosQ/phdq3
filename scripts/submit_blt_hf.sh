#!/usr/bin/env bash
# User-run submission helper. Does not contact Neuron from another machine.
set -euo pipefail
cd /scratch/r984a02/phdq3
[[ -z "${SLURM_JOB_ID:-}" ]] || { echo 'Submit from a Neuron login shell, not from inside another job' >&2; exit 2; }
case "$(hostname -s)" in glogin01|glogin02|glogin03|login01|login02|login03) ;; *) echo 'Neuron login host required' >&2; exit 2;; esac
mode=${1:-}
# These commands are informational on Neuron and may return a nonzero status
# after printing valid queue/quota information. Do not let `set -e` abort the
# submission before sbatch. The reviewed field for this project is `nlp`.
showque || echo 'Warning: showque returned nonzero; continuing with scheduler checks.' >&2
showappl || echo 'Warning: showappl returned nonzero; continuing with field=nlp.' >&2
FIELD=${FIELD:-nlp}
[[ "$FIELD" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid FIELD syntax' >&2; exit 2; }
mkdir -p artifacts/logs
case "$mode" in
  train|smoke|overfit)
    script=scripts/train_blt_hf.sh
    export TRAIN_MODE=$mode
    if [[ "$mode" == overfit ]]; then export NUM_GPUS=1 LR=${LR:-0.0001}; fi
    ;;
  eval) script=scripts/eval_blt_hf.sh; export NUM_GPUS=1;;
  score) script=scripts/score_blt_hf.sh;;
  *) echo 'Usage: RUN_ID=... bash scripts/submit_blt_hf.sh train|smoke|overfit|eval|score' >&2; exit 2;;
esac
opts=(--nodes=1 --ntasks=1 "--comment=field=${FIELD};appl=pytorch" --export=ALL)
if [[ "$mode" == score ]]; then
  partition=cpu
  cpus=${CPUS:-8}
  [[ "$cpus" =~ ^[1-9][0-9]*$ && "$cpus" -le 48 ]] || exit 2
  active=$(squeue -h -u "$USER" -p cpu -o '%i' | wc -l)
  (( active < 12 )) || { echo 'CPU active-job limit reached' >&2; exit 2; }
  opts+=(--partition=cpu --cpus-per-task="$cpus")
else
  partition=${PARTITION:-amd_a100nv_8}
  case "$partition" in amd_a100nv_8) limit=8; max=8; max_active=4;; amd_a100_4) limit=16; max=4; max_active=2;; *) echo 'Unreviewed partition' >&2; exit 2;; esac
  export NUM_GPUS=${NUM_GPUS:-1}
  [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ && "$NUM_GPUS" -le "$max" ]] || exit 2
  cpus=${CPUS:-$((8 * NUM_GPUS))}
  [[ "$cpus" =~ ^[1-9][0-9]*$ ]] && (( cpus <= limit * NUM_GPUS )) || { echo 'CPU/GPU ratio exceeds partition policy' >&2; exit 2; }
  # No job arrays: explicit shards avoid unintentionally multiplying active jobs.
  active=$(squeue -h -u "$USER" -p "$partition" -o '%i' | wc -l)
  (( active < max_active )) || { echo 'Partition active-job limit reached; wait before submitting another shard' >&2; exit 2; }
  opts+=(--partition="$partition" --gres="gpu:$NUM_GPUS" --cpus-per-task="$cpus")
fi
printf 'Submitting %s with field=%s partition=%s cpus=%s' "$mode" "$FIELD" "$partition" "$cpus"
if [[ "$mode" != score ]]; then printf ' gpus=%s' "$NUM_GPUS"; fi
printf '\n'
sbatch "${opts[@]}" "$script"
