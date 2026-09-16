# 일자별 작업 내역

## 2026-09-16 — Neuron 제출 구조를 직접 sbatch 방식으로 교정

- 실제 사용 이력이 있는 BART job script를 기준으로 train/eval/score 맨 앞에
  `#!/bin/bash`, field/appl comment, stdout/stderr, partition, node/task, 자원, wall time,
  `B:TERM@300` SBATCH header를 완성했다.
- 로그인 shell에서 Bash wrapper를 실행하는 `submit_blt_hf.sh`를 삭제했다. 사용자는
  프로젝트 root에서 각 batch script를 `sbatch --export=...`로 직접 제출한다.
- 할당된 job 안에서는 Python 및 torchrun을 `srun --ntasks=1`로 실행한다.
  `SLURM_SUBMIT_DIR`, comment, partition, CPU/GPU 조건을 compute job에서 재검사하고
  job/node/CUDA/자원 정보를 로그에 출력한다.
- shell syntax 및 SBATCH/srun 구조 회귀 테스트 5개 통과. 실제 Neuron 제출은 사용자 실행.

## 2026-09-16 — Neuron 제출 helper 수정

- 프로젝트 field를 `nlp` 기본값으로 고정하고 실제 comment 인자를
  `field=nlp;appl=pytorch` 한 개의 shell argument로 전달하도록 수정.
- `showque`/`showappl`이 정보를 출력한 뒤 nonzero를 반환해도 `set -e`로 제출이
  조기 종료되지 않도록 경고 처리. 제출 직전 mode/partition/CPU/GPU를 출력한다.
- 두 정보 명령이 각각 nonzero인 모의 Neuron 환경에서도 `sbatch` 도달 및 전체 인자를
  검사하는 회귀 테스트 추가. 로컬 전체 77개 중 57 통과/torch 관련 20 skip.

## 2026-09-16 — GitHub 공개 범위 정리

- `data/` 전체, artifacts/outputs/환경 캐시·가중치·인증정보·ssh.md를 Git에서 제외.
- 원문이 포함된 GLEU fixture 및 patch 보고서, Meta 검토용 원본 사본을 제외.
- GPL 표기가 상충하는 M2 소스·예제·원문 diff는 로컬 보존하고 공개본에서 제외.
  공개 코드에서 실행할 때 필요한 정확한 사본·hash·준비 절차는 THIRD_PARTY_NOTICES.md에 기록.
- 배포 허용 고지가 있는 HF/GLEU 파일의 저작권·라이선스와 변경 기록 유지.
- 공개 파일만 추출한 디렉터리에서 76개 검사 중 52 통과/의존성·비공개 입력 관련 24 skip.
  원본 모델 runtime 및 기존 검증 JSON은 변경하지 않았다.


## 2026-09-16 — Neuron 사용자 실행용 학습·성능 평가 구현

- ssh.md의 scratch root·A100 GPU/CPU 비율·showque/showappl·field/appl comment를 반영한
  학습/생성/CPU 채점/제출 스크립트 구현. neuron 미접속·미제출.
- DDP 누적·무패딩 학습·FP32 optimizer 상태·entropy 고정·전체 validation·checkpoint/RNG 재개.
- beam1/4 생성, 불변 batch 기록·shard 합산·전체 corpus GLEU/M2·timeout/중단 재개 구현.
- 사용자 지정 이전 프로젝트에서 GLEU/M2 scorer 확보 및 독립 이식. 이전 GLEU 4개,
  M2 2개 fixture 정확히 일치. 원본 파일/데이터/동결 패키지는 수정하지 않음.
- 로컬 76개 중 56 통과/20 skip, itcerdo HF 환경 최종 코드 76개 통과. 실제 B beam1/4 및
  FP32 parameter/bf16 autocast forward 통과. backward/optimizer는 실행하지 않음.
- Neuron 실제 학습 검사는 사용자 smoke에서 수행. 실행 문서: blt_hf/NEURON.md.

## 2026-09-16 — 검증 범위 변경 및 실제 B 전 계층 마스크 검사 완료

- 사용자 결정에 따라 P1/CLAUDE에서 전체 동질성 게이트·미확인 시 결과 폐기를 제거.
  변환 관련 검사는 가중치 보존·attention mask로 한정, 학습/평가 유효성 테스트 유지.
- 기존 515→510 텐서 전수 보존 증거 재사용, B 출력 파일 hash 및 strict load 재확인.
- itcerdo tmux에서 실제 B의 자동 패칭 6개 fixture × 59개 attention 모듈,
  강제 가상/0길이 patch fixture × 45개 모듈 검사 통과. window/EOS/causal/cross
  전체 마스크 및 미래 byte 연결 검사. 전체 단위 테스트 54개 통과.
- 보고서 `blt_hf_checks/results/p1_masks_20260916.json`, 실행 exit_code=0.
  가중치·mask passed, 학습·평가 not_run. eager/bf16/no-cache/무패딩 범위.
- neuron 미접속, backward/optimizer 미실행, 데이터·동결 패키지 미수정.
  아래 과거 기록의 동질성 pending은 당시 상태이며 현재 진행 조건이 아니다.

## 2026-09-15 — B 전수 검증 및 OSC 후보 GPU 구동

- 원본 515개 → B 510개 tensor 전수 값/dtype·coverage 검증 통과, 실제 strict load 통과.
- eager/SDPA 둘 다 forward/generate 가능하지만 영어 greedy IDs 불일치. 기준 eager 고정.
- 프로젝트 사본에 entropy/local window+EOS 및 global EOS 마스크, next-token 패치 슬롯,
  원본 entropy 산술을 적용. site-packages·동결 데이터·blt_gec는 변경하지 않음.
- bf16 threshold 반올림의 검증 규칙을 명시하고 실패/실수 기준 차이를 보존.
  독립 scalar 대비 20/20 패치 경계 일치. 실제 1,817-token forward와 생성 정상.
- 전체 동질성은 pending. neuron 접속·backward·학습은 미실행. 자세한 범위는 NOTES.md.

## 2026-09-15 — itcerdo 원본 확보·CPU 변환 완료, tmux 인계

- 사용자 제공 프로젝트 HF 토큰으로 main/entropy 고정 revision 다운로드 완료.
- 실제 HF 환경에서 단위 테스트 35개 통과. 별도/내장 entropy 129개 텐서 전수 동일.
- 직접 변환본 B 생성 완료: 510개 bf16 텐서. 상태 `converted_unvalidated`, 동질성 pending.
- 사용자 이동 요청에 따라 `phdq3-p1` tmux로 전환. CPU 비교·변환 exit_code=0 확인.
  세션은 shell 대기 중이며 추가 실행 중인 job은 없다. 명령/로그 위치는 blt_hf/COMMANDS.md.
- neuron 접속·학습은 미수행. 다음은 B 정적 검증·strict load·OSC 구동 구현.

## 2026-09-15 — P1 로컬 구현 착수

- 독립 blt_hf 데이터 어댑터·실행 manifest, 환경/데이터 점검 CLI 구현.
- Python 표준 라이브러리 unittest 25개 통과. 실제 HF/GPU 실행 검증은 미수행.
- 9개 split 재검사: 201,534행, 최대 입력 1,380, 2,048 초과 0건; TSV–M2 정합성 확인.
- 사용자 지시: neuron 접속·원격 작업 금지. itcerdo 정보 파일 수신 후 원격 단계 진행.
- 현 상태/명령/다음 단계는 blt_hf/NOTES.md, blt_hf/COMMANDS.md 참조.
