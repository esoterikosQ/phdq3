# P3a 캐시 개선 작업 분리·인계

작성 기준: 2026-09-27 `main` 커밋 `6014db5`. 상세 기술 근거는
`plan/P3a_generation_cache_20260925.md`와 `blt_hf/cache/DESIGN.md`,
Neuron 명령은 `blt_hf/NEURON.md`를 따른다. 이 문서는 **두 Codex 작업창이
동시에 움직일 때의 소유권과 인계 절차**다. 사용자는 별도 캐시 작업창을
열고, 이 파일을 첫 지시와 함께 읽힌다. 이 문서를 만들면서 원격 worktree를
생성하거나 Neuron 작업을 제출하지 않는다.

## 목표와 현재 기준선

- 현재 `global-prefix-guarded-greedy-v1`은 개발 순서상 캐시 v3다. 매 바이트
  전체 entropy/patcher·local encoder·local decoder를 실행하고, 패치 경계가
  그대로일 때 global transformer 출력만 재사용한다. 상위 두 합법 byte의
  logit 간격이 1.0 이하이면 전체 forward로 갱신한다. decoder KV 재사용은
  꺼져 있다. 기본 HF 생성 경로는 바꾸지 않는다.
- Neuron A100 10번 시험은 기존 불일치 32건+대조군 12건에서 44/44 완성
  출력 일치, global skip 후보 2,505회 중 갱신 59회였다. 이는 전체 split
  동등성의 증거가 아니다. 11번 전체 native validation 2,634건의 guarded
  생성·HF 기준 비교는 `blt_hf/NEURON.md`에 준비되어 있다.
- 10번 job 전체 시간 159→322초에는 환경·데이터 검사와 첫 문장 55의
  이상치가 섞였다. 순수 생성 합계는 v2 117.76초, guarded 124.36초다.
  원인 미확정 지연은 기록만 유지하고 캐시 개발의 선결 조건으로 삼지 않는다.

## 작업창·호스트별 책임

| 구분 | 맡을 일 | 하지 않을 일 |
| --- | --- | --- |
| **캐시 작업창 / Mac 별도 worktree** | `blt_hf/cache/`와 선택형 생성 backend의 설계·구현, 작은 테스트, itcerdo 실행 코드·보고서, 변경 커밋 | 현재 `main` checkout에서 동시 편집, 학습·checkpoint 선택 정책 변경 |
| **캐시 작업창 / itcerdo RTX 5090** | 변환 1B의 BF16 단건 생성, 동일 GPU의 HF↔캐시 출력 비교, patch 경계·단계별 시간·메모리 프로파일 | optimizer/backward/파인튜닝, Neuron 속도나 출력 동등성 주장 |
| **BLT 작업창 / Mac `main`** | 기존 학습·평가 유지, 사용자 Neuron 보고서 분석, 캐시 변경 인계·통합, 새 학습 run의 코드·명령 | 캐시 worktree의 미완성 코드를 임의로 복사하거나 실행 중인 Neuron checkout을 갱신 |
| **사용자 / Neuron A100** | 11번 전수 생성·비교 및 이후 새 통합 학습 job 제출, 로그·보고서 동기화 | 에이전트에 Neuron 접속·전송·SLURM 제출 위임 |

사용자의 이번 지시로 **itcerdo의 캐시 개선용 추론·프로파일은 허용**한다.
`CLAUDE.md`의 학습 금지와 작업 경로 제한은 그대로 적용한다. 사용자
지시에 따라 에이전트는 Neuron에 접속하거나 파일을 보내지 않는다.

## 작업공간과 파일 경계

1. Mac의 이 BLT 작업창은 현재 프로젝트 `main`을 유지한다. 새 캐시 작업창은
   Codex의 **별도 Git worktree**로 열어 임시 로컬 브랜치
   `codex/cache-p3a`에서 작업한다. 사용자 파일인 `command.md`, `papers/`,
   `tmp/`를 정리·reset·stash하지 않는다. GitHub 원격은 **`main` 하나**로
   유지하며 개발 중인 브랜치를 원격에 푸시하지 않는다.
2. itcerdo에서 캐시 전용 checkout은 허용된 프로젝트 내부인
   `~/projects/phdq3/artifacts/worktrees/cache-p3a`를 쓴다. 기존
   `~/projects/phdq3`의 `main`, 변환 모델과 데이터는 그대로 둔다.
   `/artifacts/`는 Git 무시 경로이며, 하위 worktree 생성이 가능한 구조다.
   최초 1회 기존 root가 깨끗하고 사용 중이지 않은지 확인한 뒤
   `git worktree add -b codex/cache-p3a artifacts/worktrees/cache-p3a main`으로
   만든다. 이미 있으면 재생성하지 않고 상태를 확인한다. 모든 itcerdo
   읽기·쓰기 경로는 `~/projects/phdq3` 안에 둔다.
3. 변환 모델은 기존 root의 `artifacts/converted/blt-1b-hf-own`을 **읽기
   전용 입력**으로 사용한다. 전용 worktree에는
   `artifacts/converted/blt-1b-hf-own`을 원본 디렉터리로 향하는 symlink로
   연결한다. 모델 사본을 다시 만들지 않는다. 제공 데이터가 필요한 시험은
   기존 `data/Preprocessed`의 파일 존재·해시를 먼저 확인하고 원본을 수정하지
   않는다. 전용 worktree에서 데이터가 필요하면 읽기용 symlink를 사용한다.
4. 개발 중 큰 trace·프로파일·임시 patch는 전용 worktree의
   `artifacts/cache_dev/`에 둔다. 작고 재현 가능한 최종 JSON만
   `blt_hf_checks/results/`에 명시적으로 추가한다. Neuron의 기존
   `outputs/blt_hf_eval/*`와 06~11 보고서를 덮어쓰지 않는다.
5. 캐시 창의 주 소유 파일은 `blt_hf/cache/**`, `blt_hf/generation.py`,
   캐시 진단용 `blt_hf_checks/*cache*`, `tests/test_hf_cache_*`,
   `tests/test_hf_generation_runtime.py`, 이 문서와 `blt_hf/cache/DESIGN.md`다.
   BLT 창은 `blt_hf/train.py`, `blt_hf/integrated_validation.py`,
   `blt_hf/eval.py`, Neuron 제출 스크립트, 학습·GLEU 문서를 맡는다.
   공통 파일 변경은 인계 때 한 창에서 합치고, 두 창이 동시에 편집하지 않는다.

Mac 캐시 worktree에서 itcerdo 전용 worktree로 시험할 소스만 옮길 때는
macOS 기본 `rsync -az` 또는 패치 파일을 사용한다. `--append-verify`,
`--info=progress2`, `--delete`는 사용하지 않는다. 변환 모델·데이터·기존
결과 디렉터리를 소스 동기화에 포함하지 않는다. 전송 후 itcerdo의 전용
worktree에서 `git status --short`와 코드 해시를 확인해 시험한 버전을
명시한다. 결과 JSON은 같은 전용 경로에서 Mac 캐시 worktree로 가져온다.

## 캐시 작업창의 개발 순서

1. 현재 guarded backend와 no-cache backend를 **같은 itcerdo 5090, 같은
   모델·dtype·입력**으로 비교한다. 필요하면 Neuron native 체크포인트의
   `model.safetensors`(약 8.7GB)와 `checkpoint.json`만 프로젝트 안의
   `artifacts/cache_inputs/`로 별도 전송한다. 생성 시험에는 `training.pt`
   optimizer 상태가 필요 없다. 체크포인트를 갖기 전에는 사전학습 1B의
   결과를 native fine-tuned 결과로 부르지 않는다.
2. 전체 forward를 entropy/patcher, local encoder, global, local decoder로
   나눠 GPU 동기화 후 시간을 잰다. 최초 호출·warmup·반복 호출을 분리하고
   문장 길이·출력 byte 수·patch 경계 변경·guard 갱신 횟수·peak GPU 메모리를
   기록한다. 5090 수치를 A100 예상 시간으로 환산하지 않는다.
3. 단일 구성요소를 고칠 때마다 패치 경계/byte ID·EOS·UTF-8·완성 문자열을
   **같은 GPU의 HF no-cache**와 비교한다. 경계가 뒤집히는 사례, 한글
   멀티바이트, 긴 입력, 조기 EOS를 포함한다. local encoder/decoder의
   상태를 추가 재사용한다면 patch 경계·position·mask·무효화 조건을
   `blt_hf/cache/DESIGN.md`에 명시한다. 확인되지 않은 BF16 동등성을
   가정해 캐시 상태를 확대하지 않는다.
4. 새 실험 backend 이름을 부여하고 기존 v1/v2/guarded·HF 경로를 보존한다.
   작은 실제 1B 완성 생성의 출력·시간 결과와 테스트 명령, 커밋, 모델·입력
   해시를 인계 보고서에 기록한다. itcerdo 시험만으로 운영 backend를
   바꾸지 않는다.

## BLT 작업창으로 인계하는 조건

캐시 창은 커밋 해시, 변경 파일, backend 이름, 5090 모델·입력 해시,
HF↔캐시 불일치 수, 생성 시간(준비·warmup 제외), peak 메모리, 테스트
결과와 미확인 범위를 전달한다. BLT 창은 캐시 branch의 diff를 검토해
`main`에 합치고 원격 `main`에만 푸시한다. 사용자가 Neuron에서 실행 중인
job을 마친 뒤 pull한다. 기존 Neuron run의 코드 지문·checkpoint·출력
directory는 재사용하지 않는다.

Neuron에서는 동일 A100·체크포인트·전체 native validation으로 HF 기준과
새 backend의 token ID/EOS/문자열 및 순수 생성 시간·p50/p95·메모리를
확인한다. 준비된 11번 전수 guarded 시험도 이 인계의 일부다. **전수 출력
일치와 실사용 속도 이득**이 확인되면 BLT 창이 새 학습 run의
`--validation-generation-backend` 인자, run manifest/code hash,
epoch별 전체 validation GLEU·best 선택 보고서를 연결한다. 현재 통합
학습의 HF 기본값은 유지한다. 새 backend가 beam 1·batch 1 전용이면
통합 학습도 그 조건으로 시작한다. 새 RUN_ID에서 처음부터 최대 10-epoch
schedule로 학습하고, 매 epoch 전체 validation GLEU의 최고 checkpoint를
선택한다. 연속 3 epoch 미개선이면 중단하는 규칙을 사전에 고정한다.
시간 제한에 따른 재개는 같은
코드·GPU 수·schedule을 유지한다. 기존 2-epoch run을 `EPOCHS=10`으로
늘려 재개하지 않는다.

## 새 작업창에 넣을 첫 지시

> `cache.md`와 `CLAUDE.md`, `plan/P3a_generation_cache_20260925.md`를 읽고
> 캐시 개선 작업을 별도 worktree에서 맡아라. itcerdo의 프로젝트 내부
> 전용 worktree에서 단건 BF16 추론·프로파일을 수행하고, Neuron에는 접속하지
> 마라. HF 기본 경로와 BLT 학습 코드는 바꾸지 말고, 변경·테스트·인계
> 결과를 이 문서의 범위대로 기록해라. 기존 사용자 파일은 보존해라.
