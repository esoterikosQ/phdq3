# BLT-GEC 평가 코드 개선 및 C++ 이식 계획

## 0. 구현 상태

2026-08-03 기준으로 Phase 0의 평가 절차 수정은 코드에 반영했다.

- `blt_gec/eval.py`: 병합 파일 atomic write, GLEU 선저장 및 즉시 로그 출력
- `blt_gec/m2_resumable.py`: 문장별 M2 충분통계 계산과 체크포인트 저장
- `blt_gec/m2_resumable.py`: worker process hard timeout과 단계별 timeout 증가
- `blt_gec/m2_resumable.py`: 완료, slow, timeout, error, unresolved ID 분리 저장
- `blt_gec/metrics.py`: 기존 scorer를 유지하면서 체크포인트 backend 추가
- `scripts/eval_blt.sh`: timeout, multiplier, pass 수, worker 수 환경변수 연동
- `tests/`: 기존 NUS scorer 수치 일치 및 timeout 재시작 회귀 테스트 추가

아직 구현하지 않은 부분은 C++ scorer 이식과 C++ 알고리즘 최적화다. 아래의
C++ 관련 절은 후속 구현 계획이며, Python 평가 절차 개선 부분은 현재 동작을
설명한다.

## 1. 목적

현재 BLT 평가 파이프라인은 생성된 shard를 병합한 뒤 GLEU와 M2 기반
Precision, Recall, $F_{0.5}$를 한 번에 계산한다. GLEU 계산은 비교적 빠르지만,
기존 Python M2 scorer는 일부 문장에서 edit graph가 크게 증가하면 계산 시간이
급격히 길어질 수 있다. 이 때문에 전체 aggregate 작업이 제한 시간에 종료되고,
이미 계산한 결과도 재사용하지 못하는 문제가 발생한다.

이 개선 작업의 목표는 다음과 같다.

1. GLEU와 M2 평가 단계를 분리하고 각 단계의 결과를 즉시 보존한다.
2. M2 평가를 문장 단위로 중단, 재시작 및 재시도할 수 있게 만든다.
3. 완료된 문장의 통계를 다시 계산하지 않고 누적한다.
4. 기존 Python M2 scorer와 동일한 결과를 내는 C++ 구현을 작성한다.
5. 문장 단위 CPU 병렬화와 알고리즘 최적화로 전체 처리 시간을 줄인다.
6. 일부 문장의 실패를 숨기지 않고 평가 완료율과 미완료 목록을 명시한다.

## 2. 수정 전 구현과 병목

수정 전 `blt_gec/eval.py`의 `aggregate_shards()`는 다음 순서로 동작한다.

1. `hypothesis_*.txt`, `reference_*.txt`, `source_*.txt` shard의 범위와 길이를 검증한다.
2. shard를 각각 `hypothesis.txt`, `reference.txt`, `source.txt`로 병합한다.
3. GLEU를 계산한다.
4. `blt_gec/metrics.py`를 통해 기존 Python M2 scorer를 subprocess로 실행한다.
5. 두 평가가 모두 끝난 뒤에만 `metrics.json`을 저장하고 결과를 출력한다.

따라서 M2 계산 중 SLURM 시간 제한에 도달하면 이미 끝난 GLEU도 독립된 결과
파일로 남지 않으며, M2는 다음 작업에서 처음부터 다시 시작한다. 또한 M2
subprocess의 표준 출력을 완료 시점까지 `PIPE`에 보관하므로 문장별 진행 상태를
확인할 수 없다.

기존 M2 scorer의 주요 병목 후보는 다음과 같다.

- 오류문과 예측문 사이의 Levenshtein 행렬 및 edit graph 구성
- 가능한 edit merge를 확장하는 transitive-arc 계산
- edit graph에서 최적 경로를 찾는 반복 계산
- 문장 길이와 수정 후보 수가 큰 일부 문장에서 발생하는 조합 폭증
- 전체 corpus를 단일 프로세스에서 순차 처리하는 구조

이 작업은 모델 추론이 아니라 이미 생성된 문자열의 정렬과 집계이므로 GPU보다
CPU 구현과 문장 단위 병렬화의 영향을 크게 받는다.

## 3. 제안한 4단계 방식에 대한 평가와 수정안

### 3.1 M2 시작 전에 GLEU 저장 및 출력

제안대로 적용한다. shard 병합과 GLEU 계산이 끝나면 M2를 시작하기 전에 다음
작업을 완료해야 한다.

- 병합된 세 파일을 임시 파일에 쓴 뒤 atomic rename으로 확정한다.
- 입력 파일의 행 수와 해시를 `evaluation_manifest.json`에 기록한다.
- GLEU 결과와 소요 시간을 `gleu.json`에 저장한다.
- 같은 결과를 표준 출력에 기록하고 즉시 flush한다.

M2가 실패하거나 제한 시간으로 종료되어도 GLEU는 유효한 완료 결과로 남는다.
최종 `metrics.json`은 GLEU와 M2 결과를 참조해 생성하되, M2가 미완료이면
`status: partial`과 M2 완료율을 명시한다.

### 3.2 평균보다 오래 걸리는 문장 스킵

방향은 타당하지만, 실행 중 계속 변하는 산술평균을 hard timeout으로 직접
사용하면 결과가 불안정해진다.

- 초반 몇 문장의 실행 시간에 따라 timeout 기준이 크게 달라질 수 있다.
- 매우 느린 문장 자체가 평균을 올려 이후 기준을 무력화할 수 있다.
- 평균을 조금 넘었다는 이유만으로 정상 문장을 중단하면 실행 순서에 따라 완료
  목록이 달라진다.
- Python thread만으로는 CPU-bound scorer를 안전하게 강제 중단할 수 없다.

따라서 `slow` 판정과 `timeout` 판정을 분리한다.

- `slow`: 완료는 되었지만 이전 pass의 중앙값 또는 P95보다 오래 걸린 문장
- `timeout`: 해당 pass에 고정된 hard time budget을 초과해 계산을 중단한 문장

첫 pass의 timeout은 사전 benchmark에서 정한 고정값 `T0`를 사용한다. 이후
pass는 `T1`, `T2`처럼 고정 배수로 늘린다. 중앙값/P95는 다음 실행의 `T0`를
정하는 참고 통계로만 사용하고, 진행 중인 pass의 기준은 변경하지 않는다.

C++ 구현에서는 Levenshtein, graph 확장, 경로 탐색의 긴 반복문 안에서 deadline을
주기적으로 검사해 안전하게 중단한다. C++ 이식 전 Python 임시 구현에서는 문장별
worker process를 사용해야 하며, timeout된 worker는 종료 후 재생성한다.

### 3.3 완료 점수와 미완료 문장 목록 저장

문장별 $F_{0.5}$를 저장한 뒤 이를 평균해서 corpus 점수를 만들면 안 된다. M2의
corpus 점수는 문장별 충분통계(sufficient statistics)를 합산해 계산해야 한다.

각 완료 문장마다 다음 값을 저장한다.

```text
sentence_id, correct, proposed, gold, elapsed_seconds, status,
source_tokens, hypothesis_tokens, edit_distance, pass_id, error
```

전체 corpus에서는 다음과 같이 합산한다.

```text
C = sum(correct)
P = sum(proposed)
G = sum(gold)
Precision = C / P
Recall = C / G
F0.5 = 1.25 * C / (P + 0.25 * G)
```

분모가 0인 경우는 기존 scorer의 동작과 동일하게 처리한다. 현재 평가용 M2
데이터는 annotator ID 0만 사용하는 것으로 확인되어 문장별 통계 합산이 전체
scoring과 일치한다. 향후 복수 annotator M2를 사용할 경우에는 기존 scorer의
annotator 선택 규칙까지 보존되는지 별도 검증해야 한다.

상태는 최소한 다음과 같이 구분한다.

- `completed`: 통계 계산 완료
- `slow_completed`: 제한 안에 끝났지만 slow 기준을 넘음
- `timeout`: deadline 초과, 다음 pass에서 재시도
- `invalid_input`: M2 또는 hypothesis 파싱 실패
- `internal_error`: scorer 내부 오류

`completed`와 `slow_completed`의 통계는 누적 파일에 기록하고, `timeout`은 다음
pass의 입력 목록으로 저장한다. 같은 `sentence_id`를 다시 처리할 때는 기존 완료
기록을 덮어쓰지 않는 idempotent 동작이 필요하다.

### 3.4 미완료 목록의 단계별 재시도

제안대로 timeout을 단계적으로 늘리는 pass 구조를 사용한다.

```text
pass 0: 전체 문장, timeout T0
pass 1: pass 0의 timeout 목록만, timeout T1 = T0 * k
pass 2: pass 1의 timeout 목록만, timeout T2 = T1 * k
...
final: completed 통계 합산, 남은 문장은 unresolved로 기록
```

구체적인 `T0`, 배수 `k`, 최대 pass 수는 benchmark 결과와 SLURM 제한 시간으로
결정한다. 예를 들어 30초, 2분, 10분, 1시간처럼 증가시킬 수 있지만 이 값은
초기 기본값일 뿐 실제 측정 후 확정해야 한다.

중요한 원칙은 미완료 문장을 제외한 점수를 공식 최종 점수로 표시하지 않는
것이다. 일부만 완료되었을 때는 다음 정보를 함께 출력한다.

- `status: partial`
- 전체 문장 수, 완료 수, 미완료 수, 완료율
- 완료 문장에 한정한 provisional Precision, Recall, $F_{0.5}$
- 미완료 `sentence_id` 목록

모든 문장이 끝난 경우에만 `status: complete`인 최종 M2 점수를 생성한다.

## 4. 목표 평가 파이프라인

### 단계 A: shard 병합과 입력 검증

1. shard 범위의 시작, 끝, gap, overlap을 검사한다.
2. hypothesis, reference, source의 행 수가 모두 같은지 확인한다.
3. 병합 파일을 atomic write로 저장한다.
4. 데이터셋, split, checkpoint, 파일 해시, 문장 수를 manifest에 기록한다.
5. 이전 실행의 manifest와 해시가 다르면 기존 M2 중간 결과를 재사용하지 않는다.

### 단계 B: GLEU 독립 계산

1. 병합 파일로 GLEU를 계산한다.
2. 값, 문장 수, 시작/종료 시간, 소요 시간을 `gleu.json`에 즉시 저장한다.
3. `GLEU completed` 로그를 flush한다.
4. 이후 M2 실패 여부와 관계없이 GLEU 산출물을 보존한다.

### 단계 C: 문장별 M2 통계 계산

1. hypothesis와 source-gold M2를 sentence ID로 대응시킨다.
2. 이미 완료된 ID는 건너뛴다.
3. 지정된 ID를 여러 CPU worker에 분배한다.
4. 각 worker는 문장별 `correct`, `proposed`, `gold`와 시간을 반환한다.
5. timeout 또는 오류 문장은 원인별 목록에 저장한다.
6. 일정 문장 수 또는 일정 시간마다 진행 상태를 atomic checkpoint한다.

### 단계 D: timeout 재시도

1. 이전 pass의 `timeout` ID만 읽는다.
2. 증가된 timeout으로 다시 실행한다.
3. 성공한 ID는 완료 통계에 병합한다.
4. 최대 pass 이후에도 실패한 ID는 `unresolved`로 남긴다.

### 단계 E: 최종 집계

1. sentence ID 중복과 누락을 검사한다.
2. 모든 완료 문장의 충분통계를 합산한다.
3. Precision, Recall, $F_{0.5}$를 계산한다.
4. 완료율 100%일 때만 최종 점수를 확정한다.
5. GLEU와 M2 결과를 합쳐 최종 `metrics.json`을 생성한다.

## 5. 산출물 구조

```text
outputs/blt_eval/{dataset}/{split}/{checkpoint}/
  evaluation_manifest.json
  source.txt
  reference.txt
  hypothesis.txt
  gleu.json
  metrics.json
  m2/
    run_config.json
    completed.tsv
    slow_ids.txt
    invalid_ids.txt
    unresolved_ids.txt
    progress.json
    passes/
      pass_000_pending.txt
      pass_000_timeout.txt
      pass_000_summary.json
      pass_001_pending.txt
      pass_001_timeout.txt
      pass_001_summary.json
    final_metrics.json
```

대용량 문장별 통계는 C++에서 추가 의존성 없이 빠르게 기록할 수 있도록 TSV를
기본 형식으로 사용한다. 실행 설정, 요약 결과, 진행 상태는 Python orchestration
계층에서 JSON으로 기록한다.

## 6. C++ 이식 범위와 구현 과업

### 6.1 호환성 기준 고정

최적화 전에 기존 Python scorer의 동작을 기준 결과로 고정한다.

- native, learner, union의 작은 고정 표본을 만든다.
- 삽입, 삭제, 치환, 공백 수정, 무수정 문장을 포함한다.
- 긴 문장과 edit 후보가 많은 병목 문장을 별도 fixture로 보존한다.
- Python scorer의 corpus 점수와 문장별 `correct/proposed/gold`를 저장한다.
- 동일 비용 경로가 여러 개인 경우의 tie-breaking 순서도 테스트한다.

### 6.2 C++ 프로젝트 골격

다음과 같은 독립적인 C++17 CLI를 추가한다.

```text
cpp/m2scorer/
  CMakeLists.txt
  include/
    m2_parser.hpp
    levenshtein.hpp
    edit_graph.hpp
    scorer.hpp
    deadline.hpp
  src/
    main.cpp
    m2_parser.cpp
    levenshtein.cpp
    edit_graph.cpp
    scorer.cpp
  tests/
```

Release build는 `-O3 -DNDEBUG`를 사용하고, 지원되는 환경에서는 OpenMP를
활성화한다. SLURM 노드에서 외부 패키지 다운로드가 필요하지 않도록 런타임
의존성을 최소화한다.

### 6.3 입력 파서 이식

- M2의 `S` 문장과 `A` annotation, blank-line 문장 경계를 파싱한다.
- hypothesis의 한 줄과 M2의 한 문장을 정확히 대응시킨다.
- annotation의 span, 오류 유형, 교정 문자열, annotator ID를 보존한다.
- UTF-8 바이트열을 손상시키지 않는다.
- Python `str.split()`과 Unicode whitespace 처리 차이가 결과에 영향을 주지
  않도록 호환 규칙을 명시하고 테스트한다.
- 잘못된 문장은 전체 프로세스를 종료하지 않고 `invalid_input`으로 보고한다.

### 6.4 Levenshtein 및 edit graph 이식

- Python 구현과 같은 비용 함수와 backpointer 생성 순서를 구현한다.
- 삽입, 삭제, 치환 arc의 좌표와 label을 동일하게 만든다.
- 가능한 edit merge와 transitive arc 확장 규칙을 그대로 재현한다.
- 첫 버전은 속도보다 결과 호환성을 우선한다.
- 연속 메모리 배열과 정수 index를 사용해 Python object overhead를 제거한다.

### 6.5 최적 경로 및 M2 matching 이식

- 기존 scorer의 경로 탐색과 edit sequence 선택 규칙을 재현한다.
- gold annotation과 system edit의 matching을 구현한다.
- 문장별 `correct`, `proposed`, `gold`를 반환한다.
- 복수 annotator 입력의 선택 규칙을 보존한다.
- 0으로 나누는 경우와 무수정 문장의 처리를 Python scorer와 일치시킨다.

### 6.6 deadline과 안전 중단

- 문장 시작 시 monotonic-clock deadline을 설정한다.
- Levenshtein, transitive arc, 경로 탐색 반복문에서 주기적으로 deadline을
  검사한다.
- 시간 초과 시 해당 문장의 임시 객체만 폐기하고 worker는 다음 문장으로 이동한다.
- timeout과 메모리 한계 초과를 서로 다른 상태로 기록한다.
- 한 문장의 실패가 완료 파일을 손상시키지 않도록 문장 완료 후에만 commit한다.

### 6.7 문장 단위 병렬화

- 문장은 서로 독립적이므로 sentence ID queue를 CPU worker에 분배한다.
- `--threads` 인자로 worker 수를 제어한다.
- worker가 결과 파일에 직접 동시에 쓰지 않고, 단일 writer가 ID 순서와 무관하게
  완료 레코드를 append한다.
- 최종 집계 시 sentence ID 기준으로 정렬하고 중복을 검사한다.
- `OMP_NUM_THREADS`와 `SLURM_CPUS_PER_TASK`가 일치하도록 실행 스크립트에서
  설정한다.

### 6.8 알고리즘 최적화

호환 버전 통과 후 다음 최적화를 하나씩 적용하고 매 단계 회귀 테스트를 수행한다.

- Levenshtein 행렬을 연속 메모리로 배치하고 불필요한 복사를 제거한다.
- edit graph의 adjacency lookup을 hash 또는 index 기반 구조로 교체한다.
- 명시적 삼중 반복인 transitive-arc 확장을 희소 그래프 순회로 변경한다.
- graph가 DAG임이 보장되는 구간은 Bellman-Ford 대신 topological shortest path를
  사용한다.
- 동일 상태의 merge 후보를 memoization하고 중복 arc 생성을 제거한다.
- 문장 길이와 edit distance를 이용해 작업 queue의 load balancing을 개선한다.

최적화가 tie-breaking이나 최종 edit sequence를 바꾸면 해당 변경은 채택하지
않는다. `compat` 모드와 `optimized` 모드를 일정 기간 함께 유지해 결과를 비교한다.

### 6.9 CLI 인터페이스

예상 실행 인터페이스는 다음과 같다.

```bash
m2score_cpp \
  --hypothesis hypothesis.txt \
  --source-gold test.m2 \
  --sentence-ids pass_000_pending.txt \
  --output completed.tsv \
  --timeout-seconds 30 \
  --threads 16 \
  --pass-id 0
```

CLI는 일정 간격으로 `processed`, `completed`, `timeout`, `failed`, 처리율과
경과 시간을 출력한다. 종료 코드는 전체 완료, 일부 timeout, 입력 오류를 구분한다.

### 6.10 Python 평가 코드 연동

- `blt_gec/eval.py`에서 merge, GLEU, M2 단계를 별도 함수와 별도 CLI mode로
  분리한다.
- `blt_gec/metrics.py`는 기존 Python scorer와 C++ scorer를 선택할 수 있게 한다.
- 기본값을 즉시 C++로 바꾸지 않고 `--m2_backend python|cpp`를 제공한다.
- C++ 실행 전 binary 존재 여부와 버전을 manifest에 기록한다.
- timeout pass 생성, 결과 병합, 최종 점수 계산은 Python orchestration이 맡는다.
- 기존 Python scorer는 회귀 검증 및 fallback 용도로 유지한다.

### 6.11 SLURM 실행 스크립트

- aggregate와 M2 retry를 별도 job으로 제출할 수 있게 한다.
- GPU는 요청하지 않고 CPU와 메모리를 명시한다.
- `--cpus-per-task`를 C++ thread 수에 전달한다.
- 사이트 정책에 맞는 application comment를 항상 포함한다.
- pass 간 dependency 또는 수동 재제출을 모두 지원한다.
- job 종료 signal을 받으면 새 문장 시작을 중단하고 현재 상태를 flush한다.

## 7. 구현 순서

### Phase 0: 기존 Python 파이프라인 안전장치

1. GLEU를 M2 전에 `gleu.json`으로 저장하고 로그를 flush한다.
2. merge manifest와 입력 해시를 추가한다.
3. M2 부분 결과, 완료 ID, pending ID 형식을 확정한다.
4. partial 결과에 완료율을 강제로 표시한다.
5. 문장별 timing profiler로 실제 병목 문장을 수집한다.

### Phase 1: C++ 호환 구현

1. M2 parser를 이식한다.
2. Levenshtein과 edit graph를 이식한다.
3. edit matching과 충분통계 계산을 이식한다.
4. 단일 thread에서 Python 기준 결과와 비교한다.
5. native 전체 데이터에서 최종 수치가 동일한지 검증한다.

### Phase 2: 재시작 및 timeout

1. 문장별 checkpoint writer를 구현한다.
2. deadline check와 timeout 상태를 구현한다.
3. pass별 pending 목록과 재시도 로직을 구현한다.
4. SLURM 종료 후 재실행해 완료 문장을 건너뛰는지 검증한다.

### Phase 3: 병렬화와 알고리즘 최적화

1. CPU thread 병렬화를 추가한다.
2. graph 자료구조와 경로 탐색을 최적화한다.
3. worker 수별 처리량과 메모리를 측정한다.
4. Python, C++ compat, C++ optimized의 결과를 상호 비교한다.

### Phase 4: 운영 전환

1. native, learner, union 평가를 C++ backend로 다시 실행한다.
2. 이전 Python 결과가 있는 데이터셋과 점수를 대조한다.
3. 실행 명령과 복구 절차를 `command.md`에 기록한다.
4. 충분한 검증 후 C++ backend를 기본값으로 전환한다.

## 8. 검증 기준

### 정확성

- fixture마다 Python과 C++의 `correct/proposed/gold`가 완전히 같아야 한다.
- native 전체 Precision, Recall, $F_{0.5}$가 기존 결과와 허용 오차 내에서 같아야
  한다.
- thread 수와 처리 순서가 달라도 최종 결과가 같아야 한다.
- 중단 후 재시작한 결과가 한 번에 완료한 결과와 같아야 한다.
- 누락, 중복, 입력 해시 변경을 자동으로 탐지해야 한다.

### 성능과 복구성

- 최소 30초마다 진행 상태가 로그 또는 progress 파일에 남아야 한다.
- 비정상 종료 시 손실되는 작업은 현재 처리 중인 문장으로 제한해야 한다.
- 병목 문장 하나가 전체 worker를 무기한 정지시키지 않아야 한다.
- 실제 native, learner, union benchmark 후 Python 대비 처리량과 peak memory를
  보고해야 한다.
- 최종 성능 목표는 benchmark 이후 확정하되, 우선 Python 대비 10배 이상의
  corpus 처리량 개선을 목표로 한다.

## 9. 주의할 위험

1. **점수 의미 변경**: 문장별 $F_{0.5}$ 평균은 corpus M2와 다르므로 반드시
   충분통계를 합산해야 한다.
2. **부분 점수 오용**: timeout 문장을 제외한 점수는 선택 편향이 있으므로 최종
   점수로 보고하면 안 된다.
3. **tie-breaking 변화**: 더 빠른 graph 알고리즘이 동일 비용 경로의 선택 순서를
   바꾸면 최종 edit가 달라질 수 있다.
4. **Unicode 차이**: Python과 C++의 whitespace 및 UTF-8 처리 차이가 token span을
   바꿀 수 있다.
5. **복수 annotator**: 현재 데이터에 맞춘 단순 합산을 다른 M2 corpus에 그대로
   적용하면 scorer 의미가 달라질 수 있다.
6. **timeout 구현 오류**: deadline 검사가 너무 드물면 timeout이 사실상 작동하지
   않고, 너무 잦으면 정상 문장의 처리량이 감소한다.
7. **입력 변경 후 재사용**: checkpoint 또는 hypothesis가 바뀌었는데 이전 통계를
   재사용하지 않도록 manifest hash 검증이 필수다.

## 10. 최종 권고안

사용자가 제안한 네 단계는 전체 방향이 맞다. 다만 `평균 초과 즉시 스킵`은
`고정 timeout pass + 별도 slow 통계`로 바꾸고, 저장 단위는 문장별 점수가 아니라
`correct/proposed/gold`로 바꿔야 한다. 이 두 가지를 지켜야 실행 순서와 무관하게
재현 가능한 corpus M2 점수를 얻을 수 있다.

단기적으로는 GLEU 선저장과 M2 중간 상태 저장을 Python 코드에 먼저 추가한다.
중기적으로는 기존 scorer의 결과를 그대로 재현하는 C++17 호환 구현을 완성한다.
그 다음에만 graph 최적화와 CPU 병렬화를 적용한다. 이 순서라야 속도를 개선하면서도
논문 비교에 필요한 평가 지표의 의미를 보존할 수 있다.
