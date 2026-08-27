# CLAUDE.md — P3: 조건부 후속 프로젝트 (P3a / P3b / P3c)

> 배치 위치: 착수 확정 시 각 하위 프로젝트 디렉토리로 분리 배치한다:
>   P3a → `PHDQ2/blt_hf/cache/CLAUDE.md`
>   P3b → `PHDQ2/cpp/m2scorer/CLAUDE.md`
>   P3c → `PHDQ2/byte_topk/CLAUDE.md` (또는 독립 저장소)
> 분리 전까지는 이 파일이 통합 가이드다.
> 루트 `PHDQ2/CLAUDE.md`가 항상 함께 적용된다.
> 상세 계획: `P3_조건부_후속_프로젝트_계획.md`

## 최우선 규칙: 착수 게이트

**어떤 P3 코드도 착수 판정 기록 없이 작성하지 않는다.** Claude Code 세션에서
P3 관련 구현 요청을 받으면 먼저 다음을 확인한다:

```text
1. LOG.md에 해당 하위 프로젝트의 착수 판정(착수/보류/불착수 + 근거)이 있는가
2. 판정 근거 파일이 존재하는가:
   P3a → blt_hf_checks/results/cache_probe.json + gen_bench_*.json
   P3b → blt_hf_checks/results/m2_profile_*.json
   P3c → 스코프 결정 기록 (지도교수 합의)
3. 없으면: 구현 대신 판정 자료 생성(P1의 해당 Phase)을 먼저 제안한다
```

P3 작업은 P1·P2의 미완 작업보다 우선하지 않는다. "만들 수 있다"는 착수
사유가 아니다.

---

## P3a — Patch-aware Incremental Generation Cache

### 트리거 (둘 다 충족 시에만)

(i) P1 A-6 probe의 고장 깊이 판정이 **"(다) 깊은 부재"** — 증분 상태 관리
자체가 없어 얕은 수정으로 해결 불가, **그리고**
(ii) batched no-cache 처리량으로 전체 평가가 소수 job으로 완료 불가.

### 절대 규칙

0. **얕은 고장은 P3a 스코프가 아니다.** `blt_hf/patched/` 모듈 수정으로
   해결 가능한 문제로 판명되면 작업을 P1로 반환한다 (P1 CLAUDE.md 규칙 4의
   3조건 적용). P3a 착수 시 P1의 patched 시도 기록을 Phase 0에서 먼저
   리뷰한다.
1. **upstream 우선**: 착수 전 Transformers의 BLT cache 관련 PR/이슈를 확인
   한다. 진행 중이면 자체 구현 대신 upstream 기여(테스트·리뷰)로 전환한다.
   자체 구현하더라도 최종 산출 1순위는 upstream PR이다.
2. **correctness 게이트**: Phase 1(greedy·batch 1)의 step-wise parity
   (`full_recompute` vs `incremental`, rtol/atol 명시) 통과 전에는 batching·
   beam·속도 최적화 코드를 작성하지 않는다. 속도 측정도 Phase 4 전에는
   하지 않는다.
3. **패칭 인과성 전제 검증 필수**: Phase 0의 `check_patch_causality.py`
   (전체 패칭 vs prefix 증분 패칭의 경계 불변성) 결과 없이 state 구현을
   시작하지 않는다. 비인과 규칙이 발견되면 인과적 변형을 쓰고 차이를
   정량 기록한다.
4. **불변식**: global transformer는 patch 확정 시에만 1 step 전진한다.
   이 불변식을 깨는 "최적화"는 금지.
5. backend는 `blt_hf/generation.py`의 `backend="incremental"`로 통합하고
   manifest fingerprint에 기록한다. 다른 backend shard와의 aggregate 차단은
   기존 검사에 위임한다.

### 작업 방식

- parity 실패 시 이분 격리: 컴포넌트별(entropy/encoder/global/decoder)
  hidden dump 디버그 모드로 최초 divergence 지점을 찾은 뒤 수정한다.
  허용 오차 완화로 테스트를 통과시키지 않는다.
- beam cache 재정렬은 `index_select` 기반 단순 구현 먼저. copy-on-write는
  실측으로 병목이 확인된 후에만 (선최적화 금지).
- 테스트: `tests/test_incremental_parity.py`, `test_blt_batched_beam.py`
  (no-cache 결과와의 출력·score 동등).

---

## P3b — C++ M2 Scorer 이식

### 트리거

기 구현된 병렬 Python + 체크포인트 + timeout pass 체제로 전체 M2가 운용
시간 내 미완료, 또는 unresolved 비율이 기준 초과 (기준은 판정 기록에 명시).

### 절대 규칙

1. **compat 우선**: Phase 1은 속도를 무시하고 Python scorer와의 완전 호환
   (문장별 correct/proposed/gold, tie-breaking 포함)만 목표로 한다.
   fixture 회귀를 깨는 최적화는 어떤 속도 이득이 있어도 폐기한다.
2. **점수 의미 보존**: 문장별 F0.5 평균 금지. corpus 점수는 충분통계
   (C/P/G) 합산으로만 계산한다. partial 결과는 `status: partial` + 완료율을
   강제 표기하고 최종 점수로 보고하지 않는다.
3. **런타임 의존성 0**: SLURM 노드는 오프라인일 수 있다. 외부 패키지
   다운로드가 필요한 빌드/실행 경로를 만들지 않는다. C++17 표준 + (가능 시)
   OpenMP만.
4. **Python scorer는 영구 유지**: fallback과 회귀 검증용. `--m2_backend
   python|cpp` 스위치를 제거하지 않으며, 기본값 전환은 native/learner/union
   전체 대조 통과 후에만.
5. **idempotent commit**: 문장 완료 후에만 TSV에 append. 동일 sentence_id
   재처리 시 기존 완료 기록을 덮어쓰지 않는다. 결과 파일 쓰기는 단일 writer.

### 작업 방식

- 최적화는 한 번에 하나씩, 개별 커밋 + fixture 회귀. tie-breaking이나 최종
  edit sequence가 바뀌면 그 커밋은 revert.
- deadline 검사는 3대 반복문(레벤슈타인/transitive arc/경로 탐색) 안에
  주기적으로. 너무 잦은 검사로 처리량을 깎지 않도록 반복 횟수 기반 간격.
- Unicode: Python `str.split()` 호환 whitespace 목록을 상수로 고정하고
  UTF-8 바이트를 절대 변형하지 않는다. 관련 테스트를 fixture에 포함.
- 빌드/실행:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release cpp/m2scorer && cmake --build build -j
./build/m2score_cpp --hypothesis h.txt --source-gold test.m2 \
  --sentence-ids pending.txt --output completed.tsv \
  --timeout-seconds 30 --threads $SLURM_CPUS_PER_TASK --pass-id 0
ctest --test-dir build   # fixture 회귀
```

---

## P3c — 이식성 우선 커스텀 Byte LM (byte_topk)

### 트리거

P1·P2 완료 + 논문 스코프를 "아키텍처 기여"로 확장하기로 한 명시적 결정.
자동 트리거 없음.

### 절대 규칙 (설계 하드 제약 — 위반 코드는 merge 금지)

1. 외부 entropy 모델 없음: boundary score는 encoder 위 head에서 계산.
2. 정적 shape: boundary는 고정 Top-K 선택. 입력마다 텐서 shape이 변하는
   경로를 만들지 않는다. `torch.compile` graph break 0을 CI에서 확인.
3. 모든 attention에 eager reference 구현 존재. SDPA는 기본 최적화,
   Flash/Triton은 optional. **custom autograd / C++ extension / FlexAttention
   의존 금지.** local decoder의 byte→patch 매핑은 정적 gather로 구현한다.
4. CPU에서 전체 forward/backward correctness 테스트 통과 (CI 필수 항목).
5. 가변 패치는 padding + mask(max_patch_len 고정)로 표현.
6. checkpoint는 safetensors.
7. **compute-matched 프로토콜 사전 등록**: 비교 조건(A: BPE Transformer,
   B: 고정 패칭, C: SpaceByte식 한국어 규칙, D: 제안 모델)의 모델 크기·step
   산정 근거를 학습 시작 전에 문서로 확정하고, 결과를 본 뒤 사후 조정하지
   않는다.
8. 대규모 사전학습(수백 B 바이트)은 스코프 밖. 모델 50–300M, 데이터
   5–30GB 상한을 넘는 실험을 제안하지 않는다.

### 작업 방식

- 참고 구현: goombalab/hnet(MIT) — 라우터 안정화 기법만 참고, mamba_ssm
  의존 부분은 채택 금지. MBLM — 고정 계층 baseline 구조 참고.
- Top-K 학습 불안정 시 사전 정의된 완화 순서: soft 가중 변형 → warmup
  스케줄 → 2단계 학습(경계 고정 후 본체). 즉흥 해법 전에 이 순서를 시도.
- P2 자산 재사용: 코퍼스 파이프라인, 형태소 정합 분석 스크립트, DDP/resume
  인프라. 재구현하지 않는다.
- 이식성 매트릭스 테스트({CPU, A100, V100} × {eager, SDPA} × {fp32, bf16}
  × {compile on/off})를 Phase 3에서 전 조합 실행하고 결과표를 저장한다.
- 커밋: `[P3c][mac|slurm] 설명`. 공개 준비 시 라이선스는 Apache-2.0 또는
  MIT (자체 코드이므로 선택 가능 — CC-BY-NC 코드 복사 금지 원칙 유지).

---

## 공통 컨벤션

- 커밋: `[P3a|P3b|P3c][machine] 설명`
- 판정·게이트 통과·주요 결정은 LOG.md에 날짜와 근거 파일 경로와 함께 기록
- P3 작업 중 P1/P2 코드에 버그를 발견하면 P3 브랜치에서 고치지 말고
  해당 프로젝트로 이슈를 넘긴다 (수정 커밋은 P1/P2 컨벤션으로)
