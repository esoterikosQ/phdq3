# CLAUDE.md — P2: 한국어 패처 프로젝트

> 배치 위치: `PHDQ2/ko_patcher/CLAUDE.md`
> 루트 `PHDQ2/CLAUDE.md`(클러스터·컨벤션)가 항상 함께 적용된다.
> 상세 계획: `P2_한국어_패처_프로젝트_계획.md` — 착수 전 해당 절을 먼저 읽는다.

## 프로젝트 개요

한국어 바이트 엔트로피 모델(패처)을 학습하고 세 갈래로 활용한다:
P2-1 패칭 특성 분석(논문 기여 1), P2-2 엔트로피 탐지 기반 detect-then-correct
GEC(논문 기여 2), P2-3 P1 파인튜닝용 패처 ablation(P1 완료 후).

핵심 프레임: 패처는 **오류 탐지·국소화**를 담당하고, 교정은 별도 메커니즘
(후보 rescoring 또는 국소 infilling)이 담당한다. "패치 구조를 표준 패칭에
맞게 직접 수정"하는 교정은 성립하지 않으므로 그런 방향의 코드를 만들지 않는다.

## 절대 규칙 (위반 금지)

1. **GEC 데이터(native/korean_learner/union)의 어떤 split도 사전학습 코퍼스에
   넣지 않는다.** train split도 금지다. 위반 시 P2-2의 탐지 평가 전체가
   오염된다. `corpus/clean.py`에 GEC 파일 경로 제외 필터를 하드코딩하고,
   최종 코퍼스에 대해 GEC val/test 문장과의 exact-match 검사(오염 스캔)를
   pack 단계에서 강제한다.
2. **held-out 규율**: 분석용 held-out(`analysis_heldout/`)은 코퍼스 정제
   직후, 학습 시작 전에 분리한다. 모든 분석 스크립트(P2-1)는 held-out만
   입력으로 받는다. 학습 데이터로 분석하지 않는다.
3. **BLT 호환성은 하드 제약이다.** 토크나이저(vocab 260, 특수토큰 ID),
   엔트로피→패치 세그멘테이션 규칙은 BLT와 완전 일치해야 한다. 이것은
   언어적 행동의 문제가 아니라 **기계적 인터페이스 호환성**의 문제다 —
   바이트 토크나이저는 학습되지 않는 고정 매핑이고, blt-1b의 바이트 임베딩이
   legacy ID 체계에 결박되어 있으므로 매핑이 어긋나면 P2-3(패처 주입)이
   성립하지 않는다. 호환성 테스트(`test_ko_tokenizer.py`,
   `test_ko_patching.py`)를 깨는 변경은 merge하지 않는다. fixture는 P2가
   자체 생성한다(legacy 코드 리딩으로 명세 확정 — P1 산출물에 의존하지
   않음). 세그멘테이션 로직은 "인공 entropy 값 주입 → 동일 경계"로 모델과
   분리해 검증한다.
4. **native test는 최종 1회만 평가한다.** 모든 튜닝(threshold, 신호 조합,
   λ)은 val에서 수행한다. test 결과가 나온 뒤 튜닝으로 되돌아가지 않는다.
5. **탐지 span 밖의 편집은 교정으로 인정하지 않는다** (correct_infill의
   reject 규칙). 이 규칙을 완화하는 실험을 하려면 별도 조건으로 명명하고
   기본 시스템과 혼동되지 않게 기록한다.
6. **평가는 P1 파이프라인을 재사용한다.** GLEU/M2 계산 코드를 이 패키지에
   재구현하지 않는다. `blt_hf/eval.py`의 aggregate·metrics를 import하고
   fingerprint에 system 이름을 추가한다.

## 환경

- conda env: P1과 동일한 `phdq_blt_hf` 사용 가능 (추가 의존: 형태소 분석기).
- 형태소 분석기는 kiwipiepy 우선(순수 pip 설치), 클러스터 설치 실패 시
  분석(P2-1-4)만 Mac에서 수행한다 — held-out은 소규모라 로컬로 충분.
- 학습은 A100 1–4장이면 충분하다. 8장 요청은 과할당이므로 하지 않는다.

## 디렉토리 구조

```
ko_patcher/
  corpus/       # download / clean / pack + SOURCES.md(라이선스), STATS.md
  model.py      # entropy_small(~25M, 디버깅) / entropy_base(본 학습)
  tokenizer.py  # BLT 호환 바이트 토크나이저
  train.py      # resume 1급 기능 (shard index/offset까지 복원)
  patching.py   # next_byte_entropy / segment
  analysis/     # entropy_profile, threshold_sweep, morpheme_alignment,
                #   crosslingual_compare (held-out 전용)
  gec/          # detect, candidates, correct_rescore, correct_infill, eval_detect
  ablation/     # patch_stats (P2-3)
scripts/train_ko_patcher.sh, analyze_ko_patcher.sh, eval_ko_gec.sh
outputs/ko_patcher/            # 체크포인트, figures/, 분석 결과
```

## 주요 명령

```bash
# 코퍼스 (CPU job 또는 로그인 노드 — 대용량 처리는 CPU sbatch)
python -m ko_patcher.corpus.clean --config ko_patcher/corpus/config.yaml
python -m ko_patcher.corpus.pack  --chunk-bytes 8192

# 학습: small smoke → base 본 학습 (1:55 체인, 자동 resume)
MODEL_CFG=entropy_small MAX_STEPS=2000 sbatch --gres=gpu:1 scripts/train_ko_patcher.sh
MODEL_CFG=entropy_base NUM_GPUS=2 sbatch --gres=gpu:2 scripts/train_ko_patcher.sh
# 이어서: sbatch --dependency=afterany:<jobid> ... (RESUME은 last.ckpt 자동 감지)

# 분석 (held-out 전용)
python -m ko_patcher.analysis.threshold_sweep --ckpt outputs/ko_patcher/base/best.ckpt
python -m ko_patcher.analysis.morpheme_alignment --analyzer kiwipiepy

# 탐지 평가 → 교정 → 통합 평가
python -m ko_patcher.gec.eval_detect --split val --signals surprisal,entropy,patch
sbatch scripts/eval_ko_gec.sh   # system ∈ {detect_rescore, detect_infill, ...}

# 테스트
pytest tests/test_ko_tokenizer.py tests/test_ko_patching.py tests/test_ko_detect.py -q
```

## 작업 방식

- **resume 우선**: train.py의 어떤 변경도 "중단 → 재시작 → 한 번에 학습한
  결과와 동일 궤적" 검증을 통과해야 한다 (optimizer/scheduler/데이터 위치
  복원). 1:55 제한 환경에서 resume이 깨지면 학습 전체가 무효가 된다.
- **탐지 신호는 분리 구현**: surprisal / entropy / 패치 이상을 각각 독립
  신호로 구현하고 결합은 별도 단계로 둔다. eval_detect의 비교 표가 신호별
  기여를 보여줘야 한다.
- **M2 span 변환기 주의**: gold edit은 토큰 span, 탐지는 바이트 span이다.
  변환기(`token span ↔ byte span`)는 단위 테스트 필수 — 한글 다바이트와
  공백 처리에서 off-by-one이 흔하다.
- **precision 방어 장치 유지**: rescoring에는 무편집 옵션을 항상 후보에
  포함하고, edit_cost 패널티 λ를 설정 파일로 노출한다.
- **figure 재현성**: analysis/의 모든 그림은 스크립트 1회 실행으로 재생성
  가능해야 한다. 수동 편집 금지, 난수 시드 고정.
- **커밋**: `[P2][mac|slurm] 설명`. 분석 결과 요약은 LOG.md에 갱신.

## 결정 지점 (기록 필수)

- P2-2 진입 전: eval_detect의 val F1이 기준(예: 0.4) 미달이면 교정 단계를
  축소하고 탐지·패칭 분석 중심으로 기여를 재구성한다. 판정과 근거를 LOG.md에.
- P2-3은 P1 Phase B 완료(파인튜닝 파이프라인 검증) 전에는 착수하지 않는다.
  HF 패처 주입 가능 여부는 P1의 `blt_hf/NOTES.md` 코드 리딩 결과를 따른다.
