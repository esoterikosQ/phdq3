# P1 구현·판정 기록

## 2026-09-16 — Neuron 학습·생성·평가 코드 구현

실행 정본은 `NEURON.md`다. 에이전트는 neuron에 접속·전송·제출하지 않았다.

- `train.py`: FP32 main/Adam + bf16 autocast, 고정 entropy, gradient checkpointing,
  무패딩 microbatch1, 전역 token 평균 loss, DDP/no_sync, 마지막 배치 누락·중복 방지.
  rank별 RNG/optimizer/epoch 배치 위치를 불변 checkpoint directory에 저장한다.
  전체 validation loss로 best를 선택하며 중단된 validation의 부분 점수는 채택하지 않는다.
- `generation.py`, `eval.py`: beam1/4, 동일길이 무패딩 묶음, 원래 순서 복원,
  batch 단위 원자적 기록/재개, 전체 split의 정확한 coverage·fingerprint·출력 hash 검사.
  source/target 절단이나 평가 행 제외 없이 UTF-8 오류·EOS/budget 상태도 기록한다.
- 기존 프로젝트 `/Users/esoterikos/Nextcloud/QLab/phdq`에서 scorer를 읽어 독립 이식.
  GLEU wrapper 4개 fixture, M2 CLI 2개 fixture가 이전 구현과 정확히 일치했다.
  `results/metrics_legacy_20260916.json` 및 `vendor/provenance.json`에 출처/증거를 보존.
  기존 GLEU의 소수점 6자리 반올림 후 100배 척도도 유지한다.
- M2는 CPU job에서 충분통계 저널·timeout retry·중단 재개를 사용한다.
  미완료 문장이 있으면 최종 corpus F0.5를 출력하지 않는다. `gleu.json`은 먼저 보존한다.
- 제출 helper는 ssh.md 기준 A100 partition, GPU/CPU 비율, active job 제한, showque/showappl,
  새 field/appl comment를 적용한다. array/자동 재제출/자동 환경 변경은 하지 않는다.
- **실행 증거**: Mac 76개 중 56 통과/torch 필요 20 skip. itcerdo HF 환경에서는
  최종 코드 76개 전부 통과 (`neuron_final_checks_20260916.json`). 실제 B의 8-token beam1/4 생성과 mixed-length 입력의 동일길이 묶음
  결과가 단건 실행과 같았다. FP32 parameter + bf16 autocast forward도 유한 loss.
  `p1_neuron_code_v3_20260916.json`, peak allocated 24,505,433,600 bytes.
- 첫 config hash 검사는 loader의 `_name_or_path`와 명시적 dtype 적용으로 실패했다.
  경로 메타데이터만 제외하고 factory에 loader dtype을 명시해 해결했다.
  dtype/window/patcher 등 실행 설정은 fingerprint에 남긴다. v1/v2 실패 보고서도 보존했다.
- **미실행**: 실제 backward/optimizer/DDP 및 A100 메모리·시간·전체 평가.
  `submit_blt_hf.sh smoke`는 작은 BLT backward/재개 검사 후 실제 B 긴 train 입력을
  2 update 처리한다. overfit도 별도 진단이며 본 평가 checkpoint로 사용하지 못한다.
  코드 준비 완료를 A100 학습 검증이나 GEC 성능 확인으로 해석하지 않는다.

## 2026-09-16 — 가중치·attention mask 검사 완료 (현재 기준)

사용자 지시에 따라 `plan/P1.md`와 `CLAUDE.md`에서 전체 수치 동질성 게이트를
제거했다. 원본 runtime 재구축·독립 entropy forward·계층 logits parity·C 비교는
필수 작업이 아니다. 연구 주장은 고정된 Meta 가중치의 HF 변환 및 공개한 구현
조건에서의 GEC 성능이다. 과거 `pending/confirmed/not_confirmed`는 당시 정책의
이력이며 아래 과거 절의 동질성 대기·결과 폐기 규칙은 현재 적용하지 않는다.

- 새 보고서: `blt_hf_checks/results/p1_masks_20260916.json`.
  `weight_checks=passed`, `attention_mask_checks=passed`,
  `training_checks=not_run`, `evaluation_checks=not_run`.
- 가중치: 원본 515개→B 510개 전수 exact 값·dtype·shape·coverage 검사 통과
  보고서를 재사용했다. 이번 실행에서 B의 모든 출력 파일 SHA256을 재대조하고
  실제 patched 모델 strict load도 확인했다. 원본 텐서 비교를 다시 수행한 것은 아니다.
- 실제 B의 자동 패칭 6개 fixture에서 각각 **59개 attention 모듈** 검사:
  entropy self 14, encoder self 1/cross 1, global self 25, decoder self 9/cross 9.
  EOS/연속 EOS/행별 경계가 다른 2행 배치, 길이 511/512/513/1380/2048 포함.
- 강제 가상·0길이 patch fixture는 entropy를 우회하므로 **45개 모듈**을 검사했다.
  모든 계층의 전체 additive mask가 독립 규칙과 일치했다. 빈 encoder patch가
  모든 byte를 읽는다고 보수적으로 가정해도 활성 decoder 예측으로 미래 byte가
  전달되지 않음을 patch 연결 검사로 확인했다. 원본 forward 수치 비교는 아니다.
- 허용 범위: `attention_mode=osc`, eager, bf16, no-cache, 무패딩.
  `osc|lre`는 기존 코드 호환용 실행 식별자이며 원본/legacy 동등성 인증이 아니다.
  EOS self-attention 검사만으로 hash/pooling을 포함한 전체 문서 격리를 주장하지
  않는다. GEC 샘플 packing은 하지 않으며 padding/cache 지원은 별도 검사 후 추가한다.
- itcerdo 전체 단위 테스트 **54개 통과**, backward/optimizer 실행 없음.
  GPU peak allocated **11,037,369,344 bytes** (약 11.04GB; 검사 tensor 포함).
  tmux `phdq3-p1:masks-0916` 작업은 exit_code=0으로 완료했고 shell 대기 상태다.
- neuron 접속·전송·제출은 하지 않았다. 다음 단계는 neuron 사용자 실행용
  학습 루프·메모리/backward smoke·checkpoint 및 생성/평가 경로 구현이다.
  학습·평가 성능 검증이나 P1 전체 완료로 확대 해석하지 않는다.

## 2026-09-15 — 정적 검증·GPU 구동·OSC 후보 구현 (당시 기록)

현재 단계: B 가중치 전수 검증 및 실제 GPU 구동 완료. **전체 동질성 pending**,
neuron 학습 미실행. 이전 절의 미완료 목록 중 이번에 완료한 범위는 다음과 같다.

- `weights_p1_validation_20260915.json`: 원본 main+선택 entropy 515개 텐서 →
  B 510개. hash embedding 6개를 한 텐서로 합친 차이이며, 전수 값/dtype 동일,
  누락·추가·충돌 0건. HF meta 모델 510개 키·shape도 일치. 검사는 converter를
  import하지 않고 독립 경로 변환 및 fused row slice 비교를 수행했다.
  대응표는 `manifests/mapping_p1_validation_20260915.json`에 보존했다.
- 실제 eager/SDPA `from_pretrained`에서 missing/unexpected/mismatched/error 모두 0.
  파라미터 수는 **4,633,391,872**이며 hash embedding 포함 수치다. bf16 가중치 약
  9.27GB. GPU peak allocated 약 9.28GB. 한 GEC 예제의 HF loss와 수동 shift 1회
  cross entropy가 정확히 일치했다. backward/optimizer는 실행하지 않았다.
- eager/SDPA의 영어 greedy IDs는 불일치, 한국어 사례는 일치. **기준 backend는
  eager로 결정**. SDPA 동질성은 통과 처리하지 않는다. 64-byte 생성 한도에서
  한국어 마지막 글자의 UTF-8 바이트가 잘린 사례가 있어 `valid_utf8=false`로 남겼다.
  모델 구동 성공과 출력 UTF-8/교정 품질의 성공을 구분한다.

### OSC 후보 변경과 검증 규칙

설치된 transformers 5.16.1의 사본을 `blt_hf/patched/modeling_blt.py`로 두었다.
site-packages는 변경하지 않았다. 원본 대비 전체 diff는
`blt_hf_checks/manifests/modeling_blt_osc.diff`이며 재현 명령은 COMMANDS.md에 있다.
`blt_hf/model.py`가 attention_mode와 backend를 명시해 로드하고 원본 B hash를 확인한다.

1. EOS는 자신이 닫는 구간에 속한다. `segment(i)=Σ[j<i](token[j]==EOS)`.
   local/entropy 허용 조건은 `k≤q`, `q-k<512`, `segment(q)==segment(k)`의 교집합이다.
   global은 EOS를 포함한 patch에서 구간을 닫고 같은 구간의 과거 patch만 허용한다.
   이 정의는 pinned Meta `model/utils.py`, `model/blt.py`에 근거한다.
2. 로컬 encoder/decoder 및 entropy 모든 layer의 실제 전달 마스크를 hook으로 검사했다.
   작은 인공 입력에서 미래/window/EOS/batch 분리의 독립 scalar 정의와 일치한다.
   global EOS patch 분리도 검사했다. 무패딩 동일길이 batch는 축소 모델에서 단건과
   logits 허용오차 내 일치했다. **padding·cache 입력은 OSC 경로에서 명시적으로 거부**한다.
3. 고정 패치 경계·512 미만·중간 EOS 없음 조건에서 OSC와 full-causal logits가
   정확히 일치한다(축소 fp32 모델). 수정 사본의 lre 경로와 무수정 HF도 정확히 일치한다.
   이는 통제된 mask/사본 검사이며, 원본 스택 전체 forward parity의 증거는 아니다.
4. 원본 main은 `include_next_token=True`다. 패치 시작점은 `{0,1}` 및
   `i>0`에서 `H_i>threshold`인 `i+1`; 마지막 끝점은 `S+1`이다.
   HF 기본 경로의 끝점 `S`를 OSC 후보에서 `S+1`로 보완했다.
   `blt_hf/patching.py` tensor 구현을 독립 `plain_patcher.py` scalar 구현과 비교한다.
5. 원본 entropy 산술은 같은 dtype에서 `logp=log_softmax(logits)`,
   `H=-sum(exp(logp)*logp)`다. HF Categorical 경로와 bf16 연산 결과가 달라
   OSC에 원본 순서를 적용했다. `bytelatent/transformer.py`는 logits를 반환할 때
   fp32로 올리지 않는다. 원본 소스와 보충 hash는 `entropy_source_review.json`에 있다.
6. **검증 규칙 재검토**: 원본 tensor–Python scalar 비교는 threshold도 tensor dtype으로
   변환한다. 설정값 1.335442066192627의 bf16 유효값은 1.3359375다.
   최초 이상적인 실수 비교에서는 20개 중 4개 경계가 불일치했다(v4 실패 보고서 보존).
   이를 tolerance로 통과시키지 않고, scalar 기준에 유효 threshold를 명시했다.
   v5는 실제 threshold, 유효 threshold, 실수 기준 경계와의 차이를 모두 보존하며
   원본 dtype 비교 기준에서 **20/20 일치**한다. 이 변경은 threshold 근처의 동작을
   바꾸는 runtime 수정이 아니라 source 비교 의미를 reference에 반영한 것이다.

`p1_osc_v5_20260915.json`: 실제 B에서 71·1,817 token forward/finite loss 및 64-token
생성 정상. peak allocated 9,848,814,592 bytes. 패치 보고서의 scope는 **동일 logits의
entropy 산술+경계**이며 원본 entropy 신경망 forward 동질성을 포함하지 않는다.
원격 전체 49개 테스트 통과 후 추가 동일길이 batch 테스트를 포함한 모델 4개 테스트도
통과했다(현재 테스트 항목 합계 50). 로컬은 50개 중 34개 통과, HF/torch 필요 16개 skip.

남은 실행 준비: 실제 B의 패딩/다양한 길이 생성 정책, neuron backward·학습 루프 및
메모리 smoke. 동질성 검증은 entropy 독립 forward, component parity, legacy provenance,
C 비교, A100 결과를 계속 검토한다. **전체 동질성 검증 완료를 학습 시작 조건으로
추가하지 않는다.** 현 구동 보고서만으로 학습 시작 가능 또는 전체 P1 완료를 선언하지 않는다.

## 2026-09-15 — itcerdo 실행 및 이동 전 인계 (당시 기록)

아래 로컬 착수 기록 이후 `ssh.md`와 프로젝트 HF 토큰을 받아 원격 작업을 진행했다.
작업 범위는 `/home/itcmaster/projects/phdq3`이며 neuron에는 접속하지 않았다.

- itcerdo: Python 3.11.16, torch 2.11.0+cu128, transformers 5.16.1,
  RTX 5090 sm_120, CUDA matmul 정상, xFormers 없음. 실제 pip freeze를
  `requirements.lock.txt`에 보존했다. 환경·데이터 보고서는
  `blt_hf_checks/results/*itcerdo_20260915_bootstrap.json`에 있다.
- 원본 main revision `8134b32f0b1d25d1248c30e8c7bdfd442d3bb380`,
  entropy revision `f2aae511e44e2086b1204bc4ddec6ac6c9651332` 다운로드 완료.
  토큰은 원격 프로젝트의 `artifacts/hf_home/.hf_access`에서만 읽었으며
  로컬로 전송하거나 로그에 출력하지 않았다.
- 원본 entropy 두 파일은 직렬화 hash가 다르지만 **129개 텐서의 key/shape/dtype/값이
  전부 일치**했다. B에는 별도 `facebook/blt-entropy`의 고정 snapshot을 사용했다.
  과거 legacy 실행이 실제 내려받은 revision은 아직 회수되지 않아 같다고 단정하지 않는다.
- vendored converter upstream `2cba19507be799b7bef247ca6c1c4708bf881b5b`를
  수정해 revision·dtype·tokenizer·RoPE·정규화 설정·원본 attention 설정을 보존했다.
  `vendor.diff`가 수정 기록이다. 실제 HF 환경에서 **35개 단위 테스트 통과**.
- CPU 변환 완료: `artifacts/converted/blt-1b-hf-own` (itcerdo),
  510개 텐서, 전부 bf16. 원본 LICENSE 포함. 변환 보고서는
  `blt_hf_checks/manifests/conversion_B_20260915.json`.
- **상태는 converted_unvalidated / pending**. 모델 전 텐서 A↔B 검증,
  strict model load, 실제 GPU forward/generation, OSC 구현, 전체 동질성 판정은 남아 있다.
  원본 `original_spec` 설정을 저장한 것만으로 HF가 해당 마스크를 실행하는 것은 아니다.
- 원본 entropy는 local_block_causal, window=512, EOS=2다. local encoder/decoder도
  window=512이며 global은 block_causal이다. HF 설치 코드에는 단순 causal 경로가
  있으므로 OSC 구현 시 global EOS 격리도 포함해 검토한다. legacy entropy dump를
  정답으로 취급하지 않는다. 코드 사본과 출처는 `vendor/model_sources/` 및
  `manifests/model_source_review.json`에 보존했다.
- 사용자 이동 요청으로 **tmux 세션 `phdq3-p1`**을 만들었다. 첫 실행은 기존 SSH
  다운로드와 완료 시점이 겹쳐 동일 manifest 저장을 거부하고 종료했다. 원본 보고서를
  보존하고 재시작하여 비교·변환이 **2026-09-15 03:41:58 UTC에 exit_code=0**으로 끝났다.
  현재 세션은 shell 대기 상태이며 추가 학습/검증 job이 실행 중인 상태가 아니다.
  로그·status는 원격 `artifacts/logs/p1_artifacts_20260915.{log,status}`에 있다.

재개 순서: B 전 텐서 정적 검증과 strict load → OSC 후보 구동 준비 → 사용자 실행용
neuron 학습 코드. 학습은 동질성 검증 완료와 무관하게 진행하며, 검증 트랙은 별도로
기준·fixture·허용오차를 재검토한다. 동질성 미확인 시 후속 실험을 새로 계획한다.

## 2026-09-15 — 로컬 기반 구현

이 절은 최초 착수 시점의 기록이며 현재 상태는 위 최신 기록을 따른다.

현재 단계: 실행 준비 G 진행 중. 본 학습·동질성 검증 완료 상태가 아니다.

- 사용자: itcerdo/neuron의 환경 설치 완료. 원격 버전·GPU 구동은 아직 확인하지 않음.
- 사용자 지시로 에이전트는 neuron에 접속·파일 전송·작업 제출하지 않음.
  neuron용 코드·명령은 로컬에서 준비하고 사용자가 실행한다.
- itcerdo는 사용자가 제공할 접속정보·디렉터리 구조 파일을 기다린다.
- 초기 SSH 접근은 로컬 sandbox에서 실패했고, itcerdo 재시도 승인도 거절됨.
  그 이후 원격 접속하지 않았으며 원격 파일·환경·job을 변경하지 않았다.

구현한 항목:

1. `data_adapter.py`: strict TSV reader, canonical 9개 경로, 본문 자동 특수토큰
   비활성화, BOS/EOS 명시 삽입, EOS 포함 unshifted labels, 무절단·생성 예산 검사.
   현재 vocabulary 계약은 BOS=1, EOS=2, byte+4이며 실제 tokenizer 반환 IDs가
   다르면 실패한다. HF tokenizer의 정규화나 특수문자 해석을 조용히 허용하지 않는다.
2. 검증 전 collator는 동일길이 무패딩 batch만 허용한다. 길이가 섞인 batch는
   오류를 내며, batch_size=1 경로가 가능하다. pad-aware 모델 경로는 아직 미구현.
3. `manifest.py`: P1 D-1 identity 필수 필드, OSC/LRE·config·tokenizer·dataset
   hash 비교, 전체 split 논리 순서 보존, gold M2 source 정합성 검사,
   모델 artifact SHA256 검사, 기존 결과를 덮어쓰지 않는 원자적 JSON 저장.
4. `check_env.py`: 기존 환경을 읽고 버전·xFormers 부재·GPU capability·작은 CUDA
   matmul을 점검한다. 설치/업데이트/optimizer/backward는 수행하지 않는다.
   `--cpu-only`는 정보성 결과이며 GPU 환경 통과로 해석하지 않는다.
5. `analyze_data_lengths.py`: torch 없이 제공 TSV 9개·val/test M2 6개를 읽어
   길이·정합성·파일 hash를 기록한다. 모델/토크나이저 실행 결과와 구분한다.

로컬 확인:

- Python 3.11.2 표준 라이브러리로 unittest 25개 통과.
- tokenizer 테스트는 byte+4 테스트 대역을 사용한다. 실제 HF tokenizer parity,
  torch collate 실행, CUDA 모델 forward/backward 통과를 의미하지 않는다.
- `results/data_lengths_2026-09-15.json`: 201,534행, 최대 입력 1,380,
  최대 prompt 694, 최대 target 685, 1,024 초과 9건, 2,048 초과 0건.
  val/test의 TSV–M2 source가 공백 토큰화 기준 전부 일치.
- `results/env_mac_2026-09-15.json`: 로컬에는 torch/transformers 미설치.
  기존 원격 환경을 재설치하거나 로컬에 CUDA wheel을 설치하지 않았다.

다음 구현·실행:

1. itcerdo 정보 파일에서 프로젝트 경로, 환경 Python, 저장공간, 모델/cache 경로 확인.
2. 준비된 환경에서 check_env 실행, 실제 설치 버전 기반 lock 생성.
3. 원본 main/entropy snapshot 및 legacy provenance 확인, converter upstream SHA
   고정·vendoring·revision/dtype/tokenizer 수정, B 변환. 아직 converter/B 없음.
4. B의 실제 tokenizer와 모델을 로드해 adapter 및 구동 확인. 고정한 OSC 모델 구현과
   학습 루프를 준비해 사용자 실행용 neuron 명령으로 연결.
5. 파인튜닝은 구동 준비 이후 검증 완료와 무관하게 진행한다. 별도 트랙에서
   검증 규칙을 재검토·실행하고, 동질성 미확인 시 후속 실험을 새로 계획한다.

attn backend, mask 구현, entropy 기준, cache 동작은 아직 결정·검증되지 않았다.
출처를 확인하지 않은 converter commit이나 원격 환경 lock을 임의 생성하지 않는다.
