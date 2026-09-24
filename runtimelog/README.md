# Runtime log status

`slurm-blt-hf-train-909747.*`와 `slurm-blt-hf-train-910019.*`는 폐기된 FP32
parameter/gradient/Adam 정책에서 실행된 실패 로그다. BF16 변환 artifact의 가중치·마스크
검증에는 영향을 주지 않지만, BF16 학습·DDP·메모리 검증 증거로 사용할 수 없다.

BF16 학습 코드를 새 RUN_ID로 다시 실행하고 새 로그와 report를 별도로 보존한다.

learner beam1 생성은 job 912817 → 912933 → 913148 순서로 같은 EVAL_DIR에서 재개했고,
마지막 job이 4,265/4,265건, exit 0으로 완료했다. 앞의 두 exit 75는 저장된 진행 상태를
가진 정상 중단이다. job 913151은 A100/BF16 문제가 아니라 기존 union run manifest와
현재 코드 identity 불일치로 첫 optimizer step 전에 실패했다. 기존 union checkpoint는
보존하되 다음 사이클에서는 새 RUN_ID로 처음부터 학습한다.
