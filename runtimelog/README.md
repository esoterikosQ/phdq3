# Runtime log status

`slurm-blt-hf-train-909747.*`와 `slurm-blt-hf-train-910019.*`는 폐기된 FP32
parameter/gradient/Adam 정책에서 실행된 실패 로그다. BF16 변환 artifact의 가중치·마스크
검증에는 영향을 주지 않지만, BF16 학습·DDP·메모리 검증 증거로 사용할 수 없다.

BF16 학습 코드를 새 RUN_ID로 다시 실행하고 새 로그와 report를 별도로 보존한다.
