# phdq3

HF Transformers 기반 BLT를 한국어 문법·오탈자 교정에 사용하는 연구 코드입니다.
Meta 사전학습 가중치의 직접 변환, 가중치·attention mask 검사, Neuron 학습 및
전체 split GLEU/M2 평가 경로를 포함합니다. 원본 BLT와의 수치 동등성을 주장하지 않습니다.

## 문서

- [Neuron 학습·평가 실행](blt_hf/NEURON.md)
- [P1 계획](plan/P1.md)
- [구현 및 검증 기록](blt_hf/NOTES.md)
- [외부 코드 출처 및 공개 제외 항목](THIRD_PARTY_NOTICES.md)

## 이 저장소에 없는 파일

`data/` 전체, 가중치·체크포인트, `artifacts/`, `outputs/`, 설치 환경·캐시,
`ssh.md` 및 인증정보는 커밋하지 않습니다. 데이터셋에서 발췌한 fixture·검사 보고서,
Meta 원본 코드 검토 사본, 라이선스 표기가 상충하는 M2 scorer 사본도 제외합니다.
로컬 작업 디렉터리의 파일은 삭제하지 않습니다.

학습에는 사용 권한을 확보한 원본 9개 TSV split과 val/test M2 6개 및 변환본 B가
별도로 필요합니다. 데이터는 수정·재분할하지 않습니다. 생성·M2 평가를 실행하려면
기존 실험과 동일한 M2 파일을 별도로 준비해야 합니다. 복원 대상과 hash는
[외부 코드 안내](THIRD_PARTY_NOTICES.md)에 있습니다. Git clone만으로 준비가 끝나는 것은 아닙니다.

## 검사

```bash
python -m unittest discover -s tests -p 'test_hf_*.py' -v
```

Torch가 없으면 HF 모델 검사를, 데이터·비공개 fixture/M2 사본이 없으면 해당 검사를 명시적으로
skip합니다. skip은 학습·평가 검증 통과를 뜻하지 않습니다. 실제 A100의 backward,
optimizer, DDP, 메모리 검사는 Neuron 안내서의 사용자 실행 smoke로 확인합니다.

기존 검증 JSON의 hash는 당시 파일의 기록입니다. Git 공개 범위를 정리하면서
제외한 파일을 포함할 수 있으며, 해당 JSON을 현재 공개본 전체의 실행 증명으로 해석하지 않습니다.
외부 코드는 각각의 라이선스를 따릅니다. 저장소 전체에 새로운 일괄 라이선스를 부여하지 않습니다.
