# 주요 명령어


## 깃허브 연동 관리

```bash
# 깃허브 초기화
git init -b main
git remote add origin https://github.com/esoterikosQ/phdq3.git
git status --short --branch
git remote -v
git check-ignore -v blt_gec/train.py
```

```bash
git status --short --branch # 현재 브랜치와 변경 파일 확인
git switch main
git pull --ff-only origin main # 이력이 갈라지면 멈추고 원인 확인
```

```bash
# 작업 종료시
git status --short # 커밋할 파일 확인
git add -A -- blt_hf scripts runtimelog outputs/blt_hf_eval # 실제 수정한 경로만 선택
git diff --cached --name-only
git commit -m "수정내용 입력"
git push origin main
```
