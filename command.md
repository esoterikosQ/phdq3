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
git status # 최신 상태인지 확인
git pull # 깃허브 최신 상태 동기화
```

```bash
# 작업 종료시
git status # version control 상태 확인
git add .
git commit -m "수정내용 입력"
git push
```

