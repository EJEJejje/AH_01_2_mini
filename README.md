# EJEJejje-AH_01_1-Group

* 이 저장소는 2조 프로젝트 저장소입니다
# 메인 브런치 푸쉬는 1명이상의 동의가 필요합니다. 테스트 완료후 푸쉬해주세요!
# test 브런치는 작동단계를 확인하는 곳 입니다. 확인후 PR올려주세요.
# fix 브런치는 코드 수정 후 업로드를 하는 곳 입니다.
# save 브런치는 프로젝트 기간동안 중간저장 파일을 올리는 곳 입니다.

| 상황             | 어떤 브랜치에서 시작?          | 새 브랜치 이름 예시       | PR은 어디로?  |
| -------------- | --------------------- | ----------------- | --------- |
| 새 기능/모델/EDA 추가 | `test`                | `feat/eda-본인이름`   | `test`    |
| 버그 수정          | `test` 또는 `main`      | `fix/bug-설명`      | `test`    |
| 그냥 백업하고 싶음     | `save` 또는 자기 `feat/…` | 필요시 `save`로 merge | 보통 PR 안 함 |





---
SSH 키 생성 및 GitHub 등록 (처음 한 번만)
SSH 키 생성
ssh-keygen -t ed25519 -C "본인_GitHub_이메일@example.com"
계속 Enter만 눌러도 됨

공개키 복사 (macOS):
pbcopy < ~/.ssh/id_ed25519.pub

Windows(Git Bash):
cat ~/.ssh/id_ed25519.pub
출력된 내용 전체 복사

---
GitHub 웹사이트에서:
오른쪽 위 프로필 → Settings
SSH and GPG keys → New SSH key
Title: 적당히 (예: MyLaptop)
Key: 방금 복사한 내용 붙여넣기 → Add SSH key

연결 테스트:
ssh -T git@github.com

"successfully authenticated" 문구 나오면 성공.

---
우리 팀 레포 클론(clone)
cd #원하는_폴더
git clone git@github.com:EJEJejje/EJEJejje-AH_01_1-Group.git
cd EJEJejje-AH_01_1-Group

기본 브랜치 확인
git branch -a
# main, test, fix, save 등이 보이는지 확인

---

작업 시작할 때 (항상)

```bash
cd EJEJejje-AH_01_1-Group

# test 브랜치로 이동 후 최신으로 갱신
git checkout test
git pull origin test

---
오늘 할 작업용 브랜치 만들기

예: 오늘 할 일 – “EDA 노트북 만들기”

git checkout -b feat/eda-본인이름

---
파일 수정 후 커밋/푸시

예: notebooks/01_eda_본인이름.ipynb 만들었다고 가정

git status
git add notebooks/01_eda_본인이름.ipynb
git commit -m "Add EDA notebook by 본인이름"
git push -u origin feat/eda-본인이름

---
GitHub에서 Pull Request 만들기

레포 페이지 → Compare & pull request 버튼 보이면 클릭

base: test
compare: feat/eda-본인이름

제목/내용 적기 (간단하게라도):

예: EDA notebook for basic stats

팀원 한 명을 reviewer로 지정

리뷰어는:

내용/코드 확인

필요하면 코멘트 남기기

OK면 Merge 버튼 눌러 test에 머지


