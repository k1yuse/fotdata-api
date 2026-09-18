# FotData — 프로젝트 가이드 (CLAUDE.md)

이 문서는 Claude Code가 이 저장소에서 작업할 때 항상 참고해야 하는 컨텍스트다.

## 1. 프로젝트 개요

**FotData**는 유승(GitHub: k1yuse)이 개발 중인 AI 기반 축구 경기 예측·분석 플랫폼이다. 상업적 출시를 목표로 한다.

- **장기 비전**: 유소년 축구 선수가 경기 영상을 업로드하면 AI가 선수/공을 트래킹해 히트맵, 패스/슈팅 통계, 선수 평점, 포지션 추천을 제공하는 앱
- **대상 리그**: EPL, La Liga, Bundesliga, Serie A, Ligue 1 (+ UCL 챔피언스리그)
- **현재 단계 (Stage 1)**: 데이터 수집 + ML 경기 예측 + 순위표/토너먼트/선수 스탯 웹 서비스 — 완료 및 운영 중
- **다음 단계 (Stage 2)**: YOLOv8 + ByteTrack 기반 컴퓨터 비전 (영상에서 선수/공 인식)
- **최종 단계 (Stage 3)**: 영상 업로드 → 분석 → 유소년 선수 리포트 서비스화

**대화 스타일**: 한국어 반말로 간결하게 소통. 새 세션에서도 이 톤을 유지할 것.

## 2. 라이브 서비스 & 저장소

| 항목 | 값 |
|---|---|
| 백엔드 API | https://fotdata-api.onrender.com (FastAPI, Render 무료 플랜) |
| 프론트엔드 | https://fotdata-api.vercel.app (Vercel) |
| GitHub | https://github.com/k1yuse/fotdata-api |
| 로컬 경로 | `~/fotdata-api` (맥북 프로 M5) |
| conda 환경 | `fotdata` (Python 3.10) |

**보안 사고 이력 (2026-09-19에 발견·조치 완료)**: 아래 두 건이 발견되어 즉시 조치했다. 앞으로 비슷한 실수를 반복하지 않기 위해 기록해둔다.
1. `git remote -v`의 origin URL에 GitHub Personal Access Token이 평문으로 박혀 있었음(`https://k1yuse:ghp_...@github.com/...`) — 로컬 `.git/config`에만 있던 것이라 레포 자체가 노출된 건 아니었지만, 토큰을 폐기하고 새로 발급받아 remote URL을 교체했다.
2. **(더 심각) `FotData_01.ipynb`에 football-data.org API 키가 하드코딩된 채 GitHub에 커밋되어 있었고, 레포가 public이라 실제로 전 세계에 노출된 상태였음.** 키를 재발급(무효화)하고, 노트북 코드는 `os.environ["FOOTBALL_API_KEY"]`로 환경변수에서 읽도록 수정했다. 같은 키가 들어있던 `.ipynb_checkpoints/FotData_01-checkpoint.ipynb`도 삭제하고, `.gitignore`를 추가해 체크포인트/`__pycache__`/`.env`가 다시 커밋되지 않게 했다.

**앞으로 지킬 규칙**:
- API 키/토큰은 절대 코드나 노트북에 하드코딩하지 말고 `os.environ.get(...)`으로만 읽을 것
- 새 노트북 셀을 추가할 때도 이 규칙 유지 — 노트북은 실수로 키를 박아넣기 가장 쉬운 곳
- `git remote -v` 결과에 자격증명이 보이면 즉시 정리 (credential helper로 이전)

## 3. 기술 스택

- **백엔드**: FastAPI, `joblib`로 모델 로드, CORS 전체 허용(`allow_origins=["*"]`)
- **ML**: scikit-learn (Logistic Regression — 실제 서빙용), RandomForest·XGBoost(학습·정확도 비교용, 서빙엔 LR만 사용), StandardScaler
- **프론트엔드**: Vanilla JS + HTML/CSS (프레임워크 없음), 다크 테마(#0d1117 배경, #58a6ff 포인트)
- **자동화**: GitHub Actions (`update_data.py`, 매일 UTC 18:00 = KST 03:00 실행)
- **데이터 소스**:
  - football-data.org (`FOOTBALL_API_KEY`) — 경기 결과/순위. 무료 플랜은 최근 3시즌만 제공
  - API-Football (`API_FOOTBALL_KEY`) — 선수 스탯(득점왕/도움왕). 무료 플랜은 **EPL만, 2024 시즌 고정**
- **로컬 개발환경**: Miniconda(arm64), Jupyter Notebook(`FotData_01.ipynb`), VS Code

## 4. 파일 구조

```
fotdata-api/
├── main.py                 # FastAPI 백엔드 (515줄) — 전체 API 엔드포인트
├── update_data.py          # 데이터 수집 + 피처 생성 + 모델 학습 + 파생 산출물 생성 (826줄)
├── FotData.html            # 메인 웹앱 (예측/순위/UCL 토너먼트/선수/우승예측)
├── landing.html             # 랜딩 페이지 원본
├── index.html               # 루트(`/`)에서 서빙되는 파일 — landing.html의 사본 (아래 5.1 참고)
├── FotData_01.ipynb        # 초기 개발용 주피터 노트북 — 지금은 update_data.py가 대체, 참고용
├── manifest.json            # PWA 매니페스트
├── requirements.txt          # 백엔드(Render) 의존성
├── .github/workflows/update_data.yml   # 매일 자동 업데이트 GitHub Action
└── fotdata_model/            # 모델/데이터 산출물 (전부 update_data.py가 생성·갱신)
    ├── all_matches.csv          # 24-25~26-27 전 경기 원본 (H2H, 폼, 순위 계산의 기반)
    ├── team_stats.csv           # 3시즌 블렌딩 + prestige 반영된 "현재 팀 전력" (예측용)
    ├── features.csv              # 학습용 피처 (build_features 결과, 참고용 캐시)
    ├── logistic_regression.pkl   # 실제 /predict 서빙 모델
    ├── random_forest.pkl / xgboost.pkl / label_encoder.pkl   # 비교용, 서빙에는 미사용
    ├── scaler.pkl                 # LR 입력 스케일러
    ├── team_logos.json / league_logos.json
    ├── accuracy.json             # 모델 정확도 (3개 모델 비교 + best)
    ├── ucl_tournament.json       # UCL 브래킷 데이터 (PO→R16→QF→SF→Final)
    ├── champion_predictions.json # 5대리그 우승/TOP4/강등 확률 (몬테카를로 1000회)
    └── players.json               # 득점왕/도움왕 (현재 EPL만)
```

### 4.1 `index.html` = `landing.html`의 사본 — 반드시 동기화

Vercel 루트(`/`)는 `index.html`을 서빙한다. `landing.html`을 수정하면 **반드시**:
```bash
cp landing.html index.html
```
을 실행한 뒤 커밋해야 실제 배포에 반영된다. 잊으면 "분명 고쳤는데 안 바뀐다"는 증상이 재발한다(과거에 실제로 이 문제로 여러 번 헤맸음).

## 5. 예측 모델 핵심 로직

### 5.1 3시즌 블렌딩 (`calculate_blended_stats`, update_data.py)
26-27 시즌 진행도에 따라 24-25/25-26/26-27 3개 시즌의 승률·공격력·수비력을 가중 평균한다.

| 26-27 경기 수 | 24-25 | 25-26 | 26-27 |
|---|---|---|---|
| 0~9경기 | 40% | 40% | 20% |
| 10~17경기 | 30% | 30% | 30% |
| 18경기 이상 | 30% | 30% | 40% |

### 5.2 Prestige(체급) 보정
- 24-25 + 25-26 두 시즌만으로(26-27 제외, 초반 변동성 배제) 계산: `(팀 승률 − 전체 평균 승률) × 500`
- 승격팀 등 과거 데이터 없는 팀은 prestige = 0
- UCL 경기는 계산에서 제외(리그 팀 수를 왜곡시키므로)

### 5.3 경기 예측 (`/predict`, main.py)
- ELO를 승률 기반으로 재구성: `1500 + (win_rate − 0.33) × 1000`
- 홈 어드밴티지(기본 70)에 "전력차가 클수록 감쇠"하는 팩터 적용 (`max(0.4, 1 − elo_gap/800)`) — 약팀 홈 vs 강팀 원정에서 홈 승률이 비현실적으로 치솟는 것을 막기 위한 장치
- H2H 최근 10경기 홈팀 승률도 피처로 사용
- 최종적으로 LogisticRegression 모델(`logistic_regression.pkl`)로 H/D/A 확률 산출

### 5.4 우승 예측 (`simulate_season`, update_data.py)
- 몬테카를로 시뮬레이션 1000회, 팀별 ELO에 매 경기 가우시안 노이즈(σ=50) 추가해 변동성 반영
- 결과: 우승 확률 / TOP4 확률 / 강등 확률

## 6. API 엔드포인트 (main.py)

| Method | Path | 설명 |
|---|---|---|
| GET | `/` | 헬스체크 |
| GET | `/teams` | 전체 팀 목록 |
| POST | `/predict` | 경기 결과 예측 (body: home_team, away_team) |
| GET | `/team/{team_name}` | 팀 스탯 |
| GET | `/logos` | 팀 로고 URL 맵 |
| GET | `/standings/{league_code}?season=current\|previous` | 리그 순위표 (실시간 계산) |
| GET | `/ucl/tournament` | UCL 토너먼트 브래킷 |
| GET | `/h2h?home_team=&away_team=&limit=` | 상대전적 |
| GET | `/form/{team_name}?n=` | 최근 N경기 폼 |
| GET | `/accuracy` | 모델 정확도 |
| GET | `/players/topscorers/{league_code}`, `/players/topassists/{league_code}` | 득점왕/도움왕 (현재 PL만 데이터 있음) |
| GET | `/predict/champion/{league_code}` | 리그 우승 예측 |

리그 코드: `PL`(EPL), `PD`(라리가), `BL1`(분데스리가), `SA`(세리에A), `FL1`(리그앙), `CL`(UCL)

## 7. 배포 파이프라인

1. **백엔드**: `git push` → Render가 `main` 브랜치 자동 재배포 (Render 무료 플랜은 Shell 접근 불가 — 로그 확인만 가능)
2. **프론트엔드**: `git push` → Vercel 자동 재배포
3. **일일 데이터 갱신**: GitHub Actions(`update_data.yml`, 매일 UTC 18:00)가 `update_data.py` 실행 → `fotdata_model/*` 변경분을 자동 커밋·푸시 (`자동 데이터 업데이트 YYYY-MM-DD`)
   - ⚠️ 이 워크플로우에는 `FOOTBALL_API_KEY`만 GitHub Secrets로 주입됨. `API_FOOTBALL_KEY`는 설정돼 있지 않아서, GitHub Actions 실행에서는 `fetch_top_scorers()`가 조용히 스킵된다 — 선수 데이터는 현재 **로컬에서 수동 실행할 때만** 갱신 가능. 자동화하려면 `API_FOOTBALL_KEY`도 GitHub Secrets에 추가해야 함.
   - Render 무료 플랜은 아웃바운드 API 호출이 막혀 있어서, 모든 외부 데이터(football-data.org, API-Football)는 반드시 GitHub Actions/로컬에서 미리 fetch해 JSON/CSV 캐시로 만든 뒤 서빙해야 한다. Render 서버가 직접 외부 API를 호출하는 코드는 작동하지 않는다.
   - `API_FOOTBALL_KEY`를 GitHub Secrets에 넣지 않은 것은 의도적인 선택이다: API-Football 무료 플랜은 2024 시즌 데이터만 제공해서, 자동화해도 최신 시즌 선수 스탯은 어차피 못 가져온다. 유료 플랜(Pro, $19/월)으로 업그레이드하기 전까지는 선수 데이터는 로컬에서 수동으로만 `python update_data.py`를 돌려 갱신한다.

## 8. 환경 세팅 (맥북 기준)

```bash
conda activate fotdata          # 반드시 활성화 확인 (프롬프트에 (fotdata) 표시)
which python                     # .../envs/fotdata/bin/python 인지 확인
cd ~/fotdata-api
export FOOTBALL_API_KEY=...
export API_FOOTBALL_KEY=...      # 선수 데이터 갱신 시에만 필요
python update_data.py
```
- `python3`는 시스템 파이썬을 가리킬 수 있으니 `python`(conda env 안)을 쓸 것
- VS Code 터미널을 새로 열면 conda 환경이 풀려 있을 수 있으니 매번 `conda activate fotdata` 확인

## 9. Git 작업 시 주의사항

- **rebase 충돌**: GitHub Actions가 `fotdata_model/`을 자동 커밋하므로 로컬 작업과 자주 충돌한다.
  ```bash
  git pull --rebase
  git checkout --theirs fotdata_model/all_matches.csv   # 데이터 파일은 최신(원격) 것을 채택
  git checkout --ours  <직접 수정한 코드 파일>            # 코드는 로컬 것을 채택
  git add fotdata_model/ <해당 파일>
  git rebase --continue
  git push
  ```
  **`--ours`/`--theirs`를 반대로 쓰면 최신 데이터가 스테일 데이터로 덮어써지는 회귀가 발생한 전례가 있음** — 실수했다면 `git fetch origin` + `git checkout origin/main`으로 복구.
- 빈 커밋으로 Render 강제 재배포: `git commit --allow-empty -m "trigger redeploy"`
- `landing.html` 수정 후에는 항상 `cp landing.html index.html` 후 커밋 (4.1 참고)

## 10. 알려진 이슈 / 향후 정리 과제

- ~~UCL 토너먼트 PO/R16 매칭 순서 하드코딩~~ → 2026-09-19에 `update_data.py`의 `reconstruct_bracket_order()`로 리팩토링 완료. 상위 라운드(R16/QF/SF/Final)가 실제로 확정되면 그 대진의 팀 실명으로 하위 라운드 어느 매치에서 이겼는지 역추적해서 좌우 순서를 자동 복원함(팀 이름 완전일치 기반, substring 매칭 버그도 같이 제거됨). 아직 다음 라운드가 안 열린 최전선 라운드(예: R16 발표 전의 PO)만 API 응답 순서를 그대로 씀 — 이건 UEFA의 실제 추첨 슬롯 정보가 API에 없어서 발생하는 구조적 한계로, 다음 라운드가 열리는 순간 자동으로 소급 재정렬됨.
- 모델 로드 시 구버전 scikit-learn으로 저장된 pkl과의 `multi_class` 속성 호환성 문제가 있었음 → `main.py`에 `if not hasattr(lr_model, 'multi_class')` 패치로 해결한 상태. sklearn 버전을 올릴 때 이 부분 재확인.
- `FotData_01.ipynb`는 초기 개발 단계(Stage 0~1 초반)의 유물로, 현재는 `update_data.py`가 전체 파이프라인(수집→피처→학습→저장)을 대체함. 노트북은 과거 히스토리 참고용이며 실행 경로가 아님.
- 시즌 종료 배너(`FotData.html`/`landing.html`에 HTML 주석으로 비활성화됨)는 27-28 시즌 전환 시점에 재활성화 예정.

## 11. 시즌 전환 체크리스트 (매년 반복 작업)

새 시즌이 시작될 때마다 아래를 전부 수동 확인:
- [ ] football-data.org / API-Football API 키 유효성 (로컬 + GitHub Secrets 양쪽)
- [ ] `update_data.py`의 시즌 배열 (`[2024, 2025, 2026]` 형태) 갱신
- [ ] `team_stats.csv` / `/standings` 엔드포인트의 시즌 컷오프 날짜 (여러 위치에 흩어져 있음, main.py와 update_data.py 둘 다 확인)
- [ ] `FotData.html`의 `LEAGUE_DATA` 하드코딩 팀 리스트 — 승강팀 반영
- [ ] 신규 승격팀 로고 (`update_team_logos()`가 자동 시도하지만 실패 시 수동 추가)
- [ ] 시즌 종료 배너 재활성화 여부

## 12. 다음 계획 / 수익화 로드맵

- **기능**: 선수 스탯 기능 확장(현재 EPL 무료 플랜 한정 → API-Football Pro 결제 시 전체 리그/시즌 확장 가능), README 작성
- **Stage 2**: YOLOv8 + ByteTrack 컴퓨터 비전 파이프라인 (맥북 M5는 MPS 가속 지원, `device='mps'`)
- **수익화 단기**: 도네이션 버튼(Ko-fi), Google AdSense
- **수익화 중기**: Freemium 구독(Pro 티어), RapidAPI 예측 API 판매
- **수익화 장기**: 유소년/아마추어팀 대상 SaaS 영상 분석, B2B 대시보드
- UptimeRobot으로 Render 무료 플랜 슬립 방지 설정 필요
