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
├── FotData.html            # 메인 웹앱 (예측/순위/UCL 토너먼트/선수/우승예측/승부차기 미니게임)
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
    ├── players.json               # 득점왕/도움왕 (현재 EPL만)
    ├── schedule.json              # 5대리그+UCL 26-27 시즌 전체 일정 (완료+예정) — 일정 탭 전용
    ├── team_info.json             # 팀 상세정보(홈구장/창단연도/구단색/감독/스쿼드) — 팀 클릭 정보 패널용
    ├── team_extra.json            # API-Football 스쿼드 사진/등번호 + 이적 기록 (현재 EPL 20팀 중 16팀만, 2026-09-20 추가)
    └── prediction_log.json        # AI 예측 트랙레코드 로그 (2026-09-21 추가, 5.6 참고)
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
- 강등 확률 집계 인원수는 리그별로 다름(`RELEGATION_COUNT`): 18팀 리그(분데스리가/리그앙)는
  하위 2팀만 집계(16위 강등 PO 대상은 제외, FotData.html의 `runSimulation()` releCount와
  동일 기준), 20팀 리그(EPL/라리가/세리에A)는 하위 3팀. 2026-09-19에 5개 리그 전부 하위
  3팀 고정이던 버그를 수정함(순위표 존 표시 수정과 같은 날 발견).

### 5.5 팀 상세정보 (`fetch_team_info`, update_data.py · 2026-09-19 추가)
- football-data.org의 팀 리소스(`/teams/{id}`)를 이용, 무료 플랜(기존 `FOOTBALL_API_KEY`)으로 홈구장(venue)/창단연도/구단색/스쿼드를 가져올 수 있음이 확인됨
- **감독(coach)과 등번호(shirtNumber)는 이 무료 플랜에서 항상 null** — API 자체가 제공을 안 함(신뢰성 문제가 아니라 무료 티어 제약)
- **선수 사진은 이 API에 필드 자체가 없음.** API-Football(별도 키)에는 있지만 그쪽은 EPL·2024 시즌 고정이라 현재 스쿼드와 안 맞음 — 사진 기능은 보류
- 스쿼드는 시즌 중 이적으로 계속 바뀌므로, 로고와 달리 "누락분만" 채우는 게 아니라 매번 전체 팀을 다시 fetch함 (일일 자동 업데이트에 포함됨, `FOOTBALL_API_KEY`만 있으면 되므로 GitHub Actions에서도 정상 동작)

### 5.6 AI 예측 트랙레코드 (`update_prediction_log`, update_data.py · 2026-09-21 추가)
- 예측 신뢰도를 보여주기 위해, 매일 그 시점에 예정된 경기들(향후 10일 내)에 대해 **모델 로직을 재구현하지 않고 그 순간 실제 서빙 중인 라이브 `/predict`를 그대로 호출**해서 `fotdata_model/prediction_log.json`에 미리 스냅샷 기록해둠 — main.py와 별도로 예측 로직을 두 군데서 관리하면 언젠가 어긋나서 "기록된 예측"이 실제로 사용자가 봤던 예측과 달라지는 문제를 원천 차단하기 위한 설계
- `update_data.py` 실행 순서상 모델 재학습 이후 & git push 이전에 실행되므로, 그날 아직 배포 안 된 새 모델이 아니라 "그 시점까지 실제로 서빙 중이던" 모델의 예측이 기록됨 (의도된 동작)
- 경기가 끝나면(schedule.json의 status가 FINISHED) 같은 로그 항목에 실제 결과·적중 여부(`actual`/`actual_score`/`correct`)를 채워넣음
- 결과가 확정된 항목은 최근 500건만 유지, 미확정(예정) 항목은 개수 제한 없이 계속 보관
- 프론트: `/predict` 결과 하단 "AI 모델 정확도 ... · 트랙레코드 보기" 문구 클릭 → 모달로 전체/최근 적중률 + 최근 20경기 예측-결과 비교 표시 (`/predict/track-record`)
- **첫 배포 직후에는 기록이 비어 있는 게 정상** — GitHub Actions가 매일 돌 때마다 그날 예정 경기들의 예측이 쌓이고, 그 경기들이 끝나야 적중 여부가 채워지므로 실제 트랙레코드가 유의미해지기까지 며칠~1주 정도 걸림

### 5.7 승부차기 미니게임 (FotData.html, `#page-shootout` · 2026-09-21 추가)
- 내 팀 선택 → 골키퍼 1명 + 키커 1~5번 순서를 직접 선택 → 상대팀 선택(AI 랜덤 또는 직접 선택) → 코인토스로 선축 결정 → 실제 IFAB 승부차기 규칙(5명 교대, 조기 승부 확정 시 중단, 동점이면 서든데스)으로 진행
- **선수 개별 능력치 데이터가 없음** (API-Football 무료 플랜은 EPL 득점왕/도움왕 TOP10 외엔 스탯 제공 안 함) — 그래서 상대(AI) 키커 순서는 정직하게 **포지션 우선순위(공격수→미드필더→수비수)** 로만 정하고, 혹시 그 선수가 EPL 득점왕 TOP10에 있으면 같은 포지션 내에서 골 수로 한 번 더 정렬하는 정도로 타협함 (5.5/5.6과 같은 맥락: 없는 데이터를 있는 것처럼 꾸미지 않음)
- 선수 아바타: `team_extra.json`에 사진이 있는 팀(EPL 일부)은 실제 사진, 없으면 회색 실루엣 아이콘 + 등번호로 폴백. `<img onerror="...">`에 HTML을 통째로 문자열로 박아 넣으면 따옴표 충돌로 태그가 깨지는 버그가 있었음 → `onerror="kickerAvatarError(this)"`처럼 함수 호출 + `data-num` 속성으로 값만 전달하는 방식으로 수정(실제 버그였고 재발 방지 차 기록)
- 슛/막기 조작은 6분할 존(좌/중/우 × 상/하) 클릭 방식, 슈터와 골키퍼가 각각 고른 존이 다르면 골, 같으면 선방 — 키커 존은 사용자가 고르고 상대 골키퍼(또는 상대 슈터) 존은 매번 무작위

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
| GET | `/schedule/{league_code}` | 리그 전체 시즌 일정 (완료+예정 전부) |
| GET | `/team/info/{team_name}` | 팀 상세정보 (홈구장/창단연도/구단색/감독/스쿼드) |
| GET | `/teams/prestige` | 팀별 prestige 맵 (전역 검색 기본 정렬용, 2026-09-20 추가) |
| GET | `/proxy/logo?url=` | 팀 로고 이미지 프록시 (2026-09-21 추가) — crests.football-data.org/wikimedia는 CORS 헤더가 없어서 프론트 `<canvas>`(예측 결과 공유카드)에 바로 그리면 tainted되어 내보내기가 막힘; 허용된 두 호스트로만 제한해 우리 서버(CORS 전체 허용)를 거쳐 내려줌 |
| GET | `/predict/track-record` | AI 예측 트랙레코드 요약 + 최근 20경기 (2026-09-21 추가, 5.6 참고) |

리그 코드: `PL`(EPL), `PD`(라리가), `BL1`(분데스리가), `SA`(세리에A), `FL1`(리그앙), `CL`(UCL)

## 7. 배포 파이프라인

1. **백엔드**: `git push` → Render가 `main` 브랜치 자동 재배포 (Render 무료 플랜은 Shell 접근 불가 — 로그 확인만 가능)
2. **프론트엔드**: `git push` → Vercel 자동 재배포
3. **일일 데이터 갱신**: GitHub Actions(`update_data.yml`, 매일 UTC 18:00)가 `update_data.py` 실행 → `fotdata_model/*` 변경분을 자동 커밋·푸시 (`자동 데이터 업데이트 YYYY-MM-DD`)
   - ⚠️ 이 워크플로우에는 `FOOTBALL_API_KEY`만 GitHub Secrets로 주입됨. `API_FOOTBALL_KEY`는 설정돼 있지 않아서, GitHub Actions 실행에서는 `fetch_top_scorers()`가 조용히 스킵된다 — 선수 데이터는 현재 **로컬에서 수동 실행할 때만** 갱신 가능. 자동화하려면 `API_FOOTBALL_KEY`도 GitHub Secrets에 추가해야 함.
   - Render 무료 플랜은 아웃바운드 API 호출이 막혀 있어서, 모든 외부 데이터(football-data.org, API-Football)는 반드시 GitHub Actions/로컬에서 미리 fetch해 JSON/CSV 캐시로 만든 뒤 서빙해야 한다. Render 서버가 직접 외부 API를 호출하는 코드는 작동하지 않는다.
   - `API_FOOTBALL_KEY`를 GitHub Secrets에 넣지 않은 것은 의도적인 선택이다: API-Football 무료 플랜은 2024 시즌 데이터만 제공해서, 자동화해도 최신 시즌 선수 스탯은 어차피 못 가져온다. 유료 플랜(Pro, $19/월)으로 업그레이드하기 전까지는 선수 데이터는 로컬에서 수동으로만 `python update_data.py`를 돌려 갱신한다.
   - `update_prediction_log()`(5.6)는 GitHub Actions 러너에서 **Render 배포 API로 직접 HTTP 요청**을 보낸다(로컬/러너 → Render는 인바운드라 문제 없음, Render 무료 플랜의 아웃바운드 제한과는 무관). 그 시점에 Render가 자고 있으면 첫 호출에서 콜드스타트(~50초)가 걸릴 수 있어 타임아웃을 60초로 넉넉히 잡아둠.

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
- ~~자동 업데이트 워크플로우 커밋/푸시 간헐적 실패~~ → 2026-09-19에 `.github/workflows/update_data.yml` 수정. 원인: GitHub Actions 러너가 큐에서 오래 대기하다 실행되면(수 시간 지연도 발생 가능) 그 사이 다른 커밋이 먼저 push될 수 있는데, 기존 `git pull --rebase origin main || true`는 `update_data.py`가 이미 워킹트리를 건드려놓은 상태라 원격이 움직였을 때 항상 실패하고 그 에러가 `|| true`에 조용히 삼켜져서, 결국 낡은 베이스 위에 커밋 → `push --force-with-lease` 거절로 이어짐. `git fetch` + `git reset`(mixed) + 재시도 루프로 교체해 해결. (`--soft`로 하면 인덱스가 안 갱신돼서 체크아웃 이후 원격에 새로 추가된 파일이 다음 커밋에서 삭제된 것처럼 처리되는 별도 버그가 있으니 반드시 기본/`--mixed` reset을 쓸 것.)
- ~~로컬 conda `fotdata` 환경에서 `import sklearn`이 아예 실패함~~ → 2026-09-22에 해결. 원인은 PyPI의 scipy 1.15.3 macOS arm64 공식 wheel 자체의 `_propack` 확장(`.so`)이 최신 macOS(26A428 이상)의 더 엄격해진 Mach-O `__thread_bss` 섹션 검증을 통과하지 못하는 문제였음 — `pip install --force-reinstall --no-cache-dir scipy`로 캐시를 지우고 새로 받아도 동일 wheel이라 재발(캐시 손상이 원인이 아니었음). **conda-forge 빌드(scipy 1.15.2)로 교체하니 정상 동작** — `conda install -n fotdata -c conda-forge scipy --force-reinstall -y`. 이후 `numpy 2.2.6 / scipy 1.15.2 / scikit-learn 1.7.2 / xgboost 3.2.0` 조합으로 `import main`, `update_data.py` 모두 정상 기동 확인. pkl 모델이 sklearn 1.9.1로 저장돼 지금 버전(1.7.2)과 다르다는 `InconsistentVersionWarning`이 뜨지만 실제 로드·예측은 정상 동작(기존 10번 항목의 `multi_class` 호환 패치와 같은 종류의 무해한 경고).

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
- ~~UptimeRobot으로 Render 무료 플랜 슬립 방지 설정 필요~~ → 2026-09-19에 `.github/workflows/keep_alive.yml`(10분 간격 핑)로 1차 조치했으나, 2026-09-20에 실제 실행 기록(`gh`/GitHub API로 직접 확인)을 보니 GitHub Actions의 `schedule` 크론이 best-effort라 10분 설정이 실제로는 2~5시간 간격으로만 실행되고 있었음 — 그 사이 Render가 슬립해버려 근본적 해결이 안 됐던 상태. **2026-09-20에 cron-job.org(외부 전용 크론 서비스, 10분 간격)로 교체**해 실제 해결. `keep_alive.yml`은 삭제하지 않고 보조 백업으로 유지(있어도 방해 안 됨, 어차피 못 미더우니 주력으로 의존하지 말 것).
  - 참고: 과거에 "cron-job.org로 이미 설정함"이라고 기록된 커밋(`7fc7076`)이 있었는데 그때는 실제 diff에 아무 내용이 없어 허위 기록이었음 — 이번엔 실제로 cron-job.org 콘솔에서 job 생성 확인 후 기록. 앞으로 "설정했다"는 기록은 반드시 실제 diff/동작(로그, 응답시간 등) 확인 후에만 남길 것.
