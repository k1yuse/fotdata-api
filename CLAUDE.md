# FotData — 프로젝트 가이드 (CLAUDE.md)

Claude Code가 이 저장소에서 작업할 때 참고하는 **현재 상태** 문서다. 지난 시행착오·변경 이력은 git log에 있으니 여기엔 "지금 어떻게 돼 있고, 무엇을 조심해야 하는지"만 둔다. 새 기능을 넣으면 해당 절을 **고쳐 쓰고**(덧붙이지 말고), 다시 밟으면 안 되는 함정만 짧게 남길 것. **작업이 끝날 때마다 최신화하고 200~300줄로 유지**(넘으면 오래된 세부부터 뺌).

## 1. 개요

- **FotData**: 유승(GitHub k1yuse)이 만드는 AI 축구 경기 예측·분석 웹 서비스(상업 출시 목표). 5대 리그(EPL·라리가·분데스·세리에A·리그앙) + 챔피언스리그.
- 지금(Stage 1): 경기 예측 모델 + 리그(순위·일정·순위 예측·선수·역대 기록) + 팀·선수 정보. 다음: Stage 2 컴퓨터 비전(YOLOv8 + ByteTrack), Stage 3 유소년 영상 분석 리포트.
- **대화는 한국어 반말로 간결하게.** 사용자가 직접 할 일은 단계별 방법 + 복사용 명령어로.

| 항목 | 값 |
|---|---|
| 프론트 | https://www.fotdata-official.com (랜딩) · 앱 `/app` (Vercel) — 옛 fotdata-api.vercel.app은 새 주소로 308(`/cal/` 제외) |
| 백엔드 | https://fotdata-api.onrender.com (FastAPI, Render 무료 — Shell 없음, cron-job.org가 10분마다 깨움) |
| 저장소 | https://github.com/k1yuse/fotdata-api · 로컬 `~/fotdata-api` (맥북 M5) |
| 파이썬 | conda `fotdata` (3.10) — `python3` 말고 `/opt/miniconda3/envs/fotdata/bin/python` |
| 운영 메일 | fotdata.official@gmail.com (도메인·Umami·문의·정책 페이지 연락처) |
| 도메인 | fotdata-official.com — Cloudflare Registrar(자동 갱신, 만료 2027-10-03), DNS는 Vercel CNAME **Proxy 끔** |

## 2. 보안 규칙 (과거 키 노출 사고 3번 — 반복 금지)

- API 키·토큰은 코드·노트북에 절대 하드코딩하지 말고 `os.environ.get(...)`으로만. 키 파일은 `~/.fotdata_keys`(저장소 밖, 권한 600) — `source ~/.fotdata_keys`.
- 사용자에게 키를 채팅에 붙여넣게 하지 말 것.
- `git remote -v`에 자격증명이 보이면 즉시 `https://github.com/k1yuse/fotdata-api.git`으로 정리. push 인증은 `gh`(gh auth git-credential, 키체인).
- 커밋 금지: `_test_app.html`, `promo/`, `.claude/launch.json`, API 원본 샘플 데이터, `.env`.
- Actions 실패 로그: `gh run list --workflow update_data.yml` → `gh run view <id> --log-failed`.

## 3. 기술 스택 · 데이터 소스

- **백엔드** `main.py`(FastAPI): CORS 전체 허용, GZip, IP별 요청 제한(`RateLimitMiddleware` — 1분 240회, `POST /predict` 90회, CORS보다 먼저 등록), 서빙 데이터는 하루 한 번 재배포 때만 바뀌므로 `_load_json`(lru_cache — **반환값 수정 금지, 합칠 땐 복사본**)·계산 결과 lru_cache + 서버 시작 때 미리 계산(`_warm_all` — 사용자가 먼저 보는 것부터, 한 단위마다 0.03초 쉬어 요청이 먼저, 공유 썸네일은 맨 끝·한 장마다 쉼. 한꺼번에 몰아 계산하면 Render 무료 CPU에선 재시작 뒤 몇 분 동안 모든 요청이 느려졌음). 시간 기준으로 바뀌는 `/bigmatch`·`/matches/window`·라이브 점수는 매번 계산.
- **ML**: scikit-learn LogisticRegression(C=0.1)만 서빙. RF·XGBoost는 비교용. 버전은 `requirements.txt` 하나로 학습(Actions)·서빙(Render) 일치.
- **프론트**: Vanilla JS 단일 파일 `FotData.html`(프레임워크 없음). 글꼴 Pretendard Variable + 숫자 Barlow Condensed(`--font-num`, 확률·스코어·큰 숫자만).
- **데이터 소스**
  - **football-data.org**(`FOOTBALL_API_KEY`, 무료): 경기 결과·순위·일정·스쿼드 명단·득점 순위(scorers) — 예측 모델·팀 이름의 기준. 4시즌(23-24~26-27)까지 열림. 분당 10회 → 모든 호출은 `fd_get`(429·연결 오류 재시도). 약관상 화면에 **"Football data provided by the Football-Data.org API"** 문구 필수(푸터, 문구 변경 금지). 광고·유료화 전에 상업 이용 문의 필요(daniel@football-data.org).
  - **API-Football Pro**(`API_FOOTBALL_KEY`, $19/월, 하루 7,500회, 00:00 UTC 리셋, 매달 갱신 — 다음 2026-11-06): 경기 상세(이벤트·라인업·평점·xG)·부상자·선수 프로필·경력·트로피·감독·과거 시즌(2010-11~, 선수 경기 기록은 2015-16~). 배당·`/predictions`는 안 씀(도박 비권유). 요청은 `update_data._af`(전체 0.21초 간격·재시도).
    - ⚠️ `/players?team=&season=`·`/players/topscorers`의 시즌 기록은 **지금 소속팀으로 다른 팀 기록을 합쳐 줌**(팔머가 "맨시티 22골") → 시즌 기록은 반드시 경기 상세(`/fixtures?id=` 선수별 기록) 합산으로. `/players`는 프로필(나이·키·사진)만.
    - 감독은 `/coachs`(철자 주의). 원본에 오래된 항목이 남아 있음(아스널에 벵거가 종료일 없이).
  - **위키백과/위키데이터**(키 불필요): 구단 소개·별칭·연고지·수용 인원·우승 기록·이적료 기록·지난 시즌 공식 순위 구역·역대 우승·감독 생년월일/국적 보완. CC BY-SA → **출처 링크 필수**. 나무위키는 비영리 라이선스라 사용 불가.
- **방문 통계**: Umami Cloud(쿠키 없음) — `analytics.js`의 `UMAMI_ID`, 배포 주소에서만 전송. 앱 이벤트는 `track()`.

## 4. 파일 구조

```
main.py               FastAPI 백엔드 — 전체 API
update_data.py        매일 수집 + 피처 + 모델 학습 + 파생 데이터 (CLI 옵션은 6절)
af_transform.py       API-Football 응답 → 화면용 변환(update_data·main 공용): 경기 상세, 세부 포지션, 베스트 11(pick_xi), 시즌 합산
FotData.html          앱 (/app) — 9,500줄 단일 파일
landing.html          랜딩 원본 → index.html은 사본 (4.1)
landing-story.js · ball-geometry.js · bg-ball.js · bg-ball.svg   랜딩 3D 연출·배경 공(Three.js, 공 모양 공유)
share_card.py · fonts/   경기별 공유 썸네일(1200×630) — 폰트는 Pretendard 서브셋 "FotData Card Sans"(OFL)
vercel.json           /app 리라이트, /m/*·/og/m/*·/cal/* → Render, 옛 주소 리다이렉트, 보안 헤더, 캐시
privacy.html · terms.html · legal.css · 404.html · README.md · sitemap.xml · robots.txt · manifest.json · sw.js
logos/hd/(160px)·logos/hd/l/(400px)   구단 엠블럼 고화질(generate_logos_hd.py — 위키 인포박스, EXCLUDE 7팀은 원래 로고)
logos/league/         리그 심볼 아이콘(generate_league_logos.py) — 사이트 전체 리그 로고
logos/comp/           컵대회·국가대표 대회 아이콘(generate_comp_logos.py, API-Football 대회 ID)
generate_*.py         아이콘·og-image(문구 바꾸면 재실행 + meta `?v=` 올리기)·홍보 이미지(promo/, 커밋 안 함)
fotdata_model/        전부 update_data.py가 생성 (아래)
```

**fotdata_model/** 주요 파일
- 모델: `all_matches.csv`(23-24~ 4시즌, 학습·맞대결 기준) · `team_state.json`(팀별 오늘 시점 모델 입력) · `team_stats.csv`(블렌딩 전력 — 순위 예측·검색 정렬·결과 화면 표시용) · `logistic_regression.pkl`·`scaler.pkl` · `accuracy.json` · `prediction_log.json`(트랙레코드)
- 일정·순위: `schedule.json` · `ucl_tournament.json` · `season_zones.json`(지난 시즌 공식 순위·구역·감점) · `league_history.json`(역대 우승)
- 과거 시즌(2010-11~22-23, 순위표·일정·팀 통계·역대 맞대결용 — **모델 학습엔 안 씀**): `history_matches.csv` · `history_standings.json` · `history_ucl.json` · `history_logos.json`
- 팀: `team_info.json` · `team_wiki.json` · `team_extra.json`(스쿼드 사진·등번호·이적, 5대 리그 전 팀) · `rivals.json`(수동 관리) · `team_logos*.json`
- API-Football: `af_team_map.json`(af 팀 ID → 우리 이름) · `af_fixtures.json` · `match_details/*.json.gz`(끝난 경기 상세, 한 경기 한 파일) · `match_previews.json` · `team_injuries.json`(팀별 지난 경기 결장자) · `af_players.json`(선수 프로필) · `af_profiles/<ID>.json.gz`(선수 경력·트로피·부상·시즌 기록) · `af_seasons/`(15-16~25-26 선수 시즌 합산 + index.json) · `comp_winners.json`(컵 결승·리그 1·2위 — 트로피 보강) · `af_coaches.json`(감독) · `af_leagues.json`·`af_countries.json`
- `scorers.json`(football-data 득점 순위 — API-Football 기록이 없을 때 대체)

### 4.1 `index.html` = `landing.html` 사본
Vercel 루트는 `index.html`. landing.html을 고치면 **반드시 `cp landing.html index.html`** 후 커밋.

## 5. 예측 모델

### 5.1 경기 예측 (`/predict`)
- 피처 18개는 학습·서빙이 같은 정의: `build_point_in_time_features()`가 전 경기를 시간순으로 훑으며 "그 경기 직전까지"만으로 계산(누수 없음), 마지막 상태를 `team_state.json`으로 저장 → `/predict`가 그대로 사용(`_team_snapshot` 공유).
  - 누적 ELO(K=20, 홈 +70, 챔스 포함, 승격팀은 리그 하위 4팀 평균에서 시작) · 폼(최근 5경기 승점) · 최근 10경기 득실 · 최근 38경기 공격/수비/승률 · H2H(최근 10번 맞대결 홈팀 승률).
- 정확도 = **시간순 검증**(과거 80% 학습 → 최근 20% 채점, 약 51%, 전부 홈승 기준선 약 44%) → `accuracy.json`, 서빙 모델은 전체로 재학습. 화면 문구 "학습에 안 쓴 최근 N경기로 검증".
- 5대 리그 밖 팀(UCL 기록 8경기 이상, `UCL_ONLY_MIN_MATCHES`)도 예측하되 리그 평균 쪽으로 30% 당기고(`UCL_ONLY_SHRINK`) "참고용" 안내(`limited: true`).
- 스코어 예측: 포아송(블렌딩 공격/수비로 페이스, 홈/원정 배분은 LR 승률차).
- `team_stats.csv` 블렌딩(23-24~26-27 가중 평균 + prestige)은 **예측 모델 입력이 아님** — 순위 예측 보조·결과 화면 공격/수비 표시·검색 정렬용.

### 5.2 순위 예측 (`/predict/champion/{리그}` + 브라우저 시뮬레이션)
- 서버: 현재 순위표 + 남은 경기 + 경기별 H/D/A(`_predict_hda` — `/predict`와 같은 모델), 시뮬레이션용만 리그 평균 쪽으로 10% 당김(`SEASON_SIM_SHRINK`), σ=0.15(`SEASON_SIM_SIGMA`).
- 브라우저 `simulateSeason()`: 팀별 δ~N(0,σ)로 확률 기울여 결과 추첨 → 승점 → 득실 순. 기본 1,000회(사용자 결정). 우승·UCL권·강등·예상 승점(+가운데 80% 범위)·순위 분포. 표 정렬은 화면에 보이는 숫자 기준.
- 백테스트로 정한 값(λ=0.1, σ=0.15)이니 바꿀 땐 다시 백테스트.

### 5.3 트랙레코드 (`prediction_log.json`)
- 매일 향후 10일 경기를 **라이브 `/predict`를 호출해** 미리 기록(모델 재구현 금지 — 사용자가 본 예측과 같게), 끝나면 결과·적중 채움. 이미 기록된 경기는 다시 안 찍음. 결과 확정분은 최근 500건.
- 경기 전에 기록된 예측이 있는 경기만 "AI 적중/빗나감" 표시(사후 예측으로 부풀리지 않음).

## 6. 데이터 파이프라인 (`update_data.py`)

- 매일 UTC 18:00(KST 03:00) GitHub Actions(`update_data.yml`, Secrets: `FOOTBALL_API_KEY`·`API_FOOTBALL_KEY`)가 실행 → `fotdata_model/` 자동 커밋("자동 데이터 업데이트 YYYY-MM-DD") → Render·Vercel 재배포.
- `main()`은 단계별 `step()` — 하나가 실패해도 나머지는 돌고 저장, 실패가 있으면 종료 코드 1 + 커밋은 진행.
- 순서: 경기 수집·학습 → UCL 토너먼트 → 일정 → 트랙레코드 기록 → 득점 순위 → 로고 → 팀 정보 → 위키(매일, 저장된 `en_title` 문서 그대로 — 다시 검색 안 함) → 시즌 구역 → 역대 우승 → API-Football(`af_sync`: 팀 매칭·경기 목록·끝난 경기 상세·결장자·선수 프로필) → 대회 우승 팀 → 감독 → 스쿼드·이적(`fetch_squads_transfers_all`, 5대 리그 전 팀 7일마다, 팀당 2회, 데이터 없는 팀부터) → 선수 경력·트로피(`af_profiles_sync`, 그날 남은 요청 −600 전부, 아직 없는 선수 → 출전 많은 순, 14일마다 갱신).
- API-Football 하루 한도는 00:00 UTC(KST 09:00)에 초기화 — 새벽 실행(UTC 18:00)은 **그날 낮에 쓴 몫과 같은 하루 한도**를 씀. 낮에 수동으로 많이 돌리면 그날 밤 수집이 줄어듦(`/status`로 남은 양 확인). 한도 초과·오류 응답은 `_af_ok()`로 걸러서 **기존 파일을 절대 빈 값으로 덮어쓰지 않음**(2026-10-08 새벽, 한도 초과 응답을 "경기 0개"로 저장해 이번 시즌 선수 화면이 통째로 사라진 사고 — 새 API-Football 수집 코드도 같은 규칙으로).
- 선수 경력(`af_profiles/`)은 새 선수 한 명에 약 15회(시즌 수 + 4) — 2026-10-07 기준 2,615명 중 419명, 하루 약 400명씩 채워짐. 파일이 없는 선수는 서버가 요청 때 받아 경력·트로피 탭이 5~7초 걸림.
- 따로 돌리기: `--matches-only` · `--ucl-only` · `--wiki-only [팀] [--force]` · `--zones-only` · `--league-history` · `--scorers-only` · `--transfers-only [팀 수] [--force]` · `--extra-leagues` · `--history`(지난 시즌 팀 로고·정보) · `--af-sync` · `--af-profiles [예산]` · `--coaches [--force]` · `--history-seasons [연도…]`·`--history-players [연도…] [--force]`·`--player-index`(끝난 시즌 — 한 번만).
- 저장 순서 고정(날짜→리그→홈팀) — 순서가 흔들리면 매일 파일 전체가 바뀐 것처럼 커밋되고 ELO 반영 순서도 달라짐.
- 경기 수집은 상태 필터 없이 받아 FINISHED+AWARDED만(몰수 경기 누락 방지), 승부차기 골은 빼고(`_match_goals`) 일정엔 `penalties` 따로.
- **감독**(`fetch_coaches`): 지금 감독 = 그 팀 가장 최근 경기 라인업의 감독 → `/coachs?team=`에서 같은 사람(라인업 이름이 "Enrique Luis"처럼 뒤집혀 오기도 함 — `_same_person`)의 그 팀 경력(사진·부임일) → 생년월일·국적·흔히 부르는 영어 이름·한국어 이름은 위키데이터(구단 P286). 7일마다 + 라인업 감독이 바뀌면 바로.
- **구단 최고 이적료**: 영문 위키 "List of … records and statistics" 문서만(표 — 소제목·굵은 글씨 제목 아래, 없으면 본문 글 "club record £100m"), 선수=사람·구단=축구 클럽을 위키데이터로 확인. 이 문서가 있는 팀만 나옴(약 30팀) — API-Football엔 이적료가 없고 구단 본문 글은 오래된·다른 팀 기록이 섞여서 안 씀.
- 위키 전 팀 재수집(`--wiki-only --force`) 뒤엔 "팀 이름 단어가 문서 제목에 없는 팀"을 눈으로 확인(오매칭 전례 — `WIKI_TITLE_OVERRIDE`·`WIKI_CITY_OVERRIDE`로 고정).

## 7. 서버 (main.py)

- 끝난 시즌 순위표 = 공식 순서·구역·감점(`season_zones.json` → 없으면 `history_standings.json`), 시즌은 날짜가 아니라 시즌 값으로 자름. `df_seasons`(과거+현재)는 순위표·일정·팀 통계·역대 맞대결·라이벌용, **`df_matches_all`(4시즌)만 모델·최근 폼용 — 섞지 말 것**.
- 경기 당일 점수: Render가 football-data를 직접 받아 일정 위에 덮음(`_live_overlay`, 서버 전체 1분 1회, Render 환경변수 `FOOTBALL_API_KEY` 필요). 무료 플랜은 몇 분 늦어서 "LIVE" 대신 "진행 중 N분".
- API-Football 요청 때 받기(Render 환경변수 `API_FOOTBALL_KEY` 필요): 끝난 경기 상세(파일 없으면 킥오프 110분 뒤부터), 확정 라인업(킥오프 90분 전~), 프로필 미수집 선수 경력. 동시 16개(`_AF_SEM`) + 재시도.
- 선수 시즌 기록: 이번 시즌 = `match_details` 합산(`_af_league_squads`, 시작 때 계산), 지난 시즌 = `af_seasons`. 세부 포지션은 22-23 시즌부터만 믿을 수 있음(`GRID_FROM`).
- 공유: `/share/match/{slug}`(og 태그, 사람은 앱으로) · `/og/match/{slug}.jpg`(share_card, 카드 캐시) — slug 규칙은 JS `teamSlug`와 서버 `_team_slug`가 같아야 함, 짧은 이름은 FotData.html `SHORT_NAMES` 표 하나만 관리(서버가 읽음).
- 데이터 없는 응답은 404 대신 **204**(콘솔 오류 방지).

주요 엔드포인트(전체는 main.py에서 `@app.` 검색):

| 용도 | 경로 |
|---|---|
| 예측 | `POST /predict` · `/predict/schedule/{리그}` · `/predict/champion/{리그}` · `/predict/track-record` · `/match/insights` · `/h2h?limit=0`(역대 전부) · `/form/{팀}` |
| 리그 | `/standings/{리그}?season=&view=` · `/standings/{리그}/movement` · `/schedule/{리그}?season=` · `/ucl/tournament` · `/ucl/groups` · `/league/seasons/{리그}` · `/league/history/{리그}` · `/league/players/{리그}?season=` · `/league/bestxi/{리그}?season=&round=` |
| 팀·선수 | `/team/info/{팀}`(wiki·rivals·manager) · `/team/squad/{팀}?season=`(best11·manager) · `/team/stats/{팀}` · `/team/players/{팀}` · `/player/profile/{id}` · `/player/seasons/{id}` · `/players/leaders/{리그}` |
| 경기 | `/matches/window` · `/matches/live` · `/match/detail` · `/match/preview` · `/bigmatch` |
| 기타 | `/players/search?q=`(이번 시즌 5대 리그 선수 이름 검색 — 3글자 이하는 단어 첫머리만) · `/teams` · `/teams/meta` · `/teams/ko` · `/logos` · `/meta/countries` · `/proxy/logo` · `/calendar/{slug}.ics` · `/calendar/my.ics?t=` · `/accuracy` |

리그 코드: `PL` `PD` `BL1` `SA` `FL1` `CL`

## 8. 프론트엔드 (FotData.html)

### 8.1 구조
- 페이지: 경기 예측(홈) · 선수(`#search` — 이름 검색·리그/포지션 거르기, 검색어 없으면 평점 순 `/players/search`) · 리그(`#league/PL/standings/2024` — 탭: 개요·순위·경기·순위 예측·선수·시즌, 지난 시즌은 순위 예측 없음) · 내 팀(즐겨찾기 최대 5팀, localStorage) · 승부차기 · 더보기(폰). 폰은 하단 탭바 `홈·리그·내 팀·선수·더보기`.
- 리그 패널 로더(loadStandings 등)는 예전 페이지를 옮긴 것 — 패널 안 숨겨진 리그 칩은 로더가 "늦은 응답 버리기"에 쓰므로 **지우지 말 것**. 새 리그 UI 요소는 `data-lg` 속성(`[data-league]`와 겹치면 안 됨).
- 선수 카드(`openPlayerCard`): 그 팀 그 시즌 기록에 없는 선수(떠난·예전 선수)는 우리 데이터에서 마지막으로 뛴 시즌으로 자동 이동(+ 머리에 "지금 ○○"), 우리 리그 기록이 아예 없으면 프로필·경력·트로피만으로(`noStats`). 이적 목록 선수도 누르면 카드.
- 창(모달): 팀 정보(개요·순위·경기·스쿼드·플레이어 통계·팀 통계·이적) · 경기 미리보기/결과(넓은 화면 2열) · 선수 카드 · 트랙레코드 · 대진 · 구단 둘러보기(⌘K) · 캘린더. **새 창을 만들면 `OVERLAYS`(뒤로가기)와 `SCROLL_LOCK`(뒤 스크롤 잠금)에 한 줄씩 추가.**
- 팀 정보 개요: 맨 위 감독(1/3 — 사진·국적·나이·부임·부임 후 리그·챔스 기록, 이름뿐이면 아래 정보 칸으로) + 구단 소개(2/3) → 이번 시즌 순위·다음 경기(아래 줄에 결장 N명) → 시즌별 순위 막대 → 시즌 베스트 11 → 우승·이적료 기록 → 라이벌 → 정보 칸. 스쿼드 탭 맨 위(감독 아래)에 결장·부상 + 명단 줄에 결장 배지 — `/team/squad`의 `injuries`(team_injuries.json, 그 팀 지난 경기 기준). 창을 열거나 팀 이름에 마우스를 올리면 `prefetchTeam()`이 팀 정보·팀 통계·스쿼드를 동시에 받음(요청은 promise 캐시로 한 번만 — `fetchTeamInfo`·`fetchTeamStats`·`fetchAfSquad`).
- 예측 결과: 두 독립 세로 열(`.predict-col`) — 왼쪽 예측 결과·다른 경기·결장·부상·홈팀 최근 경기 / 오른쪽 맞대결·경기 분석·라인업·주요 선수·원정팀 최근 경기. `fitMoreCard()`가 짧은 열에 "다른 경기" 카드를 넣거나 간격을 넓혀 두 열 끝을 맞춤(부가 정보가 올 때마다 다시). 폰은 `display: contents` + `order`.
- 리그 개요: 넓은 화면 두 열 끝 맞춤 `fitLoCols()`(차이 220px 이하일 때만 `.fit`).
- 실시간 경기 창(`openMatchLiveModal` — 진행 중 경기를 누르면): 점수 위 빨간 배지(전반/후반 N′·하프타임), 요약(타임라인 최신이 위)·라인업·통계, 경기 전 AI 예측 + "지금은 예측대로/다르게", 1분마다 `/match/live`(서버 45초 캐시) 다시 받고 끝나면 멈춤. 진행 중 점수는 API-Football `/fixtures?live=all`(1분 캐시)로 football-data 점수를 덮음(`_af_live_merge`).
- 공유 링크 `/m/{홈}-vs-{원정}`, 앱 주소창은 `/app?match=`.
- **뒤로가기 기록**(`navPush`): 앱 안에선 "홈(경기 예측) + 지금 화면" 두 칸만 — 홈에서 다른 화면으로 갈 때만 한 칸 쌓고(`fdDepth` 1) 그 뒤 화면·리그·탭·시즌 이동은 덮어씀, 홈으로 가면 그 칸을 되돌림 → 어디서든 뒤로 한 번 = 홈, 한 번 더 = 랜딩. 창(팀 정보 등)은 그 위 "창" 칸(`fdGuard`)으로 따로. **새로고침**은 지금 탭(주소 `#…`)은 그대로 두고 고른 경기(`?match=`)·열린 창만 초기화(공유 링크로 처음 들어온 건 그 경기 그대로), 로고 클릭도 같음(`fdReloadTab`). 주소에 탭이 있으면 `<head>`에서 `html.route-boot`로 화면을 숨겼다가 그 탭으로 바꾼 뒤 보여줌(경기 예측이 잠깐 보였다 넘어가던 깜빡임 제거). 새로고침 뒤에도 기록 칸 정보(`history.state`)를 이어 씀. 랜딩은 위 메뉴 "소개"·더보기 "FotData 소개"로(앱을 /app으로 바로 열면 뒤로가기로는 못 감). 메뉴에 화면 없는 링크를 넣을 땐 `data-page` 없이도 되게(`setActiveTab`).
- 검색(⌘K·상단 검색·더보기 "구단·선수 검색"): 구단 둘러보기 검색창이 구단(바로) + 선수(`/players/search`, 0.18초 멈춘 뒤) — 선수를 누르면 선수 카드. 경기 예측 팀 선택 팝업 맨 위엔 ★ 내 팀 줄(즐겨찾기 바로 선택).

### 8.2 디자인 규칙
- **디자인 테마**(FotData.html 맨 위 스크립트 `theme` 기본값): **`theme-a`(기본, 2026-10-10)** = 유리 층(`<style id="glass-v2">`, `html.theme-glass`) 위에 `<style id="theme-a">`를 얹음. `?theme=glass`(유리)·`?theme=default`(유리 이전)로 비교, 되돌리기는 기본값 한 글자 또는 태그 `design-glass`. **새 디자인은 덮어쓰지 말고 테마 층으로 추가**.
  - theme-a 규칙: 파랑·주황은 홈·원정 **데이터만**, 브랜드·강조는 오로라(`--iris`, 로고 공·주 버튼 테두리·고른 메뉴 아이콘 — SVG는 `url(#fdIris)`), 링크·고른 항목은 흰 계열(`--blue-text`도 흰 계열로 덮음). 층은 ① 주인공 유리(빅매치·창·위 메뉴·폰 탭바) ② 패널(`.card` — 옅은 판 + 얇은 테두리, 흐림 없음) ③ 패널 안은 상자 대신 줄. 버튼은 유리(주 버튼 = 오로라 테두리), 탭·세그먼트·칩은 어두운 홈 안에 고른 칸만 뜬 유리 알약(`--a-seg-on`). 평점 배지는 초록 단계(파랑 금지). 레이더 대신 두 팀 비교 막대(`cmpBarsHtml`, 공격·수비 칸 `#stat-boxes`는 숨김). 폰 탭바는 떠 있는 유리 알약(`--tabbar-h` 84px+안전 영역).
  - 유리 층 토큰: 알약·배지 `--gp-{blue,org,grn,red,amb,neu}-{bg,bd,tx}`.
- **움직임**(`<style id="motion-v1">`, Emil Kowalski 기준): 곡선은 `--ease-out`(나타남·사라짐)·`--ease-in-out`(화면 안 이동)·`--ease-drawer`만, UI는 0.3초 안, 닫힘이 열림보다 빠르게(창 0.26/0.15초), 누르면 `scale: .97`(transform 말고 scale 속성 — translate로 자리 잡은 버튼이 안 튐), `transition: all` 금지, 탭 전환 6px·0.22초, 키보드(⌘K)로 여는 건 애니메이션 없이. 드물게 보는 순간(예측 결과 배지)만 튀는 연출.
- **마우스 올림(`:hover`)은 전부 `@media (hover: hover) and (pointer: fine)` 안에** — 폰에서 손 뗀 뒤 올림 효과가 남지 않게. 새 `:hover` 규칙도 반드시 이 안에.
- 설명 문구는 짧게, 기준·방법 설명은 ⓘ(`TIPS`)로. "A · B · C" 식 긴 설명 줄 만들지 말 것.
- 색은 `:root` 토큰(`--bg #0a1020` 밤하늘 남색, `--surface`, `--blue #58a6ff` …), 둥글기 `--r-*`, 그림자 `--shadow-*`, 글자 `--fs-*`. 템플릿 문자열·캔버스·SVG 속성만 hex 그대로.
- **홈 파랑 `#58a6ff` / 원정 주황 `#f0883e`**, 무 회색. 결과 색은 그 화면 주인공 기준.
- **마우스 반응 규칙**: ① 팀·선수 이름(로고·사진 포함)은 어디서든 누르면 팀 정보/선수 카드, 올리면 **이름만 파랗게** ② 줄 전체가 한 곳으로 가는 목록(경기 줄·선수 줄·순위표 줄)은 **배경만**, 그 줄 안 팀 이름은 따로 안 누름 ③ 큰 카드는 **파란 테두리만**(들림·그림자 없음) ④ 버튼·탭은 글자 밝아짐/빛 ⑤ 누를 수 없는 건 반응 없음·기본 커서.
- 경고는 화면 안 상자 대신 `showToast(msg, ms, 'warn')`. 용어 도움말은 `TIPS` + `<button class="ii" data-tip="키">`.
- 시각은 `koClock()`("오후 8:30", 앞자리 0 없음), 남은 시간 `koRel()`. 짧은 팀 이름 `teamNameHtml()`(≤768px 짧은 이름).
- 이름: 경기장 위 = "M. Salah"(`mdInitialName`), 목록·카드 = 전체 이름(`afFullName`). 등번호는 이름 앞 회색.
- 포지션은 **영어 약자**(GK·CB·LB·RB·LWB·RWB·CDM·CM·CAM·LM·RM·LW·RW·ST, 묶음은 GK·DF·MF·FW — `DPOS_KO`·`POS_KO`·`POSITION_LABEL`, 이름은 예전 그대로 값만 약자). 주장은 글자 대신 초록 원 안 C(`.cap-ic` — 선수 카드는 이름 옆, 경기장 위(라인업·구단 베스트 11)는 등번호 바로 왼쪽 `capWrap`, 팀 목록(스쿼드·선수별 기록·주요 선수)은 이름 뒤 `capInline` — 팀 주장 = `/team/squad`의 `captain`. 리그 베스트 11처럼 여러 팀이 섞인 곳엔 안 넣음). 끝난 경기·예상 라인업 = 그 경기 실제 완장(경기 기록 `cap`), 확정 라인업·구단 베스트 11 = 이번 시즌 완장 많이 찬 순(`_team_captain_order`) 주장 → 부주장 → 없으면 표시 안 함. 구단 선수별 시즌 기록은 머리글(출전·골·도움·평점)을 누르면 그 순으로(`sqSort`).
- 3D 로고·구단 로고엔 원본에 없는 효과(테두리·광택)를 더하지 않음. 홍보물엔 구단 엠블럼 금지(상표).
- 카드 높이를 바꾸면 첫 화면 자리 잡기 `.slot-wait` min-height도 같이(빅매치 331/폰 175, 트랙레코드 67, 내 팀 53).

### 8.3 함정 (다시 밟지 말 것)
- `data-w`는 경기 분석 막대 애니메이션이 씀 — 다른 용도로 `data-*` 이름 겹치지 않게.
- SVG에 글자가 있는 차트는 viewBox 확대 금지 — 컨테이너 실제 폭으로 1:1로 그림.
- `backdrop-filter` 카드 안 절대 위치 팝업은 아래 카드에 가려질 수 있음 → 부모 카드에 z-index.
- 카드 `::before`(빛 테두리)가 inset -1px라 `scrollHeight`가 1px 커짐 — 넘침 판정은 여유(+2px)를 둘 것("다른 경기" 카드가 안 뜨던 전례). 예측 결과 두 열 맞춤은 `fitMoreCard` + ResizeObserver(열 안 내용만 관찰).
- flex 부모 밑 `margin: auto` 중앙 정렬 요소는 `width: 100%` 필요.
- 로고 `<img>`가 빈 src로 먼저 그려지면 onerror로 숨은 채 굳음 → `logosReady`를 기다렸다 그림.
- `onerror="..."`에 HTML 문자열을 넣지 말 것(따옴표 충돌) — 함수 호출로.
- 좁은 그리드 열은 `minmax(0, 1fr)`(긴 팀 이름 가로 넘침).
- 같은 인라인 스크립트 블록 `const` 이름 충돌 → 스크립트 전체가 멈춤. 푸시 전 인라인 스크립트 전부 `node --check`.
- main.py·FotData.html은 커서 새 함수 이름이 기존 것과 겹치기 쉬움 — 만들기 전에 `grep -n 'def 이름\|function 이름'`(2026-10-08 선수 검색용 `_norm_name`이 스쿼드 이름 맞추기용을 덮어써 스쿼드가 줄인 이름으로 나왔음 → `_search_norm`).

### 8.4 속도 (2026-10-07 라이브 실측, 한국 → Render 미국 왕복 약 0.2초 포함)
- 대부분의 API 0.2~0.6초, 팀 정보 개요 첫 표시 0.4초·전부 1초 안, 리그 개요 0.5초, 경기 미리보기 0.5초, 선수 카드 0.6초.
- 느린 곳: 선수 경력·트로피 탭(새벽 수집 전 선수 5~7초 — 6절), Render가 잠들었다 깨는 첫 요청(~50초, cron-job.org 핑으로 대부분 방지), 재배포 직후 계산 캐시가 빈 동안.
- 원칙: 한 화면이 쓰는 요청은 처음에 **동시에** 시작(앞 요청 결과를 기다렸다 다음을 부르지 말 것), 같은 요청은 promise 캐시로 한 번만, 누를 가능성이 큰 곳은 마우스 올림·손가락 댐에 미리 받기, 서버는 계산 결과 lru_cache + 시작 때 warm.
- 서비스 워커(`sw.js`)가 API GET 응답을 저장해 두고 다음엔 바로 보여준 뒤 뒤에서 새로 받음(stale-while-revalidate, 20시간까지 — 진행 중 점수·라인업·검색·공유는 제외) → 두 번째 방문부터 로딩 화면이 거의 없음. 서비스 워커를 바꾸면 `CACHE_NAME` 올리기.
- 측정: 헤드리스 크롬으로 화면 요소가 나타날 때까지 시간 + `performance.getEntriesByType('resource')`(요청 시작→끝)를 같이 봄.

## 9. 배포 · 환경

- `git push` → Render(백엔드)·Vercel(프론트) 자동 재배포. 강제 재배포: `git commit --allow-empty -m "trigger redeploy"`.
- 로컬:
  ```bash
  conda activate fotdata
  cd ~/fotdata-api && source ~/.fotdata_keys
  python update_data.py            # 또는 --옵션 (6절)
  python -m uvicorn main:app --port 8000
  ```
- 로컬 화면 확인: 정적 서버(launch.json "static", 8765) + `_test_app.html`(= FotData.html의 API 주소를 `http://127.0.0.1:8000`으로 바꾼 사본, 커밋 금지). 브라우저 창이 숨김 상태면 rAF·ResizeObserver·스크롤 애니메이션이 안 돌아 오판하기 쉬움 → 헤드리스 크롬(CDP) 캡처나 탭을 앞으로. 랜딩은 `scroll-behavior: smooth`라 `scrollTo({behavior:'instant'})`.
- scipy는 conda-forge 빌드(PyPI arm64 wheel이 최신 macOS에서 깨짐).

### Git 충돌 (Actions가 fotdata_model/을 자동 커밋)
```bash
git pull --rebase --autostash
git checkout --theirs fotdata_model/<파일>   # 데이터는 원격(최신)
git checkout --ours <내가 고친 코드>          # 코드는 로컬
git add … && git rebase --continue && git push
```
`--ours`/`--theirs`를 반대로 쓰면 최신 데이터가 낡은 데이터로 덮임 — 실수하면 `git fetch origin && git checkout origin/main -- fotdata_model/`.

## 10. 시즌 전환 체크리스트 (매년)

- [ ] 키 유효성(로컬 `~/.fotdata_keys`, GitHub Secrets, Render 환경변수)
- [ ] `update_data.py` `MATCH_SEASONS` 한 시즌 밀기, `main.py` `CURRENT_SEASON_YEAR`
- [ ] 끝난 26-27: `--history-players 2026` 한 번 + `AF_PLAYER_SEASONS` 범위 늘리기
- [ ] `FotData.html` `LEAGUE_DATA` 승강팀, 승격팀 로고(`generate_logos_hd.py` 재실행 후 눈으로 확인)
- [ ] 시즌 종료 배너(FotData.html·landing.html 주석) 재활성화 여부

## 11. 남은 일 · 보류

- 광고(AdSense) 붙이면 개인정보처리방침의 "쿠키를 사용하지 않습니다"·외부 서비스 표 수정, football-data 상업 이용 문의 먼저.
- UCL 밖 팀 2단계: 에레디비시·프리메이라리가(`extra_matches.csv` 수집 중) + 리그 수준 보정 — 백테스트 후 모델에 켜기.
- 모델 개선 후보: 4시즌 블렌딩 가중치 재설계(순위 예측용), 과거 시즌(API-Football)을 학습에 넣을지 결정(넣으면 출처가 겹치지 않게).
- 컵대회 탭(대회 아이콘·이름 `COMP_INFO` 그대로 사용).
- 보류: 예측 게임(내 예측 vs AI) — 앱이 복잡해져서. 어두운 엠블럼 밝게 처리 — 안 하기로.
- 수익화: Ko-fi(있음) → AdSense → Freemium/예측 API → 유소년 영상 분석 SaaS.
