# FotData ⚽

**AI 축구 경기 예측·분석 플랫폼** — 유럽 5대 리그(EPL·라리가·분데스리가·세리에 A·리그 1)와 UEFA 챔피언스리그의 경기 결과를 머신러닝으로 예측하고, 순위·일정·선수·구단 정보를 한곳에서 보여줍니다.

- 서비스: **https://www.fotdata-official.com**
- API: https://fotdata-api.onrender.com

## 주요 기능

| 기능 | 설명 |
|---|---|
| 경기 예측 | 홈승·무·원정승 확률, 예상 스코어, 예측 근거, 두 팀 경기 분석(순위·홈/원정 성적·파워 레이팅 추이) |
| AI 트랙레코드 | 킥오프 **전에** 기록해 둔 예측을 경기가 끝나면 그대로 채점 — 사후 수정 불가 |
| 순위 예측 | 현재 승점 + 남은 경기를 예측 모델 확률로 최대 10,000번 시뮬레이션(우승·UCL권·강등 확률) |
| 순위표·일정 | 23-24 ~ 26-27 시즌, 공식 순위 구역·승점 감점 반영, 경기 당일 결과, 일정마다 AI 예측 |
| 구단 둘러보기 | 순위 순 3D 로고 카드, 구단 소개·우승 기록·팀 통계·이적·라이벌 |
| 공유·구독 | 경기별 링크 미리보기 썸네일, 구단 경기 일정 캘린더 구독(.ics) |

## 예측 모델

- **Logistic Regression**(서빙) — RandomForest·XGBoost는 비교용
- 피처 18개: 누적 ELO, 최근 폼, 최근 득실, 공격·수비 지수, 맞대결 등 — **학습과 서빙이 같은 정의**이고 "그 경기 직전까지의 기록"만 사용(미래 정보 누수 없음)
- 정확도는 **시간순 검증**(과거로 학습 → 학습에 안 쓴 최근 경기로 채점)으로 표시: 약 51%(전부 홈승으로 찍는 기준선 약 44%)

## 구조

```
브라우저 ──▶ Vercel (FotData.html·landing.html, 정적)
   │
   └──────▶ Render (FastAPI main.py) ── fotdata_model/* (모델·데이터 캐시)
                                          ▲
GitHub Actions (매일 KST 03:00) ── update_data.py: 수집 → 피처 → 학습 → 커밋
```

- **프론트엔드**: Vanilla JS + HTML/CSS (프레임워크 없음), Three.js(배경·랜딩 3D)
- **백엔드**: FastAPI, scikit-learn, pandas, Pillow(공유 썸네일)
- **자동화**: GitHub Actions가 매일 데이터 수집·모델 재학습·커밋 → Render·Vercel 자동 배포

## 로컬 실행

```bash
conda create -n fotdata python=3.10 && conda activate fotdata
pip install -r requirements.txt
export FOOTBALL_API_KEY=...        # football-data.org 키 (코드에 넣지 말 것)
python update_data.py              # 데이터 수집·모델 학습
uvicorn main:app --reload          # API
python -m http.server 8765         # 프론트(FotData.html의 API 주소는 배포 서버를 가리킴)
```

API 키는 반드시 환경변수로만 읽습니다(코드·노트북에 하드코딩 금지).

## 데이터 출처

- Football data provided by the [Football-Data.org API](https://www.football-data.org)
- 선수 사진·이적 기록: [API-Football](https://www.api-football.com)
- 구단 소개·우승 기록·지난 시즌 공식 순위: [위키백과](https://ko.wikipedia.org)(CC BY-SA 4.0)·위키데이터(CC0)
- 구단 로고·명칭은 각 구단의 상표이며 구단을 식별하는 목적으로만 사용합니다.
- 공유 썸네일 글꼴: [Pretendard](https://github.com/orioncactus/pretendard)(SIL OFL 1.1)를 줄여 이름을 바꾼 것 — `fonts/OFL.txt`

AI 예측은 통계 모델의 참고 정보이며 경기 결과를 보장하지 않습니다.
