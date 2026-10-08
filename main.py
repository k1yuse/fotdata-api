# ── [API] FotData FastAPI 서버 ──
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import requests
from urllib.parse import urlparse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import json
import math
import re
from functools import lru_cache

app = FastAPI(title="FotData API", version="1.0.0")

# ── 요청 제한 + 기본 보안 헤더 (2026-10-04) ──
# 무료 서버 한 대라 누가 스크립트로 수천 번 부르면 모두가 느려짐 → IP마다 1분에 일반 240번·/predict 90번까지.
# 평소 사용(첫 화면 20번 안팎, 예측 1번에 5번)과 매일 자동 업데이트의 트랙레코드 기록(/predict를 1초 간격)은 넉넉히 들어옴.
# Vercel을 거쳐 오는 공유 페이지·썸네일·캘린더(/share/·/og/·/calendar/)는 IP가 Vercel 것이라 제외(어차피 캐시됨),
# 서버 깨우기 핑("/")·CORS 사전 요청(OPTIONS)도 제외. 사용자 IP를 못 찾으면 제한하지 않음(전원이 한 칸에 묶이는 사고 방지).
# 이 미들웨어를 CORS보다 먼저 등록해야 CORS가 바깥에서 감싸서 429 응답에도 CORS 헤더가 붙음(안 붙으면 브라우저엔 CORS 오류로 보임).
import time as _time
from collections import deque as _deque, Counter
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse

RATE_LIMITS = {"predict": 90, "default": 240}   # 1분당 횟수
RATE_WINDOW = 60
RATE_EXEMPT_PREFIX = ("/share/", "/og/", "/calendar/")
_rate_hits = {}
_rate_calls = 0

def _client_ip(request):
    for h in ("cf-connecting-ip", "true-client-ip", "x-real-ip"):
        v = request.headers.get(h)
        if v:
            return v.strip()
    xff = request.headers.get("x-forwarded-for")
    return xff.split(",")[0].strip() if xff else None

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        global _rate_calls
        path = request.url.path
        ip = _client_ip(request)
        limited = ip and request.method != "OPTIONS" and path != "/" and not path.startswith(RATE_EXEMPT_PREFIX)
        remaining = None
        if limited:
            bucket = "predict" if path == "/predict" else "default"
            limit, now = RATE_LIMITS[bucket], _time.monotonic()
            q = _rate_hits.setdefault((ip, bucket), _deque())
            while q and now - q[0] > RATE_WINDOW:
                q.popleft()
            if len(q) >= limit:
                retry = max(1, int(RATE_WINDOW - (now - q[0])) + 1)
                return JSONResponse({"detail": "요청이 너무 많아요. 잠시 후 다시 시도해주세요."}, status_code=429,
                                    headers={"Retry-After": str(retry), "X-RateLimit-Limit": str(limit), "X-RateLimit-Remaining": "0"})
            q.append(now)
            remaining = limit - len(q)
            _rate_calls += 1
            if _rate_calls % 500 == 0:   # 1분 넘게 조용한 IP는 정리(메모리가 계속 늘지 않게)
                for k in [k for k, v in _rate_hits.items() if not v or now - v[-1] > RATE_WINDOW]:
                    _rate_hits.pop(k, None)
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        if remaining is not None:
            response.headers["X-RateLimit-Limit"] = str(RATE_LIMITS["predict" if path == "/predict" else "default"])
            response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

app.add_middleware(RateLimitMiddleware)

# 응답 압축(2026-10-06) — 선수 목록(리그 전체 500명+)·시즌 기록처럼 큰 JSON이 늘어서. 1KB 미만은 그대로
from starlette.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS 설정 (나중에 웹/앱에서 호출 가능하게) — 요청 제한보다 나중에 등록 = 바깥에서 감쌈
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 모델 & 데이터 로드 ──
BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "fotdata_model")

lr_model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
if not hasattr(lr_model, 'multi_class'):
    lr_model.multi_class = 'auto'
scaler    = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
df_stats  = pd.read_csv(os.path.join(MODEL_DIR, "team_stats.csv"))
# ── 응답 캐시 (2026-09-29) ──
# fotdata_model/의 데이터는 하루 한 번(자동 업데이트 커밋 → Render 재배포) 바뀌고, 재배포 때 프로세스가 새로 떠서
# 캐시도 자연히 비워짐 → 파일 읽기·순위표·경기 분석·순위 예측 계산 결과를 메모리에 두고 재사용해도 안전.
# Render 무료 CPU에선 요청마다 다시 계산하느라 순위표 0.5~1.2초, 경기 분석 0.9초씩 걸렸음.
# 주의: 캐시된 dict/list를 그대로 돌려주므로 호출하는 쪽에서 수정하지 말 것(수정이 필요하면 복사본으로).
@lru_cache(maxsize=None)
def _load_json(name):
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 팀별 "오늘 시점" 모델 입력값(누적 ELO·최근 폼·최근 득실 등) — update_data.py의
# build_point_in_time_features가 학습 피처와 같은 정의로 계산해서 저장한 것
with open(os.path.join(MODEL_DIR, "team_state.json"), 'r', encoding='utf-8') as f:
    team_state = json.load(f)

# ── 5대 리그 밖 UCL 팀 (2026-09-28) ──
# 이 팀들은 데이터가 UCL 경기뿐이라, 올 시즌 UCL 일정에 있고 UCL 경기 기록이 8경기 이상인 팀만 예측을 열어줌
# (1경기뿐인 첫 출전 팀은 "예측 데이터 없음" 유지). 이런 경기는 모델이 확률을 과신해서 — 24-25·25-26 UCL
# 110경기 백테스트에서 모델 그대로는 log loss 1.083으로 "리그 평균 비율로만 찍기"(1.038)보다도 나빴음 —
# 확률을 평균 쪽으로 30% 당김(λ=0.3 → 1.024, 정확도 53.6% vs 홈승만 49.1%). 화면엔 "참고용" 표시.
UCL_ONLY_MIN_MATCHES = 8
UCL_ONLY_SHRINK = 0.30
MATCH_BASE_RATES = (0.44, 0.25, 0.31)   # 홈승/무/원정승 리그 평균 비율
def _load_ucl_only_teams():
    cl = (_load_json("schedule.json") or {}).get("CL", [])
    league_teams = set(df_stats['team'])
    cl_teams = {t for m in cl for t in (m["home_team"], m["away_team"])}
    return {t for t in cl_teams - league_teams
            if team_state.get(t, {}).get("games", 0) >= UCL_ONLY_MIN_MATCHES}
ucl_only_teams = _load_ucl_only_teams()

# 팀 이름 매핑 (HTML → API)
TEAM_NAME_MAP = {
    "Inter Milan": "FC Internazionale Milano",
    "Bayern München": "FC Bayern München",
    "Werder Bremen": "SV Werder Bremen",
    "RCD Espanyol": "RCD Espanyol de Barcelona",
    "FC St. Pauli": "FC St. Pauli 1910",
    "Pisa SC": "AC Pisa 1909",
    "Cremonese": "US Cremonese",
    "FC Heidenheim 1846": "1. FC Heidenheim 1846",
    "FC Union Berlin": "1. FC Union Berlin",
    "FSV Mainz 05": "1. FSV Mainz 05",
    "Holstein Kiel": "Holstein Kiel",
    "Hamburger SV": "Hamburger SV",
}

# 로고 딕셔너리에 HTML 팀 이름으로도 추가
# 고화질·최신 엠블럼(team_logos_hd.json, generate_logos_hd.py — 위키백과 인포박스 로고를 400px WebP로, 153팀)이 있으면
# football-data 로고(200px PNG, 옛 버전 섞임) 위에 덮어씀 → /logos·순위표·빅매치·공유 썸네일이 모두 같은 로고
def get_logos_with_mapping():
    try:
        hd = _load_json("team_logos_hd.json") or {}
        if os.environ.get("LOGO_HD_BASE"):   # 로컬 확인용: 배포 전 로고를 로컬 정적 서버에서(예: http://127.0.0.1:8765)
            hd = {t: u.replace("https://www.fotdata-official.com", os.environ["LOGO_HD_BASE"]) for t, u in hd.items()}
        logos = {**(_load_json("team_logos.json") or {}), **hd}
        print(f"로고 수: {len(logos)} (고화질 {len(hd)})")
        for html_name, api_name in TEAM_NAME_MAP.items():
            if api_name in logos:
                logos[html_name] = logos[api_name]
        for t, u in (_load_json("history_logos.json") or {}).items():   # 과거 시즌에만 나오는 옛 팀(위건 등, API-Football 로고 — 5.33)
            logos.setdefault(t, u)
        return logos
    except Exception as e:
        print(f"로고 로드 오류: {e}")
        return {}

print(f"✅ 모델 로드 완료 | 팀 수: {len(df_stats)}")
team_logos_cache = get_logos_with_mapping()

# ── 요청 형식 ──
class MatchRequest(BaseModel):
    home_team: str
    away_team: str

# ── 스코어 예측 (포아송 분포 기반) ──
# 스쿼드/부상자 데이터는 무료 API로는 구할 수 없어서, 대신 팀별 평균 득실점
# (attack_strength/defense_strength — 이미 3시즌 블렌딩된 값)만으로 기대 득점을
# 추정하는 전통적인 축구 분석 기법(Dixon-Coles류의 단순화 버전)을 사용한다.
HOME_GOAL_BOOST = 1.12   # 홈 팀 기대 득점 보정(홈 어드밴티지)
AWAY_GOAL_PENALTY = 0.92  # 원정 팀 기대 득점 보정
MAX_GOALS = 6             # 이보다 큰 스코어는 확률이 미미해 계산에서 제외(재정규화로 보정)

def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def _home_away_gap(lambda_home, lambda_away):
    """포아송(홈,원정) 조합에서 내재적으로 함의되는 (홈승률 - 원정승률)"""
    total_p = home_p = away_p = 0.0
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = poisson_pmf(i, lambda_home) * poisson_pmf(j, lambda_away)
            total_p += p
            if i > j: home_p += p
            elif i < j: away_p += p
    return (home_p - away_p) / total_p

def _solve_lambdas(base_home, base_away, target_gap, iterations=25):
    """
    승/무/패 예측 모델(누적 ELO·최근 폼 등 반영)이 내놓은 승률 격차와
    스코어 예측(포아송)이 내놓는 격차가 서로 다른 모델이라 어긋나는 문제를 보정한다.
    총 기대득점(base_home+base_away, 팀 득실점 스탯 기반의 "경기 페이스")은 그대로 유지한 채,
    홈/원정 배분 비율만 이분탐색으로 조정해서 "포아송이 내재적으로 함의하는 홈-원정 승률차"가
    실제 모델이 내놓은 승률차와 같아지게 만든다 — 이러면 압도적 승률일수록 스코어도
    자연스럽게 압도적으로(예: 3-0, 4-1) 나오게 된다.
    """
    total = base_home + base_away
    lo, hi = -total * 0.49, total * 0.49
    mid = 0.0
    for _ in range(iterations):
        mid = (lo + hi) / 2
        lh = max(0.15, total / 2 + mid)
        la = max(0.15, total / 2 - mid)
        gap = _home_away_gap(lh, la)
        if gap < target_gap:
            lo = mid
        else:
            hi = mid
    return max(0.15, total / 2 + mid), max(0.15, total / 2 - mid)

def predict_score(home_attack, away_defense, away_attack, home_defense, prediction, home_win_prob, away_win_prob):
    base_home = max(0.3, (home_attack + away_defense) / 2 * HOME_GOAL_BOOST)
    base_away = max(0.3, (away_attack + home_defense) / 2 * AWAY_GOAL_PENALTY)
    lambda_home, lambda_away = _solve_lambdas(base_home, base_away, home_win_prob - away_win_prob)

    score_probs = []
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            score_probs.append((i, j, poisson_pmf(i, lambda_home) * poisson_pmf(j, lambda_away)))
    total_p = sum(p for _, _, p in score_probs)  # MAX_GOALS 초과분 잘려나간 것 재정규화

    # 승/무/패 예측(로지스틱 회귀 모델)과 스코어 예측(포아송 분포)은 서로 다른
    # 모델이라, 필터링 없이 그냥 "가장 확률 높은 스코어"만 뽑으면 무승부 스코어
    # (0-0, 1-1 등)가 개별 확률이 커서 상위권을 차지하는 경우가 많다 — 이 경우
    # "홈팀 승리 예측"이라고 해놓고 스코어는 1-1이 뜨는 모순이 생김. 그래서
    # 승/무/패 예측과 같은 결과(홈승이면 홈득점>원정득점 등)의 스코어만 후보로
    # 남겨서, 화면에 보이는 스코어들이 위에 뜨는 예측 배지랑 항상 일치하게 한다.
    def outcome_of(i, j):
        if i > j: return 'home_win'
        if i < j: return 'away_win'
        return 'draw'

    consistent = [s for s in score_probs if outcome_of(s[0], s[1]) == prediction]
    consistent.sort(key=lambda x: x[2], reverse=True)
    top = consistent[:5] if consistent else sorted(score_probs, key=lambda x: x[2], reverse=True)[:5]
    top_scores = [{"score": f"{i}-{j}", "prob": round(p / total_p * 100, 1)} for i, j, p in top]

    return {
        "expected_goals": {"home": round(lambda_home, 2), "away": round(lambda_away, 2)},
        "most_likely": top_scores[0]["score"],
        "top_scores": top_scores,
    }

# ── 예측 설명(explainable) ──
# LogisticRegression 계수 × 스케일된 피처값 = 해당 클래스 로짓에 대한 기여도.
# 모델에는 동일 신호가 이름만 다르게 중복 입력된 피처가 있어(home_attack==home_avg_scored 등),
# 원본 피처 그대로 보여주면 사용자에게 의미 없는 나열이 되므로 축구팬이 이해할 수 있는
# 5개 개념(전력차/최근폼/공격력/수비력/승률/상대전적)으로 묶어서 기여도를 합산한다.
FEATURE_INDEX = {name: i for i, name in enumerate(scaler.feature_names_in_)}
FACTOR_GROUPS = [
    # ELO와 승률은 둘 다 경기 결과의 누적이라 서로 강하게 상관돼 있음 —
    # 따로 두면 모델이 같은 신호를 두 피처에 나눠 담으면서 계수 부호가 서로
    # 반대로 나오는 통계적 아티팩트(다중공선성)가 생겨 "ELO는 불리했다"처럼
    # 오해를 부르는 설명이 됨. 같은 개념(팀 전력)으로 묶어서 합산한다.
    ("팀 전력",   ["home_elo", "away_elo", "elo_diff", "home_win_rate", "away_win_rate", "win_rate_diff"]),
    ("최근 폼",   ["home_form", "away_form", "form_diff"]),
    ("공격력",    ["home_avg_scored", "away_avg_scored", "home_attack", "away_attack"]),
    ("수비력",    ["home_avg_conceded", "away_avg_conceded", "home_defense", "away_defense"]),
    ("상대전적(H2H)", ["h2h_home_rate"]),
]

def compute_prediction_factors(scaled_row, prediction):
    class_idx = {"away_win": "A", "draw": "D", "home_win": "H"}[prediction]
    class_pos = list(lr_model.classes_).index(class_idx)
    coef_row = lr_model.coef_[class_pos]

    raw = []
    for label, feature_names in FACTOR_GROUPS:
        contribution = sum(coef_row[FEATURE_INDEX[f]] * scaled_row[FEATURE_INDEX[f]] for f in feature_names)
        raw.append((label, contribution))

    total_abs = sum(abs(c) for _, c in raw) or 1.0
    factors = [
        {
            "label": label,
            "direction": "support" if contribution >= 0 else "against",
            "influence_pct": round(abs(contribution) / total_abs * 100, 1),
        }
        for label, contribution in raw
    ]
    factors.sort(key=lambda f: f["influence_pct"], reverse=True)
    return factors

# ── 엔드포인트 ──
@app.get("/")
def root():
    return {"message": "FotData API 서버 작동 중!", "version": "1.0.0"}

@app.get("/teams")
def get_teams():
    """사용 가능한 팀 목록 반환"""
    teams = sorted(df_stats['team'].tolist())
    # ucl_teams: 5대 리그 밖이지만 UCL 경기 기록으로 예측 가능한 팀(참고용 예측)
    return {"teams": teams, "count": len(teams), "ucl_teams": sorted(ucl_only_teams)}

# ── 모델 입력 생성 (/predict와 순위 예측이 같이 씀) ──
_h2h_cache = None
def _h2h_home_rate(home, away, n=10):
    """최근 n번 맞대결에서 home 팀 승률 (학습 피처 h2h_home_rate와 같은 정의)"""
    global _h2h_cache
    if _h2h_cache is None:
        _h2h_cache = {}
        for m in df_matches_all.sort_values('date').itertuples():
            winner = m.home_team if m.result == 'H' else (m.away_team if m.result == 'A' else None)
            _h2h_cache.setdefault(tuple(sorted((m.home_team, m.away_team))), []).append(winner)
    rec = _h2h_cache.get(tuple(sorted((home, away))), [])[-n:]
    return round(sum(w == home for w in rec) / len(rec), 3) if rec else 0.33

def _feature_row(home, away):
    hs, as_ = team_state[home], team_state[away]
    return {
        'home_elo':          hs['elo'],
        'away_elo':          as_['elo'],
        'elo_diff':          hs['elo'] - as_['elo'],
        'home_form':         hs['form'],
        'away_form':         as_['form'],
        'form_diff':         hs['form'] - as_['form'],
        'home_avg_scored':   hs['avg_scored'],
        'away_avg_scored':   as_['avg_scored'],
        'home_avg_conceded': hs['avg_conceded'],
        'away_avg_conceded': as_['avg_conceded'],
        'home_attack':       hs['attack'],
        'away_attack':       as_['attack'],
        'home_defense':      hs['defense'],
        'away_defense':      as_['defense'],
        'home_win_rate':     hs['win_rate'],
        'away_win_rate':     as_['win_rate'],
        'win_rate_diff':     hs['win_rate'] - as_['win_rate'],
        'h2h_home_rate':     _h2h_home_rate(home, away),
    }

def _predict_hda(pairs):
    """[(home, away), ...] → 각 경기 [홈승, 무, 원정승] 확률 (한 번에 계산)"""
    X = scaler.transform(pd.DataFrame([_feature_row(h, a) for h, a in pairs]))
    P = lr_model.predict_proba(X)
    cols = [list(lr_model.classes_).index(c) for c in ('H', 'D', 'A')]
    return P[:, cols]

def _display_stats(team):
    """결과 화면·스코어 예측용 공격/수비/승률. 5대 리그 팀은 블렌딩 값(team_stats.csv),
    UCL 전용 팀은 team_state의 최근 경기(=UCL 경기) 값. 둘 다 아니면 None"""
    row = df_stats[df_stats['team'] == team]
    if not row.empty:
        r = row.iloc[0]
        return {'attack_strength': r['attack_strength'], 'defense_strength': r['defense_strength'], 'win_rate': r['win_rate']}
    if team in ucl_only_teams:
        s = team_state[team]
        return {'attack_strength': s['attack'], 'defense_strength': s['defense'], 'win_rate': s['win_rate']}
    return None

# ── 일정 탭 AI 예측: 리그의 남은 경기 전체를 한 번에 (2026-09-30) ──
# 경기 하나 예측(/predict)과 같은 모델·같은 입력(_feature_row)·같은 UCL 전용 팀 보정·같은 반올림이라 숫자가 똑같음
# (일정에서 본 확률과 눌러서 연 예측이 달라 보이면 안 되므로). 예측할 수 없는 팀(첫 출전 UCL 팀 등)이 낀 경기는 뺌
@app.get("/predict/schedule/{league_code}")
def predict_schedule(league_code: str):
    code = league_code.upper()
    base = _schedule_predictions(code)
    live = _live_overlay()
    if not live:
        return base
    # 오늘 끝난 경기: 새벽 업데이트 전이라도 경기 전에 기록해둔 예측(prediction_log)으로 바로 적중 여부를 매김
    log, extra, done = _load_json("prediction_log.json") or {}, [], set()
    for m in (_load_json("schedule.json") or {}).get(code, []):
        v = live.get(_live_key(m["home_team"], m["away_team"], m["date"]))
        if m.get("status") in DONE_STATUSES or not v or v["status"] not in DONE_STATUSES or v["home_goals"] is None:
            continue
        done.add((m["home_team"], m["away_team"], m["date"]))
        e = log.get(f"{code}|{m['home_team']}|{m['away_team']}|{m['date'][:10]}")
        if not e or e.get("predicted") is None:
            continue
        hg, ag = v["home_goals"], v["away_goals"]
        actual = "home_win" if hg > ag else "away_win" if hg < ag else "draw"
        extra.append({"home_team": m["home_team"], "away_team": m["away_team"], "date": m["date"],
                      "p": [e["home_win_prob"], e["draw_prob"], e["away_win_prob"]],
                      "predicted": e["predicted"], "correct": actual == e["predicted"]})
    if not done:
        return base
    return {**base, "predictions": [x for x in base["predictions"] if (x["home_team"], x["away_team"], x["date"]) not in done],
            "results": base["results"] + extra}

@lru_cache(maxsize=8)
def _schedule_predictions(code):
    matches = (_load_json("schedule.json") or {}).get(code)
    if matches is None:
        raise HTTPException(status_code=404, detail="해당 리그 일정 없음")
    keep, pairs = [], []
    for m in matches:
        if m.get("status") in ("FINISHED", "AWARDED", "CANCELLED"):
            continue
        h, a = TEAM_NAME_MAP.get(m["home_team"], m["home_team"]), TEAM_NAME_MAP.get(m["away_team"], m["away_team"])
        if _display_stats(h) is None or _display_stats(a) is None or h not in team_state or a not in team_state:
            continue
        keep.append(m); pairs.append((h, a))
    out = []
    if pairs:
        base = dict(zip(('H', 'D', 'A'), MATCH_BASE_RATES))
        for m, (h, a), p in zip(keep, pairs, _predict_hda(pairs)):
            pd_ = dict(zip(('H', 'D', 'A'), p))
            limited = h in ucl_only_teams or a in ucl_only_teams
            if limited:
                pd_ = {c: (1 - UCL_ONLY_SHRINK) * v + UCL_ONLY_SHRINK * base[c] for c, v in pd_.items()}
            out.append({"home_team": m["home_team"], "away_team": m["away_team"], "date": m["date"],
                        "p": [round(float(pd_[c]), 3) for c in ('H', 'D', 'A')], "limited": limited})
    # 끝난 경기: 경기 전에 트랙레코드(prediction_log.json)에 실제로 기록해둔 예측만 — 결과를 보고 나서
    # 지금 모델로 다시 계산한 "사후 예측"은 보여주지 않음(적중률을 부풀릴 수 있어서)
    log = _load_json("prediction_log.json") or {}
    results = []
    for m in matches:
        if m.get("status") not in ("FINISHED", "AWARDED"):
            continue
        e = log.get(f"{code}|{m['home_team']}|{m['away_team']}|{m['date'][:10]}")
        if not e or e.get("correct") is None:
            continue
        results.append({"home_team": m["home_team"], "away_team": m["away_team"], "date": m["date"],
                        "p": [e["home_win_prob"], e["draw_prob"], e["away_win_prob"]],
                        "predicted": e["predicted"], "correct": bool(e["correct"])})
    return {"league": code, "predictions": out, "results": results}

@app.post("/predict")
def predict_match(req: MatchRequest):
    """경기 결과 예측"""
    # 팀 이름 매핑
    home_team = TEAM_NAME_MAP.get(req.home_team, req.home_team)
    away_team = TEAM_NAME_MAP.get(req.away_team, req.away_team)
    
    h, a = _display_stats(home_team), _display_stats(away_team)
    if h is None:
        raise HTTPException(status_code=404, detail=f"팀을 찾을 수 없습니다: {home_team}")
    if a is None:
        raise HTTPException(status_code=404, detail=f"팀을 찾을 수 없습니다: {away_team}")
    limited = home_team in ucl_only_teams or away_team in ucl_only_teams

    # 모델 입력은 학습 때와 같은 정의의 값(team_state.json)을 그대로 씀. 예전엔 여기서
    # 블렌딩 승률로 ELO(+prestige+홈 어드밴티지 감쇠)와 폼을 재구성했는데, 모델이 학습한
    # 입력과 의미가 달라서 강팀 홈경기를 과소평가하고 무승부를 과대평가했음(2026-09-27 수정)
    for t in (home_team, away_team):
        if t not in team_state:
            raise HTTPException(status_code=404, detail=f"예측 데이터가 없는 팀입니다: {t}")
    input_data = pd.DataFrame([_feature_row(home_team, away_team)])

    input_scaled = scaler.transform(input_data)
    proba = lr_model.predict_proba(input_scaled)[0]
    classes = lr_model.classes_
    proba_dict = dict(zip(classes, proba))
    if limited:   # UCL 경기 기록만 있는 팀 — 과신 보정(위 UCL_ONLY_SHRINK 설명)
        base = dict(zip(('H', 'D', 'A'), MATCH_BASE_RATES))
        proba_dict = {c: (1 - UCL_ONLY_SHRINK) * p + UCL_ONLY_SHRINK * base[c] for c, p in proba_dict.items()}

    h_prob = round(float(proba_dict.get('H', 0)), 3)
    d_prob = round(float(proba_dict.get('D', 0)), 3)
    a_prob = round(float(proba_dict.get('A', 0)), 3)

    if h_prob == max(h_prob, d_prob, a_prob):
        prediction = "home_win"
    elif a_prob == max(h_prob, d_prob, a_prob):
        prediction = "away_win"
    else:
        prediction = "draw"

    score_prediction = predict_score(
        home_attack=float(h['attack_strength']), away_defense=float(a['defense_strength']),
        away_attack=float(a['attack_strength']), home_defense=float(h['defense_strength']),
        prediction=prediction, home_win_prob=h_prob, away_win_prob=a_prob,
    )

    explanation = compute_prediction_factors(input_scaled[0], prediction)

    return {
        "home_team":   req.home_team,
        "away_team":   req.away_team,
        "prediction":  prediction,
        "probabilities": {
            "home_win": h_prob,
            "draw":     d_prob,
            "away_win": a_prob,
        },
        "score_prediction": score_prediction,
        "explanation": explanation,
        # True면 한쪽 이상이 5대 리그 밖 팀이라 UCL 경기 기록만으로 계산한 참고용 예측
        "limited": limited,
        "home_stats": {
            "attack":   round(float(h['attack_strength']), 3),
            "defense":  round(float(h['defense_strength']), 3),
            "win_rate": round(float(h['win_rate']), 3),
            **_stat_context(home_team),   # 결과 화면에서 "경기당 득점 1.85 · 리그 3위"처럼 읽히게(2026-09-29)
        },
        "away_stats": {
            "attack":   round(float(a['attack_strength']), 3),
            "defense":  round(float(a['defense_strength']), 3),
            "win_rate": round(float(a['win_rate']), 3),
            **_stat_context(away_team),
        }
    }

@lru_cache(maxsize=1)
def _team_league_map():
    """팀 → 가장 최근 리그(UCL 제외) 경기의 리그 코드, 전 팀을 한 번에(팀마다 _team_current_league를 부르면 느려서)"""
    m = df_matches_all[df_matches_all['league'] != 'CL']
    both = pd.concat([m[['date', 'league', 'home_team']].rename(columns={'home_team': 'team'}),
                      m[['date', 'league', 'away_team']].rename(columns={'away_team': 'team'})])
    return both.sort_values('date').groupby('team')['league'].last().to_dict()

@lru_cache(maxsize=8)
def _league_display_ranks(league):
    """예측 결과 화면 지표용: 같은 리그 팀들 사이 경기당 득점(공격) 순위·실점(수비) 순위와 리그 평균(블렌딩 값 기준)"""
    lm = _team_league_map()
    rows = df_stats[df_stats['team'].map(lambda t: lm.get(t) == league)]
    if rows.empty:
        return None
    atk = rows.sort_values('attack_strength', ascending=False)['team'].tolist()
    dfn = rows.sort_values('defense_strength')['team'].tolist()
    return {"n": len(rows), "atk": {t: i + 1 for i, t in enumerate(atk)}, "def": {t: i + 1 for i, t in enumerate(dfn)},
            "avg_atk": round(float(rows['attack_strength'].mean()), 3), "avg_def": round(float(rows['defense_strength'].mean()), 3)}

def _stat_context(team):
    """/predict home_stats·away_stats에 붙는 리그 내 순위·평균(5대 리그 팀만, UCL 전용 팀은 None)"""
    lg = _team_league_map().get(team)
    r = _league_display_ranks(lg) if lg else None
    if not r or team not in r["atk"]:
        return {}
    return {"league": lg, "league_size": r["n"], "attack_rank": r["atk"][team], "defense_rank": r["def"][team],
            "league_avg_attack": r["avg_atk"], "league_avg_defense": r["avg_def"]}

@app.get("/team/{team_name}")
def get_team_stats(team_name: str):
    """특정 팀 스탯 조회 (team_stats.csv는 3시즌 블렌딩 값이라 승/무/패/승점/득실차 같은
    실경기 집계치는 없음 — win_rate/attack/defense/prestige만 존재)"""
    team = df_stats[df_stats['team'] == team_name]
    if team.empty:
        raise HTTPException(status_code=404, detail=f"팀을 찾을 수 없습니다: {team_name}")

    t = team.iloc[0]
    return {
        "team":             t['team'],
        "games":            int(t['games']),
        "attack_strength":  round(float(t['attack_strength']), 3),
        "defense_strength": round(float(t['defense_strength']), 3),
        "win_rate":         round(float(t['win_rate']), 3),
        "prestige":         round(float(t['prestige']), 1),
    }

@app.get("/teams/prestige")
def get_teams_prestige():
    """전체 팀의 체급(prestige) 목록 — 팀 검색창 기본 정렬(파워랭킹순)용"""
    return {
        row['team']: round(float(row['prestige']), 1)
        for _, row in df_stats.iterrows()
    }
# ── 로고 API 추가 ──
# ── 로고 API 추가 ──
@app.get("/logos")
def get_logos():
    return team_logos_cache

# ── 전체 경기 데이터 로드 (H2H, 폼, 순위 계산용) ──
import json
from datetime import datetime, timezone

df_matches_all = pd.read_csv(os.path.join(MODEL_DIR, "all_matches.csv"))
df_matches_all['date'] = pd.to_datetime(df_matches_all['date'])

print(f"✅ 전체 경기 데이터 로드: {len(df_matches_all)}경기")
# 과거 시즌(2010-11~22-23, API-Football — update_data.fetch_history_seasons): 순위표·일정·팀 통계·시즌 고르기에만 씀.
# 예측 모델 입력·맞대결·최근 폼은 all_matches.csv(df_matches_all) 그대로 — 둘을 섞지 않음(5.33)
_hp = os.path.join(MODEL_DIR, "history_matches.csv")
df_history = pd.read_csv(_hp) if os.path.exists(_hp) else pd.DataFrame(columns=list(df_matches_all.columns) + ["stage", "kickoff"])
df_history['date'] = pd.to_datetime(df_history['date'])
df_seasons = pd.concat([df_history, df_matches_all], ignore_index=True)
print(f"✅ 과거 시즌 경기: {len(df_history)}경기")

# ── 리그 코드 매핑 ──
LEAGUE_MAP = {
    "PL":  "Premier League",
    "PD":  "LaLiga",
    "BL1": "Bundesliga",
    "SA":  "Serie A",
    "FL1": "Ligue 1",
    "CL":  "Champions League",
}

# ── 순위표 API ──
@app.get("/standings/{league_code}")
def get_standings(league_code: str, season: str = "current", view: str = "all"):
    if view in ("home", "away", "form"):
        return _standings_view(league_code.upper(), season, view)
    return _standings(league_code.upper(), season)

@lru_cache(maxsize=64)
def _standings_view(league_code, season, view):
    """순위표 보기 전환(2026-09-29): home = 홈 경기만, away = 원정 경기만, form = 팀마다 최근 5경기만으로 낸 순위표.
    공식 순서·구역·감점은 전체 순위표에만 해당해서 여기선 안 씀(승점 → 득실차 → 득점 순)"""
    base = _standings(league_code, season)   # 리그·시즌 검증과 이름은 전체 순위표와 공유
    yr = _season_year(season)
    f = (df_seasons['league'] == league_code) & (df_seasons['season'] == yr)   # 시즌 값으로(날짜로 자르면 19-20 세리에A처럼 8월에 끝난 시즌이 다음 시즌에 섞임)
    if league_code == 'CL':
        f = f & (df_seasons['date'] < pd.Timestamp(f'{yr + 1}-02-01'))
    df = df_seasons[f].sort_values('date')
    logos = team_logos_cache
    teams = [r["team"] for r in base["standings"]]
    rows = []
    for team in teams:
        if view == "home":
            m = df[df['home_team'] == team]
        elif view == "away":
            m = df[df['away_team'] == team]
        else:
            m = df[(df['home_team'] == team) | (df['away_team'] == team)].tail(5)
        w = d = l = gf = ga = 0
        form = []
        for _, x in m.iterrows():
            home = x['home_team'] == team
            g1, g2 = (x['home_goals'], x['away_goals']) if home else (x['away_goals'], x['home_goals'])
            gf += int(g1); ga += int(g2)
            r = 'W' if g1 > g2 else 'L' if g1 < g2 else 'D'
            w += r == 'W'; d += r == 'D'; l += r == 'L'
            form.append(r)
        rows.append({"team": team, "logo": logos.get(team, ''), "played": w + d + l, "wins": w, "draws": d, "losses": l,
                     "points": w * 3 + d, "gf": gf, "ga": ga, "gd": gf - ga, "form": form[-5:]})
    rows.sort(key=lambda r: (-r["points"], -r["gd"], -r["gf"], r["team"]))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return {"league": base["league"], "standings": rows, "official": False, "view": view}

@lru_cache(maxsize=32)
def _standings(league_code: str, season: str):
    league_name = LEAGUE_MAP.get(league_code.upper())
    if not league_name:
        raise HTTPException(status_code=404, detail="리그를 찾을 수 없습니다")

    yr = _season_year(season)   # current/previous 또는 연도(23-24~26-27, all_matches.csv가 담는 4시즌)
    cutoff = pd.Timestamp(f'{yr}-08-01')
    end = pd.Timestamp(f'{yr + 1}-08-01')
    league_name = f"{league_name} ({yr}-{(yr + 1) % 100:02d})"

    filters = (df_seasons['league'] == league_code.upper()) & (df_seasons['season'] == yr)   # 시즌 값으로(위와 같은 이유)
    if league_code.upper() == 'CL':
        if yr <= 2023:   # 23-24까지는 조별리그(4팀×8조)라 한 줄 순위표가 없음
            raise HTTPException(status_code=404, detail="조별리그 시즌")
        filters = filters & (df_seasons['date'] < pd.Timestamp(f'{yr + 1}-02-01'))   # 리그 스테이지만
    league_df = df_seasons[filters].copy()


    if league_df.empty:
        raise HTTPException(status_code=404, detail="데이터 없음")

    logos = team_logos_cache

    # 홈 스탯
    home_stats = league_df.groupby('home_team').agg(
        home_games=('result', 'count'),
        home_wins=('result', lambda x: (x=='H').sum()),
        home_draws=('result', lambda x: (x=='D').sum()),
        home_gf=('home_goals', 'sum'),
        home_ga=('away_goals', 'sum'),
    ).reset_index().rename(columns={'home_team': 'team'})

    # 원정 스탯
    away_stats = league_df.groupby('away_team').agg(
        away_games=('result', 'count'),
        away_wins=('result', lambda x: (x=='A').sum()),
        away_draws=('result', lambda x: (x=='D').sum()),
        away_gf=('away_goals', 'sum'),
        away_ga=('home_goals', 'sum'),
    ).reset_index().rename(columns={'away_team': 'team'})

    # 합치기
    merged = pd.merge(home_stats, away_stats, on='team', how='outer').fillna(0)
    merged['games']  = merged['home_games'] + merged['away_games']
    merged['wins']   = merged['home_wins'] + merged['away_wins']
    merged['draws']  = merged['home_draws'] + merged['away_draws']
    merged['losses'] = merged['games'] - merged['wins'] - merged['draws']
    merged['points'] = merged['wins'] * 3 + merged['draws']
    merged['gf']     = merged['home_gf'] + merged['away_gf']
    merged['ga']     = merged['home_ga'] + merged['away_ga']
    merged['gd']     = merged['gf'] - merged['ga']
    # 끝난 시즌: 공식 기록(영문 위키백과 시즌 표 — update_data.py fetch_season_zones)의 승점 감점·최종 순위·유럽 대항전/강등 구역
    # (23-24 에버턴 −8·노팅엄 −4, 라리가·세리에A는 동률이면 맞대결 우선이라 득실차 정렬과 순서가 다를 수 있음)
    official = ((_load_json("season_zones.json") or {}).get(league_code.upper()) or {}).get(str(yr)) \
        or ((_load_json("history_standings.json") or {}).get(league_code.upper()) or {}).get(str(yr))
    zones = {}
    if official:
        for t, pts in official.get("adjust", {}).items():
            merged.loc[merged['team'] == t, 'points'] += pts
        pos = {t: v["pos"] for t, v in official["teams"].items()}
        zones = {t: v.get("zone") for t, v in official["teams"].items()}
        merged['_pos'] = merged['team'].map(pos).fillna(99)
        merged = merged.sort_values(['_pos', 'points', 'gd', 'gf'], ascending=[True, False, False, False]).reset_index(drop=True)
    else:
        merged = merged.sort_values(['points','gd','gf'], ascending=False).reset_index(drop=True)

    rows = []
    for i, row in merged.iterrows():
        team = row['team']

        # 최근 5경기 폼
        team_matches = league_df[
            (league_df['home_team']==team) | (league_df['away_team']==team)
        ].sort_values('date').tail(5)

        form = []
        for _, m in team_matches.iterrows():
            if m['home_team'] == team:
                form.append('W' if m['result']=='H' else ('D' if m['result']=='D' else 'L'))
            else:
                form.append('W' if m['result']=='A' else ('D' if m['result']=='D' else 'L'))

        rows.append({
            "rank":   int(i + 1),
            "team":   team,
            "logo":   logos.get(team, ''),
            "played": int(row['games']),
            "wins":   int(row['wins']),
            "draws":  int(row['draws']),
            "losses": int(row['losses']),
            "points": int(row['points']),
            "gf":     int(row['gf']),
            "ga":     int(row['ga']),
            "gd":     int(row['gd']),
            "form":   form,
        })
        if official:
            rows[-1]["zone"] = zones.get(team)
            if official.get("adjust", {}).get(team):
                rows[-1]["deduction"] = official["adjust"][team]

    return {"league": league_name, "standings": rows, "official": bool(official),
            "source": official.get("source") if official else None}

# ── UCL 토너먼트 API ──
CURRENT_SEASON_YEAR = 2026   # 시즌 전환 때 같이 바꿀 것(11번 체크리스트)

def _season_year(season):
    """"current"/"previous"/"2024" 같은 값 → 시즌 시작 연도"""
    if season in (None, "", "current"):
        return CURRENT_SEASON_YEAR
    if season == "previous":
        return CURRENT_SEASON_YEAR - 1
    try:
        return int(season)
    except ValueError:
        raise HTTPException(status_code=400, detail="season은 current/previous 또는 연도(예: 2024)")

@app.get("/ucl/tournament")
def get_ucl_tournament(season: str = "current"):
    """시즌별 UCL 토너먼트 — ?season=2025. 파일이 예전 형식(한 시즌만)이면 그걸 2025 시즌으로 취급"""
    data = _load_json("ucl_tournament.json")
    if data is None:
        raise HTTPException(status_code=404, detail="UCL 토너먼트 데이터 없음")
    seasons = data["seasons"] if "seasons" in data else {"2025": data}
    seasons = {**((_load_json("history_ucl.json") or {}).get("seasons") or {}), **seasons}   # 11-12~22-23(API-Football, 5.33)
    yr = str(_season_year(season))
    stages = seasons.get(yr)
    if stages is None:
        return {"season": int(yr), "available": sorted(seasons), "stages": None}
    return {"season": int(yr), "available": sorted(seasons), "stages": stages, **stages}
   
@app.get("/ucl/groups")
def get_ucl_groups(season: str = "2023"):
    """23-24까지의 챔스 조별리그 순위표(조마다 4팀) — update_data.py가 ucl_tournament.json에 GROUPS로 저장한 경기로 계산.
    순위는 UEFA 규정대로 승점 → 맞대결 승점·득실·득점 → 전체 득실·득점. 1·2위 16강, 3위 유로파리그"""
    data = _load_json("ucl_tournament.json") or {}
    seasons = {**((_load_json("history_ucl.json") or {}).get("seasons") or {}), **data.get("seasons", {})}
    yr = str(_season_year(season))
    groups = (seasons.get(yr) or {}).get("GROUPS")
    if not groups:
        return {"season": int(yr), "groups": None}
    logos = team_logos_cache
    out = {}
    for g, ms in groups.items():
        done = [m for m in ms if m.get("home_goals") is not None]
        teams = sorted({t for m in ms for t in (m["home_team"], m["away_team"])})
        def table(sub, only=None):
            st = {t: {"played": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0, "points": 0, "form": []} for t in (only or teams)}
            for m in sub:
                h, a, hg, ag = m["home_team"], m["away_team"], m["home_goals"], m["away_goals"]
                if only and (h not in only or a not in only):
                    continue
                for t, f, ag_ in ((h, hg, ag), (a, ag, hg)):
                    r = st[t]; r["played"] += 1; r["gf"] += f; r["ga"] += ag_
                    res = "W" if f > ag_ else "D" if f == ag_ else "L"
                    r["wins" if res == "W" else "draws" if res == "D" else "losses"] += 1
                    r["points"] += 3 if res == "W" else 1 if res == "D" else 0
                    r["form"].append(res)
            return st
        full = table(done)
        def key(t):
            tied = [x for x in teams if full[x]["points"] == full[t]["points"]]
            h2h = table(done, tied)[t] if len(tied) > 1 else {"points": 0, "gf": 0, "ga": 0}
            return (-full[t]["points"], -h2h["points"], -(h2h["gf"] - h2h["ga"]), -h2h["gf"],
                    -(full[t]["gf"] - full[t]["ga"]), -full[t]["gf"])
        rows = []
        for i, t in enumerate(sorted(teams, key=key)):
            r = full[t]
            rows.append({"rank": i + 1, "team": t, "logo": logos.get(t, ""), **{k: r[k] for k in ("played", "wins", "draws", "losses", "gf", "ga", "points")},
                         "gd": r["gf"] - r["ga"], "form": r["form"][-6:],
                         "zone": "ko" if i < 2 else "el" if i == 2 else None})
        out[g] = rows
    return {"season": int(yr), "groups": out}

# ── H2H API ──
@app.get("/h2h")
def get_h2h(home_team: str, away_team: str, limit: int = 10):
    """역대 맞대결(2026-10-07 — 예전엔 최근 4시즌만): 과거 시즌 경기(2010-11~, 리그·챔스 본선)까지 합친 df_seasons에서.
    승무패 요약·total은 전체, matches는 최근 limit경기(0이면 전부). 예측 모델 입력(H2H 피처)은 그대로 4시즌 데이터"""
    df_all = df_seasons[
        ((df_seasons['home_team']==home_team) & (df_seasons['away_team']==away_team)) |
        ((df_seasons['home_team']==away_team) & (df_seasons['away_team']==home_team))
    ].dropna(subset=['home_goals', 'away_goals']).drop_duplicates(subset=['date', 'home_team', 'away_team']).sort_values('date', ascending=False)

    if df_all.empty:
        return {"home_team": home_team, "away_team": away_team, "matches": [], "total": 0, "summary": {"home_wins":0,"draws":0,"away_wins":0}}

    home_wins = away_wins = draws = 0
    matches = []
    show = len(df_all) if not limit else limit

    for i, (_, row) in enumerate(df_all.iterrows()):
        is_home = row['home_team'] == home_team
        result = row['result']

        if result == 'D':
            draws += 1
            outcome = 'D'
        elif (result == 'H' and is_home) or (result == 'A' and not is_home):
            home_wins += 1
            outcome = 'W'
        else:
            away_wins += 1
            outcome = 'L'

        if i >= show:
            continue
        matches.append({
            "date":       str(row['date'].date()),
            "home_team":  row['home_team'],
            "away_team":  row['away_team'],
            "home_goals": int(row['home_goals']) if pd.notna(row['home_goals']) else 0,
            "away_goals": int(row['away_goals']) if pd.notna(row['away_goals']) else 0,
            "result":     outcome,
        })

    return {
        "home_team": home_team,
        "away_team": away_team,
        "total": len(df_all),
        "since": str(df_all['date'].min().date()),
        "summary": {
            "home_wins": home_wins,
            "draws":     draws,
            "away_wins": away_wins,
        },
        "matches": matches
    }


# ── 팀 폼 API ──
@app.get("/form/{team_name}")
def get_team_form(team_name: str, n: int = 5):
    team_matches = df_matches_all[
        (df_matches_all['home_team']==team_name) |
        (df_matches_all['away_team']==team_name)
    ].sort_values('date', ascending=False).head(n)

    if team_matches.empty:
        raise HTTPException(status_code=404, detail=f"팀을 찾을 수 없습니다: {team_name}")

    form = []
    for _, row in team_matches.iterrows():
        is_home = row['home_team'] == team_name
        result = row['result']

        if result == 'D':
            outcome = 'D'
        elif (result == 'H' and is_home) or (result == 'A' and not is_home):
            outcome = 'W'
        else:
            outcome = 'L'

        form.append({
            "date":      str(row['date'].date()),
            "home_team": row['home_team'],
            "away_team": row['away_team'],
            "home_goals": int(row['home_goals']) if pd.notna(row['home_goals']) else 0,
            "away_goals": int(row['away_goals']) if pd.notna(row['away_goals']) else 0,
            "result":    outcome,
        })

    return {"team": team_name, "form": form}

# ── 경기 분석 API (예측 결과 화면 맞대결 카드 아래: 순위·홈/원정 성적·경기 성향·파워 레이팅 추이) ──
INSIGHT_N = 10   # 홈/원정 성적·경기 성향 집계 경기 수 (이번 시즌만 보면 홈 경기가 2~3개뿐이라 시즌을 넘어서 봄)

def _team_goals(df, team):
    """df의 각 경기에서 team 기준 (득점, 실점) 시리즈"""
    is_home = df['home_team'] == team
    gf = df['home_goals'].where(is_home, df['away_goals'])
    ga = df['away_goals'].where(is_home, df['home_goals'])
    return gf, ga

def _team_current_league(team):
    """가장 최근 리그(UCL 제외) 경기의 리그 코드"""
    league_matches = df_matches_all[
        (df_matches_all['league'] != 'CL') &
        ((df_matches_all['home_team'] == team) | (df_matches_all['away_team'] == team))
    ]
    if league_matches.empty:
        return None
    return league_matches.sort_values('date').iloc[-1]['league']

@lru_cache(maxsize=512)
def _team_insight(team, venue):
    league = _team_current_league(team)

    standing = None
    if league:
        try:
            table = get_standings(league)['standings']
            row = next((r for r in table if r['team'] == team), None)
            if row:
                standing = {k: row[k] for k in ('rank', 'played', 'wins', 'draws', 'losses', 'points', 'gd')}
                standing['total'] = len(table)
        except HTTPException:
            pass

    # 홈팀은 최근 홈 리그 경기, 원정팀은 최근 원정 리그 경기
    venue_df = df_matches_all[
        (df_matches_all['league'] != 'CL') & (df_matches_all[f'{venue}_team'] == team)
    ].sort_values('date').tail(INSIGHT_N)
    gf, ga = _team_goals(venue_df, team)
    venue_record = {
        "n":        len(venue_df),
        "wins":     int((gf > ga).sum()),
        "draws":    int((gf == ga).sum()),
        "losses":   int((gf < ga).sum()),
        "scored":   round(float(gf.mean()), 2) if len(venue_df) else None,
        "conceded": round(float(ga.mean()), 2) if len(venue_df) else None,
    }

    # 경기 성향: 대회 구분 없이 최근 경기
    recent_df = df_matches_all[
        (df_matches_all['home_team'] == team) | (df_matches_all['away_team'] == team)
    ].sort_values('date').tail(INSIGHT_N)
    rgf, rga = _team_goals(recent_df, team)
    total = rgf + rga
    tendency = {
        "n":            len(recent_df),
        "total_goals":  round(float(total.mean()), 2) if len(recent_df) else None,
        "over_2_5":     int((total > 2.5).sum()),
        "btts":         int(((rgf > 0) & (rga > 0)).sum()),
        "clean_sheets": int((rga == 0).sum()),
    }

    state = team_state.get(team, {})
    power = {
        "elo":     state.get('elo'),
        "history": [{"date": d, "elo": e} for d, e in state.get('elo_history', [])],
    }

    return {
        "team":        team,
        "league":      league,
        "league_name": LEAGUE_MAP.get(league) if league else None,
        "standing":    standing,
        "venue":       venue_record,
        "tendency":    tendency,
        "power":       power,
    }

@app.get("/match/insights")
def get_match_insights(home_team: str, away_team: str):
    home_team = TEAM_NAME_MAP.get(home_team, home_team)
    away_team = TEAM_NAME_MAP.get(away_team, away_team)
    for t in (home_team, away_team):
        if t not in team_state:
            raise HTTPException(status_code=404, detail=f"팀을 찾을 수 없습니다: {t}")
    return {
        "home": _team_insight(home_team, 'home'),
        "away": _team_insight(away_team, 'away'),
        "window": INSIGHT_N,
    }

# ── 팀 통계 API (팀 정보 모달 "팀 통계" 탭, 2026-09-29) ──
# 예전 탭은 예측 모델용 블렌딩 값(승률·공격력·수비력·prestige)만 보여줘서 실제 기록과 달랐음 → 실제 리그 경기 기록으로
# 시즌별(23-24~26-27) 지표 + 같은 리그 안 순위·리그 평균, 홈/원정, 상대 수준별, 라운드별 순위 변동, AI 파워 레이팅
STAT_SEASONS = {int(y): f"{int(y) % 100:02d}-{(int(y) + 1) % 100:02d}" for y in sorted(df_seasons.loc[df_seasons['league'] != 'CL', 'season'].dropna().unique())}   # 2010-11~(과거 시즌 포함)
# (키, 높을수록 좋은가) — 리그 안 순위 계산용
STAT_KEYS = [("ppg", True), ("gf_pg", True), ("ga_pg", False), ("cs_pct", True), ("fts_pct", False),
             ("btts_pct", None), ("over25_pct", None), ("home_ppg", True), ("away_ppg", True), ("win_pct", True)]

def _long_matches(df):
    """경기 한 줄 → 팀 관점 두 줄(team, opp, gf, ga, home 여부)"""
    h = pd.DataFrame({"date": df['date'], "team": df['home_team'], "opp": df['away_team'],
                      "gf": df['home_goals'], "ga": df['away_goals'], "home": True})
    a = pd.DataFrame({"date": df['date'], "team": df['away_team'], "opp": df['home_team'],
                      "gf": df['away_goals'], "ga": df['home_goals'], "home": False})
    out = pd.concat([h, a], ignore_index=True)
    out['pts'] = np.where(out.gf > out.ga, 3, np.where(out.gf == out.ga, 1, 0))
    return out.sort_values('date', kind='stable')

def _table(long_df):
    t = long_df.groupby('team').agg(pts=('pts', 'sum'), gf=('gf', 'sum'), ga=('ga', 'sum'), p=('pts', 'size'))
    t['gd'] = t.gf - t.ga
    t = t.sort_values(['pts', 'gd', 'gf'], ascending=False)
    t['rank'] = range(1, len(t) + 1)
    return t

@lru_cache(maxsize=None)   # 5대 리그 × 17시즌 = 85개 — 예전 64개 한도로는 과거 시즌을 넣은 뒤 캐시가 계속 밀려나 팀 통계 첫 호출이 20초 걸렸음
def _league_season_stats(league, season):
    df = df_seasons[(df_seasons['league'] == league) & (df_seasons['season'] == season)]
    df = df.dropna(subset=['home_goals', 'away_goals'])
    if df.empty:
        return None
    L = _long_matches(df)
    table = _table(L)
    n = len(table)
    top_half = set(table.index[: n // 2])
    rows = {}
    for team, g in L.groupby('team'):
        hm, aw = g[g.home], g[~g.home]
        w, d, l = int((g.gf > g.ga).sum()), int((g.gf == g.ga).sum()), int((g.gf < g.ga).sum())
        pct = lambda mask: round(float(mask.mean()) * 100, 1)
        vs_top, vs_bot = g[g.opp.isin(top_half)], g[~g.opp.isin(top_half)]
        rows[team] = {
            "played": len(g), "wins": w, "draws": d, "losses": l,
            "points": int(g.pts.sum()), "gf": int(g.gf.sum()), "ga": int(g.ga.sum()), "gd": int(g.gf.sum() - g.ga.sum()),
            "rank": int(table.loc[team, 'rank']),
            "ppg": round(float(g.pts.mean()), 2), "win_pct": pct(g.gf > g.ga),
            "gf_pg": round(float(g.gf.mean()), 2), "ga_pg": round(float(g.ga.mean()), 2),
            "cs_pct": pct(g.ga == 0), "fts_pct": pct(g.gf == 0),
            "btts_pct": pct((g.gf > 0) & (g.ga > 0)), "over25_pct": pct((g.gf + g.ga) > 2.5),
            "home": {"played": len(hm), "wins": int((hm.gf > hm.ga).sum()), "draws": int((hm.gf == hm.ga).sum()), "losses": int((hm.gf < hm.ga).sum()),
                     "gf_pg": round(float(hm.gf.mean()), 2) if len(hm) else None, "ga_pg": round(float(hm.ga.mean()), 2) if len(hm) else None},
            "away": {"played": len(aw), "wins": int((aw.gf > aw.ga).sum()), "draws": int((aw.gf == aw.ga).sum()), "losses": int((aw.gf < aw.ga).sum()),
                     "gf_pg": round(float(aw.gf.mean()), 2) if len(aw) else None, "ga_pg": round(float(aw.ga.mean()), 2) if len(aw) else None},
            "home_ppg": round(float(hm.pts.mean()), 2) if len(hm) else None,
            "away_ppg": round(float(aw.pts.mean()), 2) if len(aw) else None,
            "vs_top": {"played": len(vs_top), "ppg": round(float(vs_top.pts.mean()), 2) if len(vs_top) else None},
            "vs_bottom": {"played": len(vs_bot), "ppg": round(float(vs_bot.pts.mean()), 2) if len(vs_bot) else None},
        }
    # 같은 리그 안 순위(동률은 같은 순위)와 리그 평균
    ranks, avg = {t: {} for t in rows}, {}
    for key, higher in STAT_KEYS:
        vals = {t: r[key] for t, r in rows.items() if r[key] is not None}
        avg[key] = round(float(np.mean(list(vals.values()))), 2) if vals else None
        if higher is None:
            continue
        for t, v in vals.items():
            ranks[t][key] = 1 + sum(1 for o in vals.values() if (o > v if higher else o < v))
    return {"rows": rows, "ranks": ranks, "avg": avg, "teams": n, "long": L}

@lru_cache(maxsize=None)
def _season_rank_progress(league, season):
    """리그·시즌 전체 팀의 "경기를 치를 때마다의 순위"를 한 번에 — 날짜순으로 승점·득실·득점을 누적하며 그날 경기가 끝난
    뒤의 순위를 기록(예전엔 팀마다 경기 날짜마다 순위표를 새로 계산해서 팀 통계 첫 호출이 ~0.3초 걸렸음)"""
    L = _league_season_stats(league, season)["long"]
    tot = {t: [0, 0, 0] for t in L.team.unique()}   # 승점, 득실, 득점
    progress = {t: [] for t in tot}
    for date, day in L.groupby('date', sort=True):
        for r in day.itertuples():
            v = tot[r.team]
            v[0] += r.pts; v[1] += r.gf - r.ga; v[2] += r.gf
        order = sorted(tot, key=lambda t: (-tot[t][0], -tot[t][1], -tot[t][2]))
        rank = {t: i + 1 for i, t in enumerate(order)}
        for t in day.team.unique():
            progress[t].append(rank[t])
    return progress

def _rank_progress(league, season, team):
    return _season_rank_progress(league, season).get(team, [])

# 서버가 뜰 때 백그라운드에서 전 리그·시즌 통계를 미리 계산(재배포 직후 첫 사용자도 기다리지 않게, ~1초)
# 서버 시작 때 미리 계산 — Render 무료 CPU는 로컬보다 20~40배 느려서, 예전처럼 한 번에 몰아 계산하면 재시작 뒤 몇 분 동안
# 모든 요청이 같이 느려졌음(2026-10-08 사용자 지적 "로딩 화면이 계속 나옴"). → 사용자가 먼저 보는 것부터, 한 단위마다 잠깐 쉬어
# 요청 처리 스레드가 CPU를 먼저 쓰게 하고, 공유 썸네일 미리 그리기(로컬 8초 — 가장 무거움)는 맨 끝에
def _warm_yield():
    _time.sleep(0.03)

def _warm_all():
    steps = [lambda c=c: _af_league_squads(c) for c in ("PL", "PD", "BL1", "SA", "FL1")]   # 선수 기록(선수 탭·베스트 11·선수 카드)
    steps += [lambda: _league_rows("CL", CURRENT_SEASON_YEAR), _ps_index, _player_search_index]
    steps += [lambda lg=lg: _league_display_ranks(lg) for lg in ("PL", "PD", "BL1", "SA", "FL1")]
    steps += [lambda lg=lg: _schedule_predictions(lg) for lg in ("PL", "PD", "BL1", "SA", "FL1", "CL")]   # 일정 탭·다음 경기 예측 막대
    steps += [_team_season_league]
    for yr in sorted(STAT_SEASONS, reverse=True):   # 팀 통계(최근 시즌부터)
        for lg in ("PL", "PD", "BL1", "SA", "FL1"):
            steps.append(lambda lg=lg, yr=yr: _league_season_stats(lg, yr) and _season_rank_progress(lg, yr))
    steps += [lambda t=t: _team_stats(t) for t in list(team_state)[:200]]
    steps += [_dpos_hint]
    def share():
        import share_card
        share_card._background()
        share_card.warm_crests(team_logos_cache.values())
    steps += [share, _warm_share_cards]
    t0 = _time.time()
    for f in steps:
        try:
            f()
        except Exception as e:
            print("⚠️ 미리 계산 실패:", e)
        _warm_yield()
    print(f"✅ 미리 계산 끝 {_time.time() - t0:.1f}초")

@app.on_event("startup")
def _start_warmup():
    import threading
    threading.Thread(target=_warm_all, daemon=True).start()

@app.get("/team/stats/{team_name}")
def get_team_stats(team_name: str):
    team = TEAM_NAME_MAP.get(team_name, team_name)
    return _team_stats(team)

@lru_cache(maxsize=1)
def _team_season_league():
    """(팀, 시즌) → 그 시즌에 가장 많이 뛴 리그 — 예전엔 요청마다 3만 줄을 17번 걸러서 느렸음"""
    d = df_seasons[df_seasons['league'] != 'CL']
    long = pd.concat([d[['season', 'league', 'home_team']].rename(columns={'home_team': 'team'}),
                      d[['season', 'league', 'away_team']].rename(columns={'away_team': 'team'})])
    cnt = long.groupby(['team', 'season', 'league']).size().reset_index(name='n').sort_values('n', ascending=False)
    out = {}
    for r in cnt.itertuples():
        out.setdefault((r.team, int(r.season)), r.league)
    return out

@lru_cache(maxsize=256)
def _team_stats(team):
    league = _team_current_league(team)
    state = team_state.get(team)
    if not league and not state:
        raise HTTPException(status_code=404, detail=f"팀을 찾을 수 없습니다: {team}")
    seasons = []
    for yr, label in STAT_SEASONS.items():
        # 그 시즌에 뛴 리그(승강한 팀은 시즌마다 다를 수 있음 — 5대 리그 안에서만)
        lg = _team_season_league().get((team, yr))
        if not lg:
            continue
        st = _league_season_stats(lg, yr)
        if not st or team not in st["rows"]:
            continue
        seasons.append({
            "season": label, "year": yr, "league": lg, "league_name": LEAGUE_MAP.get(lg), "teams": st["teams"],
            "stats": st["rows"][team], "ranks": st["ranks"][team], "league_avg": st["avg"],
            "rank_progress": _rank_progress(lg, yr, team),
        })
    power = None
    if state:
        # AI 모델 입력값(최근 38경기)과 같은 리그 팀들 사이 순위
        # 비교 대상 = 가장 최근 시즌 같은 리그 팀(강등된 팀은 마지막 리그 경기가 이 리그여도 제외)
        cur = next((_league_season_stats(league, yr) for yr in sorted(STAT_SEASONS, reverse=True) if league and _league_season_stats(league, yr)), None)
        peers = [t for t in (cur["rows"] if cur else []) if t in team_state]
        rank_of = lambda key, higher=True: (1 + sum(1 for t in peers if (team_state[t][key] > state[key] if higher else team_state[t][key] < state[key]))) if team in peers else None
        power = {"elo": state.get('elo'), "elo_rank": rank_of('elo'), "attack": state.get('attack'), "attack_rank": rank_of('attack'),
                 "defense": state.get('defense'), "defense_rank": rank_of('defense', False), "peers": len(peers),
                 "history": [{"date": d, "elo": e} for d, e in state.get('elo_history', [])]}
    return {"team": team, "league": league, "seasons": seasons[::-1], "power": power}

    # ── 선수 데이터 API ──
import json as _json

# ── 모델 정확도 API ──
@app.get("/accuracy")
def get_accuracy():
    """AI 모델 정확도 정보"""
    data = _load_json("accuracy.json")
    if data is None:
        return {"best": None, "total_matches": None, "training_matches": None, "updated_at": None}
    return data

@app.get("/predict/track-record")
def get_track_record():
    """AI 예측 트랙레코드 — update_data.py가 매일 그 시점에 실제 서빙 중인 /predict를
    호출해 미리 기록해두고, 경기가 끝나면 실제 결과와 대조해 채워넣은 로그(prediction_log.json)의
    요약 + 최근 완료 경기 목록 + 채점 예정 경기"""
    log = _load_json("prediction_log.json") or {}
    now = pd.Timestamp.utcnow().tz_localize(None)

    def _row(e):
        return {k: e.get(k) for k in ("league", "home_team", "away_team", "date", "predicted", "home_win_prob",
                                      "draw_prob", "away_win_prob", "predicted_score", "actual", "actual_score", "correct")}

    # 채점 예정: 기록은 됐지만 아직 안 끝난 경기(가까운 순) — 첫 결과가 나오기 전 화면용
    pending = sorted((e for e in log.values() if e.get("actual") is None
                      and pd.Timestamp(e["date"]).tz_localize(None) > now - pd.Timedelta(hours=3)), key=lambda e: e["date"])
    upcoming = {"count": len(pending), "first_date": pending[0]["date"] if pending else None,
                "matches": [_row(e) for e in pending[:6]]}

    # total_scheduled: 예정 경기까지 포함해 기록해둔 전체 건수 (아직 결과 없는 것 포함)
    # total_resolved: 그중 실제로 경기가 끝나 적중 여부를 확정한 건수 — 적중률 계산은 이 값 기준
    total_scheduled = len(log)
    resolved = sorted((e for e in log.values() if e.get("actual") is not None), key=lambda e: e["date"])
    if not resolved:
        return {"summary": {"total_scheduled": total_scheduled, "total_resolved": 0}, "recent": [], "upcoming": upcoming}

    def _acc(rows):
        n = len(rows); c = sum(1 for e in rows if e["correct"])
        return {"n": n, "correct": c, "pct": round(c / n * 100, 1) if n else None}

    recent = resolved[-50:]
    total_correct = sum(1 for e in resolved if e["correct"])
    recent_correct = sum(1 for e in recent if e["correct"])
    # 최근 7일: 가장 최근에 끝난 기록 경기 기준 7일(A매치 휴식기에도 "지난 라운드" 성적이 보이게)
    last_dt = pd.Timestamp(resolved[-1]["date"]).tz_localize(None)
    last7 = [e for e in resolved if pd.Timestamp(e["date"]).tz_localize(None) > last_dt - pd.Timedelta(days=7)]
    # 확신 높은 예측: 가장 높은 확률이 60% 이상이었던 경기만
    confident = [e for e in resolved if max(e["home_win_prob"], e["draw_prob"], e["away_win_prob"]) >= 0.6]
    by_league = {}
    for e in resolved:
        by_league.setdefault(e.get("league") or "?", []).append(e)

    # 누적 적중률 추이: 확정된 경기를 날짜순으로 하나씩 반영했을 때 그 시점까지의
    # 누적 적중률(%)이 어떻게 움직였는지 — 차트 가독성을 위해 최근 30포인트만
    # 잘라서 보여주되, 값 자체는 처음부터 누적한 진짜 전체 누적치를 유지한다
    # (30개 구간으로 다시 시작하는 게 아니라, 긴 누적 곡선의 최근 구간만 보여주는 것).
    cum_correct = 0
    accuracy_trend = []
    for i, e in enumerate(resolved, start=1):
        if e["correct"]:
            cum_correct += 1
        accuracy_trend.append({"n": i, "accuracy_pct": round(cum_correct / i * 100, 1)})
    accuracy_trend = accuracy_trend[-30:]

    return {
        "summary": {
            "total_scheduled": total_scheduled,
            "total_resolved": len(resolved),
            "total_correct": total_correct,
            "accuracy_pct": round(total_correct / len(resolved) * 100, 1),
            "recent_n": len(recent),
            "recent_correct": recent_correct,
            "recent_accuracy_pct": round(recent_correct / len(recent) * 100, 1),
            # 비교 기준: 같은 경기를 전부 "홈승"으로 찍었다면의 적중률(정직한 비교용)
            "baseline_home_pct": round(sum(1 for e in resolved if e["actual"] == "home_win") / len(resolved) * 100, 1),
            "last7": {**_acc(last7), "from": last7[0]["date"], "to": last7[-1]["date"]},
            "confident": _acc(confident),
            "by_league": {k: _acc(v) for k, v in by_league.items()},
        },
        "accuracy_trend": accuracy_trend,
        "recent": [_row(e) for e in reversed(recent[-20:])],
        "last10": [bool(e["correct"]) for e in resolved[-10:]],
        "upcoming": upcoming,
    }

@app.get("/players/topscorers/{league_code}")
def get_top_scorers(league_code: str):
    """리그별 득점왕"""
    data = _load_json("players.json")
    if data is None:
        raise HTTPException(status_code=404, detail="선수 데이터 없음")

    scorers = data.get("topscorers", {}).get(league_code.upper(), [])
    if not scorers:
        raise HTTPException(status_code=404, detail="해당 리그 데이터 없음")
    
    return {"league": league_code.upper(), "players": scorers}

@app.get("/players/topassists/{league_code}")
def get_top_assists(league_code: str):
    """리그별 도움왕"""
    data = _load_json("players.json")
    if data is None:
        raise HTTPException(status_code=404, detail="선수 데이터 없음")

    assists = data.get("topassists", {}).get(league_code.upper(), [])
    if not assists:
        raise HTTPException(status_code=404, detail="해당 리그 데이터 없음")
    
    return {"league": league_code.upper(), "players": assists}

def _norm_name(n):
    import unicodedata
    n = unicodedata.normalize("NFKD", n or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", n).split()

def _squad_photo(team, name):
    """team_extra.json(API-Football 스쿼드)에서 같은 팀·같은 선수 사진 찾기 — 표기가 "M. Di Gregorio"처럼 이름 머리글자라
    성(마지막 단어들) + 이름 첫 글자로 맞춤. 못 찾으면 None(화면은 등번호/실루엣)"""
    squad = ((_load_json("team_extra.json") or {}).get(team) or {}).get("squad") or []
    want = _norm_name(name)
    if not want:
        return None
    for p in squad:
        got = _norm_name(p.get("name"))
        if not got or not p.get("photo"):
            continue
        if got == want or (got[-1] == want[-1] and got[0][0] == want[0][0]):
            return p["photo"]
    return None

def _done_matchday(code, upto=None):
    """득점 순위가 "몇 라운드까지" 반영됐는지: 그 날짜(upto, 수집일)까지 끝난 경기의 가장 큰 라운드.
    football-data scorers의 season.currentMatchday는 "다음(진행 중) 라운드"라서 5라운드까지 치렀는데 "6라운드 기준"으로 나왔음(2026-10-03)"""
    ms = [m for m in (_load_json("schedule.json") or {}).get(code, [])
          if m.get("status") in DONE_STATUSES and m.get("matchday") and (not upto or str(m.get("date", ""))[:10] <= upto)]
    return max((m["matchday"] for m in ms), default=None)

@lru_cache(maxsize=8)
def _leaders(code):
    """이번 시즌 득점·도움 순위(scorers.json — update_data.fetch_scorers). 도움 순위는 football-data가
    득점 순으로 준 상위 100명 안에서 다시 정렬한 것(득점 없이 도움만 많은 선수는 빠질 수 있음 — 화면에 안내)"""
    d = (_load_json("scorers.json") or {}).get(code)
    if not d or not d.get("scorers"):
        return None
    rows = [{**r, "photo": _squad_photo(r.get("team"), r.get("name"))} for r in d["scorers"]]
    goals = sorted(rows, key=lambda r: (-r["goals"], -r["assists"], r.get("played") or 99))[:20]
    assists = sorted([r for r in rows if r["assists"] > 0], key=lambda r: (-r["assists"], -r["goals"], r.get("played") or 99))[:20]
    return {"league": code, "season": d.get("season"), "matchday": _done_matchday(code, d.get("updated")) or d.get("matchday"), "updated": d.get("updated"),
            "pool": len(rows), "goals": goals, "assists": assists}

@lru_cache(maxsize=256)
def _team_players(team):
    """팀 정보 "플레이어 통계" 탭: 이번 시즌 이 팀 선수의 골·도움(scorers.json — 리그·챔스 따로, 득점 상위 100명 안)
    + 팀 리그 득점에서 차지하는 비중 + 스쿼드 구성(team_info.json의 포지션·국적)"""
    sc = _load_json("scorers.json") or {}
    lg = _team_league_map().get(team)
    comps = {}
    for code in [c for c in (lg, "CL") if c]:
        d = sc.get(code)
        if not d:
            continue
        rows = [{**r, "photo": _squad_photo(team, r.get("name"))} for r in d.get("scorers", []) if r.get("team") == team]
        rows.sort(key=lambda r: (-r["goals"], -r["assists"], r.get("played") or 99))
        comps[code] = {"season": d.get("season"), "matchday": _done_matchday(code, d.get("updated")) or d.get("matchday"), "updated": d.get("updated"), "players": rows}
    # 이번 시즌 리그 득점(비중 계산용)
    team_goals = None
    if lg:
        m = df_matches_all[(df_matches_all['league'] == lg) & (pd.to_datetime(df_matches_all['date']) >= f"{CURRENT_SEASON_YEAR}-07-01")]
        team_goals = int(m.loc[m['home_team'] == team, 'home_goals'].sum() + m.loc[m['away_team'] == team, 'away_goals'].sum())
    squad = ((_load_json("team_info.json") or {}).get(team) or {}).get("squad") or []
    pos, nat = {}, {}
    for p in squad:
        pos[p.get("position") or "?"] = pos.get(p.get("position") or "?", 0) + 1
        if p.get("nationality"):
            nat[p["nationality"]] = nat.get(p["nationality"], 0) + 1
    return {"team": team, "league": lg, "scorers_available": bool(sc), "comps": comps, "team_league_goals": team_goals,
            "squad": {"size": len(squad), "positions": pos, "nationalities": sorted(nat.items(), key=lambda x: -x[1])[:5]}}

@app.get("/team/players/{team_name}")
def get_team_players(team_name: str):
    team = TEAM_NAME_MAP.get(team_name, team_name)
    return _team_players(team)

@app.get("/players/leaders/{league_code}")
def get_leaders(league_code: str):
    """선수 탭: 이번 시즌 득점·도움 순위(5대 리그 + UCL). 아직 받은 데이터가 없는 리그는 404 → 프론트가 예전 EPL 24-25로 대체"""
    out = _leaders(league_code.upper())
    if out is None:
        raise HTTPException(status_code=404, detail="이번 시즌 득점 순위 데이터 없음")
    return out

    # ── 우승 예측 API ──
# ── 순위 예측 API ──
# 브라우저가 "현재 승점 + 남은 경기 × 모델 확률"로 몬테카를로 시뮬레이션함(FotData.html runSimulation).
# 예전엔 0점부터 전체 시즌을 블렌딩 승률 공식으로 시뮬레이션해서 현재 승점을 무시했고, 경기 예측 모델과도
# 따로 놀았음. 24-25·25-26 시즌 5개 리그 백테스트(5/10/19/28라운드 시점)에서 새 방식이 우승 확률 Brier
# 0.537→0.333, 강등 0.096→0.044, 평균 순위 오차 3.02→1.88위로 개선(2026-09-28).
SEASON_SIM_SIGMA = 0.15   # 시뮬레이션마다 팀별 전력 변동(로짓 단위) — 같은 백테스트에서 σ 0.1~0.2가 최적
# 순위 시뮬레이션용 경기 확률만 리그 평균 결과 비율 쪽으로 10% 당김. 경기 하나 예측은 보정이 잘 맞지만,
# 시즌 초 폼(예: 바르사 7전 전승 → 남은 31경기 평균 승리확률 79% → 예상 98점)이 남은 시즌 전체에 그대로
# 곱해지면서 상위 팀 최종 승점을 +1.6점 과대평가했음. 백테스트에서 λ=0.1이 편향 −0.1점으로 가장 중립적이고
# 승점 오차·우승 Brier도 소폭 개선(λ 0~0.3, 온도 보정 1.1~1.5와 비교 — 2026-09-28)
SEASON_SIM_SHRINK = 0.10
SEASON_SIM_BASE = (0.44, 0.25, 0.31)   # 홈승/무/원정승 리그 평균 비율
SIM_LEAGUES = ("PL", "PD", "BL1", "SA", "FL1")

@app.get("/predict/champion/{league_code}")
def get_champion_prediction(league_code: str):
    return _champion(league_code.upper())

@lru_cache(maxsize=8)
def _champion(code: str):
    if code not in SIM_LEAGUES:
        raise HTTPException(status_code=404, detail="해당 리그 데이터 없음")

    table = get_standings(code)["standings"]
    teams = {r["team"]: {"team": r["team"], "logo": r["logo"], "played": r["played"],
                         "points": r["points"], "gd": r["gd"]} for r in table}

    schedule = (_load_json("schedule.json") or {}).get(code, [])
    remaining = [m for m in schedule if m.get("status") not in ("FINISHED", "AWARDED", "CANCELLED")]
    for m in remaining:   # 시즌 초 아직 경기가 없는 팀도 포함
        for t in (m["home_team"], m["away_team"]):
            teams.setdefault(t, {"team": t, "logo": team_logos_cache.get(t, ''), "played": 0, "points": 0, "gd": 0})

    pairs = [(m["home_team"], m["away_team"]) for m in remaining]
    known = [i for i, (h, a) in enumerate(pairs) if h in team_state and a in team_state]
    probs = [[0.44, 0.25, 0.31]] * len(pairs)   # 데이터 없는 팀(거의 없음)은 리그 평균 결과 비율
    if known:
        P = _predict_hda([pairs[i] for i in known])
        P = (1 - SEASON_SIM_SHRINK) * P + SEASON_SIM_SHRINK * np.array(SEASON_SIM_BASE)
        for i, p in zip(known, P):
            probs[i] = [round(float(x), 4) for x in p]

    return {
        "league":    LEAGUE_MAP.get(code),
        "teams":     list(teams.values()),
        "fixtures":  [{"home": h, "away": a, "p": p} for (h, a), p in zip(pairs, probs)],
        "remaining": len(pairs),
        "sigma":     SEASON_SIM_SIGMA,
    }

# ── 빅매치 API (경기예측 탭 배너 + 랜딩 3D 연출이 같이 씀 — 페이지마다 다른 경기가 뜨지 않게 한 곳에서 결정) ──
# "빅매치" = 두 팀 모두 BIG_CLUBS(리그별 인기·시청률 상위 구단)에 속한 경기 중 **가장 가까운 경기**.
# 킥오프가 같으면 prestige 합이 큰 쪽. 경기 시작 시각이 지나면 후보에서 빠져 다음 빅매치로 넘어감.
# 예전 규칙(프론트): prestige 합 ≥80 + 가장 이른 경기일 +3일 안 최강 매치 → prestige가 최근 2시즌 승률만
# 반영해서 맨유(5.9)+토트넘(−60) 같은 부진한 인기 구단 경기가 빠지고 하루 뒤 리버풀–맨시티가 떴음.
# 사용자 결정(2026-09-28): 인기 구단끼리면 가까운 경기부터 순서대로.
BIG_CLUBS = {
    'Manchester City FC', 'Liverpool FC', 'Arsenal FC', 'Manchester United FC', 'Chelsea FC', 'Tottenham Hotspur FC',
    'Real Madrid CF', 'FC Barcelona', 'Club Atlético de Madrid',
    'FC Bayern München', 'Borussia Dortmund',
    'FC Internazionale Milano', 'AC Milan', 'Juventus FC', 'SSC Napoli', 'AS Roma',
    'Paris Saint-Germain FC', 'Olympique de Marseille',
}
BIGMATCH_HORIZON_DAYS = 30

@app.get("/bigmatch")
def get_bigmatch():
    schedule = _load_json("schedule.json") or {}
    now = pd.Timestamp.now(tz='UTC')
    horizon = now + pd.Timedelta(days=BIGMATCH_HORIZON_DAYS)
    prestige = dict(zip(df_stats['team'], df_stats['prestige'].fillna(0)))
    cands = []
    for league, matches in schedule.items():
        for m in matches:
            d = pd.Timestamp(m['date'])
            if (m.get('status') != 'FINISHED' and now < d <= horizon
                    and m['home_team'] in BIG_CLUBS and m['away_team'] in BIG_CLUBS):
                score = prestige.get(m['home_team'], 0) + prestige.get(m['away_team'], 0)
                cands.append((d, -score, league, m))
    if not cands:
        return {"match": None}
    d, _, league, m = min(cands, key=lambda c: (c[0], c[1]))
    return {"match": {
        "date": m['date'], "league": league, "matchday": m.get('matchday'),
        "home_team": m['home_team'], "away_team": m['away_team'],
        "home_logo": team_logos_cache.get(m['home_team'], ''), "away_logo": team_logos_cache.get(m['away_team'], ''),
    }}

# ── 다가오는 경기 창 API (2026-09-29) ──
# "오늘의(다음) 경기"·"내 팀"·"다른 경기도 예측해보기" 위젯용. 예전엔 이 셋 때문에 5대 리그+UCL 전체 시즌 일정 6개
# (약 1,300경기)를 통째로 받았음 → 필요한 창만 한 번에:
#   - 지금 − 30시간: 사용자 시간대(한국)의 "오늘"에 이미 끝난 경기까지 포함(날짜 판정은 브라우저가 로컬 시간으로)
#   - 지금 + 21일: A매치 휴식기(보통 2주)에도 "다음 경기"·"내 팀 다음 경기"가 들어오게. "다른 경기" 카드는 14일만 씀
# 21일 안에 내 팀 경기가 없으면 프론트가 그 팀 리그 일정만 따로 받음(예비 경로).
MATCH_WINDOW_BEFORE_H = 30
MATCH_WINDOW_AFTER_D = 21

# ── 경기 당일 결과 (2026-09-30) ──
# schedule.json은 새벽 자동 업데이트 때만 바뀌어서, 예전엔 경기가 끝나도 다음 날 아침까지 일정·다음 경기 카드에 결과가 없었음.
# 지금 −30시간 ~ +2분 사이에 킥오프했는데 일정상 아직 안 끝난 경기가 있을 때만 football-data에서 오늘 경기를 받아
# 일정 위에 덮어씀(점수·상태). 요청은 서버 전체에서 최대 1분에 1번(모든 사용자가 같은 캐시를 봄) — 무료 플랜 한도(분당 10회)는
# 새벽 자동 업데이트와 같은 키로 나눠 씀(그쪽은 429면 쉬었다 재시도, update_data.fd_get).
# 무료 플랜은 점수가 몇 분 늦게 들어옴("Scores delayed") — 화면엔 "LIVE"가 아니라 "진행 중"으로 표기.
# Render 환경변수에 FOOTBALL_API_KEY가 없으면 아무것도 안 하고 일정 그대로.
import threading
LIVE_STATUSES = {"IN_PLAY", "PAUSED", "EXTRA_TIME", "PENALTY_SHOOTOUT"}
DONE_STATUSES = {"FINISHED", "AWARDED"}
LIVE_TTL, LIVE_TTL_QUIET = 60, 900   # 진행 중 경기가 있으면 1분, 전부 끝났으면(새벽 업데이트 전까지) 15분
_live = {"at": 0.0, "data": {}, "quiet": False, "next": None}   # next: 다음 킥오프(epoch초) — 조용할 때도 그때는 다시 확인
_live_lock = threading.Lock()

def _live_key(home, away, date):
    return f"{home}|{away}|{date[:10]}"

def _live_goals(m):
    """update_data._match_goals와 같은 규칙(승부차기 골 빼기) — 진행 중이면 fullTime이 지금 점수"""
    sc = m.get("score") or {}
    ft = sc.get("fullTime") or {}
    hg, ag = ft.get("home"), ft.get("away")
    pen = sc.get("penalties") or {}
    if sc.get("duration") == "PENALTY_SHOOTOUT" and pen.get("home") is not None and hg is not None:
        return hg - pen["home"], ag - pen["away"], [pen["home"], pen["away"]]
    return hg, ag, None

def _live_needed(now):
    lo, hi = now - pd.Timedelta(hours=30), now + pd.Timedelta(minutes=2)
    for ms in (_load_json("schedule.json") or {}).values():
        for m in ms:
            if m.get("status") not in DONE_STATUSES | {"CANCELLED", "POSTPONED"} and lo <= pd.Timestamp(m["date"]) <= hi:
                return True
    return False

def _live_overlay():
    """{홈|원정|날짜: {status, home_goals, away_goals, penalties?, minute?}} — 오늘 경기 최신 상태(없으면 빈 dict)"""
    key = os.environ.get("FOOTBALL_API_KEY")
    if not key:
        return {}
    import time as _t
    now_s = _t.time()
    fresh = now_s - _live["at"] < (LIVE_TTL_QUIET if _live["quiet"] else LIVE_TTL)
    if fresh and not (_live["quiet"] and _live["next"] and now_s >= _live["next"] + 60):
        return _live["data"]
    if not _live_lock.acquire(blocking=False):   # 다른 요청이 받아오는 중이면 기다리지 않고 직전 값
        return _live["data"]
    try:
        now = pd.Timestamp.now(tz="UTC")
        _live["next"] = _next_kickoff(now)
        if not _live_needed(now):
            _live.update(at=_t.time(), data={}, quiet=True)
            return {}
        r = requests.get("https://api.football-data.org/v4/matches", headers={"X-Auth-Token": key}, timeout=8, params={
            "competitions": "PL,PD,BL1,SA,FL1,CL",
            "dateFrom": (now - pd.Timedelta(days=2)).strftime("%Y-%m-%d"), "dateTo": (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")})
        if r.status_code != 200:
            print(f"경기 당일 결과 받기 실패: {r.status_code}")
            _live["at"] = _t.time()   # 실패해도 1분은 다시 안 부름(직전 값 유지)
            return _live["data"]
        data = {}
        for m in r.json().get("matches", []):
            hg, ag, pens = _live_goals(m)
            row = {"status": m["status"], "home_goals": hg, "away_goals": ag}
            if pens:
                row["penalties"] = pens
            if m.get("minute"):   # 경기 분(무료 플랜에서도 오는지는 10/10 첫 경기에서 확인 예정 — 없으면 화면엔 "진행 중"만)
                row["minute"] = m["minute"]
            if m.get("injuryTime"):   # 추가시간(45+2 → 화면 "45+2분")
                row["injury_time"] = m["injuryTime"]
            row["utc"] = m["utcDate"]
            data[_live_key(m["homeTeam"]["name"], m["awayTeam"]["name"], m["utcDate"])] = row
        quiet = not any(v["status"] in LIVE_STATUSES for v in data.values()) and _all_started_done(data, now)
        _live.update(at=_t.time(), data=data, quiet=quiet)
        return data
    except Exception as e:
        print(f"경기 당일 결과 오류: {e}")
        _live["at"] = _t.time()
        return _live["data"]
    finally:
        _live_lock.release()

def _next_kickoff(now):
    ts = [pd.Timestamp(m["date"]) for ms in (_load_json("schedule.json") or {}).values() for m in ms
          if m.get("status") not in DONE_STATUSES | {"CANCELLED", "POSTPONED"} and pd.Timestamp(m["date"]) > now]
    return min(ts).timestamp() if ts else None

def _all_started_done(data, now):
    """이미 킥오프한 일정 경기가 전부 '끝남'으로 들어왔으면 True(그럼 새벽 업데이트 전까지 15분에 한 번만 확인)"""
    lo = now - pd.Timedelta(hours=30)
    for ms in (_load_json("schedule.json") or {}).values():
        for m in ms:
            t = pd.Timestamp(m["date"])
            if lo <= t <= now and m.get("status") not in DONE_STATUSES | {"CANCELLED", "POSTPONED"}:
                v = data.get(_live_key(m["home_team"], m["away_team"], m["date"]))
                if not v or v["status"] not in DONE_STATUSES:
                    return False
    return True

def _with_live(m, live):
    v = live.get(_live_key(m["home_team"], m["away_team"], m["date"])) if live else None
    if not v or m.get("status") in DONE_STATUSES:
        return m
    return {**m, **{k: x for k, x in v.items() if k != "utc"}, "live": True}

@app.get("/meta/countries")
def get_meta_countries():
    """나라 이름(API-Football·football-data 표기) → 국기 이미지 주소 — 선수 카드·스쿼드의 국기(af_countries.json, 한 번 받아 둔 것)"""
    return _load_json("af_countries.json") or {}

@app.get("/teams/ko")
def get_teams_ko():
    """구단 둘러보기 검색용: 팀 이름 → 한국어 구단명(위키백과 name_ko) — "맨체스터"·"바이에른"처럼 한글로도 찾게 (2026-09-30)"""
    return {t: w["name_ko"] for t, w in (_load_json("team_wiki.json") or {}).items() if w.get("name_ko")}

@app.get("/teams/meta")
def get_teams_meta():
    """구단 둘러보기: 팀 이름 → 한국어 구단명(위키) + 구단 색(football-data clubColors, 카드 뒤 빛 색) — 팀 정보를 팀마다 받지 않게 한 번에 (2026-09-30)"""
    return _teams_meta()

@lru_cache(maxsize=1)
def _teams_meta():
    wiki, info = _load_json("team_wiki.json") or {}, _load_json("team_info.json") or {}
    out = {}
    for t in set(wiki) | set(info):
        ko, col = (wiki.get(t) or {}).get("name_ko"), (info.get(t) or {}).get("clubColors")
        if ko or col:
            out[t] = {"ko": ko, "colors": col}
    return out

@app.get("/standings/{league_code}/movement")
def get_standings_movement(league_code: str):
    """구단 둘러보기 순위 변동: 지난 라운드(마지막으로 끝난 라운드 전)까지의 순위 대비 지금 순위(+면 오름). 첫 라운드뿐이면 비교 안 함 (2026-09-30)"""
    return _rank_movement(league_code.upper())

@lru_cache(maxsize=8)
def _rank_movement(code):
    ms = [m for m in (_load_json("schedule.json") or {}).get(code, []) if m.get("status") in DONE_STATUSES and m.get("home_goals") is not None]
    if not ms:
        return {"matchday": None, "delta": {}}
    last = max(m.get("matchday") or 0 for m in ms)
    if last <= 1:
        return {"matchday": last, "delta": {}}
    def ranks(sub):
        tb = {}
        for m in sub:
            for t, gf, ga in ((m["home_team"], m["home_goals"], m["away_goals"]), (m["away_team"], m["away_goals"], m["home_goals"])):
                r = tb.setdefault(t, [0, 0, 0])
                r[0] += 3 if gf > ga else 1 if gf == ga else 0
                r[1] += gf - ga
                r[2] += gf
        order = sorted(tb, key=lambda t: (-tb[t][0], -tb[t][1], -tb[t][2], t))
        return {t: i + 1 for i, t in enumerate(order)}
    now, prev = ranks(ms), ranks([m for m in ms if (m.get("matchday") or 0) < last])
    return {"matchday": last, "delta": {t: prev[t] - r for t, r in now.items() if t in prev}}

# ── API-Football(Pro, 2026-10-06 결제) — 경기 상세·팀 소식·선수 카드 ──
# 새벽 수집(update_data.af_sync): af_fixtures.json(경기 목록·API-Football ID) · match_details/(끝난 경기 상세, gzip) ·
# match_previews.json(결장·부상자) · af_players.json(선수 프로필). 시즌 기록은 경기 상세 합산(af_transform.af_season_players).
# Render 환경변수 API_FOOTBALL_KEY가 있으면 새벽 수집 전이라도 요청 때 바로 받음: 막 끝난 경기 상세, 킥오프 약 1시간 전 확정 라인업, 선수 경력·트로피·부상 이력
from af_transform import (af_match_detail, af_season_players, add_percentiles, af_player_profile, md_read, af_league_agg, pick_xi, round_xi,
                          xi_line, plain_can, GRID_FROM, DPOS_LINE, af_compact_player_season, af_compact_transfers, AF_CALENDAR_COMPS)
AF_KEY = os.environ.get("API_FOOTBALL_KEY", "")
AF_URL = "https://v3.football.api-sports.io"
MD_DIR = os.path.join(MODEL_DIR, "match_details")
_af_mem = {}   # 요청 때 받은 것(키 → (받은 시각, 값)) — 프로세스가 살아 있는 동안만
import threading as _threading
_AF_SEM = _threading.BoundedSemaphore(16)   # 서버 전체 동시 요청 10개까지(분당 300회 한도 — 여러 사람이 선수 카드를 한꺼번에 열어도 한도 오류가 안 나게)

def _af_live_get(path, ttl, **params):
    """API-Football을 요청 때 바로(키가 있을 때만). 같은 요청은 ttl초 동안 다시 안 부름(실패도 1분 기억 — 한도 보호)"""
    if not AF_KEY:
        return None
    k = path + json.dumps(params, sort_keys=True)
    hit = _af_mem.get(k)
    if hit and _time.time() - hit[0] < (ttl if hit[1] is not None else 60):
        return hit[1]
    val = None
    for attempt in range(3):   # 한꺼번에 많이 보내면 가끔 한도 오류(errors.rateLimit)가 와서 — 잠깐 쉬고 다시(예전엔 빈 값으로 넘어가 트로피가 엉뚱한 구단에 붙었음)
        try:
            with _AF_SEM:
                d = requests.get(AF_URL + path, headers={"x-apisports-key": AF_KEY}, params=params, timeout=12).json()
            if d.get("errors"):
                if re.search(r"rate|limit|requests", str(d["errors"]), re.I) and attempt < 2:
                    _time.sleep(1.0 + attempt)
                    continue
                break
            val = d.get("response")
            break
        except Exception:
            _time.sleep(0.5)
    _af_mem[k] = (_time.time(), val)
    return val

@lru_cache(maxsize=1)
def _af_index():
    """af_fixtures.json → {"홈|원정|YYYY-MM-DD": 경기} + 팀별 경기 목록(날짜순)"""
    by_key, by_team = {}, {}
    for e in (_load_json("af_fixtures.json") or {}).get("fixtures", []):
        by_key[f"{e['home_team']}|{e['away_team']}|{e['date'][:10]}"] = e
        for t in (e["home_team"], e["away_team"]):
            by_team.setdefault(t, []).append(e)
    for l in by_team.values():
        l.sort(key=lambda e: e["date"])
    return by_key, by_team

@lru_cache(maxsize=4096)
def _md_file(name):
    p = os.path.join(MD_DIR, name)
    return md_read(p) if os.path.exists(p) else None

def _md_of(e):
    """경기 상세: 저장된 파일 → 없으면(새벽 수집 전에 막 끝난 경기) 요청 때 받기"""
    d = _md_file(e["file"])
    if d:
        return d
    ko = pd.Timestamp(e["kickoff"])
    if pd.Timestamp.now(tz="UTC") < ko + pd.Timedelta(minutes=110):
        return None   # 아직 안 끝났을 시각
    r = _af_live_get("/fixtures", 3600, id=e["id"])
    if not r:
        return None
    fx = r[0]
    if (fx.get("fixture") or {}).get("status", {}).get("short") not in ("FT", "AET", "PEN", "AWD", "WO"):
        return None
    det = af_match_detail(fx)
    return {**det, **{k: e[k] for k in ("league", "home_team", "away_team", "date")}, "home_goals": fx["goals"]["home"], "away_goals": fx["goals"]["away"]}

# ── 주장 표시(2026-10-08): 끝난 경기·예상 라인업 = 그 경기에서 실제로 완장을 찬 선수(경기 기록 cap),
# 확정 라인업(경기 전 — API에 주장 정보 없음)·구단 베스트 11 = 이번 시즌 완장을 가장 많이 찬 선수 → 없으면 두 번째(부주장) → 둘 다 없으면 표시 안 함
def _mark_match_caps(d):
    """경기 상세 d의 라인업 선수에 cap(그 경기 완장) — 원본 캐시는 안 건드리고 복사본"""
    caps = {x["id"] for sd in ("home", "away") for x in ((d.get("pstats") or {}).get(sd) or []) if x.get("cap")}
    if not caps or not d.get("lineups"):
        return d
    lus = {sd: {**lu, "start": [{**p, "cap": p.get("id") in caps} for p in lu.get("start") or []],
                "subs": [{**p, "cap": p.get("id") in caps} for p in lu.get("subs") or []]} for sd, lu in d["lineups"].items() if lu}
    return {**d, "lineups": lus}

def _team_captain_order(team, rows=None):
    """[주장, 부주장] 선수 ID — 이번 시즌(또는 rows) 완장 찬 경기 수 순(같으면 출전 시간)"""
    if rows is None:
        code = _team_league_in(team, CURRENT_SEASON_YEAR)
        rows = _af_league_squads(code).get(team, []) if code else []
    cs = sorted([r for r in rows if r.get("cap_n") or r.get("captain")], key=lambda r: (-(r.get("cap_n") or 0), -(r.get("minutes") or 0)))
    return [r["id"] for r in cs[:2]]

def _pick_cap(ids, order):
    """라인업 선수 ID들 중 완장 주인 — 주장이 없으면 부주장, 둘 다 없으면 None"""
    return next((c for c in order if c in ids), None)

@app.get("/match/detail")
def get_match_detail(home_team: str, away_team: str, date: str):
    """경기 결과 창 [요약 | 라인업 | 통계] — 득점·카드·교체, 라인업(포메이션·평점), 팀 통계(xG 포함), 경기 최우수 선수.
    없으면 204(빈 응답 — 화면은 예전처럼 탭 없이)"""
    e = _af_index()[0].get(f"{home_team}|{away_team}|{date[:10]}")
    d = _md_of(e) if e else None
    if not d:
        return Response(status_code=204)
    return {k: v for k, v in _mark_match_caps(d).items() if k != "pstats"}   # 선수별 합산용 기록은 화면에 안 씀(크기만 큼)

# ── 선수 시즌 기록(이번 시즌 = match_details 합산, 지난 시즌 = af_seasons/ — 2026-10-06) ──
PS_DIR = os.path.join(MODEL_DIR, "af_seasons")

@lru_cache(maxsize=32)
def _ps_file(code, yr):
    p = os.path.join(PS_DIR, f"{code}_{yr}.json.gz")
    return md_read(p) if os.path.exists(p) else None

@lru_cache(maxsize=1)
def _ps_index():
    p = os.path.join(PS_DIR, "index.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {"players": {}, "teams": {}}

def _af_round_of(e):
    """af_fixtures 경기의 라운드(리그 = 숫자, 챔스 리그 스테이지 = 숫자) — 수집 때 넣은 값, 없으면 우리 일정의 matchday"""
    if e.get("round") is not None:
        return e["round"]
    for m in (_load_json("schedule.json") or {}).get(e["league"], []):
        if m["home_team"] == e["home_team"] and m["away_team"] == e["away_team"] and m["date"][:10] == e["date"][:10]:
            return m.get("matchday")
    return None

@lru_cache(maxsize=8)
def _cur_details(code):
    """이번 시즌 이 대회의 끝난 경기 상세(날짜순, 라운드 포함)"""
    out = []
    for e in sorted((_load_json("af_fixtures.json") or {}).get("fixtures", []), key=lambda e: e["date"]):
        if e["league"] == code and e.get("has_detail"):
            d = _md_file(e["file"])
            if d:
                out.append({**d, "round": _af_round_of(e)})
    return out

@lru_cache(maxsize=48)
def _league_rows(code, yr):
    """(팀, 선수)별 시즌 기록 목록 — 이번 시즌은 경기 상세 합산 + 프로필, 지난 시즌은 af_seasons 파일. 없으면 None"""
    if yr == CURRENT_SEASON_YEAR:
        dets = _cur_details(code)
        if not dets:
            return None
        rows = list(af_league_agg(dets, keep_matches=True).values())
        prof = _load_json("af_players.json") or {}
        pmap = {}
        for t, v in prof.items():
            for pid, pr in (v.get("players") or {}).items():
                pmap[(t, int(pid))] = pr
        keep = ("name", "full_name", "firstname", "lastname", "age", "birth_date", "birth_place", "nationality", "height", "weight", "photo", "injured")
        for r in rows:
            pr = pmap.get((r["team"], r["id"]))
            if pr:
                r.update({k: pr.get(k) for k in keep if pr.get(k) is not None})
                if pr.get("pos"):
                    r["pos_profile"] = pr["pos"]
            r["full_name"] = r.get("full_name") or r.get("name")
        # 90분당 순위 기준: 450분(5경기) — 시즌 초엔 최다 출전의 절반으로 낮춤(5라운드면 450분을 다 채운 선수가 거의 없음)
        add_percentiles(rows, min_minutes=min(450, 0.5 * max([r["minutes"] or 0 for r in rows] or [0])))
        return rows
    d = _ps_file(code, yr)
    if not d:
        return None
    rows = d["players"]
    for r in rows:
        r.setdefault("full_name", r.get("name"))
        r.setdefault("photo", f"https://media.api-sports.io/football/players/{r['id']}.png")
    return rows

@lru_cache(maxsize=1)
def _dpos_hint():
    """선수 ID → 22-23 시즌 이후 리그 선발 세부 포지션 횟수(지난 시즌 파일 + 이번 시즌)"""
    from collections import Counter as _C
    out = {}
    for code in ("PL", "PD", "BL1", "SA", "FL1"):
        for yr in range(GRID_FROM, CURRENT_SEASON_YEAR + 1):
            for r in (_league_rows(code, yr) or []):
                if r.get("dpos"):
                    out.setdefault(r["id"], _C()).update(r["dpos"])
    return out

def _xi_player(r, key="rating"):
    dp = r.get("dpos") or {}
    top = max(dp, key=dp.get) if dp else None
    tot = sum(dp.values())
    can = {k for k, v in dp.items() if v >= 0.25 * tot}   # 선발의 25% 이상 선 자리는 다 후보(측면 공격수가 양쪽 다 서는 경우 등)
    if not can:   # 21-22 이전 시즌: 같은 선수의 22-23 이후 세부 포지션을 빌려 씀(같은 줄일 때만 — 포지션을 바꾼 선수 방지), 없으면 기록으로 추정
        h = _dpos_hint().get(r["id"])
        if h:
            ht = sum(h.values())
            hc = {k for k, v in h.items() if v >= 0.25 * ht}
            if hc and all(DPOS_LINE.get(k) == r.get("pos") for k in hc):
                can, top = hc, max(h, key=h.get)
        if not can:
            can = plain_can(r)
    return {"id": r["id"], "name": r.get("name"), "full_name": r.get("full_name"), "photo": r.get("photo"), "team": r["team"],
            "rating": r.get(key), "line": xi_line(r), "can": can, "dpos": top,
            "goals": r.get("goals") or 0, "assists": r.get("assists") or 0, "num": r.get("number"), "apps": r.get("apps")}

def _xi_out(xi):
    if not xi:
        return None
    return {"formation": xi["formation"], "avg": xi["avg"], "lines": [[{k: p.get(k) for k in ("id", "name", "full_name", "photo", "team", "rating", "dpos", "slot", "goals", "assists", "num", "apps", "line")} for p in l] for l in xi["lines"]]}

def _xi_caps(xi, order):
    """구단 베스트 11에 주장(없으면 부주장) 표시"""
    if xi:
        cap = _pick_cap({p["id"] for l in xi["lines"] for p in l}, order)
        for l in xi["lines"]:
            for p in l:
                p["cap"] = p["id"] == cap
    return xi

def _season_xi(rows, share=0.4):
    """시즌 베스트 11: 출전 시간이 (그 묶음) 최다의 share 이상인 선수 중 평점 순"""
    mx = max([r["minutes"] or 0 for r in rows] or [0])
    return pick_xi([_xi_player(r) for r in rows if r.get("rating") and (r["minutes"] or 0) >= mx * share])

def _af_league_squads(code):
    """(예전 이름 유지) 이번 시즌 리그 선수 기록 → {팀: 선수 목록}. 이번 시즌 리그 경기를 안 뛴 선수도 선수 카드가 열리게 프로필만으로 추가"""
    rows = _league_rows(code, CURRENT_SEASON_YEAR) or []
    out = {}
    for r in rows:
        out.setdefault(r["team"], []).append(r)
    for t, v in (_load_json("af_players.json") or {}).items():
        if v.get("league") != code:
            continue
        have = {r["id"] for r in out.get(t, [])}
        for pid, pr in (v.get("players") or {}).items():
            if int(pid) not in have:
                out.setdefault(t, []).append({**pr, "id": int(pid), "team": t, "apps": 0, "starts": 0, "minutes": 0, "goals": 0, "assists": 0,
                                              "yellow": 0, "red": 0, "rating": None, "matches": [], "dpos": {}})
    for t, l in out.items():   # 포지션: 시즌 프로필(Attacker 등)을 우선 — 경기 기록의 G/D/M/F는 라인업 줄 기준이라 4-2-3-1 측면 공격수가 미드필더로 잡힘
        for r in l:
            if r.get("pos_profile"):
                r["pos"] = r["pos_profile"]
    return out

def _team_league_in(team, yr):
    """그 시즌 이 팀의 자국 리그 코드(지난 시즌 = 선수 기록 색인, 이번 시즌 = af_players.json)"""
    if yr == CURRENT_SEASON_YEAR:
        return ((_load_json("af_players.json") or {}).get(team) or {}).get("league")
    codes = ((_ps_index().get("teams") or {}).get(team) or {}).get(str(yr)) or []
    return next((c for c in codes if c != "CL"), codes[0] if codes else None)

# ── 선수 검색(2026-10-08): 이번 시즌 5대 리그 선수(af_players.json, 약 2,600명) — 이름 일부로(악센트·대소문자 무시), 이번 시즌 출전 시간 많은 선수부터 ──
def _norm_name(s):
    import unicodedata
    return unicodedata.normalize("NFKD", (s or "").replace("ß", "ss").replace("ø", "o").replace("Ø", "o").replace("ł", "l")).encode("ascii", "ignore").decode().lower()

@lru_cache(maxsize=1)
def _player_search_index():
    stat = {}
    for code in ("PL", "PD", "BL1", "SA", "FL1"):
        for rows in (_af_league_squads(code) or {}).values():
            for r in rows:
                cur = stat.get(r["id"])
                if not cur or (r.get("minutes") or 0) > (cur.get("minutes") or 0):
                    stat[r["id"]] = r
    out = []
    for team, v in (_load_json("af_players.json") or {}).items():
        for pid, p in (v.get("players") or {}).items():
            full = p.get("full_name") or p.get("name") or ""
            keys = " ".join(sorted({_norm_name(x) for x in (full, p.get("name"), p.get("firstname"), p.get("lastname")) if x}))
            r = stat.get(int(pid)) or {}
            out.append({"id": int(pid), "team": team, "league": v.get("league"), "name": full, "photo": p.get("photo"),
                        "pos": r.get("pos_profile") or p.get("pos") or r.get("pos"), "nationality": p.get("nationality"), "age": p.get("age"),
                        "apps": r.get("apps") or 0, "minutes": r.get("minutes") or 0, "goals": r.get("goals") or 0, "assists": r.get("assists") or 0,
                        "rating": r.get("rating"), "_k": keys})
    return out

@app.get("/players/search")
def players_search(q: str = "", league: str = None, pos: str = None, limit: int = 12, offset: int = 0):
    """선수 검색(이번 시즌 5대 리그) — q가 있으면 이름(3글자 이하는 단어 첫머리만), 없으면 꾸준히 뛴 선수(270분 이상) 평점 순.
    league(PL·PD·BL1·SA·FL1)·pos(GK·DF·MF·FW)로 거름"""
    qn = _norm_name(q).strip()
    toks = qn.split()
    idx = [p for p in _player_search_index() if (not league or p["league"] == league) and (not pos or p["pos"] == pos)]
    hits = []
    if toks:
        for p in idx:
            words = p["_k"].split()
            ok = all(any(w.startswith(t) for w in words) if len(t) <= 3 else t in p["_k"] for t in toks)
            if ok:
                score = 2 * any(w.startswith(toks[-1]) for w in words) + any(w.startswith(toks[0]) for w in words)
                hits.append((-score, -(p["minutes"] or 0), p))
    else:
        hits = [(-(p["rating"] or 0), -(p["minutes"] or 0), p) for p in idx if (p["minutes"] or 0) >= 270 and p["rating"]]
    hits.sort(key=lambda x: (x[0], x[1]))
    out, seen = [], set()
    for _, _, p in hits:   # 시즌 중 팀을 옮긴 선수(임대 등)는 한 번만 — 출전 시간 많은 쪽
        if p["id"] in seen:
            continue
        seen.add(p["id"]); out.append(p)
    lim = max(1, min(limit, 60))
    return {"total": len(out), "players": [{k: v for k, v in p.items() if not k.startswith("_")} for p in out[offset:offset + lim]]}

@app.get("/team/squad/{team_name}")
def get_team_squad(team_name: str, season: int = None):
    """선수 카드·구단 베스트 11·주요 선수 — 이번 시즌: 프로필(af_players.json) + 리그 경기 상세 합산, 지난 시즌(15-16~): af_seasons. 없으면 204
    응답의 best11 = 시즌 평균 평점 베스트 11(출전 시간이 팀 최다의 25% 이상), seasons = 선수 기록이 있는 시즌 목록"""
    team = TEAM_NAME_MAP.get(team_name, team_name)
    yr = season or CURRENT_SEASON_YEAR
    code = _team_league_in(team, yr)
    if not code:
        return Response(status_code=204)
    if yr == CURRENT_SEASON_YEAR:
        players = _af_league_squads(code).get(team, [])
    else:
        players = [r for r in (_league_rows(code, yr) or []) if r["team"] == team]
    if not players:
        return Response(status_code=204)
    seasons = sorted({int(y) for y, cs in (((_ps_index().get("teams") or {}).get(team)) or {}).items() if any(c != "CL" for c in cs)}
                     | ({CURRENT_SEASON_YEAR} if _team_league_in(team, CURRENT_SEASON_YEAR) else set()), reverse=True)
    v = (_load_json("af_players.json") or {}).get(team) or {}
    return {"team": team, "league": code, "season": yr, "updated": v.get("updated"), "players": players, "seasons": seasons,
            "manager": _manager_of(team) if yr == CURRENT_SEASON_YEAR else None,
            "best11": _xi_caps(_xi_out(_season_xi([r for r in players if r.get("minutes")], share=0.25)), _team_captain_order(team, players))}

_LP_KEYS = ("id", "team", "full_name", "name", "photo", "pos", "apps", "starts", "minutes", "goals", "assists", "rating", "rated", "shots", "shots_on",
            "key_passes", "dribbles_won", "tackles", "interceptions", "blocks", "duels_won", "clean_sheets", "saves", "yellow", "red", "pen_scored", "number", "nationality")

@lru_cache(maxsize=48)
def _league_players_compact(code, yr):
    rows = _league_rows(code, yr)
    if not rows:
        return None
    by = {}
    for r in rows:   # 시즌 중 같은 리그 안에서 팀을 옮긴 선수는 하나로(팀 = 더 오래 뛴 팀)
        cur = by.get(r["id"])
        x = {k: r.get(k) for k in _LP_KEYS}
        dp = r.get("dpos") or {}
        x["dpos"] = max(dp, key=dp.get) if dp else None
        if yr == CURRENT_SEASON_YEAR and r.get("pos_profile"):
            x["pos"] = r["pos_profile"]
        if not cur:
            by[r["id"]] = x
            continue
        main = cur if (cur["minutes"] or 0) >= (x["minutes"] or 0) else x
        merged = dict(main)
        for k in ("apps", "starts", "minutes", "goals", "assists", "shots", "shots_on", "key_passes", "dribbles_won", "tackles", "interceptions",
                  "blocks", "duels_won", "clean_sheets", "saves", "yellow", "red", "pen_scored", "rated"):
            merged[k] = (cur.get(k) or 0) + (x.get(k) or 0)
        n1, n2 = cur.get("rated") or 0, x.get("rated") or 0
        merged["rating"] = round(((cur.get("rating") or 0) * n1 + (x.get("rating") or 0) * n2) / (n1 + n2), 2) if n1 + n2 else None
        merged["teams"] = sorted({cur["team"], x["team"]})
        by[r["id"]] = merged
    return list(by.values())

@app.get("/league/players/{league_code}")
def get_league_players(league_code: str, season: int = None):
    """선수 탭: 리그 전체 선수 시즌 기록(골·도움·평점·출전 시간 등 — 순위 카드·전체 목록용). 이번 시즌 = 경기 상세 합산, 지난 시즌 15-16~"""
    code, yr = league_code.upper(), season or CURRENT_SEASON_YEAR
    ps = _league_players_compact(code, yr)
    if not ps:
        return Response(status_code=204)
    mx = max([p["minutes"] or 0 for p in ps] or [0])
    n = len(_cur_details(code)) if yr == CURRENT_SEASON_YEAR else ((_ps_file(code, yr) or {}).get("matches"))
    return {"league": code, "season": yr, "matches": n, "max_minutes": mx, "players": ps}

@app.get("/league/bestxi/{league_code}")
def get_league_bestxi(league_code: str, season: int = None, round: str = None):
    """리그 베스트 11 — 이번 시즌: 라운드별 '이번 라운드의 팀'(그 라운드 45분 이상 뛴 선수 평점 순) + 시즌 베스트,
    지난 시즌: 시즌 베스트(출전 시간이 리그 최다의 40% 이상). round = 숫자 | season"""
    code, yr = league_code.upper(), season or CURRENT_SEASON_YEAR
    if yr == CURRENT_SEASON_YEAR:
        dets = _cur_details(code)
        if not dets:
            return Response(status_code=204)
        rounds = sorted({d["round"] for d in dets if isinstance(d.get("round"), int)})
        if round == "season":
            rows = _league_rows(code, yr) or []
            return {"league": code, "season": yr, "mode": "season", "rounds": rounds, "xi": _xi_out(_season_xi(rows))}
        r = int(round) if round and round.isdigit() else None
        if r is None:   # 기본 = 가장 최근에 (거의) 다 끝난 라운드
            cnt = Counter(d["round"] for d in dets if isinstance(d.get("round"), int))
            full = max(cnt.values()) if cnt else 0
            done = [x for x in rounds if cnt[x] >= full * 0.8]
            r = max(done) if done else (rounds[-1] if rounds else None)
        if r is None:
            return Response(status_code=204)
        xi = _round_xi_cached(code, r)
        return {"league": code, "season": yr, "mode": "round", "round": r, "rounds": rounds, "matches": sum(1 for d in dets if d.get("round") == r), "xi": xi}
    rows = _league_rows(code, yr)
    if not rows:
        return Response(status_code=204)
    return {"league": code, "season": yr, "mode": "season", "rounds": [], "xi": _xi_out(_season_xi(rows))}

@lru_cache(maxsize=128)
def _round_xi_cached(code, r):
    return _xi_out(round_xi([d for d in _cur_details(code) if d.get("round") == r]))

@app.get("/player/seasons/{player_id}")
def get_player_seasons(player_id: int):
    """선수 카드 시즌 고르기: 우리 데이터(5대 리그·챔스, 15-16~)에 이 선수 기록이 있는 [리그, 시즌, 팀] — 이번 시즌 포함"""
    out = [list(x) for x in (_ps_index().get("players") or {}).get(str(player_id), [])]
    for code in ("PL", "PD", "BL1", "SA", "FL1", "CL"):
        for r in _league_rows(code, CURRENT_SEASON_YEAR) or []:
            if r["id"] == player_id:
                out.append([code, CURRENT_SEASON_YEAR, r["team"]])
    out.sort(key=lambda x: (-x[1], x[0] == "CL"))
    return {"id": player_id, "seasons": out}

# 선수 카드 경력·트로피(2026-10-06 개편): 풋몹처럼 구단별 기간(이적 날짜)·경기 수·골 + 트로피를 구단별로 묶고 대회 로고
# 요청 때 API-Football: 경력 시즌 목록 1 + 이적 1 + 트로피 1 + 부상 1 + 시즌별 기록(/players?id=&season=) 시즌 수만큼(2010~, 최근 16시즌까지) — 하루 캐시
AF_NATIONAL_COMPS = {1, 4, 5, 6, 7, 9, 10, 480, 29, 30, 31, 32, 33, 34, 960, 21, 22}   # 국가대표 대회(월드컵·유로·네이션스리그·코파·아프리카·아시안컵·올림픽·예선·친선)
AF_YOUTH_TEAM_RE = re.compile(r"\bU\d{2}\b|\bYouth\b|\bReserves?\b|\bII$|\sB$|\bJong\b|Primavera|Academy", re.I)

def _af_comp_id(name, country):
    lg = _load_json("af_leagues.json") or {}
    for c in (country, "World", "Europe"):
        v = lg.get(f"{name}|{c}")
        if v:
            return v[0]
    return None

def _af_season_stats(pid, yr):
    ttl = 86400 if yr >= CURRENT_SEASON_YEAR - 1 else 86400 * 30
    r = _af_live_get("/players", ttl, id=pid, season=yr)
    return (r or [None])[0]

def _af_profile_file(pid):
    p = os.path.join(MODEL_DIR, "af_profiles", f"{pid}.json.gz")
    return md_read(p) if os.path.exists(p) else None

def _profile_sources(pid):
    """(경력 시즌 목록, 트로피, 부상, 이적, {시즌: 대회별 기록}) — 새벽에 받아 둔 파일(af_profiles/) 먼저, 없으면 요청 때 API-Football
    (한 번에 동시 요청 — 요청 하나가 0.4~1초라 Render에서 처음 여는 선수는 수 초 걸림)"""
    f = _af_profile_file(pid)
    if f:
        return f["teams"], f["trophies"], f["sidelined"], f["transfers"], {int(k): v for k, v in f["stats"].items()}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(20) as ex:
        fut_tm = ex.submit(_af_live_get, "/players/teams", 86400, player=pid)
        fut_tr = ex.submit(_af_live_get, "/trophies", 86400, player=pid)
        fut_sd = ex.submit(_af_live_get, "/sidelined", 86400, player=pid)
        fut_tf = ex.submit(_af_live_get, "/transfers", 86400, player=pid)
        teams = fut_tm.result()
        if teams is None:
            return None
        teams = [{**t, "seasons": [int(y) for y in (t.get("seasons") or []) if str(y).isdigit()]} for t in teams]   # 시즌이 가끔 문자열로 옴
        years = sorted({y for t in teams for y in t["seasons"] if y >= 2010}, reverse=True)[:16]
        stats = dict(zip(years, ex.map(lambda y: af_compact_player_season(_af_season_stats(pid, y)), years)))
        return teams, fut_tr.result() or [], fut_sd.result() or [], af_compact_transfers(fut_tf.result()), stats

@lru_cache(maxsize=1)
def _comp_names():
    """API-Football 대회 ID → (이름, 나라) — af_leagues.json 거꾸로"""
    return {v[0]: tuple(k.split("|", 1)) for k, v in (_load_json("af_leagues.json") or {}).items()}

def _missing_trophies(trophies, by_season):
    """API-Football 트로피 목록에 아직 없는 우승·준우승(25-26 시즌 등)을 결승·최종 순위 결과(comp_winners.json)로 채움 —
    그 시즌 그 팀 소속으로 한 경기라도 뛴 선수만"""
    cw = _load_json("comp_winners.json") or {}
    have = set()
    for t in trophies or []:
        cid = _af_comp_id(t.get("league"), t.get("country"))
        if cid and str(t.get("season") or "")[:4].isdigit():
            have.add((cid, int(str(t["season"])[:4])))
    out = []
    for cid, seasons in cw.items():
        cid = int(cid)
        name, country = _comp_names().get(cid, (None, None))
        if not name:
            continue
        for yr, wr in seasons.items():
            yr = int(yr)
            if (cid, yr) in have:
                continue
            mine = {c["team_id"] for c in by_season.get(yr, []) if c.get("apps")}
            place = "Winner" if wr.get("w") in mine else "2nd Place" if wr.get("r") in mine else None
            if not place:
                continue
            season = str(yr) if cid in AF_CALENDAR_COMPS else f"{yr}/{yr + 1}"
            out.append({"league": name, "country": country, "season": season, "place": {"Winner": "winner", "2nd Place": "runner_up"}[place], "youth": False})
    return out

@app.get("/player/profile/{player_id}")
def get_player_profile(player_id: int):
    """선수 카드 경력·트로피·부상 이력 + 기본 정보(나이·국적·키 — 지난 시즌 선수처럼 우리 프로필 파일에 없는 선수용). 키가 없거나 못 받으면 204"""
    src = _profile_sources(player_id)
    if src is None:
        return Response(status_code=204)
    teams, trophies, sidelined, tlist, stats = src
    years = sorted(stats, reverse=True)
    latest = next((stats[y] for y in years if stats.get(y)), None)
    pl = (latest or {}).get("player") or {}
    nat = pl.get("nationality")
    flags = _load_json("af_countries.json") or {}
    bio = {"name": pl.get("name"), "firstname": pl.get("firstname"), "lastname": pl.get("lastname"), "age": pl.get("age"),
           "birth_date": (pl.get("birth") or {}).get("date"), "birth_place": (pl.get("birth") or {}).get("place"),
           "birth_country": (pl.get("birth") or {}).get("country"), "nationality": nat, "flag": flags.get(nat),
           "height": pl.get("height"), "weight": pl.get("weight"), "photo": pl.get("photo")} if pl else None
    # 구단별 경기 수·골(모든 대회 합산) + 시즌별 소속(트로피를 구단에 붙이는 데 씀)
    agg, by_season = {}, {}
    for y, d in stats.items():
        for st in (d or {}).get("statistics") or []:
            tm, lg, g = st.get("team") or {}, st.get("league") or {}, st.get("games") or {}
            if not tm.get("id"):
                continue
            a = agg.setdefault(tm["id"], {"apps": 0, "goals": 0, "assists": 0})
            a["apps"] += g.get("appearences") or 0
            a["goals"] += (st.get("goals") or {}).get("total") or 0
            a["assists"] += (st.get("goals") or {}).get("assists") or 0
            by_season.setdefault(y, []).append({"team_id": tm["id"], "team": tm.get("name"), "logo": tm.get("logo"), "league_id": lg.get("id"),
                                                "country": lg.get("country"), "apps": g.get("appearences") or 0})
    def is_nat(name):
        return bool(nat) and (name == nat or name.startswith(nat + " "))
    # 이적 날짜 → 구단별 들어온/나간 날(풋몹 "2022년 7월 - 지금")
    ins, outs = {}, {}
    for tr in tlist or []:
        dt, tt = tr.get("date"), tr.get("teams") or {}
        if not dt:
            continue
        i, o = (tt.get("in") or {}).get("id"), (tt.get("out") or {}).get("id")
        if i: ins.setdefault(i, []).append(dt)
        if o: outs.setdefault(o, []).append(dt)
    base = af_player_profile(teams, [], sidelined, nationality=nat, current_season=CURRENT_SEASON_YEAR)
    for c in base["career"]:
        tid = c.get("team_id")
        a = agg.get(tid) or {}
        c.update({"apps": a.get("apps"), "goals": a.get("goals"), "assists": a.get("assists"),
                  "youth": c["youth"] or bool(AF_YOUTH_TEAM_RE.search(c["team"] or "")), "national": is_nat(c["team"] or "")})
        if ins.get(tid):
            c["from_date"] = min(ins[tid])[:7]
        if outs.get(tid) and c.get("to"):
            c["to_date"] = max(outs[tid])[:7]
        if c["national"]:
            c["flag"] = flags.get(nat) if c["team"] == nat else None
    # 트로피 → 구단별 묶음
    groups, seen = {}, set()
    tro = af_player_profile([], trophies, [], nationality=nat)["trophies"] + _missing_trophies(trophies, by_season)
    for t in tro:
        if t["youth"] or t["place"] not in ("winner", "runner_up"):
            continue
        cid = _af_comp_id(t["league"], t.get("country"))
        yr = int(str(t.get("season") or "0")[:4] or 0)
        cands = by_season.get(yr, []) or by_season.get(yr - 1, [])
        if cid in AF_NATIONAL_COMPS or (t.get("country") == "World" and cid not in (15, 2, 3, 848, 531, 1168) and not any(c["league_id"] == cid for c in cands)):
            nt = next((c for c in cands if is_nat(c["team"] or "") and c["team"] == nat), None)
            key = ("nat", nat)
            info = {"team": nat or "국가대표", "logo": flags.get(nat), "national": True, "country": nat}
            if nt:
                info["team_id"] = nt["team_id"]
        else:
            clubs = [c for c in cands if not is_nat(c["team"] or "")]
            hit = [c for c in clubs if c["league_id"] == cid] or [c for c in clubs if c.get("country") == t.get("country")] or clubs
            hit.sort(key=lambda c: -c["apps"])
            c = hit[0] if hit else None
            key = ("club", c["team_id"] if c else t.get("country"))
            info = {"team": c["team"] if c else (t.get("country") or "-"), "team_id": c["team_id"] if c else None, "logo": c["logo"] if c else None,
                    "national": False, "country": t.get("country")}
        g = groups.setdefault(key, {**info, "items": {}})
        it = g["items"].setdefault(t["league"], {"name": t["league"], "comp_id": cid, "country": t.get("country"), "wins": [], "runner": []})
        dk = (key, t["league"], t["place"], t.get("season"))
        if dk in seen:
            continue
        seen.add(dk)
        (it["wins"] if t["place"] == "winner" else it["runner"]).append(t.get("season") or "")
    tgroups = []
    for g in groups.values():
        items = sorted(g.pop("items").values(), key=lambda x: (-len(x["wins"]), -len(x["runner"])))
        g["items"], g["wins"] = items, sum(len(x["wins"]) for x in items)
        tgroups.append(g)
    # 구단 순서 = 경력 순서(지금 팀 → 최근에 떠난 팀 → 그 전), 국가대표는 맨 뒤 — 예전엔 우승 횟수 순이라 옛 팀이 위로 올라왔음
    order = {c.get("team_id"): i for i, c in enumerate(base["career"]) if not c.get("national")}
    last = lambda g: max([int(str(s)[:4]) for it in g["items"] for s in it["wins"] + it["runner"] if str(s)[:4].isdigit()] or [0])
    tgroups.sort(key=lambda g: (g["national"], order.get(g.get("team_id"), 999), -last(g)))
    return {**base, "bio": bio, "trophy_groups": tgroups,
            "trophies": [t for t in tro if not t["youth"]]}   # 예전 화면 호환(평평한 목록)

@app.get("/match/preview")
def get_match_preview(home_team: str, away_team: str, date: str = None):
    """경기 미리보기·예측 결과의 '팀 소식': 라인업(킥오프 약 1시간 전 확정 — 요청 때 받음, 그 전엔 각 팀 지난 경기 선발 = 예상 라인업)
    + 결장·부상자(match_previews.json). 없으면 204"""
    by_key, by_team = _af_index()
    if not date:   # 날짜가 없으면(예측 탭에서 두 팀만 고른 경우) 다가오는 같은 대진 날짜로
        now = pd.Timestamp.now(tz="UTC")
        nxt = sorted(e["date"] for e in by_team.get(home_team, []) if e["away_team"] == away_team and pd.Timestamp(e["date"]) > now - pd.Timedelta(hours=3))
        date = nxt[0] if nxt else ""
    key = f"{home_team}|{away_team}|{date[:10]}"
    pv = dict((_load_json("match_previews.json") or {}).get(key) or {})
    e = by_key.get(key)
    if e:
        now, ko = pd.Timestamp.now(tz="UTC"), pd.Timestamp(e["kickoff"])
        if ko - pd.Timedelta(minutes=90) <= now <= ko + pd.Timedelta(hours=3):   # 확정 라인업은 보통 킥오프 약 1시간 전
            lu = _af_live_get("/fixtures/lineups", 180, fixture=e["id"])
            if lu and len(lu) == 2:
                lus = af_match_detail({"teams": {"home": {"id": e["home_id"]}}, "lineups": lu})["lineups"]
                for sd, team in (("home", home_team), ("away", away_team)):
                    if lus.get(sd):
                        cap = _pick_cap({p["id"] for p in lus[sd]["start"]}, _team_captain_order(team))
                        for p in lus[sd]["start"]:
                            p["cap"] = p["id"] == cap
                pv["lineups"] = lus
    if not (pv.get("lineups") or {}).get("home"):
        def last_xi(team):
            """이 팀의 가장 최근 경기 선발(평점·교체 표시는 뺌) — 예상 라인업"""
            for x in reversed(by_team.get(team, [])):
                if not x.get("has_detail") or x["date"][:10] >= date[:10]:
                    continue
                d = _md_file(x["file"])
                sd = "home" if x["home_team"] == team else "away"
                lu = ((d or {}).get("lineups") or {}).get(sd)
                if lu and lu.get("start"):
                    caps = {p["id"] for p in ((d.get("pstats") or {}).get(sd) or []) if p.get("cap")}
                    return {**lu, "subs": [], "from": x["date"][:10],
                            "start": [{**{k: v for k, v in p.items() if k in ("id", "name", "number", "pos", "grid", "photo")}, "cap": p.get("id") in caps} for p in lu["start"]]}
            return None
        pred = {"home": last_xi(home_team), "away": last_xi(away_team)}
        if pred["home"] or pred["away"]:
            pv["predicted"] = pred
    # 경기별 결장자 명단은 킥오프 1~2일 전에야 올라와서 그 전엔 비어 있었음(2026-10-07) → 비어 있는 팀은 그 팀 "가장 최근 경기"의
    # 결장자(team_injuries.json — 새벽 수집, /injuries 리그·시즌 전체에서 팀별 마지막 경기)로 대신하고 그 경기 날짜를 붙임
    ti = _load_json("team_injuries.json") or {}
    inj = {k: list(v) for k, v in (pv.get("injuries") or {}).items()}
    for sd, team in (("home", home_team), ("away", away_team)):
        if not inj.get(sd) and (ti.get(team) or {}).get("players"):
            inj[sd] = [{**x, "last": ti[team]["date"]} for x in ti[team]["players"]]
    if inj.get("home") or inj.get("away") or pv.get("injuries"):
        pv["injuries"] = {"home": inj.get("home", []), "away": inj.get("away", [])}
    if not pv.get("lineups") and not pv.get("predicted") and not pv.get("injuries"):
        return Response(status_code=204)
    return {**pv, "date": date}



@app.get("/matches/live")
def get_matches_live():
    """오늘 경기 최신 상태만(가벼움) — 화면이 진행 중 경기가 있을 때 1분마다 부름"""
    live = _live_overlay()
    return {"matches": [{"home_team": k.split("|")[0], "away_team": k.split("|")[1], **v} for k, v in live.items()],
            "enabled": bool(os.environ.get("FOOTBALL_API_KEY")),
            "af_enabled": bool(os.environ.get("API_FOOTBALL_KEY"))}   # API-Football 요청 때 받기(확정 라인업·선수 경력)가 켜졌는지 — 값은 안 내보냄

@app.get("/matches/window")
def get_matches_window():
    now = pd.Timestamp.now(tz='UTC')
    lo, hi = now - pd.Timedelta(hours=MATCH_WINDOW_BEFORE_H), now + pd.Timedelta(days=MATCH_WINDOW_AFTER_D)
    live = _live_overlay()
    out = []
    for league, matches in (_load_json("schedule.json") or {}).items():
        for m in matches:
            if lo <= pd.Timestamp(m['date']) <= hi:
                out.append({**_with_live(m, live), "league": league})
    out.sort(key=lambda m: m['date'])
    return {"from": lo.isoformat(), "to": hi.isoformat(), "matches": out}

# ── 리그 페이지 시즌 선택(2026-10-06) ──
# 시즌 목록은 데이터에서: 경기(all_matches.csv — 지금 football-data 무료 4시즌)가 있는 시즌만, 선수(득점·도움)는 이번 시즌만(scorers.json).
# API-Football Pro를 붙여 과거 시즌 경기·선수를 받으면 여기 목록만 늘어나면 됨(프론트는 이 목록대로 드롭다운·탭을 그림)
@lru_cache(maxsize=8)
def _league_seasons(code):
    if code not in LEAGUE_MAP:
        raise HTTPException(status_code=404, detail="리그를 찾을 수 없습니다")
    yrs = sorted({int(y) for y in df_seasons.loc[df_seasons['league'] == code, 'season'].dropna()}, reverse=True)
    if CURRENT_SEASON_YEAR not in yrs:
        yrs.insert(0, CURRENT_SEASON_YEAR)
    sc = (_load_json("scorers.json") or {}).get(code)
    out = []
    for y in yrs:
        cur = y == CURRENT_SEASON_YEAR
        out.append({"year": y, "label": f"{y}/{(y + 1) % 100:02d}", "current": cur,
                    "players": bool(cur and ((sc and sc.get("scorers")) or _cur_details(code))) or os.path.exists(os.path.join(PS_DIR, f"{code}_{y}.json.gz")),
                    "simulation": cur and code in SIM_LEAGUES,
                    "group_stage": code == "CL" and y <= 2023})
    return out

@app.get("/league/seasons/{league_code}")
def get_league_seasons(league_code: str):
    return {"league": league_code.upper(), "seasons": _league_seasons(league_code.upper())}

@lru_cache(maxsize=32)
def _past_schedule(code, yr):
    """끝난 시즌 일정 = all_matches.csv(시각은 없음 — 날짜만). 챔스 토너먼트는 ucl_tournament.json의 1·2차전으로 라운드·시각·승부차기를 찾음"""
    df = df_seasons[(df_seasons['league'] == code) & (df_seasons['season'] == yr)].sort_values(['date', 'home_team'])
    if df.empty:
        raise HTTPException(status_code=404, detail="해당 시즌 데이터 없음")
    legs = {}
    if code == "CL":
        for stage, ties in (((_load_json("ucl_tournament.json") or {}).get("seasons") or {}).get(str(yr)) or {}).items():
            if not isinstance(ties, list):
                continue
            for t in ties:
                for lg in t.get("legs") or []:
                    legs[(lg.get("home_team"), lg.get("away_team"), str(lg.get("date", ""))[:10])] = (stage, lg.get("date"), t.get("pens") if lg is (t.get("legs") or [None])[-1] else None)
    league_end = pd.Timestamp(f"{yr + 1}-02-01")
    out = []
    for _, r in df.iterrows():
        d = r['date'].strftime('%Y-%m-%d')
        ko = r.get('kickoff') if 'kickoff' in r else None
        m = {"date": ko if isinstance(ko, str) and ko else f"{d}T00:00:00Z", "date_only": not (isinstance(ko, str) and ko),
             "matchday": int(r['matchday']) if pd.notna(r['matchday']) and r['matchday'] > 0 else None,
             "stage": "REGULAR_SEASON", "home_team": r['home_team'], "away_team": r['away_team'],
             "home_goals": int(r['home_goals']), "away_goals": int(r['away_goals']), "status": "FINISHED"}
        if isinstance(r.get('stage'), str) and r['stage']:   # 과거 시즌(API-Football)은 라운드가 기록에 있음
            m["stage"] = r['stage']
            if code == "CL" and r['stage'] != "GROUP_STAGE":
                m["matchday"] = None
        elif code == "CL":
            hit = legs.get((r['home_team'], r['away_team'], d))
            if hit:
                m["stage"], m["date"], m["date_only"] = hit[0], hit[1] or m["date"], not hit[1]
                m["matchday"] = None
                if hit[2]:
                    m["penalties"] = hit[2]
            else:
                m["stage"] = ("GROUP_STAGE" if yr <= 2023 else "LEAGUE_STAGE") if r['date'] < league_end else "KNOCKOUT"
        out.append(m)
    return out

# ── 역대 우승·준우승(리그 페이지 "시즌" 탭) — league_history.json(update_data.fetch_league_history, 영문 위키백과 우승 목록) ──
@app.get("/league/history/{league_code}")
def get_league_history(league_code: str):
    code = league_code.upper()
    h = (_load_json("league_history.json") or {}).get(code)
    if not h:
        raise HTTPException(status_code=404, detail="기록 없음")
    have = {s["year"] for s in _league_seasons(code)}
    logos = team_logos_cache
    side = lambda x: {**x, "logo": logos.get(x.get("team") or "", "")}
    return {"league": code, "source": h.get("source"),
            "seasons": [{"season": r["season"], "label": f"{r['season']}/{(r['season'] + 1) % 100:02d}", "has_data": r["season"] in have,
                         "champion": side(r["champion"]), "runner_up": side(r["runner_up"])} for r in h["seasons"]]}

# ── 전체 일정 API ──
@app.get("/schedule/{league_code}")
def get_schedule(league_code: str, season: str = "current"):
    """리그 전체 시즌 일정 (완료 + 예정 경기 전부). season=연도면 끝난 시즌(all_matches.csv)"""
    yr = _season_year(season)
    if yr != CURRENT_SEASON_YEAR:
        return {"league": league_code.upper(), "season": yr, "matches": _past_schedule(league_code.upper(), yr)}
    data = _load_json("schedule.json")
    if data is None:
        raise HTTPException(status_code=404, detail="일정 데이터 없음")

    matches = data.get(league_code.upper())
    if matches is None:
        raise HTTPException(status_code=404, detail="해당 리그 데이터 없음")

    live = _live_overlay()
    if live:   # 오늘 경기만 최신 점수·상태로(캐시된 원본은 건드리지 않고 복사)
        matches = [_with_live(m, live) for m in matches]
    return {"league": league_code.upper(), "matches": matches}

def _squad_full_names(af_squad, fd_squad):
    """API-Football 스쿼드(사진·등번호)의 이름을 football-data 1군 명단의 전체 이름으로 바꿈(2026-09-30).
    API-Football은 "I. Meslier"·"R. Calafiori"처럼 이니셜로 줄여 와서 football-data 명단을 쓰는 다른 팀·화면과 표기가 달랐음.
    성(마지막 단어, 한 단어 이름이면 그 단어)이 같고 이니셜이 맞는 사람이 딱 한 명일 때만 바꾸고, 애매하면 원래 이름 그대로"""
    # 바꾸는 경우는 두 가지뿐: "I. Meslier"처럼 이니셜로 줄인 이름(이니셜이 맞아야 함), "Kepa"처럼 한 단어 이름.
    # 이미 전체 이름("Maroan Sannadi")이면 그대로 — football-data 쪽이 더 짧은 경우("Sannadi")도 있어서.
    # 성만 같은 다른 선수(유스 "Z. Christie" ↔ 1군 "Ryan Christie", "Vitor Nunes" ↔ "Matheus Nunes")로 바뀌지 않게 이니셜은 엄격히
    fd = [(p.get("name"), _norm_name(p.get("name"))) for p in fd_squad if p.get("name")]
    picks = []
    for p in af_squad:
        raw = p.get("name") or ""
        toks = _norm_name(raw)
        abbrev = bool(re.match(r"^\S\.\s", raw))
        cand = None
        if toks and (abbrev or len(raw.split()) == 1):
            last = toks[-1]
            cands = [n for n, t in fd if last in t and len(t) > len(toks) - (1 if abbrev else 0)]
            if abbrev and len(toks) > 1:
                cands = [n for n in cands if _norm_name(n)[0].startswith(toks[0][0])]
            if len(cands) == 1:
                cand = cands[0]
        picks.append(cand)
    used = {n: picks.count(n) for n in picks if n}
    return [{**p, "name": n, "short_name": p.get("name")} if n and used[n] == 1 else p for p, n in zip(af_squad, picks)]

def _manager_of(team):
    """지금 감독(af_coaches.json — update_data.fetch_coaches: 최근 경기 라인업 감독 + API-Football 사진·부임일 + 위키데이터 생년월일·국적)"""
    c = (_load_json("af_coaches.json") or {}).get(team)
    if not c or not c.get("name"):
        return None
    out = {k: v for k, v in c.items() if k != "updated"}
    b = c.get("birth_date")
    if b:
        try:
            bd, today = datetime.strptime(b[:10], "%Y-%m-%d").date(), datetime.now(timezone.utc).date()
            out["age"] = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
        except ValueError:
            pass
    st = c.get("start")
    if st:   # 부임 후 기록(우리 데이터의 리그·챔스 경기 — 부임일이 월 단위라 그 달 초 경기가 섞일 수 있음)
        h = df_seasons[((df_seasons.home_team == team) | (df_seasons.away_team == team)) & (df_seasons.date >= pd.Timestamp(st, tz=df_seasons.date.dt.tz))].dropna(subset=["home_goals"]) \
            .drop_duplicates(subset=["date", "home_team", "away_team"])
        if len(h):
            gf, ga = _team_goals(h, team)
            out["record"] = {"played": len(h), "wins": int((gf > ga).sum()), "draws": int((gf == ga).sum()), "losses": int((gf < ga).sum()),
                             "since": str(h.date.min())[:10]}
    return out

@app.get("/team/info/{team_name}")
def get_team_info(team_name: str):
    """팀 상세 정보 (홈구장, 창단연도, 구단색, 스쿼드) — update_data.py의 fetch_team_info()가 생성한 캐시"""
    data = _load_json("team_info.json")
    if data is None:
        raise HTTPException(status_code=404, detail="팀 정보 데이터 없음")

    info = data.get(team_name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"팀 정보를 찾을 수 없습니다: {team_name}")

    # API-Football 스쿼드 사진/등번호 + 이적 기록 (현재는 PL만 — fetch_squad_transfers() 참고)
    extra = (_load_json("team_extra.json") or {}).get(team_name)
    if extra:
        info = {**info}   # 캐시된 원본을 건드리지 않게 복사본에 합침
        if extra.get("squad"):
            info["squad"] = _squad_full_names(extra["squad"], info.get("squad") or [])
        if extra.get("transfers") is not None:
            info["transfers"] = extra["transfers"]

    # 구단 소개·별칭·홈구장 수용 인원·감독·연고지(위키백과/위키데이터 — update_data.py fetch_team_wiki)
    wiki = (_load_json("team_wiki.json") or {}).get(team_name)
    if wiki:
        info = {**info, "wiki": wiki}
    mgr = _manager_of(team_name)
    if mgr:
        info = {**info, "manager": mgr}
    # 주요 라이벌(수동 관리 rivals.json) + 우리 데이터에 있는 맞대결 전적(2010-11 이후 리그·챔스 — 2026-10-07 예전엔 최근 4시즌)
    rivals = (_load_json("rivals.json") or {}).get(team_name)
    if rivals:
        logos = team_logos_cache
        out = []
        for r in rivals:
            opp = r["opponent"]
            h = df_seasons[((df_seasons.home_team == team_name) & (df_seasons.away_team == opp)) |
                           ((df_seasons.home_team == opp) & (df_seasons.away_team == team_name))].dropna(subset=['home_goals']).drop_duplicates(subset=['date', 'home_team', 'away_team'])
            gf, ga = _team_goals(h, team_name) if len(h) else (pd.Series(dtype=float), pd.Series(dtype=float))
            out.append({**r, "logo": logos.get(opp, ""), "played": len(h),
                        "wins": int((gf > ga).sum()), "draws": int((gf == ga).sum()), "losses": int((gf < ga).sum())})
        info = {**info, "rivals": out}

    return info

# ── 팀 로고 이미지 프록시 ──
# crests.football-data.org / wikimedia는 CORS 헤더를 안 내려줘서, 프론트에서
# <canvas>에 로고를 그려 예측 결과 공유카드 이미지를 만들 때 canvas가 tainted되어
# toDataURL/toBlob이 막힌다. 허용 호스트로 제한한 프록시를 거쳐 우리 서버(CORS 전체 허용)
# 응답으로 내려주면 crossOrigin="anonymous"로 안전하게 로드해 캔버스에 사용할 수 있다.
SITE_HOSTS = {"www.fotdata-official.com", "fotdata-official.com", "fotdata-api.vercel.app"}   # 우리 사이트 주소(옛 vercel.app 포함)
PROXY_ALLOWED_HOSTS = {"crests.football-data.org", "upload.wikimedia.org"} | SITE_HOSTS
_HD_LOGO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logos", "hd")

@app.get("/proxy/logo")
def proxy_logo(url: str):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in PROXY_ALLOWED_HOSTS:
        raise HTTPException(status_code=400, detail="허용되지 않은 이미지 URL")
    # 우리 고화질 로고(logos/hd)는 저장소에 같이 배포돼 있으니 디스크에서 바로 (Vercel까지 왕복 안 함)
    m = re.fullmatch(r"/logos/hd/((?:l/)?[a-z0-9-]+\.webp)", parsed.path)
    if parsed.hostname in SITE_HOSTS and m and os.path.exists(os.path.join(_HD_LOGO_DIR, m.group(1))):
        with open(os.path.join(_HD_LOGO_DIR, m.group(1)), "rb") as f:
            return Response(content=f.read(), media_type="image/webp", headers={"Cache-Control": "public, max-age=86400"})
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
    except Exception:
        raise HTTPException(status_code=502, detail="이미지를 불러올 수 없습니다")
    return Response(
        content=resp.content,
        media_type=resp.headers.get("Content-Type", "image/png"),
        headers={"Cache-Control": "public, max-age=86400"},
    )
# ── 경기별 링크 공유 미리보기 (2026-09-30) ──
# 카톡·페북·X의 링크 미리보기는 JS를 실행하지 않아서 앱 주소(/app?match=..., 예전 FotData.html?match=...)로는 사이트 공통 썸네일만 떴음.
# 공유 링크를 https://www.fotdata-official.com/m/{홈}-vs-{원정} 으로 바꾸고, Vercel이 이 주소를 여기로 넘겨줌(vercel.json rewrites):
#   /m/{slug}          → /share/match/{slug}  : 그 경기 예측이 담긴 og 태그 + 사람은 JS로 앱 예측 화면으로 이동
#   /og/m/{slug}.jpg   → /og/match/{slug}.jpg : 1200×630 예측 카드(share_card.py)
# slug 규칙은 FotData.html teamSlug()와 같아야 함(앱이 만든 링크를 여기서 풀어야 하므로)
SITE_URL = "https://www.fotdata-official.com"   # 2026-10-03 도메인 이전(예전 https://fotdata-api.vercel.app)
LEAGUE_KO = {"PL": "프리미어리그", "PD": "라리가", "BL1": "분데스리가", "SA": "세리에 A", "FL1": "리그 1", "CL": "챔피언스리그"}
_SLUG_CLUB = re.compile(r"\b(fc|afc|cf|ac|sc|ssc|us|as|rc|rcd|cd|sv|tsg|vfb|vfl|ogc|aj|bc)\b", re.ASCII)

def _team_slug(name):
    import unicodedata
    s = re.sub(r"^\d+\.\s*", "", name or "")
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not ("̀" <= c <= "ͯ")).lower()
    s = _SLUG_CLUB.sub(" ", s)
    return re.sub(r"[^a-z0-9]+", "-", s, flags=re.ASCII).strip("-")

@lru_cache(maxsize=1)
def _slug_map():
    """slug → 예측에 쓰는 팀 이름. 앱이 쓰는 이름(LEAGUE_DATA 표기·API 표기) 어느 쪽 slug로 와도 찾게"""
    names = set(team_logos_cache) | set(df_stats['team']) | set(ucl_only_teams) | set(TEAM_NAME_MAP)
    out = {}
    for n in sorted(names):
        api = TEAM_NAME_MAP.get(n, n)
        s = _team_slug(n)
        if s and (s not in out or (api in team_state and out[s] not in team_state)):
            out[s] = api
    return out

@lru_cache(maxsize=1)
def _short_names():
    """FotData.html의 SHORT_NAMES 표를 그대로 읽어 씀(표를 두 군데서 관리하지 않으려고) — 못 읽으면 빈 표"""
    try:
        with open(os.path.join(BASE, "FotData.html"), encoding="utf-8") as f:
            html = f.read()
        block = html[html.index("const SHORT_NAMES = {"):]
        block = block[:block.index("};")]
        return dict(re.findall(r"'([^']+)':\s*'([^']+)'", block))
    except Exception:
        return {}

def _short_name(name):
    """FotData.html shortName()과 같은 규칙"""
    if name in _short_names():
        return _short_names()[name]
    t = re.sub(r"\b(FC|CF|AFC|BC|SC|AC|SK|FK|SSC|US|KV)\b|\b\d{4}\b", "", re.sub(r"^\d+\.\s*", "", name))
    return re.sub(r"\s+", " ", t).strip() or name

def _fixture_meta(home, away):
    """다가오는 같은 대진(홈·원정 그대로)이 일정에 있으면 '챔피언스리그 · 10월 21일(수) 04:00'(한국 시간)"""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    best = None
    for code, ms in (_load_json("schedule.json") or {}).items():
        for m in ms:
            if m.get("status") in ("FINISHED", "AWARDED", "CANCELLED"):
                continue
            if TEAM_NAME_MAP.get(m["home_team"], m["home_team"]) != home or TEAM_NAME_MAP.get(m["away_team"], m["away_team"]) != away:
                continue
            dt = datetime.fromisoformat(m["date"].replace("Z", "+00:00"))
            if dt > now - timedelta(hours=3) and (best is None or dt < best[0]):
                best = (dt, code)
    if not best:
        return None
    k = best[0] + timedelta(hours=9)
    return f"{LEAGUE_KO.get(best[1], best[1])} · {k.month}월 {k.day}일({'월화수목금토일'[k.weekday()]}) {k:%H:%M}"

@lru_cache(maxsize=512)
def _share_data(slug):
    """slug → 공유 미리보기에 쓸 예측 요약. 풀 수 없거나 예측할 수 없는 경기면 None"""
    if "-vs-" not in slug:
        return None
    hs, as_ = slug.split("-vs-", 1)
    sm = _slug_map()
    home, away = sm.get(hs), sm.get(as_)
    if not home or not away or home == away:
        return None
    try:
        r = predict_match(MatchRequest(home_team=home, away_team=away))
    except HTTPException:
        return None
    p = r["probabilities"]
    return {
        "slug": f"{hs}-vs-{as_}", "home": home, "away": away,
        "home_short": _short_name(home), "away_short": _short_name(away),
        "probs": (p["home_win"], p["draw"], p["away_win"]),
        "prediction": r["prediction"], "score": (r.get("score_prediction") or {}).get("most_likely"),
        "limited": r.get("limited", False), "meta": _fixture_meta(home, away),
    }

def _data_version():
    """썸네일 주소에 붙이는 버전 — 매일 예측이 바뀌면 메신저가 새 이미지를 받아가게(모델 학습 시각)"""
    acc = _load_json("accuracy.json") or {}
    return re.sub(r"\D", "", str(acc.get("updated_at") or ""))[:10] or "1"

@app.get("/share/match/{slug}")
def share_match_page(slug: str):
    from fastapi.responses import HTMLResponse
    from html import escape
    slug = slug.lower().strip()
    app_url = f"{SITE_URL}/app?match={slug}"   # 앱 주소 /app(vercel.json이 FotData.html로 넘김)
    d = _share_data(slug)
    if d:
        hp, dp, ap = (round(x * 100) for x in d["probs"])
        pick = (f"{d['home_short']} 승 {hp}%" if d["prediction"] == "home_win"
                else f"{d['away_short']} 승 {ap}%" if d["prediction"] == "away_win" else f"무승부 {dp}%")
        title = f"{d['home_short']} vs {d['away_short']} — AI 예측: {pick}"
        desc = f"홈 {hp}% · 무 {dp}% · 원정 {ap}%"
        if d["score"]:
            desc += f" · 예상 스코어 {d['score']}"
        desc += f" | {d['meta']}" if d["meta"] else ""
        desc += " — FotData AI 축구 경기 예측"
        image = f"{SITE_URL}/og/m/{d['slug']}.jpg?v={_data_version()}"
    else:
        title, desc, image = "FotData — AI 축구 경기 예측", "5대 리그 + 챔피언스리그 AI 경기 예측", f"{SITE_URL}/og-image.png?v=6"
    e = lambda s: escape(s, quote=True)
    html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)}</title>
<meta name="robots" content="noindex">
<meta name="description" content="{e(desc)}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="FotData">
<meta property="og:locale" content="ko_KR">
<meta property="og:title" content="{e(title)}">
<meta property="og:description" content="{e(desc)}">
<meta property="og:url" content="{e(f'{SITE_URL}/m/{slug}')}">
<meta property="og:image" content="{e(image)}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{e(title)}">
<meta name="twitter:description" content="{e(desc)}">
<meta name="twitter:image" content="{e(image)}">
<meta name="theme-color" content="#0d1117">
<script>location.replace({json.dumps(app_url)});</script>
</head><body style="margin:0;background:#0d1117;color:#e6edf3;font-family:-apple-system,sans-serif;display:grid;place-items:center;min-height:100vh">
<a href="{e(app_url)}" style="color:#58a6ff">FotData에서 예측 보기</a>
</body></html>"""
    # 사람은 곧바로 앱으로 넘어가고, 미리보기 봇(JS 실행 안 함)만 위 태그를 읽음. 링크 미리보기 봇이 og:url을
    # 다시 긁어가도 같은 페이지라 안전(og:url을 앱 주소로 두면 페북이 그쪽 공통 태그로 덮어씀)
    return HTMLResponse(html, headers={"Cache-Control": "public, max-age=600, s-maxage=3600"})

@app.get("/og/match/{name}")
def share_match_image(name: str):
    slug = re.sub(r"\.(jpe?g|png)$", "", name.lower())
    d = _share_data(slug)
    if not d:
        return Response(status_code=302, headers={"Location": f"{SITE_URL}/og-image.png?v=6"})
    return Response(_share_jpg(d["home"], d["away"]), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400, s-maxage=43200"})

@lru_cache(maxsize=256)
def _share_jpg(home, away):
    """대진(예측에 쓰는 팀 이름) 기준 캐시 — 같은 경기를 다른 표기의 slug로 불러도 한 번만 그림"""
    import share_card   # Pillow는 이 기능에서만 씀 — 서버 시작 시간에 영향 없게 지연 import
    d = _share_data(f"{_team_slug(home)}-vs-{_team_slug(away)}")
    return share_card.render(d["home_short"], d["away_short"], team_logos_cache.get(home), team_logos_cache.get(away),
                             d["probs"], d["score"], d["prediction"], d["meta"], d["limited"])

def _warm_share_cards(days=10):
    """앞으로 며칠 안의 실제 경기 카드를 미리 그려 둠 — 공유는 대부분 곧 열릴 경기라, 첫 미리보기 봇도 바로 받게"""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    for ms in (_load_json("schedule.json") or {}).values():
        for m in ms:
            if m.get("status") in ("FINISHED", "AWARDED", "CANCELLED"):
                continue
            dt = datetime.fromisoformat(m["date"].replace("Z", "+00:00"))
            if not (now <= dt <= now + timedelta(days=days)):
                continue
            h, a = TEAM_NAME_MAP.get(m["home_team"], m["home_team"]), TEAM_NAME_MAP.get(m["away_team"], m["away_team"])
            try:
                if _share_data(f"{_team_slug(h)}-vs-{_team_slug(a)}"):
                    _share_jpg(h, a)
            except Exception:
                pass
            _time.sleep(0.1)   # 한 장마다 쉬어서 사용자 요청이 먼저(서버 시작 뒤 미리 그리는 중에도 화면이 느려지지 않게)

# ── 구단 일정 캘린더 구독(.ics) (2026-10-03, 정식 출시 전 킵 항목) ──
# 앱의 "캘린더에 추가"가 webcal:// 주소로 구독하게 함 → 킥오프 시각이 바뀌거나(방송 일정 확정) 결과가 나면 캘린더 앱이
# 다음 새로고침 때 따라 바뀜(파일 한 번 내려받기는 안 바뀜). Vercel /cal/* 이 여기로 넘겨줌(vercel.json)
#   /calendar/liverpool.ics            한 팀(slug는 공유 링크와 같은 규칙)
#   /calendar/my.ics?t=liverpool,arsenal  즐겨찾기 여러 팀(최대 5)
CAL_EVENT_MINUTES = 115   # 경기 시간(전후반 + 하프타임 + 추가시간)

def _ics_text(s):
    """iCalendar 텍스트 값 이스케이프(RFC 5545 3.3.11)"""
    return str(s).replace("\\", "\\\\").replace(";", "\;").replace(",", "\\,").replace("\n", "\\n")

def _ics_fold(line):
    """한 줄 75바이트 넘으면 접기(한글은 3바이트라 글자 중간에서 자르지 않게 바이트 단위로 셈)"""
    out, cur = [], ""
    for ch in line:
        if len((cur + ch).encode("utf-8")) > (75 if not out else 74):
            out.append(cur); cur = ch
        else:
            cur += ch
    out.append(cur)
    return "\r\n ".join(out)

@lru_cache(maxsize=256)
def _team_calendar(teams, version):
    """teams: 예측에 쓰는 팀 이름 튜플. version은 캐시 무효화용(일정·라이브 반영 시각)"""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    want = set(teams)
    events = []
    for code, ms in (_load_json("schedule.json") or {}).items():
        preds, results = {}, {}
        try:
            sp = _schedule_predictions(code)
            preds = {(p["home_team"], p["away_team"], p["date"]): p for p in sp["predictions"]}
            results = {(p["home_team"], p["away_team"], p["date"]): p for p in sp["results"]}   # 경기 전에 기록된 예측의 적중 여부
        except Exception:
            pass
        for m in ms:
            h, a = TEAM_NAME_MAP.get(m["home_team"], m["home_team"]), TEAM_NAME_MAP.get(m["away_team"], m["away_team"])
            if (h not in want and a not in want) or m.get("status") == "CANCELLED":
                continue
            dt = datetime.fromisoformat(m["date"].replace("Z", "+00:00"))
            if dt < now - timedelta(days=45):   # 지난 경기는 최근 45일만(구독 파일이 커지지 않게)
                continue
            hs, as_ = _short_name(h), _short_name(a)
            done = m.get("status") in DONE_STATUSES and m.get("home_goals") is not None
            if done:
                summary = f"{hs} {m['home_goals']}-{m['away_goals']} {as_}"
            else:
                summary = f"{hs} vs {as_}" + (" (연기)" if m.get("status") == "POSTPONED" else "")
            lines = [f"{LEAGUE_KO.get(code, code)}" + (f" · {m['matchday']}라운드" if m.get("matchday") and code != "CL" else "")]
            if m.get("status") == "SCHEDULED":
                lines.append("킥오프 시각 미정 — 확정되면 자동으로 바뀌어요")
            p = preds.get((m["home_team"], m["away_team"], m["date"]))
            if p and not done:
                hp, dp, ap = (round(x * 100) for x in p["p"])
                lines.append(f"AI 예측: {hs} 승 {hp}% · 무 {dp}% · {as_} 승 {ap}%" + (" (참고용)" if p.get("limited") else ""))
            res = results.get((m["home_team"], m["away_team"], m["date"]))
            if res and done:
                lines.append(f"AI 예측 {'적중' if res['correct'] else '빗나감'} (경기 전 기록)")
            if m.get("penalties"):
                lines.append(f"승부차기 {m['penalties'][0]}-{m['penalties'][1]}")
            url = f"{SITE_URL}/m/{_team_slug(h)}-vs-{_team_slug(a)}"
            lines.append(f"AI 분석 보기: {url}")
            uid = f"{m['date'][:10]}-{_team_slug(h)}-{_team_slug(a)}@fotdata"
            end = dt + timedelta(minutes=CAL_EVENT_MINUTES)
            events.append([
                "BEGIN:VEVENT", f"UID:{uid}", f"DTSTAMP:{stamp}",
                f"DTSTART:{dt.strftime('%Y%m%dT%H%M%SZ')}", f"DTEND:{end.strftime('%Y%m%dT%H%M%SZ')}",
                f"SUMMARY:{_ics_text(summary)}", f"DESCRIPTION:{_ics_text(chr(10).join(lines))}",
                f"URL:{url}", "TRANSP:TRANSPARENT", "END:VEVENT"])
    name = _short_name(teams[0]) if len(teams) == 1 else "내 팀"
    head = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//FotData//Team Fixtures//KO", "CALSCALE:GREGORIAN", "METHOD:PUBLISH",
            f"X-WR-CALNAME:{_ics_text(f'FotData · {name} 경기 일정')}",
            # 일정 제목엔 그림을 넣을 수 없어서(캘린더 앱 공통) 이모지 대신 브랜드 색 + 캘린더 아이콘(RFC 7986, 지원하는 앱만)
            "COLOR:dodgerblue", "X-APPLE-CALENDAR-COLOR:#58A6FF",
            f"IMAGE;VALUE=URI;DISPLAY=BADGE;FMTTYPE=image/png:{SITE_URL}/icon-192.png",
            f"X-WR-CALDESC:{_ics_text('FotData가 매일 갱신하는 경기 일정·결과와 AI 예측 — ' + SITE_URL)}",
            "X-WR-TIMEZONE:Asia/Seoul", "REFRESH-INTERVAL;VALUE=DURATION:PT6H", "X-PUBLISHED-TTL:PT6H"]
    body = head + [l for ev in sorted(events, key=lambda e: e[3]) for l in ev] + ["END:VCALENDAR"]
    return "\r\n".join(_ics_fold(l) for l in body) + "\r\n"

@app.get("/calendar/{name}")
def team_calendar(name: str, t: str = ""):
    sm = _slug_map()
    slugs = [s for s in (t.split(",") if t else [re.sub(r"\.ics$", "", name.lower())]) if s][:5]
    teams = tuple(dict.fromkeys(sm[s.strip().lower()] for s in slugs if s.strip().lower() in sm))
    if not teams:
        raise HTTPException(status_code=404, detail="팀을 찾을 수 없어요")
    ics = _team_calendar(teams, _data_version())
    fname = f"fotdata-{_team_slug(teams[0]) if len(teams) == 1 else 'my-teams'}.ics"
    return Response(ics, media_type="text/calendar; charset=utf-8",
                    headers={"Content-Disposition": f'inline; filename="{fname}"', "Cache-Control": "public, max-age=1800"})
