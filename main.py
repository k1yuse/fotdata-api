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
from functools import lru_cache

app = FastAPI(title="FotData API", version="1.0.0")

# CORS 설정 (나중에 웹/앱에서 호출 가능하게)
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
def get_logos_with_mapping():
    logo_path = os.path.join(MODEL_DIR, "team_logos.json")
    print(f"로고 파일 경로: {logo_path}")
    print(f"파일 존재: {os.path.exists(logo_path)}")
    try:
        with open(logo_path, 'r', encoding='utf-8') as f:
            logos = json.load(f)
        print(f"로고 수: {len(logos)}")
        for html_name, api_name in TEAM_NAME_MAP.items():
            if api_name in logos:
                logos[html_name] = logos[api_name]
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
        },
        "away_stats": {
            "attack":   round(float(a['attack_strength']), 3),
            "defense":  round(float(a['defense_strength']), 3),
            "win_rate": round(float(a['win_rate']), 3),
        }
    }

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
from datetime import datetime

df_matches_all = pd.read_csv(os.path.join(MODEL_DIR, "all_matches.csv"))
df_matches_all['date'] = pd.to_datetime(df_matches_all['date'])

print(f"✅ 전체 경기 데이터 로드: {len(df_matches_all)}경기")

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
def get_standings(league_code: str, season: str = "current"):
    return _standings(league_code.upper(), season)

@lru_cache(maxsize=32)
def _standings(league_code: str, season: str):
    league_name = LEAGUE_MAP.get(league_code.upper())
    if not league_name:
        raise HTTPException(status_code=404, detail="리그를 찾을 수 없습니다")

    if season == "previous":
        cutoff = pd.Timestamp('2025-08-01')
        end = pd.Timestamp('2026-08-01')
        league_name = f"{league_name} (2025-26)"
    else:
        cutoff = pd.Timestamp('2026-08-01')
        end = pd.Timestamp('2027-08-01')
        league_name = f"{league_name} (2026-27)"

    filters = (df_matches_all['league'] == league_code.upper()) & (df_matches_all['date'] >= cutoff) & (df_matches_all['date'] < end)
    if league_code.upper() == 'CL':
        if season == "previous":
            filters = filters & (df_matches_all['date'] < pd.Timestamp('2026-02-01'))
        else:
            filters = filters & (df_matches_all['date'] < pd.Timestamp('2027-02-01'))
    league_df = df_matches_all[filters].copy()


    if league_df.empty:
        raise HTTPException(status_code=404, detail="데이터 없음")

    logos = _load_json("team_logos.json") or {}

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

    return {"league": league_name, "standings": rows}

# ── UCL 토너먼트 API ──
@app.get("/ucl/tournament")
def get_ucl_tournament():
    data = _load_json("ucl_tournament.json")
    if data is None:
        raise HTTPException(status_code=404, detail="UCL 토너먼트 데이터 없음")
    return data
   
# ── H2H API ──
@app.get("/h2h")
def get_h2h(home_team: str, away_team: str, limit: int = 10):
    df_h2h = df_matches_all[
        ((df_matches_all['home_team']==home_team) & (df_matches_all['away_team']==away_team)) |
        ((df_matches_all['home_team']==away_team) & (df_matches_all['away_team']==home_team))
    ].sort_values('date', ascending=False).head(limit)

    if df_h2h.empty:
        return {"home_team": home_team, "away_team": away_team, "matches": [], "summary": {"home_wins":0,"draws":0,"away_wins":0}}

    home_wins = away_wins = draws = 0
    matches = []

    for _, row in df_h2h.iterrows():
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
    요약 + 최근 완료 경기 목록"""
    log = _load_json("prediction_log.json")
    if log is None:
        return {"summary": {"total_scheduled": 0, "total_resolved": 0}, "recent": []}

    # total_scheduled: 예정 경기까지 포함해 기록해둔 전체 건수 (아직 결과 없는 것 포함)
    # total_resolved: 그중 실제로 경기가 끝나 적중 여부를 확정한 건수 — 적중률 계산은 이 값 기준
    total_scheduled = len(log)
    resolved = [e for e in log.values() if e.get("actual") is not None]
    resolved.sort(key=lambda e: e["date"])
    if not resolved:
        return {"summary": {"total_scheduled": total_scheduled, "total_resolved": 0}, "recent": []}

    recent = resolved[-50:]
    total_correct = sum(1 for e in resolved if e["correct"])
    recent_correct = sum(1 for e in recent if e["correct"])

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
        },
        "accuracy_trend": accuracy_trend,
        "recent": list(reversed(recent[-20:])),
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

@app.get("/matches/window")
def get_matches_window():
    now = pd.Timestamp.now(tz='UTC')
    lo, hi = now - pd.Timedelta(hours=MATCH_WINDOW_BEFORE_H), now + pd.Timedelta(days=MATCH_WINDOW_AFTER_D)
    out = []
    for league, matches in (_load_json("schedule.json") or {}).items():
        for m in matches:
            if lo <= pd.Timestamp(m['date']) <= hi:
                out.append({**m, "league": league})
    out.sort(key=lambda m: m['date'])
    return {"from": lo.isoformat(), "to": hi.isoformat(), "matches": out}

# ── 전체 일정 API ──
@app.get("/schedule/{league_code}")
def get_schedule(league_code: str):
    """리그 전체 시즌 일정 (완료 + 예정 경기 전부)"""
    data = _load_json("schedule.json")
    if data is None:
        raise HTTPException(status_code=404, detail="일정 데이터 없음")

    matches = data.get(league_code.upper())
    if matches is None:
        raise HTTPException(status_code=404, detail="해당 리그 데이터 없음")

    return {"league": league_code.upper(), "matches": matches}

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
            info["squad"] = extra["squad"]
        if extra.get("transfers") is not None:
            info["transfers"] = extra["transfers"]

    return info

# ── 팀 로고 이미지 프록시 ──
# crests.football-data.org / wikimedia는 CORS 헤더를 안 내려줘서, 프론트에서
# <canvas>에 로고를 그려 예측 결과 공유카드 이미지를 만들 때 canvas가 tainted되어
# toDataURL/toBlob이 막힌다. 허용 호스트로 제한한 프록시를 거쳐 우리 서버(CORS 전체 허용)
# 응답으로 내려주면 crossOrigin="anonymous"로 안전하게 로드해 캔버스에 사용할 수 있다.
PROXY_ALLOWED_HOSTS = {"crests.football-data.org", "upload.wikimedia.org"}

@app.get("/proxy/logo")
def proxy_logo(url: str):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in PROXY_ALLOWED_HOSTS:
        raise HTTPException(status_code=400, detail="허용되지 않은 이미지 URL")
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