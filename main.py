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
    승/무/패 예측 모델(ELO+prestige+홈어드밴티지 등 반영)이 내놓은 승률 격차와
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
    # ELO와 승률은 ELO 자체가 승률에서 파생된 값이라 서로 강하게 상관돼 있음 —
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
    return {"teams": teams, "count": len(teams)}

@app.post("/predict")
def predict_match(req: MatchRequest):
    """경기 결과 예측"""
    # 팀 이름 매핑
    home_team = TEAM_NAME_MAP.get(req.home_team, req.home_team)
    away_team = TEAM_NAME_MAP.get(req.away_team, req.away_team)
    
    h = df_stats[df_stats['team'] == home_team]
    a = df_stats[df_stats['team'] == away_team]

    if h.empty:
        raise HTTPException(status_code=404, detail=f"팀을 찾을 수 없습니다: {home_team}")
    if a.empty:
        raise HTTPException(status_code=404, detail=f"팀을 찾을 수 없습니다: {away_team}")

    h, a = h.iloc[0], a.iloc[0]

   # H2H 홈팀 승률 계산
    h2h_df = df_matches_all[
        ((df_matches_all['home_team']==home_team) & (df_matches_all['away_team']==away_team)) |
        ((df_matches_all['home_team']==away_team) & (df_matches_all['away_team']==home_team))
    ].tail(10)
    if len(h2h_df) > 0:
        h2h_home_wins = len(h2h_df[((h2h_df['home_team']==home_team) & (h2h_df['result']=='H')) |
                                    ((h2h_df['away_team']==home_team) & (h2h_df['result']=='A'))])
        h2h_rate = round(h2h_home_wins / len(h2h_df), 3)
    else:
        h2h_rate = 0.33

    # ELO 점수 추정 (승률 + prestige + 전력차 기반 홈 어드밴티지 감쇠)
    home_prestige = h['prestige'] if 'prestige' in h.index and pd.notna(h['prestige']) else 0
    away_prestige = a['prestige'] if 'prestige' in a.index and pd.notna(a['prestige']) else 0

    base_home_elo = 1500 + (h['win_rate'] - 0.33) * 1000
    base_away_elo = 1500 + (a['win_rate'] - 0.33) * 1000
    elo_gap = abs(base_home_elo - base_away_elo)

    home_advantage_base = 70
    home_advantage_factor = max(0.4, 1 - elo_gap / 800)
    home_advantage_bonus = home_advantage_base * home_advantage_factor

    home_elo = base_home_elo + home_prestige + home_advantage_bonus
    away_elo = base_away_elo + away_prestige

    input_data = pd.DataFrame([{
        'home_elo':          home_elo,
        'away_elo':          away_elo,
        'elo_diff':          home_elo - away_elo,
        'home_form':         h['win_rate'] * 15,
        'away_form':         a['win_rate'] * 15,
        'form_diff':         (h['win_rate'] - a['win_rate']) * 15,
        'home_avg_scored':   h['attack_strength'],
        'away_avg_scored':   a['attack_strength'],
        'home_avg_conceded': h['defense_strength'],
        'away_avg_conceded': a['defense_strength'],
        'home_attack':       h['attack_strength'],
        'away_attack':       a['attack_strength'],
        'home_defense':      h['defense_strength'],
        'away_defense':      a['defense_strength'],
        'home_win_rate':     h['win_rate'],
        'away_win_rate':     a['win_rate'],
        'win_rate_diff':     h['win_rate'] - a['win_rate'],
        'h2h_home_rate':     h2h_rate,
    }])

    input_scaled = scaler.transform(input_data)
    proba = lr_model.predict_proba(input_scaled)[0]
    classes = lr_model.classes_
    proba_dict = dict(zip(classes, proba))

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

    print(f"🔍 {league_code} 데이터: {len(league_df)}경기")
    print(f"🔍 전체 리그: {df_matches_all['league'].unique()}")

    if league_df.empty:
        raise HTTPException(status_code=404, detail="데이터 없음")

    # 로고 불러오기
    logo_path = os.path.join(MODEL_DIR, "team_logos.json")
    logos = {}
    if os.path.exists(logo_path):
        with open(logo_path, 'r', encoding='utf-8') as f:
            logos = json.load(f)

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
    tournament_path = os.path.join(MODEL_DIR, "ucl_tournament.json")
    if not os.path.exists(tournament_path):
        raise HTTPException(status_code=404, detail="UCL 토너먼트 데이터 없음")
    with open(tournament_path, 'r', encoding='utf-8') as f:
        return json.load(f)
   
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

    # ── 선수 데이터 API ──
import json as _json

# ── 모델 정확도 API ──
@app.get("/accuracy")
def get_accuracy():
    """AI 모델 정확도 정보"""
    accuracy_path = os.path.join(MODEL_DIR, "accuracy.json")
    if not os.path.exists(accuracy_path):
        return {
            "best": 56.3,
            "total_matches": 4567,
            "training_matches": 4059,
            "updated_at": None
        }
    
    with open(accuracy_path, 'r', encoding='utf-8') as f:
        return _json.load(f)

@app.get("/predict/track-record")
def get_track_record():
    """AI 예측 트랙레코드 — update_data.py가 매일 그 시점에 실제 서빙 중인 /predict를
    호출해 미리 기록해두고, 경기가 끝나면 실제 결과와 대조해 채워넣은 로그(prediction_log.json)의
    요약 + 최근 완료 경기 목록"""
    log_path = os.path.join(MODEL_DIR, "prediction_log.json")
    empty = {"summary": {"total_logged": 0}, "recent": []}
    if not os.path.exists(log_path):
        return empty

    with open(log_path, 'r', encoding='utf-8') as f:
        log = _json.load(f)

    resolved = [e for e in log.values() if e.get("actual") is not None]
    resolved.sort(key=lambda e: e["date"])
    if not resolved:
        return empty

    recent = resolved[-50:]
    total_correct = sum(1 for e in resolved if e["correct"])
    recent_correct = sum(1 for e in recent if e["correct"])

    return {
        "summary": {
            "total_logged": len(resolved),
            "total_correct": total_correct,
            "accuracy_pct": round(total_correct / len(resolved) * 100, 1),
            "recent_n": len(recent),
            "recent_correct": recent_correct,
            "recent_accuracy_pct": round(recent_correct / len(recent) * 100, 1),
        },
        "recent": list(reversed(recent[-20:])),
    }

@app.get("/players/topscorers/{league_code}")
def get_top_scorers(league_code: str):
    """리그별 득점왕"""
    players_path = os.path.join(MODEL_DIR, "players.json")
    if not os.path.exists(players_path):
        raise HTTPException(status_code=404, detail="선수 데이터 없음")
    
    with open(players_path, 'r', encoding='utf-8') as f:
        data = _json.load(f)
    
    scorers = data.get("topscorers", {}).get(league_code.upper(), [])
    if not scorers:
        raise HTTPException(status_code=404, detail="해당 리그 데이터 없음")
    
    return {"league": league_code.upper(), "players": scorers}

@app.get("/players/topassists/{league_code}")
def get_top_assists(league_code: str):
    """리그별 도움왕"""
    players_path = os.path.join(MODEL_DIR, "players.json")
    if not os.path.exists(players_path):
        raise HTTPException(status_code=404, detail="선수 데이터 없음")
    
    with open(players_path, 'r', encoding='utf-8') as f:
        data = _json.load(f)
    
    assists = data.get("topassists", {}).get(league_code.upper(), [])
    if not assists:
        raise HTTPException(status_code=404, detail="해당 리그 데이터 없음")
    
    return {"league": league_code.upper(), "players": assists}

    # ── 우승 예측 API ──
@app.get("/predict/champion/{league_code}")
def get_champion_prediction(league_code: str):
    """리그 우승 예측"""
    path = os.path.join(MODEL_DIR, "champion_predictions.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="우승 예측 데이터 없음")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    league = data.get(league_code.upper())
    if not league:
        raise HTTPException(status_code=404, detail="해당 리그 데이터 없음")
    
    # 로고 추가
    logo_path = os.path.join(MODEL_DIR, "team_logos.json")
    logos = {}
    if os.path.exists(logo_path):
        with open(logo_path, 'r', encoding='utf-8') as f:
            logos = json.load(f)
    
    # for team in league['teams']:
    #     team['logo'] = logos.get(team['team'], '')
    
    # return league

    # 팀 승률 + prestige 추가 (사용자 시뮬레이션용)
    stats_path = os.path.join(MODEL_DIR, "team_stats.csv")
    win_rates = {}
    prestiges = {}
    if os.path.exists(stats_path):
        df_s = pd.read_csv(stats_path)
        for _, row in df_s.iterrows():
            win_rates[row['team']] = float(row['win_rate'])
            prestiges[row['team']] = float(row['prestige']) if 'prestige' in row and pd.notna(row['prestige']) else 0.0

    for team in league['teams']:
        team['logo'] = logos.get(team['team'], '')
        team['win_rate'] = win_rates.get(team['team'], 0.33)
        team['prestige'] = prestiges.get(team['team'], 0.0)

    return league

# ── 전체 일정 API ──
@app.get("/schedule/{league_code}")
def get_schedule(league_code: str):
    """리그 전체 시즌 일정 (완료 + 예정 경기 전부)"""
    path = os.path.join(MODEL_DIR, "schedule.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="일정 데이터 없음")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    matches = data.get(league_code.upper())
    if matches is None:
        raise HTTPException(status_code=404, detail="해당 리그 데이터 없음")

    return {"league": league_code.upper(), "matches": matches}

@app.get("/team/info/{team_name}")
def get_team_info(team_name: str):
    """팀 상세 정보 (홈구장, 창단연도, 구단색, 스쿼드) — update_data.py의 fetch_team_info()가 생성한 캐시"""
    path = os.path.join(MODEL_DIR, "team_info.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="팀 정보 데이터 없음")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    info = data.get(team_name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"팀 정보를 찾을 수 없습니다: {team_name}")

    # API-Football 스쿼드 사진/등번호 + 이적 기록 (현재는 PL만 — fetch_squad_transfers() 참고)
    extra_path = os.path.join(MODEL_DIR, "team_extra.json")
    if os.path.exists(extra_path):
        with open(extra_path, 'r', encoding='utf-8') as f:
            extra = json.load(f).get(team_name)
        if extra:
            info = {**info}
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