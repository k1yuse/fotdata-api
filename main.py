# ── [API] FotData FastAPI 서버 ──
from fastapi import FastAPI, HTTPException
import requests
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import json

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

lr_model  = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
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

    home_advantage = 0.45  # 데이터 기반 홈 어드밴티지

    input_data = pd.DataFrame([{
        'home_attack':    h['attack_strength'],
        'away_attack':    a['attack_strength'],
        'home_defense':   h['defense_strength'],
        'away_defense':   a['defense_strength'],
        'home_form':      h['win_rate'] * 15,
        'away_form':      a['win_rate'] * 15,
        'form_diff':      (h['win_rate'] - a['win_rate']) * 15,
        'home_win_rate':  h['win_rate'],
        'away_win_rate':  a['win_rate'],
        'win_rate_diff':  h['win_rate'] - a['win_rate'],
        'home_goal_diff': h['goal_diff'],
        'away_goal_diff': a['goal_diff'],
        'attack_diff':    h['attack_strength'] - a['attack_strength'],
        'defense_diff':   h['defense_strength'] - a['defense_strength'],
        'home_advantage': 0.43,
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

    return {
        "home_team":   req.home_team,
        "away_team":   req.away_team,
        "prediction":  prediction,
        "probabilities": {
            "home_win": h_prob,
            "draw":     d_prob,
            "away_win": a_prob,
        },
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
    """특정 팀 스탯 조회"""
    team = df_stats[df_stats['team'] == team_name]
    if team.empty:
        raise HTTPException(status_code=404, detail=f"팀을 찾을 수 없습니다: {team_name}")
    
    t = team.iloc[0]
    return {
        "team":             t['team'],
        "games":            int(t['games']),
        "wins":             int(t['wins']),
        "draws":            int(t['draws']),
        "losses":           int(t['losses']),
        "points":           int(t['points']),
        "attack_strength":  round(float(t['attack_strength']), 3),
        "defense_strength": round(float(t['defense_strength']), 3),
        "goal_diff":        int(t['goal_diff']),
        "win_rate":         round(float(t['win_rate']), 3),
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
def get_standings(league_code: str):
    league_name = LEAGUE_MAP.get(league_code.upper())
    if not league_name:
        raise HTTPException(status_code=404, detail="리그를 찾을 수 없습니다")

    cutoff = pd.Timestamp('2025-08-01')
    filters = (df_matches_all['league'] == league_code.upper()) & (df_matches_all['date'] >= cutoff)
    if league_code.upper() == 'CL':
        filters = filters & (df_matches_all['date'] < pd.Timestamp('2026-02-01'))
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