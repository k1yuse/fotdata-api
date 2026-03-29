# ── [API] FotData FastAPI 서버 ──
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

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

print(f"✅ 모델 로드 완료 | 팀 수: {len(df_stats)}")

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
    h = df_stats[df_stats['team'] == req.home_team]
    a = df_stats[df_stats['team'] == req.away_team]

    if h.empty:
        raise HTTPException(status_code=404, detail=f"팀을 찾을 수 없습니다: {req.home_team}")
    if a.empty:
        raise HTTPException(status_code=404, detail=f"팀을 찾을 수 없습니다: {req.away_team}")

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
        'home_advantage': home_advantage,
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
@app.get("/logos")
def get_logos():
    import json
    logo_path = os.path.join(MODEL_DIR, "team_logos.json")
    with open(logo_path, 'r', encoding='utf-8') as f:
        logos = json.load(f)
    return logos