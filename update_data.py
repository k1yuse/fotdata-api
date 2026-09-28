# ── 자동 데이터 업데이트 스크립트 ──
import re
import requests
import time
from collections import Counter
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

API_KEY = os.environ.get('FOOTBALL_API_KEY', '')
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}
MODEL_DIR = "fotdata_model"

# API-Football (선수 데이터용)
API_FOOTBALL_KEY = os.environ.get('API_FOOTBALL_KEY', '')
API_FOOTBALL_URL = "https://v3.football.api-sports.io"
API_FOOTBALL_HEADERS = {"x-apisports-key": API_FOOTBALL_KEY}

# API-Football 리그 ID
LEAGUE_IDS = {
    "PL":  39,   # EPL
    "PD":  140,  # LaLiga
    "BL1": 78,   # Bundesliga
    "SA":  135,  # Serie A
    "FL1": 61,   # Ligue 1
}

LEAGUES_V2 = {
    "PL":  "EPL (잉글랜드)",
    "PD":  "라리가 (스페인)",
    "BL1": "분데스리가 (독일)",
    "SA":  "세리에A (이탈리아)",
    "FL1": "리그앙 (프랑스)",
    "CL":  "챔피언스리그",
}

def _match_goals(m):
    """football-data 경기 → (홈 득점, 원정 득점, 승부차기 (홈, 원정) 또는 None).
    승부차기로 끝난 경기는 fullTime에 승부차기 골까지 더해져 옴(예: 아스널–포르투 1-0 + 승부차기 4-2 → fullTime 5-2)
    — 그대로 쓰면 결과·ELO·맞대결이 틀려서(리버풀 1-0 승리가 1-5 패배로 기록됐었음, 2026-09-29 발견) 승부차기 골을 뺌"""
    sc = m.get("score") or {}
    ft = sc.get("fullTime") or {}
    hg, ag = ft.get("home"), ft.get("away")
    pen = sc.get("penalties") or {}
    if sc.get("duration") == "PENALTY_SHOOTOUT" and pen.get("home") is not None and hg is not None:
        return hg - pen["home"], ag - pen["away"], (pen["home"], pen["away"])
    if m.get("status") == "AWARDED":
        # 판정 결과(몰수 등): 점수가 없거나 경기장 점수라 판정과 안 맞으면 판정 승자 기준 표준 점수(승 2-0, 무 0-0)
        w = sc.get("winner")
        ok = hg is not None and ((w == "HOME_TEAM" and hg > ag) or (w == "AWAY_TEAM" and ag > hg) or (w == "DRAW" and hg == ag))
        if not ok and w in ("HOME_TEAM", "AWAY_TEAM", "DRAW"):
            return {"HOME_TEAM": (2, 0), "AWAY_TEAM": (0, 2), "DRAW": (0, 0)}[w] + (None,)
    return hg, ag, None

def fetch_matches(league_code, season):
    url = f"{BASE_URL}/competitions/{league_code}/matches"
    # 상태 필터 없이 받아서 FINISHED + AWARDED(몰수·판정승 — 공식 순위에 포함됨)만 남김. 예전엔 FINISHED만 받아서
    # 24-25 우니온–보훔(관중석 물체 투척으로 보훔 판정승) 같은 경기가 빠져 공식 순위표와 1경기씩 달랐음(2026-09-29)
    params = {"season": season}
    name = LEAGUES_V2.get(league_code, league_code)
    print(f"  [{name}] 수집 중...")
    res = requests.get(url, headers=HEADERS, params=params)
    if res.status_code != 200:
        print(f"  ❌ 오류: {res.status_code}")
        return pd.DataFrame()
    matches = [m for m in res.json().get("matches", []) if m.get("status") in ("FINISHED", "AWARDED")]
    print(f"  ✅ {len(matches)}경기")
    rows = []
    for m in matches:
        hg, ag, _ = _match_goals(m)
        if hg is None:
            continue
        rows.append({
            "match_id":   m["id"],
            "date":       m["utcDate"][:10],
            "league":     league_code,
            "home_team":  m["homeTeam"]["name"],
            "away_team":  m["awayTeam"]["name"],
            "home_goals": hg,
            "away_goals": ag,
            "matchday":   m.get("matchday"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    def get_result(row):
        if row["home_goals"] > row["away_goals"]:   return "H"
        elif row["home_goals"] < row["away_goals"]: return "A"
        else:                                        return "D"
    df["result"] = df.apply(get_result, axis=1)
    return df.sort_values("date").reset_index(drop=True)

def calculate_team_stats(df):
    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    stats = []
    for team in teams:
        home = df[df['home_team'] == team]
        away = df[df['away_team'] == team]
        games = len(home) + len(away)
        if games == 0:
            continue
        goals_scored   = home['home_goals'].sum() + away['away_goals'].sum()
        goals_conceded = home['away_goals'].sum() + away['home_goals'].sum()
        wins  = len(home[home['result']=='H']) + len(away[away['result']=='A'])
        draws = len(home[home['result']=='D']) + len(away[away['result']=='D'])
        losses = games - wins - draws
        points = wins * 3 + draws
        stats.append({
            "team":             team,
            "games":            games,
            "wins":             wins,
            "draws":            draws,
            "losses":           losses,
            "points":           points,
            "goals_scored":     goals_scored,
            "goals_conceded":   goals_conceded,
            "goal_diff":        goals_scored - goals_conceded,
            "attack_strength":  round(goals_scored / games, 3),
            "defense_strength": round(goals_conceded / games, 3),
            "win_rate":         round(wins / games, 3),
        })
    return pd.DataFrame(stats).sort_values("points", ascending=False).reset_index(drop=True)
    
def calculate_blended_stats(df_total):
    """3시즌(24-25/25-26/26-27) 혼합 + prestige 보정이 반영된 팀 스탯 계산"""
    df_2425 = df_total[(df_total['date'] >= '2024-08-01') & (df_total['date'] < '2025-08-01') & (df_total['league'] != 'CL')]
    df_2526 = df_total[(df_total['date'] >= '2025-08-01') & (df_total['date'] < '2026-08-01') & (df_total['league'] != 'CL')]
    df_2627 = df_total[(df_total['date'] >= '2026-08-01') & (df_total['league'] != 'CL')]

    stats_2425 = calculate_team_stats(df_2425).set_index('team')
    stats_2526 = calculate_team_stats(df_2526).set_index('team')
    stats_2627 = calculate_team_stats(df_2627).set_index('team')

    current_teams = stats_2627.index.tolist()

    def get_val(stats_df, team, col, default):
        return stats_df.loc[team, col] if team in stats_df.index else default

    rows = []
    for team in current_teams:
        games_2627 = stats_2627.loc[team, 'games']

        if games_2627 < 10:
            w2425, w2526, w2627 = 0.4, 0.4, 0.2
        elif games_2627 < 18:
            w2425, w2526, w2627 = 0.3, 0.3, 0.3
        else:
            w2425, w2526, w2627 = 0.3, 0.3, 0.4

        blended_win_rate = (
            w2425 * get_val(stats_2425, team, 'win_rate', 0.33) +
            w2526 * get_val(stats_2526, team, 'win_rate', 0.33) +
            w2627 * get_val(stats_2627, team, 'win_rate', 0.33)
        )
        blended_attack = (
            w2425 * get_val(stats_2425, team, 'attack_strength', 1.3) +
            w2526 * get_val(stats_2526, team, 'attack_strength', 1.3) +
            w2627 * get_val(stats_2627, team, 'attack_strength', 1.3)
        )
        blended_defense = (
            w2425 * get_val(stats_2425, team, 'defense_strength', 1.3) +
            w2526 * get_val(stats_2526, team, 'defense_strength', 1.3) +
            w2627 * get_val(stats_2627, team, 'defense_strength', 1.3)
        )

        rows.append({
            "team": team,
            "games": int(games_2627),
            "win_rate": round(blended_win_rate, 3),
            "attack_strength": round(blended_attack, 3),
            "defense_strength": round(blended_defense, 3),
        })

        df_blended = pd.DataFrame(rows)

    # prestige: 26-27 시즌 영향 배제하고, 24-25+25-26 두 시즌만으로 "안정적인 체급" 계산
    df_prior2 = df_total[(df_total['date'] >= '2024-08-01') & (df_total['date'] < '2026-08-01') & (df_total['league'] != 'CL')]
    stats_prior2 = calculate_team_stats(df_prior2).set_index('team')

    prior_win_rates = []
    for team in df_blended['team']:
        wr = stats_prior2.loc[team, 'win_rate'] if team in stats_prior2.index else None
        prior_win_rates.append(wr)
    df_blended['_prior_win_rate'] = prior_win_rates

    valid_prior = df_blended['_prior_win_rate'].dropna()
    prior_league_avg = valid_prior.mean() if len(valid_prior) > 0 else 0.33

    def calc_prestige(wr):
        if pd.isna(wr):
            return 0  # 24-25/25-26 기록 없는 신규 승격팀 등은 보정 없음
        return round((wr - prior_league_avg) * 500, 1)

    df_blended['prestige'] = df_blended['_prior_win_rate'].apply(calc_prestige)
    df_blended = df_blended.drop(columns=['_prior_win_rate'])

    return df_blended

FEATURES = [
    'home_elo','away_elo','elo_diff',
    'home_form','away_form','form_diff',
    'home_avg_scored','away_avg_scored','home_avg_conceded','away_avg_conceded',
    'home_attack','away_attack','home_defense','away_defense',
    'home_win_rate','away_win_rate','win_rate_diff',
    'h2h_home_rate'
]

ELO_K = 20
ELO_HOME_ADVANTAGE = 70
FORM_N, GOALS_N, STATS_N, H2H_N = 5, 10, 38, 10
ELO_TRAIL_N = 20   # 예측 결과 화면 "파워 레이팅 추이" 그래프용으로 저장하는 최근 경기 수
MIN_HISTORY = 5   # 이보다 경기 기록이 적은 팀이 낀 경기는 학습에서 제외

def _team_snapshot(history):
    """팀의 "지금까지" 경기 기록으로 피처 값 계산 — 학습(경기 직전 시점)과
    서빙(team_state.json, 오늘 시점)이 반드시 이 함수 하나를 같이 써야 함"""
    form = history[-FORM_N:]
    recent = history[-GOALS_N:]
    season = history[-STATS_N:]
    return {
        'form':         sum(x['pts'] for x in form),
        'avg_scored':   float(np.mean([x['gf'] for x in recent])),
        'avg_conceded': float(np.mean([x['ga'] for x in recent])),
        'attack':       float(np.mean([x['gf'] for x in season])),
        'defense':      float(np.mean([x['ga'] for x in season])),
        'win_rate':     float(np.mean([x['pts'] == 3 for x in season])),
    }

def build_point_in_time_features(df):
    """경기를 시간순으로 훑으면서 "그 경기 직전까지의 기록"만으로 피처를 만든다.

    예전 build_features는 ELO·폼은 경기 직전 값을 썼지만 공격력/수비력/승률은
    전체 기간 합산(미래 경기 포함)을 붙였고, 서빙(/predict)은 이 값들을 저장해두지
    않아서 블렌딩 승률로 ELO와 폼을 재구성해 넣었다 — 모델이 배운 입력과 실제
    예측 입력의 의미가 달랐음. 시간순 검증(2026-09-27)에서 이 방식이 정확도
    48.9→50.8%(26.01~), 44.7→51.8%(25.08~), log loss도 모든 구간에서 개선됐고,
    확률 보정(예측 확률 ≈ 실제 적중 비율)도 크게 좋아짐.

    반환: (피처 DataFrame(date 포함), 팀별 최종 상태 dict — team_state.json용)
    """
    df = df.sort_values('date').reset_index(drop=True)
    data_start = df['date'].min()
    elo = {}
    history = {}   # team -> [{'gf','ga','pts','league'}]
    h2h = {}       # (팀A, 팀B) 정렬 튜플 -> [승리팀 또는 None]
    elo_trail = {} # team -> [(날짜, 경기 후 ELO)]

    def initial_elo(league, date):
        # 데이터 첫 시즌 초반에 등장한 팀은 1500에서 시작. 이후 처음 등장하는 팀(승격팀 등)은
        # 같은 리그 현재 하위 4팀 평균에서 시작 — 1500(중상위권)에서 시작하면 승격팀이
        # 과대평가됨(실험에서 log loss 소폭 개선 확인)
        if date < data_start + pd.Timedelta(days=60):
            return 1500.0
        league_elos = sorted(elo[t] for t, h in history.items() if h and h[-1]['league'] == league)
        return float(np.mean(league_elos[:4])) if len(league_elos) >= 4 else 1400.0

    rows = []
    for _, m in df.iterrows():
        home, away, date, league = m['home_team'], m['away_team'], m['date'], m['league']
        for t in (home, away):
            if t not in elo:
                elo[t] = initial_elo(league, date)
                history[t] = []

        if len(history[home]) >= MIN_HISTORY and len(history[away]) >= MIN_HISTORY:
            hs, as_ = _team_snapshot(history[home]), _team_snapshot(history[away])
            rec = h2h.get(tuple(sorted((home, away))), [])[-H2H_N:]
            rows.append({
                'date':              date,
                'home_elo':          elo[home],
                'away_elo':          elo[away],
                'elo_diff':          elo[home] - elo[away],
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
                'h2h_home_rate':     round(sum(w == home for w in rec) / len(rec), 3) if rec else 0.33,
                'result':            m['result'],
            })

        # 결과 반영 (ELO: 홈 어드밴티지 포함 기대승률 대비 실제 결과)
        expected_home = 1 / (1 + 10 ** ((elo[away] - (elo[home] + ELO_HOME_ADVANTAGE)) / 400))
        actual_home = {'H': 1.0, 'D': 0.5, 'A': 0.0}[m['result']]
        change = ELO_K * (actual_home - expected_home)
        elo[home] += change
        elo[away] -= change
        home_pts = {'H': 3, 'D': 1, 'A': 0}[m['result']]
        away_pts = {'A': 3, 'D': 1, 'H': 0}[m['result']]
        history[home].append({'gf': m['home_goals'], 'ga': m['away_goals'], 'pts': home_pts, 'league': league})
        history[away].append({'gf': m['away_goals'], 'ga': m['home_goals'], 'pts': away_pts, 'league': league})
        winner = home if m['result'] == 'H' else (away if m['result'] == 'A' else None)
        h2h.setdefault(tuple(sorted((home, away))), []).append(winner)
        day = date.strftime('%Y-%m-%d')
        elo_trail.setdefault(home, []).append((day, round(elo[home], 1)))
        elo_trail.setdefault(away, []).append((day, round(elo[away], 1)))

    team_state = {}
    for team, hist in history.items():
        if not hist:
            continue
        snap = _team_snapshot(hist)
        team_state[team] = {
            'elo': round(elo[team], 2),
            'games': len(hist),
            **{k: round(v, 4) for k, v in snap.items()},
            'elo_history': elo_trail[team][-ELO_TRAIL_N:],
        }
    return pd.DataFrame(rows), team_state

def update_team_logos():
    """team_stats.csv에 있는 팀 중 로고 없는 팀을 자동으로 채워넣기"""
    import json
    print("\n[로고 자동 업데이트] 확인 중...")
    
    logo_path = f"{MODEL_DIR}/team_logos.json"
    logos = {}
    if os.path.exists(logo_path):
        with open(logo_path, 'r', encoding='utf-8') as f:
            logos = json.load(f)
    
    df_stats = pd.read_csv(f"{MODEL_DIR}/team_stats.csv")
    all_teams = set(df_stats['team'].tolist())
    missing_teams = all_teams - set(logos.keys())
    
    if not missing_teams:
        print("  ✅ 누락된 로고 없음")
        return
    
    print(f"  ⚠️ 로고 없는 팀 {len(missing_teams)}개 발견: {missing_teams}")
    
    # 5대 리그 + CL 전체 팀 목록을 API로 조회해서 매칭
    for code in list(LEAGUES_V2.keys()):
        try:
            url = f"https://api.football-data.org/v4/competitions/{code}/teams"
            headers = {"X-Auth-Token": os.environ.get("FOOTBALL_API_KEY")}
            res = requests.get(url, headers=headers)
            data = res.json()
            for team in data.get('teams', []):
                name = team['name']
                if name in missing_teams and team.get('crest'):
                    logos[name] = team['crest']
                    missing_teams.discard(name)
            time.sleep(6)
        except Exception as e:
            print(f"  ❌ {code} 로고 조회 실패: {e}")
    
    with open(logo_path, 'w', encoding='utf-8') as f:
        json.dump(logos, f, ensure_ascii=False, indent=2)
    
    if missing_teams:
        print(f"  ⚠️ 여전히 못 찾은 팀: {missing_teams}")
    print(f"  ✅ team_logos.json 업데이트 완료")
    
POSITION_ORDER = {"Goalkeeper": 0, "Defence": 1, "Midfield": 2, "Offence": 3}

def fetch_team_info():
    """팀 상세 정보(홈구장/창단연도/구단색/스쿼드) 수집 — 팀 클릭 시 정보 패널용.
    team_stats.csv에 있는(=현재 서빙 중인) 팀만 대상으로 한다."""
    import json
    print("\n[팀 상세정보] 수집 중...")

    df_stats = pd.read_csv(f"{MODEL_DIR}/team_stats.csv")
    target_teams = set(df_stats['team'].tolist())

    # 1. 경쟁 리그별 팀 목록으로 이름→ID 매핑 확보
    name_to_id = {}
    for code in list(LEAGUES_V2.keys()):
        try:
            res = requests.get(f"{BASE_URL}/competitions/{code}/teams", headers=HEADERS)
            data = res.json()
            for team in data.get('teams', []):
                name_to_id[team['name']] = team['id']
            time.sleep(6)
        except Exception as e:
            print(f"  ❌ {code} 팀 목록 조회 실패: {e}")

    # 2. 대상 팀별 상세 정보 조회
    team_info = {}
    missing = []
    for name in sorted(target_teams):
        team_id = name_to_id.get(name)
        if team_id is None:
            missing.append(name)
            continue
        try:
            res = requests.get(f"{BASE_URL}/teams/{team_id}", headers=HEADERS)
            if res.status_code != 200:
                print(f"  ❌ {name} 조회 실패: {res.status_code}")
                time.sleep(6)
                continue
            d = res.json()
            squad = sorted(
                [
                    {
                        "name": p.get("name"),
                        "position": p.get("position"),
                        "nationality": p.get("nationality"),
                        "shirtNumber": p.get("shirtNumber"),
                    }
                    for p in d.get("squad", [])
                ],
                key=lambda p: POSITION_ORDER.get(p["position"], 99)
            )
            coach = d.get("coach") or {}
            team_info[name] = {
                "venue": d.get("venue"),
                "founded": d.get("founded"),
                "clubColors": d.get("clubColors"),
                "coach": coach.get("name"),
                "squad": squad,
            }
            print(f"  ✅ {name} ({len(squad)}명)")
        except Exception as e:
            print(f"  ❌ {name} 예외: {e}")
        time.sleep(6)

    if missing:
        print(f"  ⚠️ ID 못 찾은 팀: {missing}")

    # 현재 팀만 새로 받고, 예전 시즌 팀(강등팀·지난 챔스 팀 — fetch_history_teams가 채움)은 그대로 둠.
    # 예전엔 매일 현재 팀만으로 파일을 새로 써서 그 팀들의 정보가 사라졌음(2026-09-29)
    old_info = {}
    if os.path.exists(f"{MODEL_DIR}/team_info.json"):
        with open(f"{MODEL_DIR}/team_info.json", 'r', encoding='utf-8') as f:
            old_info = json.load(f)
    team_info = {**{k: v for k, v in old_info.items() if k not in team_info}, **team_info}
    with open(f"{MODEL_DIR}/team_info.json", 'w', encoding='utf-8') as f:
        json.dump(team_info, f, ensure_ascii=False, indent=2)
    print(f"  ✅ team_info.json 저장 완료 ({len(team_info)}팀)")

def fetch_history_teams():
    """지난 시즌(23-24~) 리그·챔스에 나왔던 팀의 로고·팀 정보 채우기 — 강등팀·지난 챔스 팀은 로고가 없어서 순위표·대진표에 빈칸이었음.
    /competitions/{리그}/teams?season=연도 한 번에 로고(crest)·홈구장·창단·구단 색이 같이 와서 팀별 호출이 필요 없음(리그 6 × 시즌 4 = 24회).
    스쿼드는 그 시즌 기준이라 헷갈리지 않게 현재 시즌에 나온 팀만 넣음. 이미 있는 팀 정보는 안 건드림."""
    import json
    print("\n[지난 시즌 팀 로고·정보]")
    logo_path, info_path = f"{MODEL_DIR}/team_logos.json", f"{MODEL_DIR}/team_info.json"
    logos = json.load(open(logo_path, encoding='utf-8')) if os.path.exists(logo_path) else {}
    info = json.load(open(info_path, encoding='utf-8')) if os.path.exists(info_path) else {}
    added_logo, added_info = 0, 0
    for code in LEAGUES_V2:
        for yr in MATCH_SEASONS:
            res = requests.get(f"{BASE_URL}/competitions/{code}/teams", headers=HEADERS, params={"season": yr})
            time.sleep(6)
            if res.status_code != 200:
                print(f"  ❌ {code} {yr}: {res.status_code}"); continue
            for t in res.json().get("teams", []):
                name = t.get("name")
                if not name:
                    continue
                if not logos.get(name) and t.get("crest"):
                    logos[name] = t["crest"]; added_logo += 1
                if name not in info:
                    coach = t.get("coach") or {}
                    info[name] = {"venue": t.get("venue"), "founded": t.get("founded"), "clubColors": t.get("clubColors"),
                                  "coach": coach.get("name"), "squad": [], "last_season": yr}
                    added_info += 1
                if yr == max(MATCH_SEASONS) and not info[name].get("squad") and t.get("squad"):
                    info[name]["squad"] = sorted([{"name": p.get("name"), "position": p.get("position"), "nationality": p.get("nationality"),
                                                   "shirtNumber": p.get("shirtNumber")} for p in t["squad"]],
                                                 key=lambda p: POSITION_ORDER.get(p["position"], 99))
    with open(logo_path, 'w', encoding='utf-8') as f:
        json.dump(logos, f, ensure_ascii=False, indent=2)
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 로고 {added_logo}개, 팀 정보 {added_info}개 추가")

def _history_missing():
    """경기 데이터에 있는데 로고가 없는 팀이 있으면 True(매일 실행 때 그때만 fetch_history_teams)"""
    import json
    df = pd.read_csv(f"{MODEL_DIR}/all_matches.csv")
    logos = json.load(open(f"{MODEL_DIR}/team_logos.json", encoding='utf-8')) if os.path.exists(f"{MODEL_DIR}/team_logos.json") else {}
    return any(not logos.get(t) for t in set(df.home_team) | set(df.away_team))

SQUAD_POSITION_MAP = {"Goalkeeper": "Goalkeeper", "Defender": "Defence", "Midfielder": "Midfield", "Attacker": "Offence"}

def _normalize_team_name(name):
    n = name.lower().strip()
    n = re.sub(r'^(afc|ac|cf|sk|bk)\s+', '', n)
    for suf in (" fc", " afc", " cf", " sk", " bk", " ac"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    n = n.replace('&', ' ').replace('  ', ' ')
    return n.strip()

LEAGUE_COUNTRY = {"PL": "England", "PD": "Spain", "BL1": "Germany", "SA": "Italy", "FL1": "France"}

def _search_af_team_id_query(query, country):
    res = requests.get(f"{API_FOOTBALL_URL}/teams", headers=API_FOOTBALL_HEADERS, params={"search": query})
    candidates = res.json().get("response", [])
    for c in candidates:
        t = c.get("team", {})
        name = (t.get("name") or "")
        # 유스/리저브팀 제외: " U18" 하나가 빠져 있어서 Newcastle이 U18팀 ID로
        # 잘못 매칭된 전례가 있었음(스쿼드가 전부 유스 선수로 채워짐) — 나이대 표기를
        # 정규식으로 통째로 잡아서 앞으로 U15~U23 등 어떤 연령대가 와도 걸러지게 함.
        if t.get("country") == country and not re.search(r"\bU1[5-9]\b|\bU2[0-3]\b", name.upper()) \
                and not any(x in name.upper() for x in (" II", " B", " W", " RES.")):
            return t.get("id")
    return None

def _search_af_team_id(team_name, country):
    """이름으로 API-Football 팀ID 검색 (시즌 제한 없는 엔드포인트).
    같은 이름의 유스/리저브팀·해외 동명팀을 걸러내기 위해 국가로 필터링.
    football-data.org의 공식 풀네임(예: "Brighton & Hove Albion FC",
    "AFC Bournemouth")과 API-Football의 짧은 통칭("Brighton", "Bournemouth")이
    달라서 전체 이름으로 검색이 실패하면 원래 이름의 첫 단어로 한 번 더 시도한다."""
    query = _normalize_team_name(team_name)
    af_id = _search_af_team_id_query(query, country)
    if af_id:
        return af_id
    first_word = team_name.split()[0].lower()
    if first_word != query:
        return _search_af_team_id_query(first_word, country)
    return None

def _af_get(url, params):
    """레이트리밋(errors 응답) 대비 1회 재시도가 포함된 API-Football 호출."""
    res = requests.get(url, headers=API_FOOTBALL_HEADERS, params=params)
    data = res.json()
    if data.get("errors"):
        print(f"    ⚠️ API 응답 오류({data['errors']}), 20초 후 재시도")
        time.sleep(20)
        res = requests.get(url, headers=API_FOOTBALL_HEADERS, params=params)
        data = res.json()
    return data

def _clean_squad(squad_raw, team_name, team_info):
    """API-Football의 /players/squads는 무료 플랜에서도 1군 외에 유스/후보 선수까지
    구분 없이 섞어서 반환하는데, 이 유스/후보 선수들이 1군과 등번호가 겹치는 경우가
    실제로 다수 발견됨(예: Arsenal #1을 David Raya와 후보 골키퍼가 동시에 사용).
    football-data.org 쪽 공식 스쿼드(team_info.json, 등번호는 없지만 1군만 깨끗하게
    제공됨)의 선수 이름을 화이트리스트 삼아, 성(姓)이 그 목록에 없는 선수는 노이즈로
    간주해 제외한다. 화이트리스트가 없는 팀(team_info 미수집)은 필터링 없이 그대로 둔다.
    필터링 후에도 등번호가 겹치면(둘 다 화이트리스트에 있는 경우 등, 어느 쪽이 진짜
    현재 1군 번호인지 판단할 근거가 없으므로) 틀린 번호를 확신 없이 보여주는 대신
    해당 번호를 비워서 최소한 오정보는 피한다."""
    whitelist = [p.get("name", "") for p in (team_info.get(team_name, {}) or {}).get("squad", [])]
    if whitelist:
        whitelist_lower = [n.lower() for n in whitelist]
        filtered = []
        for p in squad_raw:
            name = p.get("name") or ""
            tokens = name.split()
            key = tokens[-1].lower() if tokens else name.lower()
            if any(key in wl for wl in whitelist_lower):
                filtered.append(p)
        squad_raw = filtered

    counts = Counter(p.get("shirtNumber") for p in squad_raw if p.get("shirtNumber") is not None)
    for p in squad_raw:
        if p.get("shirtNumber") is not None and counts[p["shirtNumber"]] > 1:
            p["shirtNumber"] = None
    return squad_raw

# ── 구단 소개(위키백과) + 별칭·홈구장 수용 인원·감독(위키데이터) — 팀 정보 모달 "개요" 탭 (2026-09-29) ──
# football-data.org 무료 플랜은 감독이 항상 null이고 구단 설명이 없어서, 키 없이 쓸 수 있는 위키미디어 API로 채움.
# 위키백과 글은 CC BY-SA라 화면에 출처(위키백과)와 원문 링크를 반드시 같이 표시할 것.
# 팀 이름 → 영어 위키백과 검색 → 위키데이터 항목이 "축구 클럽"(Q476028)인지 확인해서 동명 도시·경기장 문서를 거름.
# 감독은 바뀌므로 WIKI_REFRESH_DAYS마다 다시 받음(매일 전 팀을 다 받지 않고 오래된 것만).
WIKI_UA = {"User-Agent": "FotData/1.0 (https://fotdata-api.vercel.app)"}
WIKI_REFRESH_DAYS = 7
# 검색이 엉뚱한 문서(동명 다른 클럽 등)를 고르는 팀: {"팀 이름": "영어 위키백과 문서 제목"} — 2026-09-29 전 팀 확인 후 추가
WIKI_TITLE_OVERRIDE = {
    "Manchester United FC": "Manchester United F.C.",   # FC United of Manchester가 먼저 잡힘
    "Borussia Dortmund": "Borussia Dortmund",
    "VfB Stuttgart": "VfB Stuttgart",
    "Real Sociedad de Fútbol": "Real Sociedad",
    "Atalanta BC": "Atalanta BC",
    "TSG 1899 Hoffenheim": "TSG Hoffenheim",
    "Venezia FC": "Venezia FC",
    "Newcastle United FC": "Newcastle United F.C.",   # 호주 Newcastle Jets가 잡힘
    "PAE Olympiakos SFP": "Olympiacos F.C.",          # football-data 표기(PAE … SFP)로는 검색이 안 됨
    "Galatasaray SK": "Galatasaray S.K. (football)",   # 본문서는 종합 스포츠 클럽(농구·휠체어농구 우승까지 섞임)
    "Fenerbahçe SK": "Fenerbahçe S.K. (football)",
    "Sport Lisboa e Benfica": "S.L. Benfica",
    "Sporting Clube de Portugal": "Sporting CP",       # "Atlético Clube de Portugal"이 잡힘
    "Feyenoord Rotterdam": "Feyenoord",                # 같은 도시 Excelsior Rotterdam이 잡힘
}
_WIKI_SKIP_TITLE = re.compile(r"(\sII\b|\s[BC]$|\b(reserves?|women|femenino|féminin|frauen|femminile|u-?\d\d|under-\d\d|youth|academy|primavera)\b)", re.I)
WIKI_NAME_KO_OVERRIDE = {"Venezia FC": "베네치아 FC"}   # 위키데이터 한국어 이름이 옛 명칭인 경우

def _wiki_get(url, params=None):
    for attempt in range(5):   # 429(요청 과다)면 Retry-After만큼(없으면 점점 길게) 쉬었다가 재시도
        r = requests.get(url, params=params, headers=WIKI_UA, timeout=20)
        if r.status_code != 429:
            break
        time.sleep(float(r.headers.get("Retry-After") or 2 * (attempt + 1)))
    r.raise_for_status()
    return r.json()

def _wd_entities(ids, props="claims|labels|sitelinks"):
    if not ids:
        return {}
    return _wiki_get("https://www.wikidata.org/w/api.php", {
        "action": "wbgetentities", "ids": "|".join(ids), "props": props,
        "languages": "ko|en", "sitefilter": "kowiki|enwiki", "format": "json"}).get("entities", {})

def _wd_label(ent):
    labels = (ent or {}).get("labels", {})
    return (labels.get("ko") or labels.get("en") or {}).get("value")

def _wd_current(claims, prop):
    """현재 값(종료일 P582 없는 것) 중 선호 순위 → 시작일(P580) 최신 순으로 하나"""
    cands = []
    for c in claims.get(prop, []):
        if c.get("rank") == "deprecated" or "P582" in c.get("qualifiers", {}):
            continue
        start = ((c.get("qualifiers", {}).get("P580") or [{}])[0].get("datavalue", {}).get("value", {}) or {}).get("time", "")
        cands.append((c.get("rank") == "preferred", start, c))
    if not cands:
        return None
    return max(cands, key=lambda x: (x[0], x[1]))[2].get("mainsnak", {}).get("datavalue", {}).get("value")

# 연고지: 위키데이터 소재지 값은 훈련장 건물·구(區)·동네인 경우가 많아서(예: 바르셀로나 → "La Maternitat i Sant Ramon",
# 아스널 → 이즐링턴구) 행정구역(P131)을 따라 올라가며 "도시"로 분류된 첫 항목을 씀
_CITY_KINDS = {"Q515", "Q1549591", "Q1637706", "Q5119", "Q200250", "Q22865", "Q42744322", "Q484170", "Q747074",
               "Q2074737", "Q3957", "Q1093829", "Q902814", "Q7930989", "Q15284"}
_CITY_NAME_FIX = {"Q23306": "런던"}   # 그레이터런던 → 런던
# 위키데이터상 훈련장·경기장이 옆 도시에 있어서 팬들이 아는 연고지와 다르게 나오는 팀(2026-09-29 전 팀 확인)
WIKI_CITY_OVERRIDE = {"Cagliari Calcio": "칼리아리", "Nottingham Forest FC": "노팅엄", "Olympique Lyonnais": "리옹",
                      "Manchester City FC": "맨체스터", "Manchester United FC": "맨체스터", "Lille OSC": "릴", "SS Lazio": "로마",
                      "Aston Villa FC": "버밍엄"}
_wd_cache = {}
def _wd_city(qid, depth=0):
    if qid in _CITY_NAME_FIX:
        return _CITY_NAME_FIX[qid]
    if depth > 4:
        return None
    if qid not in _wd_cache:
        _wd_cache[qid] = _wd_entities([qid], "claims|labels").get(qid, {})
    ent = _wd_cache[qid]
    kinds = {c.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id") for c in ent.get("claims", {}).get("P31", [])}
    if kinds & _CITY_KINDS:
        return _wd_label(ent)
    up = _wd_current(ent.get("claims", {}), "P131")
    return _wd_city(up["id"], depth + 1) if up else None

# ── 우승 기록·구단 최고 이적료 파서(영문 위키백과) — 팀 정보 개요 탭 (2026-09-29) ──
# 우승 기록: 구단 문서 "Honours" 절의 표(대회·횟수·시즌) 또는 목록(* 대회 / ** Winners: 시즌…)을 읽어 주요 대회별 횟수로.
#   유스·2군·여자팀·지역 대회·친선 대회는 제외, "(level 2)" 표기는 이름보다 우선(잉글랜드 1992~2004 "First Division"은 2부).
#   2026-09-29 96팀 전부 출력을 원문과 대조해 규칙을 다듬음(시즌 링크 이중 계산, {{lang}}/<sup> 속 숫자, 소제목 단위 제외 등).
# 최고 이적료: "List of … records and statistics" 문서의 paid/received 표 — 순위 칸이 있으면 1위, 없으면 최고액.
# 대회 이름 → 분류 (순서 중요: 하위 리그·UEFA를 먼저 거름)
_HON_RULES = [
    ("skip", r"Istanbul|Ankara|İzmir|Izmir|Athens|Attica|Piraeus|Lisbon Championship|Porto Championship|(?i:women|femenin|féminin|frauen|femminile|Reina|ladies)|Catalunya|Catalan|Catalonia|Galici|Cantabri|Gipuzkoa|Guipúzcoa|Biscay|Vizcaya|Levante Championship|Valencian|Andalusia|Castil|Madrid Cup|Madrid Championship|Alsace|Dordogne|Brittany|Bretagne|Bezirksliga|Kreisliga|Verbandsliga|Landesliga|Gauliga|Oberliga|Southern German|South German|West German|North German|Berlin|Hessen|Hesse|Baden|Württemberg|Bavaria|Bayern Cup|Saarland|Westphalia|Lombard|Piedmont|Tuscan|Campania|Sicil|Sardin|Emilia|Veneto|Liguria|Lazio Cup|Ile-de-France|Paris Cup|Coupe de Paris|Normandie|Provence|Nord|Centenary Trophy|Western (Football )?League|patronages|Sheriff of London|Football League Super Cup|Hallenpokal|Screen Sport|Under[- ]?\d\d|Junior|Juvenil|Allievi|Gambardella|Intertoto|Fairs Cup|Youth|Reserve|Women|Primavera|U-?\d\d|Regional|Sussex|Lancashire|Liverpool Senior|Kent|Isthmian|Southern League|Premier League 2|Premier League Asia|Emirates Cup|Joan Gamper|Amsterdam|Teresa Herrera|Ramón de Carranza|Trofeo|Torneo|Coppa delle Alpi|Mitropa|Latin Cup|Pequeña|Anglo|Watney|Texaco|Full Members|Zenith|Simod|Mercantile|Wartime|War Cup|League North|League South|Coppa Italia (Serie C|Lega Pro|Serie D)|Supercoppa (di Serie C|Lega Pro)|Copa Federación|Copa Eva|Copa de Oro|Ligapokal Pre|Supercoppa di Serie"),
    ("uecl", r"Conference League"),
    ("l3", r"Third Division|Fourth Division|League One|League Two|Serie C|Serie D|3\. Liga|Regionalliga|Oberliga|Segunda División B|Segunda Federación|Tercera|Primera Federación|Championnat National|National 2|CFA|Lega Pro|Prima Divisione|Seconda Divisione|Division 3|Division d'Honneur|Football Conference|National League|Amateur"),
    ("ucl", r"UEFA Champions League|European Cup(?! Winners)|European Champion Clubs"),
    ("cwc", r"Cup Winners'? Cup"),
    ("usc", r"UEFA Super Cup|European Super Cup"),
    ("uel", r"UEFA Europa League(?! Conference)|UEFA Cup(?! Winners)"),
    ("uecl", r"Conference League"),
    ("world", r"Intercontinental Cup|Club World Cup|FIFA Club World"),
    ("l2", r"Second Division|EFL Championship|Football League Championship|^\W*Championship|Segunda División|Segunda Division|2\. Bundesliga|Serie B|Ligue 2|Division 2|Zweite"),
    ("lcup", r"League Cup|EFL Cup|Football League Cup|Coupe de la Ligue|Copa de la Liga|Ligapokal|Taça da Liga"),
    ("super", r"Community Shield|Charity Shield|Supercopa|DFL-Supercup|DFB-Supercup|German Super ?Cup|Supercoppa|Trophée des [Cc]hampions|Super Cup"),
    # 5대 리그 밖(챔스에 나오는 네덜란드·포르투갈·튀르키예·그리스·스코틀랜드 등) 자국 컵·리그도 인식(2026-09-29)
    ("cup", r"FA Cup|Copa del Rey|Copa del Generalísimo|Copa de España|DFB-Pokal|German Cup|Tschammer|Coppa Italia|Coupe de France|Copa del Presidente"
            r"|KNVB Cup|Taça de Portugal|Turkish Cup|Türkiye Kupası|Greek (Football )?Cup|Scottish Cup|Belgian Cup|Austrian Cup|ÖFB|Swiss Cup|Czech Cup|Danish Cup"
            r"|DBU Pokalen|Ukrainian Cup|Croatian (Football )?Cup|Serbian Cup|Norwegian (Football )?Cup|Kazakhstan Cup|Cypriot Cup|Azerbaijan Cup|Slovak Cup|Soviet Cup|Yugoslav Cup"),
    ("league", r"Premier League|First Division|La Liga|Primera División|Primera Division|Bundesliga|German (football )?champ|Serie A|Italian (football )?champ|Ligue 1|Division 1|French (football )?champ|Championnat de France|English champions|Spanish champ|Football League(?! Cup| Trophy)|Scudetto|Divisione Nazionale"
               r"|Eredivisie|Netherlands Football League Championship|Dutch champ|Primeira Liga|Primeira Divisão|Portuguese champ|Süper Lig|Turkish (Football )?Championship|Super League Greece|Alpha Ethniki|Panhellenic Championship|Greek champ"
               r"|Scottish Premiership|Scottish Premier League|Scottish (Football )?League(?! Cup)|Scottish champ|Belgian Pro League|Belgian First Division|Belgian champ|Austrian (Football )?Bundesliga|Austrian champ"
               r"|Swiss Super League|Nationalliga A|Swiss champ|Czech First League|Czechoslovak First League|Danish Superliga|Danish champ|Ukrainian Premier League|Soviet Top League|Croatian First (Football )?League|Prva HNL|HNL"
               r"|Serbian SuperLiga|Yugoslav First League|Eliteserien|Norwegian champ|Kazakhstan Premier League|Azerbaijan Premier League|Slovak (First Football League|Super Liga)|Niké liga|Fortuna liga"),
]
_HON_SEASON = re.compile(r"(?<![\d/])((?:18|19|20)\d\d(?:\s*[–\-/]\s*(?:\d{4}|\d{2}))?)(?!\d)")

_HON_CUT = re.compile(r"^={2,4}\s*(European (Achievements|record|results|history)|Record in|Seasons|League history|Youth|Reserve|Academy|Women|Ladies|Doubles|Trebles|Regional|Friendly|Friendlies|Minor|Invitational|Other|Pre-season|Individual|Awards|Records|Unofficial|Amateur|Junior|B team|II team|Futsal|Basketball|Handball|Esports|Feminine|Femenino|Reserves|Second team|Minor titles|Minor trophies|Other titles|Other competitions)", re.I | re.M)

def _hon_strip(w):
    # 유스·2군·여자팀·더블·지역 대회·친선 대회 소제목은 그 소제목 구간만 버림(다음 같은/상위 단계 소제목 전까지)
    out, skip_level = [], None
    for line in w.split("\n"):
        h = re.match(r"^(={2,5})\s*(.*?)\s*\1\s*$", line)
        if h:
            lv = len(h.group(1))
            if skip_level is not None and lv <= skip_level:
                skip_level = None
            if skip_level is None and _HON_CUT.match(line):
                skip_level = lv
        if skip_level is None:
            out.append(line)
    w = "\n".join(out)
    w = re.sub(r"<sup[^>]*>.*?</sup>", "", w, flags=re.S)
    # 메달 아이콘 같은 그림 링크 제거 — "[[File:Gold medal icon.svg]] Winners (1): …"에서 File:의 콜론 때문에 줄이 잘못 나뉘었음(칼리아리·슬로반)
    w = re.sub(r"\[\[(?:File|Image):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", "", w)
    w = re.sub(r"\{\{(?:lang|nowrap|nobr)\|(?:[a-z-]+\|)?([^{}|]*(?:\[\[[^\]]*\]\][^{}|]*)*)[^{}]*\}\}", r"\1", w)
    w = re.sub(r"<ref[^>]*/>", "", w)
    w = re.sub(r"<ref[^>]*>.*?</ref>", "", w, flags=re.S)
    w = re.sub(r"<!--.*?-->", "", w, flags=re.S)
    return w

def _hon_link_text(s):
    # [[A|B]] → "A B" (분류는 대상·표시 둘 다로), 템플릿 제거
    s = re.sub(r"\{\{(?:flagicon|fbaicon|nowrap|sort|small|refn|efn|sfn|abbr)[^{}]*\}\}", " ", s)
    s = re.sub(r"\[\[([^\]|]*)\|([^\]]*)\]\]", r"\1 / \2", s)
    s = re.sub(r"\[\[([^\]]*)\]\]", r"\1", s)
    return re.sub(r"'''?|\{\{[^{}]*\}\}|style=\"[^\"]*\"|scope=\"?\w+\"?|align=\"?\w+\"?", " ", s)

def _hon_classify(name):
    lv = re.search(r"(?:level|tier)\s*(\d)|\((I{2,4}|IV|V)\)", name, re.I)
    if lv:   # "(level 2)" 표기가 있으면 이름보다 우선(잉글랜드 1992~2004 "First Division"은 2부였음)
        n = int(lv.group(1)) if lv.group(1) else {"II": 2, "III": 3, "IIII": 4, "IV": 4, "V": 5}[lv.group(2)]
        if n >= 2 and not re.search(r"UEFA|European|Cup", name):
            return "l2" if n == 2 else "l3"
    for cat, rx in _HON_RULES:
        if re.search(rx, name):
            return cat
    return None

def _hon_display(s):
    # 링크는 보이는 글자만([[1993–94 Coupe de France|1994]] → 1994) — 대상 제목의 시즌까지 세면 두 번 셌음
    s = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[\[([^\]]*)\]\]", r"\1", s)
    return re.sub(r"\{\{[^{}]*\}\}", " ", s)

def _hon_count(text):
    return len(set(m.group(1).replace(' ', '') for m in _HON_SEASON.finditer(_hon_display(text))))

def _parse_honours(wikitext):
    """→ {분류: 우승 횟수}, [(대회명, 분류, 횟수, 방식)] — 표(대회·횟수·시즌) 또는 목록(* 대회 / ** Winners: 시즌…)"""
    w = _hon_strip(wikitext)
    found = []
    # 1) 표: 행마다 대회(! 셀) + 횟수(숫자만 있는 셀) + 시즌
    for table in re.findall(r"\{\|.*?\n\|\}", w, flags=re.S):
        if not re.search(r"Titles|Winners|Competition|Honou?rs|Trophies", table[:800], re.I):
            continue   # 우승 기록 표만(브뤼헤처럼 같은 절에 시즌별 순위 표가 같이 있는 경우 제외)
        for row in re.split(r"\n\|-[^\n]*", table):
            cells = [c.strip() for c in re.split(r"\n[!|]|\|\||!!", "\n" + row) if c.strip()]
            comp = None; n = None; seasons = ""
            for c in cells:
                val = c.split("|", 1)[-1] if re.match(r'^\s*(style|scope|align|rowspan|colspan|class|width|bgcolor|data-sort-value)', c) else c
                val = val.strip()
                t = _hon_link_text(val).strip()
                if comp is None and re.search(r"[A-Za-z]{3}", t) and not re.match(r"\s*(18|19|20)\d\d", t) and not re.fullmatch(r"(Domestic|Continental|International|European|Worldwide|Regional|National|Type|Competition|Titles|Seasons|Friendly|Other)s?\W*", t, re.I):
                    if _hon_classify(t) or re.search(r"Cup|League|Liga|Champion|Serie|Pokal|Coppa|Coupe|Copa|Shield|Trophy|Division|Bundesliga", t):
                        comp = t; continue
                if comp and n is None and re.fullmatch(r"\d{1,2}", t):   # 우승 횟수는 두 자리까지(연도·시즌 표 숫자 오인 방지 — 브뤼헤 1645회)
                    n = int(t); continue
                if comp and n is None and re.match(r"^\W*(Winners|Champions)\W*:?", t, re.I) and ":" in t:
                    n = _hon_count(t.split(":", 1)[1]); continue
                if comp and n is not None:
                    seasons += " " + val
            if comp and n:
                found.append((comp, _hon_classify(comp), n, "table", _hon_count(seasons)))
    # 2) 목록
    if not found:
        comp = None
        for line in w.split("\n"):
            m = re.match(r"^\*\s*(?!\*)(.*)", line)
            if m:
                comp = _hon_link_text(m.group(1)).strip()
                comp_done = False   # 대회 줄에서 이미 횟수를 셌으면 아래 줄(시즌 목록)은 안 셈 — 스포르팅 컵위너스컵 2회로 이중 계산됐었음
                # "* [[FA Cup]]: 1965, 1974" 같이 한 줄에 우승 시즌이 같이 있는 형식(준우승 줄이 따로 있으면 그쪽은 무시)
                rest = re.split(r":(?![^\[]*\]\])", m.group(1), maxsplit=1)   # 링크 안의 콜론은 제외
                if len(rest) == 2 and not re.search(r"runner|finalist|second", rest[0], re.I):
                    k = _hon_count(rest[1])
                    num = re.fullmatch(r"\W*(\d{1,2})\W*", rest[1].strip())
                    if not k and num:   # 대회 줄에 "대회: 21" 횟수만 있고 시즌은 다음 줄(스포르팅 형식)
                        k = int(num.group(1))
                    if k and not re.search(r"runner", rest[1], re.I):
                        found.append((comp, _hon_classify(comp), k, "inline", k))
                        comp_done = True
                continue
            m = re.match(r"^(?:\*\*+|:+)\s*(.*)", line)   # "** Winners: …" 또는 ": Winners (19): …"(페네르바흐체 형식)
            if m and comp and not comp_done:
                sub = m.group(1)
                head = _hon_display(sub.split(":", 1)[0]) if ":" in sub else _hon_display(sub)   # 링크 대상 제목의 "Winners' Cup"을 우승 줄로 오인하지 않게 보이는 글자만
                if re.search(r"winner|champion", head, re.I) and not re.search(r"runner|play-?off", head, re.I):
                    k = _hon_count(sub.split(":", 1)[1] if ":" in sub else sub)
                    if k:
                        found.append((comp, _hon_classify(comp), k, "list", k))
    totals = {}
    for comp, cat, n, how, k in found:
        if cat and cat not in ("skip", "l3"):
            totals[cat] = totals.get(cat, 0) + n
    return totals, found


# ── 구단 최고 이적료(영입/방출): "List of … records and statistics" 문서의 이적 표 첫 줄 ──
_HON_FEE = re.compile(r"([£€$])\s?([\d]+(?:[.,]\d+)*)\s*(million|m\b|bn)?", re.I)
_HON_FEE_TRAIL = re.compile(r"(?<![\d.,])([\d]+(?:[.,]\d+)*)\s*(million|m\b)?\s*(€|euros?|£|pounds)", re.I)
def _hon_fees(row):
    out = []
    row = re.sub(r"\[\[[^\]|]*\|([£€$])\]\]", r"\1", row)   # [[Euro|€]]32 → €32
    for m in _HON_FEE.finditer(row.replace("&nbsp;", " ")):
        cur, num, unit = m.groups()
        v = float(num.replace(",", ""))
        if not unit and v >= 1e5:   # £40,200,000 → 40.2m
            v, unit = v / 1e6, "m"
        if v > 0 and (unit or v < 1000):
            out.append((cur, round(v, 1)))
    if not out:
        for m in _HON_FEE_TRAIL.finditer(row.replace("&nbsp;", " ")):
            num, unit, cur = m.groups()
            v = float(num.replace(",", ""))
            if not unit and v >= 1e5:
                v, unit = v / 1e6, "m"
            if v > 0 and (unit or v < 1000):
                out.append(("€" if cur.lower().startswith(("€", "euro")) else "£", round(v, 1)))
    return out

def _hon_record_row(row, prefer_eur):
    row = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", row, flags=re.S)
    row = re.sub(r"\{\{(?:efn|refn)[^{}]*\}\}", " ", row)
    row = re.sub(r"\{\{#tag:ref.*", " ", row, flags=re.S)   # 비고 칸 각주(다른 금액이 섞여 있음)는 버림
    row = re.sub(r"\{\{(?:[Uu]pdated|[Aa]bbr)[^{}]*\}\}", " ", row)
    # {{sortname|Philippe|Coutinho|(링크 대상)}} → [[대상|Philippe Coutinho]] (바르셀로나 표 형식)
    rank = re.match(r"\s*\|?\s*(?:[a-z]+=\"?[^|\n]*\"?\s*\|)?\s*(\d+)\s*(?:\n|\|\|)", row)
    row = re.sub(r"(?i)\{\{sortname\|([^|{}]+)\|([^|{}]*)(?:\|([^|{}]*))?[^{}]*\}\}",
                 lambda m: f"[[{(m.group(3) or '').strip() or (m.group(1) + ' ' + m.group(2)).strip()}|{(m.group(1) + ' ' + m.group(2)).strip()}]]", row)
    row = re.sub(r"\{\{(?:flagicon|fbaicon|flag|nowrap|sort)[^{}]*\}\}|\{\{[A-Z]{3}\}\}", " ", row)
    links = [(a.strip(), (b or a).strip()) for a, b in re.findall(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", row)
             if not re.match(r"(File|Image|:?[a-z]{2}:)", a) and not re.fullmatch(r"[\d–\-/ ]+", (b or a).strip())
             and not re.search(r"\d{4}.*(season|window|transfer|League|Liga)", a)]
    fees = _hon_fees(row)
    years = re.findall(r"(?<!\d)((?:19|20)\d\d)(?!\d)", re.sub(r"\[\[[^\]]*\|", "", row))
    rank = int(rank.group(1)) if rank else None
    if len(links) < 2 or (not fees and rank != 1):
        return None
    if not fees:   # 1위인데 이적료 비공개(예: 브렌트퍼드 상가레)
        return {"player_title": links[0][0], "player": links[0][1], "club_title": links[1][0], "club": links[1][1],
                "fee": None, "year": years[-1] if years else None, "_v": 0, "_rank": 1}
    eur = [f for f in fees if f[0] == "€"]
    cur, v = eur[0] if (prefer_eur and eur) else fees[0]
    # 크기 비교용(파운드·달러는 대략 유로로 환산)
    value = v * {"€": 1, "£": 1.17, "$": 0.92}[cur]
    return {"player_title": links[0][0], "player": links[0][1], "club_title": links[1][0], "club": links[1][1],
            "fee": f"{cur}{v:g}m", "year": years[-1] if years else None, "_v": value, "_rank": rank}

def _parse_record_transfers(w, prefer_eur=False):
    """→ {"paid": {...}, "received": {...}} — 소제목(paid/received, in/out)마다 첫 표에서 이적료가 가장 큰 줄
    (날짜순으로 정렬된 표도 있어서 첫 줄을 그대로 쓰면 안 됨 — 바르셀로나 표가 최근 이적 순이었음)"""
    out = {}
    parts = re.split(r"^(={2,5}[^=\n]+={2,5})\s*$", w, flags=re.M)
    heads = [("", parts[0])] + [(parts[i], parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
    for head, body in heads:
        h = head.lower()
        kind = "paid" if re.search(r"paid|\bin\b|purchase|signing|bought", h) else "received" if re.search(r"received|\bout\b|sale|sold", h) else None
        if not kind or kind in out:
            continue
        t = re.search(r"\{\|.*?\n\|\}", body, flags=re.S)
        if not t:
            continue
        rows = [r for r in (_hon_record_row(x, prefer_eur) for x in re.split(r"\n\|-[^\n]*", t.group(0))[1:]) if r]
        if rows:
            ranked = [r for r in rows if r["_rank"] == 1]
            best = ranked[0] if ranked else max(rows, key=lambda r: r["_v"])   # 순위 칸이 있으면 1위, 없으면 최고액
            best.pop("_v"); best.pop("_rank")
            out[kind] = best
    return out

# ── 지난 시즌 공식 최종 순위·유럽 대항전 진출·강등·승점 감점 (영문 위키백과 시즌 문서의 Sports table) — 순위표 (2026-09-29) ──
# 순위표의 존 색을 "표준 배정(LEAGUE_ZONES)"이 아니라 그 시즌 실제 결과로: 컵 우승 팀의 유로파행(예: 23-24 맨유 8위 → 유로파),
# 리그컵 우승 팀 자리가 6위로 내려간 것 등. 승점 감점(23-24 에버턴 −8·노팅엄 −4)도 여기서 받아 서버 순위표에 반영.
SEASON_ZONE_TITLE = {"PL": "Premier League", "PD": "La Liga", "BL1": "Bundesliga", "SA": "Serie A", "FL1": "Ligue 1"}

def _zone_of(text):
    t = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text or "")
    t = re.sub(r"\{\{nowrap\|(.*)\}\}", r"\1", t)
    if re.search(r"relegation play-?off", t, re.I): return "rel-po"
    if re.search(r"^\W*relegat", t, re.I): return "rel"
    if re.search(r"Champions League", t):
        return "cl-q" if re.search(r"qualifying|play-?off", t, re.I) else "cl"
    if re.search(r"Europa League", t): return "el"
    if re.search(r"Conference League", t): return "ecl"
    return None

def _norm_club(n):
    import unicodedata
    n = unicodedata.normalize("NFD", n or "").encode("ascii", "ignore").decode().lower()
    n = re.sub(r"\b(f\.?c\.?|c\.?f\.?|a\.?f\.?c\.?|s\.?c\.?|a\.?c\.?|calcio|club|de|futbol|football|u\.?d\.?|ssc|ss|as|us|rc|rcd|ca|sv|vfl|vfb|tsg|1899|1909|1913|1907|1901|1848|63|29|05|04|07|98|1919|hsc|ogc|osc|aj|es|sco|ac)\b", " ", n)
    return re.sub(r"[^a-z0-9]+", " ", n).strip()

def fetch_season_zones():
    import json
    path = f"{MODEL_DIR}/season_zones.json"
    out = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            out = json.load(f)
    df = pd.read_csv(f"{MODEL_DIR}/all_matches.csv")
    wiki = {}
    if os.path.exists(f"{MODEL_DIR}/team_wiki.json"):
        with open(f"{MODEL_DIR}/team_wiki.json", 'r', encoding='utf-8') as f:
            wiki = json.load(f)
    by_title = {v.get("en_title"): t for t, v in wiki.items() if v.get("en_title")}
    print("\n[지난 시즌 공식 순위 구역]")
    for code, name in SEASON_ZONE_TITLE.items():
        for yr in [y for y in MATCH_SEASONS if y < max(MATCH_SEASONS)]:   # 끝난 시즌만
            title = f"{yr}–{str(yr + 1)[2:]} {name}"
            try:
                page = _wiki_get("https://en.wikipedia.org/w/api.php", {"action": "parse", "page": title, "prop": "wikitext", "format": "json", "redirects": 1})
                w = page["parse"]["wikitext"]["*"]
            except Exception as e:
                print(f"  ❌ {title}: {e}"); continue
            m = re.search(r"\{\{#invoke:\s*Sports table.*?(?=</onlyinclude>|\n==)", w, re.S)
            if not m:   # 라리가·분데스·세리에·리그앙 시즌 문서는 표를 {{2023–24 La Liga table}} 틀로 따로 둠
                tpl = re.search(r"\{\{\s*(\d{4}–\d{2} [^{}|]*? table)\s*\}\}", w)
                if tpl:
                    try:
                        tw = _wiki_get("https://en.wikipedia.org/w/api.php", {"action": "parse", "page": "Template:" + tpl.group(1),
                                                                                 "prop": "wikitext", "format": "json", "redirects": 1})["parse"]["wikitext"]["*"]
                        m = re.search(r"\{\{#invoke:\s*Sports table.*?(?=</onlyinclude>|\n==|$)", tw, re.S)
                    except Exception:
                        m = None
            if not m:
                print(f"  ⚠️ {title}: 표 없음"); continue
            tb = m.group(0)
            order = [x.strip() for x in re.search(r"\|\s*team_order\s*=\s*([^|\n]+)", tb).group(1).split(",") if x.strip()]
            results = {int(k): v.strip() for k, v in re.findall(r"\|\s*result(\d+)\s*=\s*([^|\s]+)", tb)}
            texts = {k: v for k, v in re.findall(r"\|\s*text_([^\s=|]+)\s*=\s*([^\n]+)", tb)}
            names = {k: re.findall(r"\[\[([^\]|]+)", v) for k, v in re.findall(r"\|\s*name_([^\s=|]+)\s*=\s*([^\n]+)", tb)}
            adjust = {k: int(v) for k, v in re.findall(r"\|\s*adjust_points_([^\s=|]+)\s*=\s*([+-]?\d+)", tb)}
            ours = sorted(set(df[(df.league == code) & (df.season == yr)].home_team))
            norm_ours = {_norm_club(t): t for t in ours}
            def match(abbr, pos):
                for tgt in names.get(abbr, []):
                    if tgt in by_title and by_title[tgt] in ours:
                        return by_title[tgt]
                    n = _norm_club(tgt)
                    if n in norm_ours:
                        return norm_ours[n]
                    cand = [t for k, t in norm_ours.items() if k and (k in n or n in k)]
                    if len(cand) == 1:
                        return cand[0]
                return None
            zones, adj, unmatched = {}, {}, []
            for i, abbr in enumerate(order, start=1):
                team = match(abbr, i)
                if not team:
                    unmatched.append(abbr); continue
                code_r = results.get(i)
                z = _zone_of(texts.get(code_r, "")) if code_r else None
                zones[team] = {"pos": i, "zone": z}
                if abbr in adjust:
                    adj[team] = adjust[abbr]
            if unmatched or len(zones) != len(ours):
                print(f"  ⚠️ {title}: 매칭 안 된 팀 {unmatched} (우리 {len(ours)}팀 / 위키 {len(order)}팀)")
                if len(zones) < len(ours) - 1:
                    continue
            out.setdefault(code, {})[str(yr)] = {"teams": zones, "adjust": adj,
                                                  "source": f"https://en.wikipedia.org/wiki/{requests.utils.quote(title.replace(' ', '_'))}"}
            print(f"  ✅ {title}: {len(zones)}팀, 감점 {adj or '없음'}")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)

def _wiki_sections(title):
    return _wiki_get("https://en.wikipedia.org/w/api.php", {"action": "parse", "page": title, "prop": "sections",
                                                            "format": "json", "redirects": 1}).get("parse", {}).get("sections")

def _wiki_section_text(title, index):
    return _wiki_get("https://en.wikipedia.org/w/api.php", {"action": "parse", "page": title, "prop": "wikitext", "section": index,
                                                            "format": "json", "redirects": 1})["parse"]["wikitext"]["*"]

def _wiki_honours_records(en_title, prefer_eur=False):
    out = {}
    secs = _wiki_sections(en_title) or []
    hon = [x for x in secs if x["level"] == "2" and re.match(r"(honours|honors|achievements|trophies|titles)", x["line"], re.I)]
    if hon:
        totals, _ = _parse_honours(_wiki_section_text(en_title, hon[0]["index"]))
        out["honours"] = {k: v for k, v in totals.items() if v}
        out["honours_url"] = f"https://en.wikipedia.org/wiki/{requests.utils.quote(en_title.replace(' ', '_'))}#{requests.utils.quote(hon[0]['anchor'])}"
    base = re.sub(r"\s*\(.*\)$", "", en_title)
    for rt in (f"List of {base} records and statistics", f"List of {base.replace('F.C.', 'FC')} records and statistics"):
        rsecs = _wiki_sections(rt)
        if not rsecs:
            continue
        tr = [x for x in rsecs if re.search(r"transfer", x["line"], re.I)]
        text = "".join(_wiki_section_text(rt, x["index"]) + "\n" for x in tr[:3])
        recs = _parse_record_transfers(text, prefer_eur) if text else {}
        if recs:
            # 선수·구단 한국어 이름(한국어 위키백과 문서 제목, 없으면 영어 그대로)
            titles = sorted({r[k] for r in recs.values() for k in ("player_title", "club_title")})
            ko = {}
            q = _wiki_get("https://en.wikipedia.org/w/api.php", {"action": "query", "titles": "|".join(titles), "prop": "langlinks",
                                                                  "lllang": "ko", "redirects": 1, "format": "json"})["query"]
            alias = {r["from"]: r["to"] for r in q.get("redirects", []) + q.get("normalized", [])}
            for pg in q.get("pages", {}).values():
                if pg.get("langlinks"):
                    ko[pg["title"]] = pg["langlinks"][0]["*"]
            for r in recs.values():
                for k in ("player", "club"):
                    t = r.pop(f"{k}_title")
                    t = alias.get(t, t)
                    if t in ko:
                        r[f"{k}_ko"] = re.sub(r"\s*\(.*\)$", "", ko[t])
            out["records"] = recs
            out["records_url"] = f"https://en.wikipedia.org/wiki/{requests.utils.quote(rt.replace(' ', '_'))}"
        break
    return out

def _wiki_one(team_name, league=None):
    title = WIKI_TITLE_OVERRIDE.get(team_name)
    qid = None
    candidates = [title] if title else [h["title"] for h in _wiki_get("https://en.wikipedia.org/w/api.php", {
        "action": "query", "list": "search", "srsearch": f"{team_name} football club", "srlimit": 5, "format": "json"})["query"]["search"]]
    candidates = [c for c in candidates if title or not _WIKI_SKIP_TITLE.search(c)]   # 2군·유스·여자팀 문서 제외
    if not candidates:
        return None
    pages = _wiki_get("https://en.wikipedia.org/w/api.php", {
        "action": "query", "titles": "|".join(candidates), "prop": "pageprops", "ppprop": "wikibase_item",
        "redirects": 1, "format": "json"})["query"]
    by_title = {p["title"]: p.get("pageprops", {}).get("wikibase_item") for p in pages.get("pages", {}).values()}
    for r in pages.get("redirects", []) + pages.get("normalized", []):
        by_title.setdefault(r["from"], by_title.get(r["to"]))
    qids = [by_title.get(c) for c in candidates if by_title.get(c)]
    ents = _wd_entities(qids, "claims")
    for q in qids:   # 검색 순서대로, 축구 클럽인 첫 문서
        kinds = {c.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id") for c in ents.get(q, {}).get("claims", {}).get("P31", [])}
        if "Q476028" in kinds or "Q103229495" in kinds:
            qid = q
            break
    if not qid:
        return None
    ent = _wd_entities([qid])[qid]
    claims = ent.get("claims", {})
    site = ent.get("sitelinks", {})
    out = {"qid": qid, "name_ko": WIKI_NAME_KO_OVERRIDE.get(team_name) or ent.get("labels", {}).get("ko", {}).get("value")}
    # 소개 글: 한국어 문서가 있으면 한국어, 없으면 영어 — 첫 문단(요약 API)이 아니라 머리말 전체
    # (리버풀 한국어 요약은 44자 한 문장, 머리말 전체는 1,399자)
    for lang in ("ko", "en"):
        link = site.get(f"{lang}wiki")
        if not link:
            continue
        pg = next(iter(_wiki_get(f"https://{lang}.wikipedia.org/w/api.php", {
            "action": "query", "prop": "extracts|info", "explaintext": 1, "exintro": 1, "inprop": "url",
            "titles": link["title"], "redirects": 1, "format": "json"})["query"]["pages"].values()))
        text = re.sub(r"\n{2,}", "\n", (pg.get("extract") or "")).strip()
        if text:
            out.update({"extract": text, "lang": lang, "url": pg.get("fullurl"), "title": link["title"]})
            break
    en_title = site.get("enwiki", {}).get("title")
    if en_title:
        out["en_title"] = en_title
        try:
            out.update(_wiki_honours_records(en_title, prefer_eur=league not in (None, "PL")))
        except Exception as e:
            print(f"  ⚠️ {team_name} 우승·이적 기록 실패(기존 유지): {e}")
    # 별칭(P1449, 한국어 → 영어)
    nick = [c["mainsnak"]["datavalue"]["value"] for c in claims.get("P1449", []) if c.get("mainsnak", {}).get("datavalue")]
    nick = [n for n in nick if n.get("language") == "ko"] or [n for n in nick if n.get("language") == "en"]
    if nick:
        out["nickname"] = nick[0]["text"]
    # 홈구장(P115) 수용 인원(P1083), 감독(P286)
    venue = _wd_current(claims, "P115")
    coach = _wd_current(claims, "P286")
    city_cands = [_wd_current(claims, "P159"), _wd_current(claims, "P131")]   # 본부 소재지 → 구단 소재지 → (아래) 경기장 소재지
    refs = _wd_entities([v["id"] for v in (venue, coach) if v], "claims|labels")
    if venue and venue["id"] in refs:
        vent = refs[venue["id"]]
        # 위키데이터 "현재" 홈구장(종료일 없는 값). football-data 경기장 이름은 옛 이름·옛 경기장인 경우가 많아서
        # (에버턴 Goodison Park, 칼리아리 Sardegna Arena, 마인츠 Opel Arena 등) 화면엔 이쪽을 우선 표시
        out["stadium"] = _wd_label(vent)
        cap = _wd_current(vent.get("claims", {}), "P1083")
        if cap:
            out["capacity"] = int(float(cap["amount"]))
        city_cands.append(_wd_current(vent.get("claims", {}), "P131"))
    for c in ([] if team_name in WIKI_CITY_OVERRIDE else city_cands):
        name = _wd_city(c["id"]) if c else None
        if name:
            out["city"] = name
            break
    if "city" not in out:   # "도시"로 분류된 항목을 못 찾으면 첫 후보 이름 그대로
        first = next((c for c in city_cands if c), None)
        if first:
            out["city"] = _wd_label(_wd_cache.get(first["id"]) or _wd_entities([first["id"]], "labels").get(first["id"]))
    if team_name in WIKI_CITY_OVERRIDE:
        out["city"] = WIKI_CITY_OVERRIDE[team_name]
    if coach and coach["id"] in refs:
        out["coach"] = _wd_label(refs[coach["id"]])
    return out

def fetch_team_wiki(force=False):
    import json
    from datetime import date, timedelta
    path = f"{MODEL_DIR}/team_wiki.json"
    wiki = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            wiki = json.load(f)
    with open(f"{MODEL_DIR}/team_info.json", 'r', encoding='utf-8') as f:
        teams = sorted(json.load(f))
    only = [a for a in __import__("sys").argv[2:] if not a.startswith("--")]   # --wiki-only "팀 이름" ... 이면 그 팀만
    if only:
        teams, force = [t for t in teams if t in only], True
    stale = (date.today() - timedelta(days=WIKI_REFRESH_DAYS)).isoformat()
    todo = [t for t in teams if force or wiki.get(t, {}).get("fetched", "") < stale]
    print(f"\n[구단 소개(위키)] {len(todo)}/{len(teams)}팀 갱신")
    from concurrent.futures import ThreadPoolExecutor
    league_of = {}
    try:
        with open(f"{MODEL_DIR}/schedule.json", 'r', encoding='utf-8') as f:
            for code, ms in json.load(f).items():
                if code != 'CL':
                    for m in ms:
                        league_of[m['home_team']] = league_of[m['away_team']] = code
    except Exception:
        pass
    def one(t):
        try:
            return t, _wiki_one(t, league_of.get(t)), None
        except Exception as e:
            return t, None, e
    with ThreadPoolExecutor(max_workers=2) as ex:   # 위키미디어 권장 범위의 적은 동시 요청(팀당 요청 5~6개라 순차면 10분+)
        for t, info, err in ex.map(one, todo):
            if err:
                print(f"  ❌ {t}: {err}")   # 실패하면 기존 값 유지
            elif not info:
                print(f"  ⚠️ 문서를 못 찾음: {t}")
            else:
                prev = wiki.get(t, {})
                # 우승 횟수는 줄어들 수 없음 — 위키 편집으로 표 형식이 바뀌어 적게 읽히면 이전 값을 유지
                old_h, new_h = prev.get("honours") or {}, info.get("honours")
                if old_h and not force and (new_h is None or any(new_h.get(k, 0) < v for k, v in old_h.items())):   # --force(파서 수정 후 재수집)면 새 값
                    print(f"  ⚠️ {t} 우승 기록이 줄어들게 읽힘 → 이전 값 유지 ({old_h} → {new_h})")
                    info["honours"] = old_h
                    info["honours_url"] = prev.get("honours_url", info.get("honours_url"))
                if prev.get("records") and not info.get("records"):
                    info["records"], info["records_url"] = prev["records"], prev.get("records_url")
                info["fetched"] = date.today().isoformat()
                wiki[t] = info
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(wiki, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"  ✅ team_wiki.json 저장 ({len(wiki)}팀)")

def _fetch_af_transfers(af_id, limit=30):
    """API-Football /transfers → 최근 이적 기록(이 팀이 관련된 것만). 선수 사진·양쪽 구단 로고·방향(in/out)까지 저장
    — 예전엔 이름·날짜·유형만 저장해서 화면에 글자만 나왔음(2026-09-29 확장). 사진은 선수 ID로 만드는 고정 주소."""
    raw = _af_get(f"{API_FOOTBALL_URL}/transfers", {"team": af_id}).get("response", [])
    flat = []
    for item in raw:
        player = item.get("player") or {}
        pid = player.get("id")
        for t in item.get("transfers", []):
            tin = t.get("teams", {}).get("in") or {}
            tout = t.get("teams", {}).get("out") or {}
            if af_id not in (tin.get("id"), tout.get("id")):
                continue
            flat.append({
                "player": player.get("name"),
                "player_id": pid,
                "photo": f"https://media.api-sports.io/football/players/{pid}.png" if pid else None,
                "date": t.get("date"),
                "type": t.get("type"),
                "from": tout.get("name"), "from_logo": tout.get("logo"),
                "to": tin.get("name"), "to_logo": tin.get("logo"),
                "dir": "in" if tin.get("id") == af_id else "out",
            })
    flat.sort(key=lambda x: x["date"] or "", reverse=True)
    return flat[:limit]

def refresh_transfers():
    """이적 기록만 다시 받기(스쿼드는 그대로): python update_data.py --transfers-only
    team_extra.json에 있는 팀 전부 — 팀 ID를 저장해 둔 팀은 호출 1번, 아니면 검색 포함 2번(무료 한도 100회/일, 분당 10회)."""
    import json
    if not API_FOOTBALL_KEY:
        print("  ⚠️ API_FOOTBALL_KEY가 없어 건너뜀")
        return
    extra_path = f"{MODEL_DIR}/team_extra.json"
    with open(extra_path, 'r', encoding='utf-8') as f:
        extra = json.load(f)
    with open(f"{MODEL_DIR}/schedule.json", 'r', encoding='utf-8') as f:
        schedule = json.load(f)
    league_of = {t: code for code, ms in schedule.items() if code != 'CL' for m in ms for t in (m['home_team'], m['away_team'])}
    print(f"\n[이적 기록 갱신] {len(extra)}팀")
    for team_name, entry in extra.items():
        af_id = entry.get("af_id")
        if not af_id:
            country = LEAGUE_COUNTRY.get(league_of.get(team_name))
            af_id = _search_af_team_id(team_name, country) if country else None
            time.sleep(7)
            if not af_id:
                print(f"  ⚠️ 매칭 실패: {team_name}")
                continue
            entry["af_id"] = af_id
        try:
            transfers = _fetch_af_transfers(af_id)
        except Exception as e:
            print(f"  ❌ {team_name} 예외: {e}")
            continue
        if transfers:   # 한도 초과 등으로 빈 응답이면 기존 기록 유지
            entry["transfers"] = transfers
            with open(extra_path, 'w', encoding='utf-8') as f:
                json.dump(extra, f, ensure_ascii=False, indent=2)
        print(f"  {'✅' if transfers else '⚠️ 빈 응답(기존 유지)'} {team_name} ({len(transfers)}건)")
        time.sleep(7)

def fetch_squad_transfers(league_code):
    """API-Football 무료 플랜으로 스쿼드(사진/등번호)+이적 기록 수집.
    시즌 제한이 있는 통계 엔드포인트와 달리, 스쿼드/이적 엔드포인트는
    무료 플랜에서도 시즌 제약 없이 현재 데이터를 준다 — 요청 한도(100회/일)만
    문제라 리그 단위로 나눠서 점진적으로 채운다.
    팀ID는 /teams?league=&season=으로 한 번에 못 가져온다 — 이 조합은
    시즌 제한에 걸려 무료 플랜에서 빈 배열만 옴. 대신 팀 이름 검색
    (/teams?search=, 시즌 무관)으로 팀별로 하나씩 찾는다.
    이미 처리된 팀은 건너뛰어서(팀당 3회 호출·10회/분 한도라 20팀 전체를
    한 번에 다 못 돌 수도 있음) 재실행 시 이어서 채울 수 있게 한다."""
    import json
    print(f"\n[{league_code} 스쿼드+이적] 수집 중...")

    if not API_FOOTBALL_KEY:
        print("  ⚠️ API_FOOTBALL_KEY가 없어 건너뜀")
        return

    country = LEAGUE_COUNTRY.get(league_code)
    if not country:
        print(f"  ❌ {league_code}: 국가 매핑 없음")
        return

    schedule_path = f"{MODEL_DIR}/schedule.json"
    with open(schedule_path, 'r', encoding='utf-8') as f:
        schedule = json.load(f)
    our_teams = sorted({m['home_team'] for m in schedule.get(league_code, [])} |
                        {m['away_team'] for m in schedule.get(league_code, [])})

    extra_path = f"{MODEL_DIR}/team_extra.json"
    extra = {}
    if os.path.exists(extra_path):
        with open(extra_path, 'r', encoding='utf-8') as f:
            extra = json.load(f)

    team_info_path = f"{MODEL_DIR}/team_info.json"
    team_info = {}
    if os.path.exists(team_info_path):
        with open(team_info_path, 'r', encoding='utf-8') as f:
            team_info = json.load(f)

    for team_name in our_teams:
        if extra.get(team_name, {}).get("squad"):
            continue  # 이미 처리됨 — 재실행 시 스킵

        af_id = _search_af_team_id(team_name, country)
        time.sleep(7)
        if not af_id:
            print(f"  ⚠️ 매칭 실패: {team_name}")
            continue

        entry = extra.get(team_name, {})
        try:
            squad_raw = (_af_get(f"{API_FOOTBALL_URL}/players/squads", {"team": af_id}).get("response") or [{}])[0].get("players", [])
            squad_clean = [
                {
                    "name": p.get("name"),
                    "position": SQUAD_POSITION_MAP.get(p.get("position"), p.get("position")),
                    "shirtNumber": p.get("number"),
                    "photo": p.get("photo"),
                }
                for p in squad_raw
            ]
            entry["squad"] = _clean_squad(squad_clean, team_name, team_info)
            time.sleep(7)

            entry["af_id"] = af_id
            entry["transfers"] = _fetch_af_transfers(af_id)

            extra[team_name] = entry
            with open(extra_path, 'w', encoding='utf-8') as f:
                json.dump(extra, f, ensure_ascii=False, indent=2)
            print(f"  ✅ {team_name} (스쿼드 {len(entry['squad'])}명, 이적 {len(entry['transfers'])}건)")
            time.sleep(7)
        except Exception as e:
            print(f"  ❌ {team_name} 예외: {e}")

    print(f"  ✅ team_extra.json 저장 완료 ({sum(1 for t in our_teams if extra.get(t, {}).get('squad'))}/{len(our_teams)}팀)")

def reconstruct_bracket_order(stages):
    """하드코딩 없이 실제 대진(팀 실명)으로 이전 라운드의 좌우 배치를 역추적한다.

    상위 라운드가 확정되면 그 대진의 두 팀이 각각 하위 라운드 어느 매치에서
    이겼는지는 팀 이름으로 100% 정확히 역산 가능하다 (PO→R16은 1:1, 그 외는
    2:1이지만 팀 소속 여부만 보면 되므로 같은 로직으로 처리된다 — R16 매치의
    두 팀 중 시즌 내내 직행한 팀은 PO 소속이 아니라 자동으로 걸러진다).
    아직 다음 라운드가 확정되지 않은 최전선 라운드는 API 응답 순서를 그대로 둔다.
    """
    STAGE_ORDER = ["PLAYOFFS", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL"]
    present = [i for i, s in enumerate(STAGE_ORDER) if stages.get(s)]
    if not present:
        return stages

    ordered = {s: list(stages.get(s, [])) for s in STAGE_ORDER}
    top_idx = present[-1]

    def reorder_lower_by_upper(upper_ties, lower_raw):
        lower_by_team = {}
        for tie in lower_raw:
            lower_by_team[tie['team1']] = tie
            lower_by_team[tie['team2']] = tie
        result, used = [], set()
        for upper_tie in upper_ties:
            for team in (upper_tie['team1'], upper_tie['team2']):
                src = lower_by_team.get(team)
                if src is not None and id(src) not in used:
                    result.append(src)
                    used.add(id(src))
        for tie in lower_raw:
            if id(tie) not in used:
                result.append(tie)  # 대진 미확정 등 예외 상황 안전망
        return result

    for i in range(top_idx, 0, -1):
        upper_stage, lower_stage = STAGE_ORDER[i], STAGE_ORDER[i - 1]
        if not ordered[lower_stage]:
            continue
        ordered[lower_stage] = reorder_lower_by_upper(ordered[upper_stage], ordered[lower_stage])

    return ordered

def fetch_full_schedule():
    """5대리그+UCL 26-27 시즌 전체 일정(완료+예정 전부) 수집 — 일정 탭 전용.
    all_matches.csv/team_stats.csv 등 모델 학습 파이프라인과는 무관한 별도 캐시."""
    import json
    print("\n[전체 일정] 수집 중...")
    CURRENT_SEASON = 2026
    schedule = {}
    for code, name in LEAGUES_V2.items():
        print(f"  [{name}] 일정 수집 중...")
        res = requests.get(
            f"{BASE_URL}/competitions/{code}/matches",
            headers=HEADERS,
            params={"season": CURRENT_SEASON}
        )
        if res.status_code != 200:
            print(f"  ❌ {name} 일정 오류: {res.status_code}")
            time.sleep(6)
            continue

        rows = []
        for m in res.json().get("matches", []):
            hg, ag, pens = _match_goals(m)
            row = {
                "date":       m["utcDate"],
                "matchday":   m.get("matchday"),
                "stage":      m.get("stage"),
                "home_team":  m["homeTeam"]["name"],
                "away_team":  m["awayTeam"]["name"],
                "home_goals": hg,
                "away_goals": ag,
                "status":     m["status"],
            }
            if pens:
                row["penalties"] = list(pens)
            rows.append(row)
        rows.sort(key=lambda r: r["date"])
        schedule[code] = rows
        print(f"  ✅ {name} {len(rows)}경기")
        time.sleep(6)

    with open(f"{MODEL_DIR}/schedule.json", 'w', encoding='utf-8') as f:
        json.dump(schedule, f, ensure_ascii=False, indent=2)
    print(f"  ✅ schedule.json 저장 완료")
    return schedule

def update_prediction_log(schedule):
    """
    AI 예측 트랙레코드용 로그 갱신 (fotdata_model/prediction_log.json).
    - 모델 예측 로직을 여기서 재구현하지 않고, 그 시점에 실제 서빙 중인 라이브
      /predict를 그대로 호출해서 예정 경기들의 예측을 미리 스냅샷으로 저장해둔다.
      main.py와 별도로 예측 로직을 두 군데 관리하면 언젠가 반드시 어긋나서
      "기록된 예측"과 "그때 사용자가 실제로 본 예측"이 달라지는 문제가 생기므로,
      항상 라이브 API 응답을 그대로 기록하는 방식으로 그 문제 자체를 없앤다.
    - 이미 기록된 예측 중 경기가 끝난 것들은 실제 결과와 대조해 적중 여부를 채운다.
    """
    import json
    print("\n[예측 트랙레코드] 갱신 중...")
    API_BASE = "https://fotdata-api.onrender.com"
    log_path = f"{MODEL_DIR}/prediction_log.json"

    log = {}
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            log = json.load(f)

    now = pd.Timestamp.utcnow().tz_localize(None)

    # 1) 완료된 경기 결과로 기존 로그 백필
    filled = 0
    for code, rows in schedule.items():
        for m in rows:
            if m["status"] != "FINISHED" or m["home_goals"] is None or m["away_goals"] is None:
                continue
            key = f"{code}|{m['home_team']}|{m['away_team']}|{m['date'][:10]}"
            entry = log.get(key)
            if not entry or entry.get("actual") is not None:
                continue
            hg, ag = m["home_goals"], m["away_goals"]
            actual = "home_win" if hg > ag else ("away_win" if hg < ag else "draw")
            entry["actual"] = actual
            entry["actual_score"] = f"{hg}-{ag}"
            entry["correct"] = (actual == entry["predicted"])
            filled += 1
    if filled:
        print(f"  ✅ {filled}건 결과 대조 완료")

    # 2) 새로 예정된 경기들 미리 예측해서 기록 (앞으로 25일 내, 아직 안 찍힌 것만)
    # 2026-09-22에 발견: 국제 A매치 기간처럼 5대 리그+UCL이 동시에 2주 이상
    # 쉬는 구간이 있으면 10일 윈도우 안에 걸리는 경기가 하나도 없어서 트랙레코드가
    # 계속 텅 비는 문제가 있었음(코드 버그가 아니라 윈도우가 실제 리그 휴식기보다
    # 짧았던 것) — 어떤 휴식기에도 다음 라운드가 걸리도록 25일로 넉넉하게 늘림.
    horizon = now + pd.Timedelta(days=25)
    logged = 0
    for code, rows in schedule.items():
        for m in rows:
            if m["status"] not in ("SCHEDULED", "TIMED"):
                continue
            try:
                match_dt = pd.Timestamp(m["date"]).tz_localize(None)
            except Exception:
                continue
            if not (now < match_dt <= horizon):
                continue
            key = f"{code}|{m['home_team']}|{m['away_team']}|{m['date'][:10]}"
            if key in log:
                continue
            try:
                resp = requests.post(
                    f"{API_BASE}/predict",
                    json={"home_team": m["home_team"], "away_team": m["away_team"]},
                    timeout=60,
                )
                if resp.status_code != 200:
                    continue
                pred = resp.json()
                log[key] = {
                    "league": code,
                    "home_team": m["home_team"],
                    "away_team": m["away_team"],
                    "date": m["date"],
                    "predicted": pred["prediction"],
                    "home_win_prob": pred["probabilities"]["home_win"],
                    "draw_prob": pred["probabilities"]["draw"],
                    "away_win_prob": pred["probabilities"]["away_win"],
                    "predicted_score": (pred.get("score_prediction") or {}).get("most_likely"),
                    "logged_at": now.isoformat(),
                    "actual": None,
                    "actual_score": None,
                    "correct": None,
                }
                logged += 1
                time.sleep(1)
            except Exception as e:
                print(f"  ⚠️ 예측 기록 실패 ({m['home_team']} vs {m['away_team']}): {e}")
    if logged:
        print(f"  ✅ {logged}건 신규 예측 기록")

    # 3) 로그 크기 관리 — 결과가 확정된 것 중 오래된 건 정리(최근 500건만 유지), 미확정 건은 계속 보관
    resolved_keys = sorted(
        [k for k, e in log.items() if e.get("actual") is not None],
        key=lambda k: log[k]["date"],
    )
    if len(resolved_keys) > 500:
        for k in resolved_keys[:-500]:
            del log[k]

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"  ✅ prediction_log.json 저장 완료 (총 {len(log)}건, 결과 확정 {len(resolved_keys)}건)")

UCL_KO_STAGES = ["PLAYOFFS", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL"]

def _ucl_bracket(matches, logos):
    """한 시즌 CL 경기 → 라운드별 대진(두 경기 합산, 승부차기 반영, 실제 대진표 순서)"""
    stages = {s: [] for s in UCL_KO_STAGES}
    groups = {}   # 23-24까지 조별리그(GROUP_STAGE, group "GROUP_A") — 조별 순위표용
    agg = {}
    for m in matches:
        stage = m.get("stage", "")
        if stage == "GROUP_STAGE" and m.get("group"):
            hg, ag, _ = _match_goals(m)
            groups.setdefault(m["group"].replace("GROUP_", ""), []).append({
                "home_team": m["homeTeam"].get("name"), "away_team": m["awayTeam"].get("name"),
                "home_goals": hg, "away_goals": ag, "date": m.get("utcDate"), "status": m.get("status")})
            continue
        if stage not in stages:
            continue
        home, away = m["homeTeam"].get("name"), m["awayTeam"].get("name")
        if not home or not away:
            continue
        hg, ag, pens = _match_goals(m)
        key = (stage,) + tuple(sorted([home, away]))
        if key not in agg:
            agg[key] = {"stage": stage, "team1": home, "team2": away, "team1_goals": 0, "team2_goals": 0, "legs": [],
                        "status": "FINISHED", "pens": None}
        v = agg[key]
        leg = {"home_team": home, "away_team": away, "home_goals": hg, "away_goals": ag, "date": m.get("utcDate")}
        if hg is not None:
            if v["team1"] == home:
                v["team1_goals"] += hg; v["team2_goals"] += ag
            else:
                v["team1_goals"] += ag; v["team2_goals"] += hg
            if pens:   # 승부차기는 팀1 기준으로 저장
                v["pens"] = list(pens) if v["team1"] == home else [pens[1], pens[0]]
                leg["penalties"] = list(pens)
        if m["status"] not in ("FINISHED", "AWARDED"):
            v["status"] = "UPCOMING"
        v["legs"].append(leg)
    for v in agg.values():
        v["legs"].sort(key=lambda l: l.get("date") or "")
        t1, t2, g1, g2 = v["team1"], v["team2"], v["team1_goals"], v["team2_goals"]
        winner = None
        if v["status"] == "FINISHED":
            if g1 != g2:
                winner = t1 if g1 > g2 else t2
            elif v["pens"]:
                winner = t1 if v["pens"][0] > v["pens"][1] else t2
        stages[v["stage"]].append({
            "team1": t1, "team2": t2, "team1_goals": g1, "team2_goals": g2,
            "team1_logo": logos.get(t1, ""), "team2_logo": logos.get(t2, ""),
            "winner": winner, "status": v["status"], "legs": v["legs"], "pens": v["pens"],
        })
    out = reconstruct_bracket_order(stages)
    if groups:
        out["GROUPS"] = {g: sorted(v, key=lambda x: x["date"] or "") for g, v in sorted(groups.items())}
    return out

def fetch_ucl_tournament():
    """시즌별 UCL 토너먼트(23-24~26-27) → ucl_tournament.json {"seasons": {"2026": {...}, ...}}
    예전엔 2025 시즌만 하드코딩해서, 26-27 시즌이 시작돼도 지난 시즌 대진이 계속 나왔음(2026-09-29 수정)"""
    import json
    print("\n[UCL 토너먼트] 수집 중...")
    path = f"{MODEL_DIR}/ucl_tournament.json"
    old = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            old = json.load(f)
    seasons = dict(old.get("seasons", {})) if "seasons" in old else ({"2025": old} if old else {})
    logos = {}
    if os.path.exists(f"{MODEL_DIR}/team_logos.json"):
        with open(f"{MODEL_DIR}/team_logos.json", 'r', encoding='utf-8') as f:
            logos = json.load(f)
    for yr in MATCH_SEASONS:
        res = requests.get(f"{BASE_URL}/competitions/CL/matches", headers=HEADERS, params={"season": yr})
        time.sleep(6)
        if res.status_code != 200:
            print(f"  ❌ {yr} 시즌 오류: {res.status_code} (기존 유지)")
            continue
        seasons[str(yr)] = _ucl_bracket(res.json().get("matches", []), logos)
        print(f"  ✅ {yr}-{(yr + 1) % 100:02d}: " + ", ".join(f"{k} {len(v)}" for k, v in seasons[str(yr)].items() if v))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({"seasons": seasons}, f, ensure_ascii=False, indent=2)
    print(f"  ✅ UCL 토너먼트 저장 완료")

def train_models(df_total):
    """피처 생성 → 시간순 검증 → 전체 기간으로 서빙 모델 재학습 → 모델/team_state/accuracy 저장"""
    # 4. Feature 생성 — 모든 피처를 "그 경기 직전까지의 기록"으로 계산(미래 정보 누수 없음).
    # 서빙(/predict)은 같은 정의로 만든 팀별 최종 상태(team_state.json)를 그대로 읽어서 씀.
    print("\nFeature 생성 중...")
    df_features, team_state = build_point_in_time_features(df_total)
    df_features = df_features.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES).reset_index(drop=True)
    print(f"학습 데이터: {len(df_features)}경기")

    # 5. 시간순 검증 — 과거 80%로 학습하고 가장 최근 20% 경기로 채점. 예전의 무작위 분할은
    # 미래 경기로 학습한 모델이 과거 경기를 맞히는 셈이라 정확도가 부풀려졌음(54.7% → 실제 약 51%)
    split = int(len(df_features) * 0.8)
    train_df, test_df = df_features.iloc[:split], df_features.iloc[split:]
    X_train, y_train = train_df[FEATURES], train_df['result']
    X_test, y_test = test_df[FEATURES], test_df['result']

    scaler_eval = StandardScaler()
    X_train_scaled = scaler_eval.fit_transform(X_train)
    X_test_scaled  = scaler_eval.transform(X_test)

    lr_eval = LogisticRegression(max_iter=2000, random_state=42, C=0.1, solver='lbfgs')
    lr_eval.fit(X_train_scaled, y_train)
    acc_lr = accuracy_score(y_test, lr_eval.predict(X_test_scaled))
    logloss_lr = log_loss(y_test, lr_eval.predict_proba(X_test_scaled), labels=lr_eval.classes_)
    print(f"✅ Logistic Regression: {acc_lr:.1%} (log loss {logloss_lr:.4f})")

    def make_rf():
        return RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42)
    def make_xgb():
        return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.02,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                             random_state=42, eval_metric='mlogloss', verbosity=0)

    acc_rf = accuracy_score(y_test, make_rf().fit(X_train, y_train).predict(X_test))
    print(f"✅ Random Forest: {acc_rf:.1%}")

    le = LabelEncoder()
    le.fit(df_features['result'])
    xgb_eval = make_xgb().fit(X_train, le.transform(y_train))
    acc_xgb = accuracy_score(y_test, le.inverse_transform(xgb_eval.predict(X_test)))
    print(f"✅ XGBoost: {acc_xgb:.1%}")

    baseline_home = (y_test == 'H').mean()
    print(f"   (기준선: 전부 홈승으로 찍으면 {baseline_home:.1%})")

    # 6. 서빙용 모델은 전체 기간으로 다시 학습 (가장 최근 경기까지 반영). 화면에 표시하는
    # 정확도는 위 시간순 검증 결과.
    X_all, y_all = df_features[FEATURES], df_features['result']
    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=2000, random_state=42, C=0.1, solver='lbfgs')
    lr.fit(scaler.fit_transform(X_all), y_all)
    rf = make_rf().fit(X_all, y_all)
    xgb = make_xgb().fit(X_all, le.transform(y_all))

    joblib.dump(lr,     f"{MODEL_DIR}/logistic_regression.pkl")
    joblib.dump(rf,     f"{MODEL_DIR}/random_forest.pkl")
    joblib.dump(xgb,    f"{MODEL_DIR}/xgboost.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(le,     f"{MODEL_DIR}/label_encoder.pkl")

    import json as _json
    with open(f"{MODEL_DIR}/team_state.json", 'w', encoding='utf-8') as f:
        _json.dump(team_state, f, ensure_ascii=False, indent=1)
    print(f"✅ team_state.json 저장 완료 ({len(team_state)}팀)")

    # 정확도 저장 (시간순 검증 결과)
    accuracy_data = {
        "logistic_regression": round(acc_lr * 100, 1),
        "random_forest": round(acc_rf * 100, 1),
        "xgboost": round(acc_xgb * 100, 1),
        "best": round(max(acc_lr, acc_rf, acc_xgb) * 100, 1),
        "log_loss": round(logloss_lr, 4),
        "baseline_home_win": round(baseline_home * 100, 1),
        "evaluation": "time_split",
        "test_matches": len(test_df),
        "test_from": test_df['date'].min().strftime('%Y-%m-%d'),
        "test_to": test_df['date'].max().strftime('%Y-%m-%d'),
        "total_matches": len(df_total),
        "training_matches": len(df_features),
        "updated_at": pd.Timestamp.now().isoformat(),
    }
    with open(f"{MODEL_DIR}/accuracy.json", 'w', encoding='utf-8') as f:
        _json.dump(accuracy_data, f, ensure_ascii=False, indent=2)
    print(f"✅ accuracy.json 저장 완료")
    return accuracy_data

# football-data.org 무료 플랜이 접근 가능한 시즌(최근 4시즌, 2022 이하는 403 — 2026-09-23 실측)
MATCH_SEASONS = [2023, 2024, 2025, 2026]
MATCH_COLUMNS = ['match_id', 'date', 'league', 'home_team', 'away_team',
                 'home_goals', 'away_goals', 'matchday', 'result', 'season']

def collect_matches():
    """전 시즌 경기를 매번 다시 받아서 기존 all_matches.csv와 합집합으로 병합해 저장.

    예전엔 매일 받아오면서도 현재 시즌(2026-08-01 이후)만 새 데이터로 바꾸고 그 이전은 "기존 CSV
    유지"라, 과거 어느 시점에 덜 받아진 24-25 시즌(BL1 272/306, FL1 255/306, PD·SA 323/380)과
    23-24 시즌(BL1·SA·FL1 통째로 없음)이 그대로 굳어 있었음(2026-09-27 발견).
    - 같은 경기(날짜+홈+원정)는 새로 받은 쪽을 우선
    - 요청이 실패한 리그·시즌(한도 초과 등)은 기존 데이터를 그대로 유지 — 실패가 삭제로 이어지지 않게
    """
    fetched, failed = [], []
    for season in MATCH_SEASONS:
        print(f"\n[{season}-{season+1} 시즌]")
        for code in LEAGUES_V2:
            df_s = fetch_matches(code, season)
            if df_s.empty:
                failed.append(f"{code} {season}")
            else:
                df_s['season'] = season
                fetched.append(df_s)
            time.sleep(6)
    if failed:
        print(f"⚠️ 수집 실패(기존 데이터 유지): {', '.join(failed)}")

    parts = fetched[:]
    existing_path = f"{MODEL_DIR}/all_matches.csv"
    if os.path.exists(existing_path):
        df_existing = pd.read_csv(existing_path)
        df_existing['date'] = pd.to_datetime(df_existing['date'])
        parts.append(df_existing)
    if not parts:
        raise RuntimeError("경기 데이터를 하나도 받지 못했고 기존 파일도 없음")

    df_total = pd.concat(parts, ignore_index=True)
    df_total = df_total[[c for c in MATCH_COLUMNS if c in df_total.columns]]
    df_total = df_total.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='first')

    # match_id 없는 예전 행(초기 노트북 시절 데이터)은 "Bayern München"/"Inter Milan"/"RCD Espanyol"처럼
    # 팀 이름 표기가 달라서 위 중복 제거에 안 걸리고 같은 경기가 두 번 들어가 있었음(24-25 BL1·PD·SA 237경기,
    # 전부 같은 날·같은 스코어의 정식 경기가 따로 있음을 확인 — 2026-09-27). 정식 데이터(match_id 있음)가
    # 있는 리그·시즌에서는 예전 행을 버림
    season_year = df_total['date'].dt.year.where(df_total['date'].dt.month >= 7, df_total['date'].dt.year - 1)
    has_id = df_total['match_id'].notna()
    official = set(zip(df_total.loc[has_id, 'league'], season_year[has_id]))
    legacy_dup = ~has_id & pd.Series([k in official for k in zip(df_total['league'], season_year)], index=df_total.index)
    if legacy_dup.any():
        print(f"🧹 이름 표기가 다른 예전 중복 행 {int(legacy_dup.sum())}개 제거")
    df_total = df_total[~legacy_dup]
    # 같은 날짜 경기끼리도 순서를 고정(날짜만으로 정렬하면 실행마다 순서가 바뀌어 파일 전체가 바뀐 것처럼
    # 커밋되고, 같은 날 경기의 ELO 반영 순서도 달라짐)
    df_total = df_total.sort_values(['date', 'league', 'home_team'], kind='mergesort').reset_index(drop=True)
    df_total['match_id'] = df_total['match_id'].astype('Int64')   # 예전 행은 match_id가 없어 float로 바뀌는 것 방지
    df_total.to_csv(existing_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 전체 경기 데이터: {len(df_total)}경기 (새로 받은 {sum(len(d) for d in fetched)}경기 + 기존 병합)")
    return df_total

def main():
    print("=== FotData 자동 업데이트 시작 ===")

    # 1~2. 경기 데이터 수집 + 기존 CSV와 병합 → all_matches.csv
    df_total = collect_matches()

    # 3. 3시즌 혼합 스탯 (예측용) — 경기 수 구간별 가중치 + prestige 보정
    df_stats_current = calculate_blended_stats(df_total)
    df_stats_current.to_csv(f"{MODEL_DIR}/team_stats.csv", index=False, encoding='utf-8-sig')
    
    # 4~6. 피처 생성 + 시간순 검증 + 서빙 모델 학습/저장 (+ team_state.json, accuracy.json)
    accuracy_data = train_models(df_total)

# UCL 토너먼트
    fetch_ucl_tournament()

    # 전체 일정 (일정 탭용)
    schedule = fetch_full_schedule()

    # AI 예측 트랙레코드 (라이브 /predict를 호출하므로 반드시 위 모델 학습 이후,
    # 그리고 아직 이번 실행분 커밋을 push하기 전에 실행 — 그래야 "그 시점에 실제
    # 서빙 중이던 모델"의 예측을 기록하게 됨)
    try:
        update_prediction_log(schedule)
    except Exception as e:
        print(f"⚠️ 예측 트랙레코드 갱신 실패(다음 실행에서 재시도): {e}")

    fetch_top_scorers()

    # 로고 자동 업데이트
    update_team_logos()

    # 팀 상세정보 (홈구장/스쿼드 등)
    fetch_team_info()

    # 지난 시즌 팀 중 로고 없는 팀이 있으면 그때만(보통 새 시즌 첫날 한 번)
    try:
        if _history_missing():
            fetch_history_teams()
    except Exception as e:
        print(f"  ⚠️ 지난 시즌 팀 로고·정보 실패(기존 유지): {e}")

    # 구단 소개·별칭·홈구장 수용 인원·감독(위키백과/위키데이터, 키 불필요, 오래된 팀만)
    try:
        fetch_team_wiki()
    except Exception as e:
        print(f"  ⚠️ 구단 소개 갱신 실패(기존 유지): {e}")
    try:
        fetch_season_zones()
    except Exception as e:
        print(f"  ⚠️ 지난 시즌 순위 구역 갱신 실패(기존 유지): {e}")

    # 스쿼드 사진/등번호 + 이적 기록 (API-Football, 현재는 PL만 — 요청 한도 때문에 리그별로 점진 확대 예정)
    fetch_squad_transfers('PL')

    # (순위 예측은 main.py /predict/champion이 요청 때 현재 승점·남은 일정·모델 확률로 계산 — 2026-09-28부터
    #  예전 simulate_season/champion_predictions.json은 화면에 안 쓰여서 제거)

    print(f"\n🏆 업데이트 완료!")
    print(f"   데이터: {len(df_total)}경기")
    print(f"   서빙 모델(LR) 시간순 검증 정확도: {accuracy_data['logistic_regression']}%")

def fetch_top_scorers():
    """리그별 득점왕/도움왕 데이터 수집"""
    import json
    print("\n[선수 스탯] 수집 중...")
    
    if not API_FOOTBALL_KEY:
        print("  ⚠️ API_FOOTBALL_KEY가 없어 건너뜀")
        return
    
    # 기존 파일을 먼저 읽어두고, 새로 제대로 받아온 리그만 교체한다 — API-Football은
    # 요청 한도 초과 시에도 200 응답에 errors + 빈 response를 주기 때문에, 받은 그대로
    # 저장하면 멀쩡하던 득점왕/도움왕이 빈 배열로 덮어써짐(선수 탭 404)
    players_path = f"{MODEL_DIR}/players.json"
    all_players = {"topscorers": {}, "topassists": {}}
    if os.path.exists(players_path):
        with open(players_path, 'r', encoding='utf-8') as f:
            all_players = json.load(f)
        all_players.setdefault("topscorers", {})
        all_players.setdefault("topassists", {})

    # 무료 플랜은 24-25 시즌만 가능
    SEASON = 2024
    updated = False

    def fetch_ranking(endpoint, label, league_id, league_name):
        print(f"  [{league_name}] {label} 수집 중...")
        try:
            res = requests.get(
                f"{API_FOOTBALL_URL}/players/{endpoint}",
                headers=API_FOOTBALL_HEADERS,
                params={"league": league_id, "season": SEASON}
            )
            if res.status_code != 200:
                print(f"    ❌ 오류: {res.status_code} — 기존 데이터 유지")
                return None
            data = res.json()
            players = data.get("response") or []
            if data.get("errors") or not players:
                print(f"    ⚠️ 빈 응답/오류({data.get('errors')}) — 기존 데이터 유지")
                return None
            print(f"    ✅ {len(players)}명")
            return players
        except Exception as e:
            print(f"    ❌ 예외: {e} — 기존 데이터 유지")
            return None

    for code, league_id in LEAGUE_IDS.items():
        league_name = LEAGUES_V2.get(code, code)

        # 무료 플랜은 EPL만 가능 (다른 리그는 유료)
        if code != "PL":
            print(f"  [{league_name}] 무료 플랜 미지원, 건너뜀")
            continue

        scorers = fetch_ranking("topscorers", "득점왕", league_id, league_name)
        if scorers is not None:
            all_players["topscorers"][code] = scorers
            updated = True
        time.sleep(2)

        assists = fetch_ranking("topassists", "도움왕", league_id, league_name)
        if assists is not None:
            all_players["topassists"][code] = assists
            updated = True
        time.sleep(2)

    if not updated:
        print("  ⚠️ 새로 받은 선수 데이터 없음 — players.json 그대로 둠")
        return
    with open(players_path, 'w', encoding='utf-8') as f:
        json.dump(all_players, f, ensure_ascii=False, indent=2)
    print(f"  ✅ players.json 저장 완료")

if __name__ == "__main__":
    import sys
    if "--matches-only" in sys.argv:
        # 경기 데이터만 다시 받아서 all_matches.csv 갱신(학습·다른 산출물은 건드리지 않음)
        collect_matches()
    elif "--zones-only" in sys.argv:
        fetch_season_zones()
    elif "--wiki-only" in sys.argv:
        fetch_team_wiki(force="--force" in sys.argv)
    elif "--history" in sys.argv:
        # 한 번만: 지난 시즌 팀 로고·정보 + 챔스 조별리그(23-24) + 새 팀들 구단 소개 + 지난 시즌 공식 순위 구역
        fetch_history_teams()
        fetch_ucl_tournament()
        fetch_team_wiki()
        fetch_season_zones()
    elif "--ucl-only" in sys.argv:
        fetch_ucl_tournament()
    elif "--transfers-only" in sys.argv:
        refresh_transfers()
    else:
        main()