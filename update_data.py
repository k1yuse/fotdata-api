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

def fd_get(url, **kw):
    """football-data GET — 429(분당 10회 초과)면 서버가 알려준 초기화 시간만큼, 연결 오류면 10·20·30초 쉬고 최대 3번 재시도.
    2026-09-30부터 Render 서버도 경기 당일 결과를 같은 키로 받아와서(1분에 최대 1회) 이 작업의 6초 간격(=분당 10회)과
    겹치면 429가 날 수 있음 — 예전엔 429면 그 리그를 조용히 건너뛰었음"""
    kw.setdefault("timeout", 30)
    for attempt in range(4):
        try:
            r = requests.get(url, **kw)
        except requests.exceptions.RequestException as e:
            # 연결이 중간에 끊기는 일시 오류(2026-10-01 SSL EOF로 라리가 일정 요청이 실패해 그날 업데이트 전체가 멈춤)도 재시도
            if attempt == 3:
                raise
            wait = 10 * (attempt + 1)
            print(f"  ⚠️ 연결 오류({type(e).__name__}) — {wait}초 쉬고 다시")
            time.sleep(wait)
            continue
        if r.status_code != 429 or attempt == 3:
            return r
        wait = int(r.headers.get("X-RequestCounter-Reset") or 0) or 15 * (attempt + 1)
        print(f"  ⏳ 요청 한도(429) — {wait}초 쉬고 다시")
        time.sleep(min(wait, 65) + 1)
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
    res = fd_get(url, headers=HEADERS, params=params)
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
            res = fd_get(url, headers=HEADERS)   # 429·연결 오류 재시도 공통
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
            res = fd_get(f"{BASE_URL}/competitions/{code}/teams", headers=HEADERS)
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
            res = fd_get(f"{BASE_URL}/teams/{team_id}", headers=HEADERS)
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
            res = fd_get(f"{BASE_URL}/competitions/{code}/teams", headers=HEADERS, params={"season": yr})
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
# 감독·홈구장 같은 건 바뀌므로 WIKI_REFRESH_DAYS마다 다시 받음. 2026-09-30 7일 → 1일(매일): 감독 교체가 위키에 반영돼도
# 우리 쪽이 최대 7일 늦게 받아서 "위키백과가 느리다"로 보였음. 갱신 때는 검색하지 않고 이전에 찾은 문서를 그대로 씀(_wiki_one)
WIKI_UA = {"User-Agent": "FotData/1.0 (https://www.fotdata-official.com)"}
WIKI_REFRESH_DAYS = 1
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
    # 2026-09-30 전 팀 점검(팀 이름 단어가 문서 제목에 없는 경우를 전부 확인)에서 나온 오매칭 — 검색 결과가 실행마다 바뀌어 생김
    "Sheffield United FC": "Sheffield United F.C.",    # 다른 클럽 Sheffield F.C.가 잡힘
    "RC Celta de Vigo": "RC Celta de Vigo",            # 2군 RC Celta Fortuna가 잡힘
    "FK Kairat": "FC Kairat",                          # Qarabağ FK가 잡힘
    "PAE AEK": "AEK Athens F.C.",                      # Panathinaikos가 잡힘
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
                      "Aston Villa FC": "버밍엄", "AS Saint-Étienne": "생테티엔", "PAE AEK": "아테네", "Real Oviedo": "오비에도",
                      "Stade de Reims": "랭스", "Viking FK": "스타방에르", "SV 07 Elversberg": "슈피젠엘버스베르크"}
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

# ── 역대 우승·준우승(리그 페이지 "시즌" 탭, 2026-10-06) ──
# 영문 위키백과 리그별 우승 목록 문서 한 장씩(키 불필요). 문서마다 표 모양이 달라서 "시즌 링크가 있는 줄 → 그다음 구단 칸 둘"로 읽음
# (챔스 결승 목록은 시즌 · 나라 · 우승 · 점수 · 준우승 순 — 나라 틀·점수 칸은 건너뜀). 끝난 시즌만, HISTORY_FROM부터
LEAGUE_HISTORY_PAGES = {"PL": "List of English football champions", "PD": "List of Spanish football champions",
                        "BL1": "List of German football champions", "SA": "List of Italian football champions",
                        "FL1": "List of French football champions", "CL": "List of European Cup and UEFA Champions League finals"}
HISTORY_FROM = 2000

def _wiki_cells(row):
    """위키 표 한 줄 → 칸 목록(속성 부분 'style=…|' 은 떼고 내용만)"""
    row = row.replace("||", "\n|").replace("!!", "\n!")
    cells = []
    for line in row.split("\n"):
        if not line.startswith(("|", "!")) or line.startswith(("|}", "|+", "{|")):
            continue
        c = line[1:]
        # 첫 '|'가 [[ ]]·{{ }} 밖에 있으면 그 앞은 속성
        depth, cut = 0, None
        for i, ch in enumerate(c):
            if c.startswith(("[[", "{{"), i): depth += 1
            elif c.startswith(("]]", "}}"), i): depth -= 1
            elif ch == "|" and depth == 0:
                cut = i; break
        cells.append((c[cut + 1:] if cut is not None and "=" in c[:cut] else c).strip())
    return cells

def _club_of(cell):
    """칸 → (보이는 이름, 영문 문서 제목) — 구단이 아닌 칸(나라 틀·점수·숫자)이면 None"""
    stripped = re.search(r"<del>(.*?)</del>", cell)   # 박탈된 우승(세리에A 04-05 유벤투스 — 칼초폴리)
    if stripped:
        return re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]", r"\1", stripped.group(1)).strip(), None, True
    s = re.sub(r"<sup.*?</sup>|<ref.*?(</ref>|/>)|<!--.*?-->", "", cell, flags=re.S)
    s = re.sub(r"\{\{sortname\|([^{}|]*)\|([^{}|]*)[^{}]*\}\}", r"\1 \2", s)
    s = re.sub(r"\{\{(?:nowrap|small|center)\|((?:[^{}]|\{\{[^{}]*\}\})*)\}\}", r"\1", s)
    link = re.search(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", s)
    s2 = re.sub(r"\{\{[^{}]*\}\}", "", s)
    s2 = re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]", r"\1", s2)
    s2 = re.sub(r"'''?|†|‡|\*|\(\d+\)|\[\w\]", "", s2).strip(" ,")
    if not s2 or re.fullmatch(r"[\d\s–\-:.,()a-z]*", s2) or re.search(r"\d+\s*[–-]\s*\d+", s2):
        return None
    title = link.group(1) if link and (link.group(2) or link.group(1)).strip() in s2 else None
    return s2, title, False

def fetch_league_history():
    import json
    path = f"{MODEL_DIR}/league_history.json"
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            out = json.load(f)
    wiki = {}
    if os.path.exists(f"{MODEL_DIR}/team_wiki.json"):
        with open(f"{MODEL_DIR}/team_wiki.json", encoding="utf-8") as f:
            wiki = json.load(f)
    by_title = {v.get("en_title"): t for t, v in wiki.items() if v.get("en_title")}
    ours = {}
    if os.path.exists(f"{MODEL_DIR}/team_logos.json"):
        with open(f"{MODEL_DIR}/team_logos.json", encoding="utf-8") as f:
            ours = {_norm_club(t): t for t in json.load(f)}
    def to_ours(name, title):
        if title and title in by_title:
            return by_title[title]
        for n in (_norm_club(title or ""), _norm_club(name)):
            if n and n in ours:
                return ours[n]
        n = _norm_club(name)
        cand = [t for k, t in ours.items() if n and k and (k.startswith(n + " ") or k.endswith(" " + n) or n.startswith(k + " "))]
        return cand[0] if len(cand) == 1 else None
    print("\n[역대 우승·준우승]")
    for code, page in LEAGUE_HISTORY_PAGES.items():
        try:
            w = _wiki_get("https://en.wikipedia.org/w/api.php", {"action": "parse", "page": page, "prop": "wikitext", "format": "json", "redirects": 1})["parse"]["wikitext"]["*"]
        except Exception as e:
            print(f"  ❌ {code}: {e}"); continue
        seasons = {}
        for row in re.split(r"\n\|-", w):
            cells = _wiki_cells(row.split("\n|}")[0])   # 표 끝(|}) 뒤에 붙은 다른 틀은 떼어냄
            if any(re.match(r"^\s*[a-z_][a-z_0-9 ]*=", c) for c in cells):   # 인포박스 칸(| champions = …)은 표가 아님
                continue
            for i, c in enumerate(cells):
                m = re.search(r"\[\[[^\]|]*\|?\s*(\d{4})[–-](\d{2}|\d{4})\s*\]\]", c)
                if m and not re.search(r"\]\]\s*,", c):   # 구단별 표("1936–37, 1967–68, …")는 건너뜀
                    yr = int(m.group(1))
                    clubs = [x for x in (_club_of(c2) for c2 in cells[i + 1:]) if x][:2]
                    if yr >= HISTORY_FROM and yr not in seasons and len(clubs) == 2:
                        seasons[yr] = clubs
                    break
        rows = []
        for yr in sorted(seasons, reverse=True):
            (cn, ct, cx), (rn, rt, _) = seasons[yr]
            champ = {"name": cn, "team": to_ours(cn, ct)}
            if cx:
                champ["stripped"] = True   # 화면: "우승 박탈" 표시
            rows.append({"season": yr, "champion": champ, "runner_up": {"name": rn, "team": to_ours(rn, rt)}})
        if len(rows) < 10:
            print(f"  ⚠️ {code}: {len(rows)}시즌만 읽힘 (기존 유지)"); continue
        out[code] = {"source": f"https://en.wikipedia.org/wiki/{requests.utils.quote(page.replace(' ', '_'))}", "seasons": rows}
        miss = sorted({x["name"] for r in rows for x in (r["champion"], r["runner_up"]) if not x["team"] and not x.get("stripped")})
        print(f"  ✅ {code}: {rows[-1]['season']}–{rows[0]['season']} {len(rows)}시즌{' · 우리 팀 목록에 없는 구단 ' + ', '.join(miss) if miss else ''}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)

# ── API-Football Pro 수집(2026-10-06 결제) — 경기 상세·부상자·선수 프로필 ──
# 이번 시즌 5대 리그 + 챔스. 매일 새벽 실행(키가 있을 때만), 따로: python update_data.py --af-sync
#   af_team_map.json   API-Football 팀 ID → 우리 팀 이름(같은 날 경기를 이름 비슷함·킥오프 시각으로 맞춰 투표 — "Wolves"↔"Wolverhampton Wanderers FC")
#   af_fixtures.json   이번 시즌 경기 목록(API-Football 경기 ID·상태·우리 이름·우리 일정 날짜) — 서버가 요청 때 라인업·상세를 바로 받을 때도 씀
#   match_details/     끝난 경기 상세 한 경기 한 파일(이벤트·라인업·평점·팀 통계·선수별 기록 — 한 번 받으면 안 바뀌어서 파일만 늘어남)
#   match_previews.json 앞으로 4일 경기의 결장·부상자
#   af_players.json    팀별 선수 프로필(나이·키·몸무게·국적·사진·부상 여부) — 6일마다 갱신
# 시즌 기록(골·도움·평점·90분당 순위)은 서버가 match_details를 합산해서 만듦(af_transform.af_season_players)
AF_COMPS = {"PL": 39, "PD": 140, "BL1": 78, "SA": 135, "FL1": 61, "CL": 2}
AF_DONE = {"FT", "AET", "PEN", "AWD", "WO"}
MD_DIR = f"{MODEL_DIR}/match_details"
AF_PROFILE_DAYS = 6

import threading as _thr
_AF_LOCK, _AF_LAST = _thr.Lock(), [0.0]

def _af(path, **params):
    """API-Football 호출 — 분당 300회 한도라 전체(여러 스레드 합쳐) 0.21초 간격, 429·한도 오류면 쉬었다 최대 3번"""
    for attempt in range(3):
        with _AF_LOCK:
            wait = _AF_LAST[0] + 0.21 - time.time()
            if wait > 0:
                time.sleep(wait)
            _AF_LAST[0] = time.time()
        try:
            r = requests.get(API_FOOTBALL_URL + path, headers=API_FOOTBALL_HEADERS, params=params, timeout=40)
            d = r.json()
        except Exception as e:
            print(f"    ⚠️ {path} {params}: {e}"); time.sleep(5 * (attempt + 1)); continue
        if r.status_code == 429 or (d.get("errors") and re.search(r"rate|limit|requests", str(d["errors"]), re.I)):
            time.sleep(15 * (attempt + 1)); continue
        pass   # 간격은 위 전체 제한(0.21초)이 맞춤
        if d.get("errors"):
            print(f"    ⚠️ {path} {params}: {d['errors']}")
        return d
    return {}

def _af_tokens(n):
    return [t for t in _norm_club(n).split() if t]

def _af_name_sim(a, b):
    """이름 비슷함(0~1): 단어가 같거나 앞 4글자가 같으면(“wolves”↔“wolverhampton”) 맞는 단어로"""
    ta, tb = _af_tokens(a), _af_tokens(b)
    if not ta or not tb:
        return 0
    hit = sum(1 for x in ta if any(x == y or (len(x) >= 4 and len(y) >= 4 and x[:4] == y[:4]) for y in tb))
    return hit / max(len(ta), len(tb))

def af_sync(season=None):
    import json
    from collections import Counter, defaultdict
    if not API_FOOTBALL_KEY:
        print("  ⚠️ API_FOOTBALL_KEY가 없어 API-Football 수집 건너뜀"); return
    season = season or max(MATCH_SEASONS)
    rd = lambda n, d: json.load(open(f"{MODEL_DIR}/{n}", encoding="utf-8")) if os.path.exists(f"{MODEL_DIR}/{n}") else d
    def wr(n, obj, indent=None):
        with open(f"{MODEL_DIR}/{n}", "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent, sort_keys=True, separators=None if indent else (",", ":"))
    sched = rd("schedule.json", {})
    teammap = rd("af_team_map.json", {})
    print(f"\n[API-Football 수집] {season}-{(season + 1) % 100:02d} 시즌")
    # 1) 대회별 경기 목록
    fx_all = {}
    for code, lid in AF_COMPS.items():
        fx_all[code] = _af("/fixtures", league=lid, season=season).get("response") or []
        print(f"  {code}: {len(fx_all[code])}경기")
    # 2) 팀 이름 맞추기 — 같은 날 우리 일정에서 홈·원정 이름이 가장 비슷한 경기(킥오프 시각이 같으면 가산)
    votes = defaultdict(Counter)
    for code, fxs in fx_all.items():
        by_day = defaultdict(list)
        for m in sched.get(code, []):
            by_day[m["date"][:10]].append(m)
        for f in fxs:
            ko = pd.Timestamp(f["fixture"]["date"]).tz_convert("UTC")
            h, a = f["teams"]["home"], f["teams"]["away"]
            best, bs = None, 0
            for m in by_day.get(ko.strftime("%Y-%m-%d"), []):
                sc = _af_name_sim(h["name"], m["home_team"]) + _af_name_sim(a["name"], m["away_team"])
                if abs((pd.Timestamp(m["date"]) - ko).total_seconds()) < 600:
                    sc += 0.6
                if sc > bs:
                    best, bs = m, sc
            if best and bs >= 1.2:
                votes[h["id"]][best["home_team"]] += 1
                votes[a["id"]][best["away_team"]] += 1
    changed = []
    for tid, c in votes.items():
        name, n = c.most_common(1)[0]
        if n >= 2 and teammap.get(str(tid)) != name:
            changed.append((teammap.get(str(tid)), name)); teammap[str(tid)] = name
    clash = [n for n, k in Counter(teammap.values()).items() if k > 1]
    print(f"  팀 매칭: {len(teammap)}팀{' · 새로/바뀜 ' + str(len(changed)) if changed else ''}{' · ⚠️ 한 이름에 ID 여러 개: ' + ', '.join(clash) if clash else ''}")
    wr("af_team_map.json", teammap, indent=1)
    # 3) 경기 목록(우리 이름·우리 일정 날짜)
    os.makedirs(MD_DIR, exist_ok=True)
    ours_by_pair = defaultdict(list)
    for code, ms in sched.items():
        for m in ms:
            ours_by_pair[(code, m["home_team"], m["away_team"])].append(m)
    index, unmapped = [], set()
    for code, fxs in fx_all.items():
        for f in fxs:
            h, a = teammap.get(str(f["teams"]["home"]["id"])), teammap.get(str(f["teams"]["away"]["id"]))
            if not h or not a:
                unmapped.update(x["name"] for x, ok in ((f["teams"]["home"], h), (f["teams"]["away"], a)) if not ok); continue
            ko = pd.Timestamp(f["fixture"]["date"]).tz_convert("UTC")
            cands = ours_by_pair.get((code, h, a), [])
            om = min(cands, key=lambda m: abs((pd.Timestamp(m["date"]) - ko).total_seconds())) if cands else None
            date = om["date"] if om and abs((pd.Timestamp(om["date"]) - ko).days) <= 3 else ko.strftime("%Y-%m-%dT%H:%M:%SZ")
            e = {"id": f["fixture"]["id"], "league": code, "home_team": h, "away_team": a, "date": date, "kickoff": ko.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "status": f["fixture"]["status"]["short"], "home_id": f["teams"]["home"]["id"], "away_id": f["teams"]["away"]["id"],
                 "home_goals": f["goals"]["home"], "away_goals": f["goals"]["away"], "file": md_file(h, a, date),
                 "round": int(rm.group(1)) if (rm := re.search(r"^(?:Regular Season|League Stage) - (\d+)$", f["league"].get("round") or "")) else None}
            index.append(e)
    if unmapped:
        print(f"  ⚠️ 우리 이름을 못 찾은 팀(그 경기는 건너뜀): {', '.join(sorted(unmapped))}")
    # 4) 끝난 경기 상세 — 아직 파일이 없는 것만, 한 번에 20경기씩
    need = [e for e in index if e["status"] in AF_DONE and not os.path.exists(f"{MD_DIR}/{e['file']}")]
    by_id = {e["id"]: e for e in need}
    got = 0
    for i in range(0, len(need), 20):
        ids = [e["id"] for e in need[i:i + 20]]
        for fx in _af("/fixtures", ids="-".join(map(str, ids))).get("response") or []:
            e = by_id.get(fx["fixture"]["id"])
            if not e:
                continue
            det = af_match_detail(fx)
            if not det["lineups"] and not det["events"]:
                continue   # 아직 상세가 안 올라온 경기 — 다음 실행 때
            det.update({k: e[k] for k in ("league", "home_team", "away_team", "date", "home_goals", "away_goals")})
            md_write(f"{MD_DIR}/{e['file']}", det)
            got += 1
    for e in index:
        e["has_detail"] = os.path.exists(f"{MD_DIR}/{e['file']}")
    print(f"  경기 상세: 새로 {got}경기 · 전체 {sum(e['has_detail'] for e in index)}경기")
    wr("af_fixtures.json", {"season": season, "updated": pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ'), "fixtures": index})
    # 5) 앞으로 4일 경기 결장·부상자(대회별 한 번씩)
    now = pd.Timestamp.now(tz="UTC")
    soon = [e for e in index if e["status"] in ("NS", "TBD") and now <= pd.Timestamp(e["kickoff"]) <= now + pd.Timedelta(days=4)]
    previews = {}
    for code in sorted({e["league"] for e in soon}):
        inj = _af("/injuries", league=AF_COMPS[code], season=season).get("response") or []
        by_fx = defaultdict(list)
        for x in inj:
            by_fx[(x.get("fixture") or {}).get("id")].append(x)
        for e in (e for e in soon if e["league"] == code):
            rows = by_fx.get(e["id"], [])
            previews[f"{e['home_team']}|{e['away_team']}|{e['date'][:10]}"] = {"fixture_id": e["id"],
                "injuries": {"home": af_injury_rows(rows, e["home_id"]), "away": af_injury_rows(rows, e["away_id"])}}
    wr("match_previews.json", previews, indent=1)
    print(f"  결장·부상자: {len(previews)}경기")
    # 팀별 "가장 최근 경기" 결장자(경기별 명단은 킥오프 1~2일 전에야 올라와서 그 전엔 이걸로 지금 빠져 있는 선수를 보여줌) — 대회마다 1번
    ti = {}
    for code, lid in AF_COMPS.items():
        rows = _af("/injuries", league=lid, season=season).get("response") or []
        last = {}
        for x in rows:
            nm = teammap.get(str((x.get("team") or {}).get("id")))
            dt = ((x.get("fixture") or {}).get("date") or "")[:10]
            if nm and dt and dt <= now.strftime("%Y-%m-%d") and dt >= last.get(nm, ""):
                last[nm] = dt
        for x in rows:
            nm = teammap.get(str((x.get("team") or {}).get("id")))
            dt = ((x.get("fixture") or {}).get("date") or "")[:10]
            if not nm or dt != last.get(nm) or (ti.get(nm) and ti[nm]["date"] > dt):
                continue
            ent = ti.setdefault(nm, {"date": dt, "players": []})
            if ent["date"] != dt:
                ti[nm] = ent = {"date": dt, "players": []}
            pl = x.get("player") or {}
            if pl.get("id") and all(p["id"] != pl["id"] for p in ent["players"]):
                ent["players"].append({"id": pl["id"], "name": pl.get("name"), "photo": pl.get("photo"),
                                       "status": "doubtful" if pl.get("type") == "Questionable" else "out", "reason": pl.get("reason")})
    wr("team_injuries.json", ti, indent=1)
    print(f"  팀별 최근 경기 결장자: {len(ti)}팀 · {sum(len(v['players']) for v in ti.values())}명")
    # 6) 선수 프로필(5대 리그 팀, 6일 넘은 팀만)
    players = rd("af_players.json", {})
    league_team = {}
    for e in index:
        if e["league"] != "CL":
            league_team[e["home_team"]] = (e["league"], e["home_id"]); league_team[e["away_team"]] = (e["league"], e["away_id"])
    today = now.strftime("%Y-%m-%d")
    stale = [t for t in league_team if (pd.Timestamp(today) - pd.Timestamp((players.get(t) or {}).get("updated", "2000-01-01"))).days >= AF_PROFILE_DAYS]
    for t in stale:
        code, tid = league_team[t]
        rows, page, pages = {}, 1, 1
        while page <= pages and page <= 6:
            d = _af("/players", team=tid, season=season, page=page)
            pages = (d.get("paging") or {}).get("total") or 1
            for p in d.get("response") or []:
                r = af_profile_row(p)
                if r["id"]:
                    rows[str(r["id"])] = r
            page += 1
        if rows:
            players[t] = {"af_id": tid, "league": code, "season": season, "updated": today, "players": rows}
    for t in [t for t in players if t not in league_team]:   # 강등 등으로 빠진 팀
        players.pop(t)
    wr("af_players.json", players)
    print(f"  선수 프로필: {len(stale)}팀 갱신 · 전체 {len(players)}팀 {sum(len(v['players']) for v in players.values())}명")

# ── 선수 경력·트로피 미리 받기(2026-10-07) ──
# 서버가 선수 카드를 열 때마다 API-Football을 15~20번 불러 처음 여는 선수가 5~7초 걸렸음(라이브 측정) → 새벽에 5대 리그 선수의
# 경력 시즌 목록·트로피·부상 이력·이적·시즌별 기록(/players?id=&season=)을 받아 fotdata_model/af_profiles/<ID>.json.gz로 저장.
# 지난 시즌 기록은 한 번 받으면 다시 안 받고, 이번 시즌·트로피·부상·이적만 AF_PROFILE_REFRESH_DAYS마다. 하루 요청 한도를 나눠 쓰므로
# 그날 남은 요청(/status) 안에서만(서버 몫 600회는 남김) — 처음엔 며칠에 걸쳐 전원 채움(출전 시간 많은 선수부터)
AF_PROFILE_DIR = f"{MODEL_DIR}/af_profiles"
AF_PROFILE_REFRESH_DAYS = 14
AF_PROFILE_BUDGET = None   # None = 그날 남은 요청 − 600 (af_profiles_sync)

def _af_remaining():
    try:
        d = requests.get(API_FOOTBALL_URL + "/status", headers=API_FOOTBALL_HEADERS, timeout=20).json()["response"]["requests"]
        return d["limit_day"] - d["current"]
    except Exception:
        return 0

def fetch_comp_winners():
    """컵대회 결승 승자·준우승, 5대 리그 최종 1·2위(API-Football 팀 ID) → comp_winners.json {대회 ID: {시즌: {w, r}}}.
    이미 받은 시즌은 다시 안 받음(결승 전이면 다음 날 다시)"""
    import json
    if not API_FOOTBALL_KEY:
        return
    path = f"{MODEL_DIR}/comp_winners.json"
    cw = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    cur = max(MATCH_SEASONS)
    got = 0
    for cid in AF_CUP_COMPS + AF_LEAGUE_COMPS:
        for yr in (cur - 2, cur - 1, cur):
            if str(yr) in cw.get(str(cid), {}):
                continue
            if cid in AF_LEAGUE_COMPS:
                if yr >= cur:
                    continue   # 진행 중인 시즌 리그는 우승이 아직 없음
                st = ((_af("/standings", league=cid, season=yr).get("response") or [{}])[0].get("league") or {}).get("standings") or []
                t = st[0] if st else []
                if len(t) >= 2:
                    cw.setdefault(str(cid), {})[str(yr)] = {"w": t[0]["team"]["id"], "r": t[1]["team"]["id"]}; got += 1
                continue
            for f in _af("/fixtures", league=cid, season=yr, round="Final").get("response") or []:
                if f["fixture"]["status"]["short"] not in AF_DONE:
                    continue
                h, a = f["teams"]["home"], f["teams"]["away"]
                if h.get("winner") is None and a.get("winner") is None:
                    continue
                w, r = (h, a) if h.get("winner") else (a, h)
                cw.setdefault(str(cid), {})[str(yr)] = {"w": w["id"], "r": r["id"]}; got += 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cw, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"  대회 우승 팀: 새로 {got}개")

# ── 감독(2026-10-07) ──
# 지금 감독 = 그 팀 가장 최근 경기 라인업의 감독(가장 확실). 사진·국적·생년월일·부임일은 /coachs?team= 의 그 팀 경력에서 —
# 원본에 오래된 항목이 남아 있어(아스널에 벵거가 종료일 없이 남아 있음) 라인업 이름과 맞는 사람을 고르고, 없으면 종료일 없는 것 중 최근 부임.
# 생년월일·국적이 비어 있는 감독(아르테타·과르디올라 등)은 위키데이터(구단 P286 현재 감독)로 채움. 7일마다 + 라인업 감독이 바뀌면 바로.
AF_COACH_DAYS = 7

def _person_key(n):
    import unicodedata
    n = unicodedata.normalize("NFKD", (n or "").replace("ß", "ss")).encode("ascii", "ignore").decode().lower()
    return [t for t in re.split(r"[^a-z]+", n) if t]

def _same_person(a, b):
    """'Mikel Arteta' ↔ 'M. Arteta' ↔ 'Arteta' — 성(마지막 단어)이 같고, 둘 다 이름이 있으면 첫 글자도 같음"""
    ta, tb = _person_key(a), _person_key(b)
    if not ta or not tb:
        return False
    if ta[-1] != tb[-1]:   # 라인업 이름이 "성 이름"으로 뒤집혀 오는 팀이 있음("Enrique Luis", "Piero Gasperini Gian") → 단어 두 개 이상 겹치면 같은 사람
        return len(set(ta) & set(tb)) >= 2
    return len(ta) == 1 or len(tb) == 1 or ta[0][0] == tb[0][0]

def _lineup_coaches():
    """팀 → (날짜, 라인업 감독 이름) — 가장 최근 경기"""
    import json
    fx = json.load(open(f"{MODEL_DIR}/af_fixtures.json", encoding="utf-8")).get("fixtures", [])
    out = {}
    for f in sorted([f for f in fx if f.get("has_detail") and f.get("file")], key=lambda f: f["date"], reverse=True):
        need = [sd for sd in ("home", "away") if f[f"{sd}_team"] not in out]
        if not need:
            continue
        try:
            d = md_read(f"{MD_DIR}/{f['file']}")
        except Exception:
            continue
        for sd in need:
            c = ((d.get("lineups") or {}).get(sd) or {}).get("coach")
            if c:
                out[f[f"{sd}_team"]] = (f["date"][:10], c)
    return out

def _wd_coach(qid):
    """위키데이터: 구단의 현재 감독 → {name_en, name_ko, birth, country}(국적은 영어 이름 — API-Football 나라 이름과 같은 표기)"""
    club = _wd_entities([qid], "claims").get(qid) or {}
    c = _wd_current(club.get("claims", {}), "P286")
    if not c:
        return None
    since = None   # 부임일 = 그 감독 항목의 시작일(P580)
    for cl0 in club.get("claims", {}).get("P286", []):
        if (cl0.get("mainsnak", {}).get("datavalue", {}).get("value") or {}).get("id") == c["id"] and "P582" not in cl0.get("qualifiers", {}):
            t = ((cl0.get("qualifiers", {}).get("P580") or [{}])[0].get("datavalue", {}).get("value") or {}).get("time", "")
            since = since or (t[1:11].replace("-00", "-01") if t else None)
    ent = _wd_entities([c["id"]], "claims|labels").get(c["id"]) or {}
    cl, lb = ent.get("claims", {}), ent.get("labels", {})
    val = lambda p: ((cl.get(p) or [{}])[0].get("mainsnak", {}).get("datavalue", {}) or {}).get("value")
    birth = (val("P569") or {}).get("time", "")[1:11] or None
    nat = val("P1532") or val("P27")   # 대표팀 국적(P1532) 우선 — 이중국적이면 P27 첫 값이 엉뚱할 수 있음
    country = None
    if nat and nat.get("id"):
        country = ((_wd_entities([nat["id"]], "labels").get(nat["id"]) or {}).get("labels", {}).get("en") or {}).get("value")
    return {"name_en": (lb.get("en") or {}).get("value"), "name_ko": (lb.get("ko") or {}).get("value"),
            "birth": birth, "country": country, "start": since}

def fetch_coaches(force=False):
    import json
    if not API_FOOTBALL_KEY:
        return
    path = f"{MODEL_DIR}/af_coaches.json"
    old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    teams = json.load(open(f"{MODEL_DIR}/af_players.json", encoding="utf-8"))
    wiki = json.load(open(f"{MODEL_DIR}/team_wiki.json", encoding="utf-8")) if os.path.exists(f"{MODEL_DIR}/team_wiki.json") else {}
    lineup = _lineup_coaches()
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    out, done = {}, 0
    for team, info in teams.items():
        prev, lc = old.get(team), (lineup.get(team) or (None, None))[1]
        fresh = prev and (pd.Timestamp(today) - pd.Timestamp(prev.get("updated", "2000-01-01"))).days < AF_COACH_DAYS
        if not force and fresh and (not lc or _same_person(lc, prev.get("name") or "")):
            out[team] = prev; continue
        tid = info["af_id"]
        cands = []
        for c in (_af("/coachs", team=tid).get("response") or []):
            for car in c.get("career") or []:
                if (car.get("team") or {}).get("id") == tid:
                    cands.append((c, car))
        pick = None
        if lc:
            m = [x for x in cands if _same_person(lc, x[0].get("name") or "") or _same_person(lc, f"{x[0].get('firstname') or ''} {x[0].get('lastname') or ''}")]
            pick = max(m, key=lambda x: x[1].get("start") or "", default=None)
        if not pick:
            pick = max([x for x in cands if not x[1].get("end")], key=lambda x: x[1].get("start") or "", default=None)
            if pick and lc:   # 라인업 감독이 목록에 없음(신임·대행) → 이름만
                pick = None
        c, car = pick if pick else ({}, {})
        flip = lc and c.get("name") and _person_key(lc)[-1:] != _person_key(c["name"])[-1:]   # 뒤집힌 라인업 이름이면 감독 목록 이름으로
        full = " ".join(x for x in (c.get("firstname"), c.get("lastname")) if x) or (c.get("name") if flip else None)
        row = {"id": c.get("id"), "name": c["name"] if flip else (lc or c.get("name")), "full_name": full or lc or c.get("name"),
               "photo": c.get("photo"), "nationality": c.get("nationality"),
               "birth_date": (c.get("birth") or {}).get("date"), "start": car.get("start"), "updated": today}
        qid = (wiki.get(team) or {}).get("qid")
        if qid:
            try:
                wd = _wd_coach(qid)
            except Exception as e:
                wd = None; print(f"    ⚠️ 위키데이터 감독 {team}: {e}")
            if wd and (_same_person(row["name"] or "", wd.get("name_en") or "") or _same_person(row["full_name"] or "", wd.get("name_en") or "")):
                ko = wd.get("name_ko")   # 한국어 라벨이 본명 전체("루이스 엔리케 마르티네스 가르시아")면 안 씀
                row["name_ko"] = ko if ko and len(ko.split()) <= 3 else None
                row["birth_date"] = row["birth_date"] or wd.get("birth")
                row["nationality"] = row["nationality"] or wd.get("country")
                row["start"] = row["start"] or wd.get("start")
                if wd.get("name_en"):   # 위키데이터 영어 이름 = 흔히 부르는 이름("José Mourinho" — API-Football은 본명 전체)
                    row["full_name"] = wd["name_en"]
        if row["name"]:
            out[team] = row; done += 1
        elif prev:
            out[team] = prev
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"  감독: {done}팀 갱신 · 전체 {len(out)}팀 · 사진 {sum(1 for v in out.values() if v.get('photo'))} · 생년월일 {sum(1 for v in out.values() if v.get('birth_date'))}")

def _af_profile_minutes():
    """이번 시즌 리그 출전 시간(경기 상세 합산) — 출전 많은 선수부터 받으려고"""
    import glob
    from collections import Counter
    mins = Counter()
    for p in glob.glob(f"{MD_DIR}/*.json.gz"):
        try:
            d = md_read(p)
        except Exception:
            continue
        for sd in ("home", "away"):
            for x in (d.get("pstats") or {}).get(sd) or []:
                mins[x["id"]] += x.get("min") or 0
    return mins

def af_profiles_sync(budget=None):
    import json
    from concurrent.futures import ThreadPoolExecutor
    if not API_FOOTBALL_KEY:
        print("  ⚠️ API_FOOTBALL_KEY가 없어 건너뜀"); return
    left = _af_remaining()
    # 새벽 실행: 그날(UTC) 남은 요청에서 600회만 남기고 전부 — 아직 못 받은 선수가 많으면(처음 며칠) 하루 400명 안팎씩 채움.
    # 남긴 600회는 서버가 요청 때 쓰는 몫(확정 라인업·경력 미수집 선수) — 한도는 00:00 UTC(KST 09:00)에 초기화
    budget = min(budget, max(0, left - 600)) if budget else max(0, left - 600)
    print(f"\n[선수 경력·트로피] 오늘 남은 요청 {left} · 이번에 쓸 수 있는 {budget}")
    if budget <= 0:
        return
    os.makedirs(AF_PROFILE_DIR, exist_ok=True)
    cur = max(MATCH_SEASONS)
    players = json.load(open(f"{MODEL_DIR}/af_players.json", encoding="utf-8"))
    ids = {int(pid) for v in players.values() for pid in (v.get("players") or {})}
    mins = _af_profile_minutes()
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    def load(pid):
        p = f"{AF_PROFILE_DIR}/{pid}.json.gz"
        return md_read(p) if os.path.exists(p) else None
    todo = []
    for pid in ids:
        c = load(pid)
        age = (pd.Timestamp(today) - pd.Timestamp(c["updated"])).days if c else 9999
        if age >= AF_PROFILE_REFRESH_DAYS:
            todo.append((c is not None, -mins.get(pid, 0), -age, pid))
    todo.sort()   # 아직 없는 선수 → 출전 시간 많은 순 / 그다음 오래된 순
    used, done = [0], [0]
    def one(pid):
        old = load(pid) or {}
        teams = (_af("/players/teams", player=pid).get("response"))
        if teams is None:
            return 1
        for t in teams:   # 시즌이 가끔 문자열("2023")로 옴
            t["seasons"] = [int(y) for y in (t.get("seasons") or []) if str(y).isdigit()]
        years = sorted({y for t in teams for y in t["seasons"] if y >= 2010}, reverse=True)[:16]
        stats = {int(k): v for k, v in (old.get("stats") or {}).items()}
        need = [y for y in years if y >= cur - 1 or y not in stats]   # 지난 시즌은 한 번만, 이번·지난 시즌은 다시
        for y in need:
            stats[y] = af_compact_player_season(((_af("/players", id=pid, season=y).get("response")) or [None])[0])
        rec = {"id": pid, "updated": today,
               "teams": [{"team": {k: (t.get("team") or {}).get(k) for k in ("id", "name", "logo")}, "seasons": t.get("seasons")} for t in teams],
               "trophies": _af("/trophies", player=pid).get("response") or [],
               "sidelined": [{k: x.get(k) for k in ("type", "start", "end")} for x in (_af("/sidelined", player=pid).get("response") or [])],
               "transfers": af_compact_transfers(_af("/transfers", player=pid).get("response")),
               "stats": {str(y): v for y, v in stats.items() if y in years}}
        md_write(f"{AF_PROFILE_DIR}/{pid}.json.gz", rec)
        return 5 + len(need)
    est = lambda new: 18 if not new else 7   # 처음 받는 선수 ~18회, 갱신 ~7회
    batch = []
    for has, _, _, pid in todo:
        cost = est(not has)
        if used[0] + cost > budget:
            break
        used[0] += cost
        batch.append(pid)
    def safe(pid):
        try:
            return one(pid)
        except Exception as e:
            print(f"    ⚠️ 선수 {pid}: {type(e).__name__}: {e}")
            return None
    with ThreadPoolExecutor(8) as ex:   # 요청 하나가 0.5~1초라 4개로는 초당 2.4회뿐이었음(한도는 전체 0.21초 간격이 지킴)
        for n in ex.map(safe, batch):
            done[0] += n is not None
    print(f"  ✅ {done[0]}명 저장(대기 {len(todo) - len(batch)}명) · 전체 {len(os.listdir(AF_PROFILE_DIR))}명")

# ── 과거 시즌 구단 기록(2026-10-06, API-Football Pro) — 2010-11 ~ 2022-23 ──
# 순위표·일정·팀 통계·시즌 고르기에만 씀(예측 모델 학습 데이터 all_matches.csv와는 따로 — 섞지 않음). 끝난 시즌이라 한 번만:
#   python update_data.py --history-seasons   (대회 6개 × 13시즌 경기 목록 + 공식 순위표, 약 170회)
#   history_matches.csv   경기(우리 이름 — 지금 우리 목록에 없는 옛 팀(위건 등)은 API-Football 이름 그대로)
#   history_standings.json 공식 순위표: 최종 순위·유럽대항전/강등 구역(순위표 description)·승점 감점(공식 승점 − 경기로 센 승점), 챔스는 조 편성
#   history_logos.json    우리 로고 목록에 없는 옛 팀 로고(API-Football)
HISTORY_SEASONS = list(range(2010, 2023))
AF_ROUND_STAGE = {"Group Stage": "GROUP_STAGE", "8th Finals": "LAST_16", "Round of 16": "LAST_16", "Quarter-finals": "QUARTER_FINALS",
                  "Semi-finals": "SEMI_FINALS", "Final": "FINAL"}

def _af_desc_zone(desc):
    d = (desc or "").lower()
    if not d:
        return None
    if "relegation" in d:
        return "rel-po" if ("play" in d or "(relegation)" in d) else "rel"
    if "champions league" in d:
        return "cl-q" if ("qualif" in d or "play-off" in d) else "cl"
    if "europa league" in d:
        return "el"
    if "conference" in d:
        return "ecl"
    return None

def _af_ucl_season(fxs, groups, nm):
    """옛 챔스 한 시즌(API-Football 경기) → ucl_tournament.json과 같은 모양: 토너먼트 라운드별 대진(두 경기 합산·승부차기·진출 팀) + GROUPS(조별 경기).
    진출 팀은 다음 라운드에 나온 팀(원정 다득점 규칙 시절도 그대로 맞음), 결승은 점수 → 승부차기"""
    order = ["LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL"]
    stages, ties, team_group = {s: [] for s in ["PLAYOFFS"] + order}, {}, {t: g for g, ts in groups.items() for t in ts}
    gm = {}
    for f in sorted(fxs, key=lambda f: f["fixture"]["date"]):
        if f["fixture"]["status"]["short"] not in AF_DONE or f["goals"]["home"] is None:
            continue
        base = (f["league"].get("round") or "").split(" - ")[0].strip()
        if not (base.startswith("Group") or AF_ROUND_STAGE.get(base) in order):
            continue   # 예선(이름 매기기 전에 거름 — 예선 팀이 옛 팀 목록에 섞이지 않게)
        h, a = nm(f["teams"]["home"]), nm(f["teams"]["away"])
        hg, ag = int(f["goals"]["home"]), int(f["goals"]["away"])
        date = pd.Timestamp(f["fixture"]["date"]).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        if base.startswith("Group"):
            g = team_group.get(h)
            if g:
                gm.setdefault(g, []).append({"home_team": h, "away_team": a, "home_goals": hg, "away_goals": ag, "date": date, "status": "FINISHED"})
            continue
        st = AF_ROUND_STAGE.get(base)
        if st not in order:
            continue
        pen = (f.get("score") or {}).get("penalty") or {}
        key = (st,) + tuple(sorted([h, a]))
        v = ties.setdefault(key, {"stage": st, "team1": h, "team2": a, "team1_goals": 0, "team2_goals": 0, "legs": [], "pens": None,
                                  "team1_logo": "", "team2_logo": "", "status": "FINISHED"})
        leg = {"home_team": h, "away_team": a, "home_goals": hg, "away_goals": ag, "date": date}
        if v["team1"] == h:
            v["team1_goals"] += hg; v["team2_goals"] += ag
        else:
            v["team1_goals"] += ag; v["team2_goals"] += hg
        if pen.get("home") is not None:
            leg["penalties"] = [pen["home"], pen["away"]]
            v["pens"] = [pen["home"], pen["away"]] if v["team1"] == h else [pen["away"], pen["home"]]
        v["legs"].append(leg)
    in_stage = {s: {t for k, v in ties.items() if v["stage"] == s for t in (v["team1"], v["team2"])} for s in order}
    for v in ties.values():
        i = order.index(v["stage"])
        nxt = in_stage[order[i + 1]] if i + 1 < len(order) else set()
        if v["team1"] in nxt or v["team2"] in nxt:
            v["winner"] = v["team1"] if v["team1"] in nxt else v["team2"]
        elif v["team1_goals"] != v["team2_goals"]:
            v["winner"] = v["team1"] if v["team1_goals"] > v["team2_goals"] else v["team2"]
        elif v["pens"]:
            v["winner"] = v["team1"] if v["pens"][0] > v["pens"][1] else v["team2"]
        else:
            v["winner"] = None
        stages[v.pop("stage")].append(v)
    out = reconstruct_bracket_order(stages)
    out["GROUPS"] = {g: ms for g, ms in sorted(gm.items())}
    return out

def fetch_history_seasons(seasons=None):
    import json
    from collections import Counter, defaultdict
    if not API_FOOTBALL_KEY:
        print("  ⚠️ API_FOOTBALL_KEY가 없어 건너뜀"); return
    seasons = seasons or HISTORY_SEASONS
    teammap = json.load(open(f"{MODEL_DIR}/af_team_map.json", encoding="utf-8")) if os.path.exists(f"{MODEL_DIR}/af_team_map.json") else {}
    df = pd.read_csv(f"{MODEL_DIR}/all_matches.csv")
    print(f"\n[과거 시즌 구단 기록] {seasons[0]}-{seasons[-1]}")
    # 1) 23-24~25-26 시즌으로 팀 이름 맞추기를 넓힘(강등된 팀 등 — 같은 날·같은 점수·이름 비슷함으로 투표)
    votes = defaultdict(Counter)
    for yr in [y for y in MATCH_SEASONS if y < max(MATCH_SEASONS)]:
        for code, lid in AF_COMPS.items():
            ours = df[(df.league == code) & (df.season == yr)]
            by_day = defaultdict(list)
            for r in ours.itertuples():
                by_day[str(r.date)[:10]].append(r)
            for f in _af("/fixtures", league=lid, season=yr).get("response") or []:
                if f["goals"]["home"] is None:
                    continue
                day = pd.Timestamp(f["fixture"]["date"]).tz_convert("UTC").strftime("%Y-%m-%d")
                h, a = f["teams"]["home"], f["teams"]["away"]
                best, bs = None, 0
                for r in by_day.get(day, []):
                    sc = _af_name_sim(h["name"], r.home_team) + _af_name_sim(a["name"], r.away_team)
                    if int(r.home_goals) == f["goals"]["home"] and int(r.away_goals) == f["goals"]["away"]:
                        sc += 0.6
                    if sc > bs:
                        best, bs = r, sc
                if best is not None and bs >= 1.2:
                    votes[h["id"]][best.home_team] += 1; votes[a["id"]][best.away_team] += 1
    added = 0
    for tid, c in votes.items():
        name, n = c.most_common(1)[0]
        if n >= 2 and str(tid) not in teammap:
            teammap[str(tid)] = name; added += 1
    with open(f"{MODEL_DIR}/af_team_map.json", "w", encoding="utf-8") as f:
        json.dump(teammap, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"  팀 매칭 넓힘: +{added}팀 (전체 {len(teammap)})")
    # 2) 과거 시즌 경기 + 공식 순위표
    rows, stand, hlogos, ucl_hist = [], {}, {}, {}
    used_names = defaultdict(set)   # 우리 이름 → API-Football ID들(한 이름에 여러 팀이 붙으면 경고)
    idname = {}   # 우리 목록 밖 옛 팀: 처음 본 이름으로 고정(경기 목록 "Bastia" ↔ 순위표 다른 표기처럼 같은 팀이 응답마다 달랐음)
    def nm(t):
        n = teammap.get(str(t["id"]))
        if not n:
            n = idname.setdefault(t["id"], t["name"])
            hlogos[n] = hlogos.get(n) or t.get("logo")
        used_names[n].add(t["id"])
        return n
    for yr in seasons:
        for code, lid in AF_COMPS.items():
            fxs = _af("/fixtures", league=lid, season=yr).get("response") or []
            if not fxs:
                continue
            n0 = len(rows)
            for f in fxs:
                st = f["fixture"]["status"]["short"]
                if st not in AF_DONE or f["goals"]["home"] is None:
                    continue
                rnd = (f["league"].get("round") or "")
                base = rnd.split(" - ")[0].strip()
                if code == "CL":
                    stage = "GROUP_STAGE" if base.startswith("Group") else AF_ROUND_STAGE.get(base)   # 시즌마다 "Group Stage - 1"·"Group A - 1"
                    if not stage:
                        continue   # 예선·본선 전 플레이오프는 뺌
                else:
                    if not base.startswith("Regular Season"):
                        continue   # 승강 플레이오프(분데스 16위·리그앙 18위·세리에 동률 결정전)는 리그 순위에 안 들어감
                    stage = "REGULAR_SEASON"
                md = re.search(r" - (\d+)$", rnd)
                hg, ag = int(f["goals"]["home"]), int(f["goals"]["away"])
                rows.append({"date": pd.Timestamp(f["fixture"]["date"]).tz_convert("UTC").strftime("%Y-%m-%d"), "league": code,
                             "home_team": nm(f["teams"]["home"]), "away_team": nm(f["teams"]["away"]), "home_goals": hg, "away_goals": ag,
                             "matchday": int(md.group(1)) if md else None, "result": "H" if hg > ag else "A" if hg < ag else "D",
                             "season": yr, "stage": stage, "kickoff": pd.Timestamp(f["fixture"]["date"]).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")})
            sres = _af("/standings", league=lid, season=yr).get("response") or []
            tables = (sres[0]["league"]["standings"] if sres else [])
            if code == "CL":
                groups = {}
                for t in tables:
                    m = re.search(r"Group ([A-H])\s*$", (t[0].get("group") or "") if t else "")   # "Group A" / "Uefa Champions League: Group A"
                    g = m.group(1) if m else ""
                    if g:
                        groups[g] = [nm(r["team"]) for r in t]
                ucl_hist[str(yr)] = _af_ucl_season(fxs, groups, nm)
            elif tables:
                sub = [r for r in rows[n0:]]
                pts = Counter()
                for r in sub:
                    pts[r["home_team"]] += 3 if r["result"] == "H" else 1 if r["result"] == "D" else 0
                    pts[r["away_team"]] += 3 if r["result"] == "A" else 1 if r["result"] == "D" else 0
                teams, adjust = {}, {}
                for r in tables[0]:
                    t = nm(r["team"])
                    teams[t] = {"pos": r["rank"], "zone": _af_desc_zone(r.get("description"))}
                    if r["points"] != pts.get(t, 0):
                        adjust[t] = r["points"] - pts.get(t, 0)
                stand.setdefault(code, {})[str(yr)] = {"teams": teams, "adjust": adjust, "source": "API-Football"}
            print(f"  {code} {yr}-{(yr + 1) % 100:02d}: {len(rows) - n0}경기{' · 감점/조정 ' + str(stand.get(code, {}).get(str(yr), {}).get('adjust')) if stand.get(code, {}).get(str(yr), {}).get('adjust') else ''}")
    clash = {n: ids for n, ids in used_names.items() if len(ids) > 1}
    if clash:
        print(f"  ⚠️ 한 이름에 여러 팀: {clash}")
    out = pd.DataFrame(rows).sort_values(["date", "league", "home_team"])
    out.to_csv(f"{MODEL_DIR}/history_matches.csv", index=False)
    with open(f"{MODEL_DIR}/history_standings.json", "w", encoding="utf-8") as f:
        json.dump(stand, f, ensure_ascii=False, indent=1, sort_keys=True)
    with open(f"{MODEL_DIR}/history_ucl.json", "w", encoding="utf-8") as f:
        json.dump({"seasons": ucl_hist}, f, ensure_ascii=False, indent=1, sort_keys=True)
    with open(f"{MODEL_DIR}/history_logos.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in hlogos.items() if v}, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"  ✅ {len(out)}경기 · 옛 팀(우리 목록 밖) {len(hlogos)}팀")

# ── 과거 시즌 선수 기록(2026-10-06, API-Football Pro) — 14-15 ~ 25-26 ──
# 경기별 선수 기록(평점·골·도움·출전 시간 + 17-18부터 포메이션·칸)을 받아 리그·시즌마다 합산만 저장(경기 원본은 무거워서 안 남김):
#   fotdata_model/af_seasons/<리그>_<연도>.json.gz  {"players": [(팀, 선수)별 시즌 기록 + 90분당 백분위], "teams", "matches"}
#   fotdata_model/af_seasons/index.json             {"players": {선수 ID: [[리그, 연도, 팀], …]}, "teams": {팀: {연도: [리그…]}}}
# 선수 탭(과거 시즌 순위·전체 목록)·리그 시즌 베스트 11·구단 시즌 베스트 11·선수 카드 시즌 기록이 씀. 이번 시즌은 match_details/에서 바로 합산.
# 끝난 시즌이라 한 번만: python update_data.py --history-players [연도…]  (리그·시즌마다 경기 목록 1번 + 20경기씩 상세, 전부 약 1,250회)
# 선수별 경기 기록이 있는 시즌은 리그마다 다름(/leagues coverage.statistics_players — EPL 14-15부터, 나머지 15-16부터) → 없는 시즌은 건너뜀
AF_PLAYER_SEASONS = list(range(2014, 2026))
PS_DIR = f"{MODEL_DIR}/af_seasons"

def _history_name_votes(code, yr, fxs, teammap):
    """API-Football 팀 ID → 우리 순위표·일정에 쓰는 이름(그 리그·시즌 경기 파일과 같은 날·같은 점수 + 이름 비슷함으로 투표)"""
    import json
    from collections import defaultdict
    src = f"{MODEL_DIR}/history_matches.csv" if yr < min(MATCH_SEASONS) else f"{MODEL_DIR}/all_matches.csv"
    df = pd.read_csv(src)
    df = df[(df.league == code) & (df.season == yr)]
    by_day = defaultdict(list)
    for r in df.itertuples():
        by_day[str(r.date)[:10]].append(r)
    votes = defaultdict(Counter)
    for f in fxs:
        if f["goals"]["home"] is None:
            continue
        day = pd.Timestamp(f["fixture"]["date"]).tz_convert("UTC").strftime("%Y-%m-%d")
        h, a = f["teams"]["home"], f["teams"]["away"]
        best, bs = None, 0
        for r in by_day.get(day, []):
            sc = _af_name_sim(h["name"], r.home_team) + _af_name_sim(a["name"], r.away_team)
            if int(r.home_goals) == f["goals"]["home"] and int(r.away_goals) == f["goals"]["away"]:
                sc += 0.8
            if sc > bs:
                best, bs = r, sc
        if best is not None and bs >= 1.0:
            votes[h["id"]][best.home_team] += 1; votes[a["id"]][best.away_team] += 1
    out = {str(t): c.most_common(1)[0][0] for t, c in votes.items()}
    return lambda t: out.get(str(t["id"])) or teammap.get(str(t["id"])) or t["name"]

def _history_players_one(code, yr, teammap):
    import json
    lid = AF_COMPS[code]
    fxs = _af("/fixtures", league=lid, season=yr).get("response") or []
    keep = []
    for f in fxs:
        if f["fixture"]["status"]["short"] not in AF_DONE or f["goals"]["home"] is None:
            continue
        base = (f["league"].get("round") or "").split(" - ")[0].strip()
        if code == "CL":
            if not (base.startswith("Group") or base.startswith("League Stage") or AF_ROUND_STAGE.get(base) or base == "Knockout Round Play-offs"):
                continue   # 예선 제외
        elif not base.startswith("Regular Season"):
            continue
        keep.append(f)
    if not keep:
        return None
    nm = _history_name_votes(code, yr, keep, teammap)
    dets = []
    for i in range(0, len(keep), 20):
        ids = [f["fixture"]["id"] for f in keep[i:i + 20]]
        for fx in _af("/fixtures", ids="-".join(map(str, ids))).get("response") or []:
            det = af_match_detail(fx)
            if not det["pstats"]["home"] and not det["pstats"]["away"]:
                continue
            md = re.search(r" - (\d+)$", fx["league"].get("round") or "")
            dets.append({"home_team": nm(fx["teams"]["home"]), "away_team": nm(fx["teams"]["away"]),
                         "date": pd.Timestamp(fx["fixture"]["date"]).tz_convert("UTC").strftime("%Y-%m-%d"),
                         "home_goals": fx["goals"]["home"], "away_goals": fx["goals"]["away"], "round": int(md.group(1)) if md else None,
                         "detail": {"pstats": det["pstats"], "lineups": det["lineups"]}})
    if len(dets) < 0.5 * len(keep):
        print(f"  ⚠️ {code} {yr}: 선수 기록이 있는 경기 {len(dets)}/{len(keep)} — 저장 안 함")
        return None
    rows = list(af_league_agg(dets, use_grid=yr >= GRID_FROM).values())
    add_percentiles(rows, min_minutes=450)
    for r in rows:
        r.pop("matches", None)
    teams = sorted({d["home_team"] for d in dets} | {d["away_team"] for d in dets})
    md_write(f"{PS_DIR}/{code}_{yr}.json.gz", {"league": code, "season": yr, "matches": len(dets), "fixtures": len(keep),
                                                "grid": yr >= GRID_FROM,
                                                "teams": teams, "players": rows})
    print(f"  ✅ {code} {yr}-{(yr + 1) % 100:02d}: {len(dets)}/{len(keep)}경기 · 선수 {len(rows)}명 · 팀 {len(teams)}")
    return True

def build_player_season_index():
    import json, glob
    from collections import defaultdict
    players, teams = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for p in sorted(glob.glob(f"{PS_DIR}/*_*.json.gz")):
        d = md_read(p)
        for r in d["players"]:
            players[str(r["id"])].append([d["league"], d["season"], r["team"]])
        for t in d["teams"]:
            teams[t][str(d["season"])].append(d["league"])
    with open(f"{PS_DIR}/index.json", "w", encoding="utf-8") as f:
        json.dump({"players": players, "teams": teams}, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    print(f"  index.json: 선수 {len(players)}명 · 팀 {len(teams)}")

def fetch_history_players(seasons=None, force=False):
    import json
    from concurrent.futures import ThreadPoolExecutor
    if not API_FOOTBALL_KEY:
        print("  ⚠️ API_FOOTBALL_KEY가 없어 건너뜀"); return
    seasons = seasons or AF_PLAYER_SEASONS
    os.makedirs(PS_DIR, exist_ok=True)
    teammap = json.load(open(f"{MODEL_DIR}/af_team_map.json", encoding="utf-8"))
    cov = {}
    for code, lid in AF_COMPS.items():
        r = (_af("/leagues", id=lid).get("response") or [{}])[0]
        cov[code] = {s["year"] for s in r.get("seasons") or [] if ((s.get("coverage") or {}).get("fixtures") or {}).get("statistics_players")}
    jobs = [(c, y) for y in seasons for c in AF_COMPS if y in cov[c] and (force or not os.path.exists(f"{PS_DIR}/{c}_{y}.json.gz"))]
    print(f"\n[과거 시즌 선수 기록] {len(jobs)}개 리그·시즌")
    def run(j):
        try:
            return _history_players_one(j[0], j[1], teammap)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ❌ {j}: {e}")
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(run, jobs))
    build_player_season_index()

def _wiki_sections(title):
    return _wiki_get("https://en.wikipedia.org/w/api.php", {"action": "parse", "page": title, "prop": "sections",
                                                            "format": "json", "redirects": 1}).get("parse", {}).get("sections")

def _wiki_section_text(title, index):
    return _wiki_get("https://en.wikipedia.org/w/api.php", {"action": "parse", "page": title, "prop": "wikitext", "section": index,
                                                            "format": "json", "redirects": 1})["parse"]["wikitext"]["*"]

# 기록 문서에 표 대신 글로만 적힌 팀(아스널: Declan Rice … club record £100m)·굵은 글씨 제목(Fees Paid) 아래 표(애스턴 빌라)용.
# 선수·구단 링크는 위키데이터로 사람(Q5)·축구 클럽인지 확인 — 표 칸이 밀려 'Winger'가 구단으로 잡히는 일(아약스)을 막음.
# 구단 본문 문서의 글까지 읽으면 오래된 기록·다른 팀 기록이 섞여서(PSV·레스터) 기록 문서(List of … records and statistics)만 씀.
_REC_CLUB_P31 = {'Q476028', 'Q847017', 'Q12973014', 'Q103229495', 'Q1194951', 'Q17270000'}
def _wiki_classify(titles):
    titles = [t for t in titles if t]
    if not titles: return {}
    out = {}
    for i in range(0, len(titles), 40):
        q = _wiki_get("https://en.wikipedia.org/w/api.php", {"action": "query", "titles": "|".join(titles[i:i+40]), "prop": "pageprops",
                         "ppprop": "wikibase_item", "redirects": 1, "format": "json"})["query"]
        alias = {}
        for r in q.get("normalized", []) + q.get("redirects", []): alias[r["from"]] = r["to"]
        qid = {pg["title"]: pg.get("pageprops", {}).get("wikibase_item") for pg in q.get("pages", {}).values()}
        ents = _wd_entities([v for v in qid.values() if v], "claims")
        for t in titles[i:i+40]:
            tt = t
            while tt in alias: tt = alias[tt]
            e = ents.get(qid.get(tt) or "", {})
            p31 = {(c.get("mainsnak", {}).get("datavalue", {}).get("value") or {}).get("id") for c in e.get("claims", {}).get("P31", [])}
            out[t] = "human" if "Q5" in p31 else "club" if p31 & _REC_CLUB_P31 else None
    return out

_REC_PAID = re.compile(r"(record|highest|most expensive|biggest|largest)\b[^.]{0,90}?\b(signing|purchase|paid|outlay|acquisition|transfer fee|spent|signed|bought)|\b(paid|spent)\b[^.]{0,60}\brecord", re.I)
_REC_RECV = re.compile(r"(record|highest|biggest|largest|most expensive)\b[^.]{0,90}?\b(received|sale|sold|departure|fee received)|\b(sold|sale|received)\b[^.]{0,60}\brecord", re.I)
_REC_OLD = re.compile(r"at the time|then[- ]club|previous|former record|until|broke the|was broken|surpass|world|British|Premier League history|Italian football|Spanish football|German football|French football", re.I)

def _parse_record_prose(text, prefer_eur=False):
    t = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    t = re.sub(r"\[\[(?:File|Image):[^|\]]*\|(?:[^|\]]*\|)*", "", t)
    t = re.sub(r"\{\{(?:efn|refn|cite)[^{}]*\}\}", " ", t, flags=re.I)
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z\[\'])|\n", t)
    cands = {}
    for s in sents:
        for kind, rx in (("paid", _REC_PAID), ("received", _REC_RECV)):
            if kind in cands or not rx.search(s) or _REC_OLD.search(s): continue
            if kind == "paid" and _REC_RECV.search(s) and not _REC_PAID.search(s): continue
            links = [(a.strip(), (b or a).strip()) for a, b in re.findall(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", s) if not re.match(r"(File|Image|Category|:?[a-z]{2}:)", a)]
            fees = _hon_fees(s)
            if not fees or len(links) < 2: continue
            cands.setdefault(kind, (s, links, fees))
    if not cands: return {}
    cls = _wiki_classify(sorted({a for v in cands.values() for a, _ in v[1]}))
    out = {}
    for kind, (s, links, fees) in cands.items():
        pl = next((l for l in links if cls.get(l[0]) == "human"), None)
        cl = next((l for l in links if cls.get(l[0]) == "club"), None)
        if not pl or not cl: continue
        eur = [f for f in fees if f[0] == "€"]
        cur, v = eur[0] if (prefer_eur and eur) else fees[0]
        yrs = re.findall(r"(?<!\d)((?:19|20)\d\d)(?!\d)", re.sub(r"\[\[[^\]]*\|", "", s))
        out[kind] = {"player_title": pl[0], "player": pl[1], "club_title": cl[0], "club": cl[1], "fee": f"{cur}{v:g}m", "year": yrs[-1] if yrs else None}
    return out

def _record_bold_split(text):
    # '''Fees Paid''' 같은 굵은 제목도 소제목으로
    return re.sub(r"^'''([^'\n]{3,40})'''\s*$", r"==== \1 ====", text, flags=re.M)


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
        tr = [x for x in rsecs if re.search(r"transfer|fee", x["line"], re.I)]
        text = "".join(_wiki_section_text(rt, x["index"]) + "\n" for x in tr[:3])
        recs = _parse_record_transfers(_record_bold_split(text), prefer_eur) if text else {}
        if text and len(recs) < 2:
            for k, v in _parse_record_prose(text, prefer_eur).items():
                recs.setdefault(k, v)
        if recs:   # 선수 = 사람, 구단 = 축구 클럽인지 확인
            cls = _wiki_classify(sorted({r[k] for r in recs.values() for k in ("player_title", "club_title")}))
            recs = {k: r for k, r in recs.items() if cls.get(r["club_title"]) == "club" and cls.get(r["player_title"]) != "club"}
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

def _wiki_one(team_name, league=None, known_title=None):
    # known_title: 이전에 찾아 둔(2026-09-30 전 팀 점검을 거친) 영어 문서 제목 — 갱신 때 다시 검색하면 검색 결과가 실행마다
    # 바뀌어 엉뚱한 문서가 잡힐 수 있어서(셰필드 유나이티드 → Sheffield F.C. 등 전례) 한 번 찾은 문서를 계속 씀.
    # 새 팀만 검색. 문서를 다시 찾고 싶으면 team_wiki.json에서 그 팀 항목을 지우고 실행
    title = WIKI_TITLE_OVERRIDE.get(team_name) or known_title
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
            return t, _wiki_one(t, league_of.get(t), wiki.get(t, {}).get("en_title")), None
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

# 경기 상세·선수 기록·부상자·선수 경력 변환은 af_transform.py(서버 main.py와 같이 씀 — 2026-10-06 분리)
from af_transform import *   # noqa: F401,F403

def _fetch_af_transfers(af_id, limit=30):
    """API-Football /transfers → 최근 이적 기록(이 팀이 관련된 것만). 선수 사진·양쪽 구단 로고·방향(in/out)까지 저장
    — 예전엔 이름·날짜·유형만 저장해서 화면에 글자만 나왔음(2026-09-29 확장). 사진은 선수 ID로 만드는 고정 주소."""
    raw = _af("/transfers", team=af_id).get("response") or []
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
    # API-Football이 같은 이적을 날짜만 하루이틀 다르게 두세 번 주는 경우가 많음(고레츠카 8.26·8.27) → 같은 선수·같은 경로는 60일 안이면 하나만
    out, seen = [], {}
    for x in flat:
        k = (x["player_id"] or x["player"], x["from"], x["to"])
        d = pd.Timestamp(x["date"]) if x["date"] else None
        if k in seen and d is not None and seen[k] is not None and abs((seen[k] - d).days) <= 60:
            continue
        seen[k] = d
        out.append(x)
    return out[:limit]

# ── 스쿼드(사진·등번호)·이적 기록 — 5대 리그 전 팀(2026-10-07) ──
# 예전 fetch_squad_transfers는 무료 플랜 시절 것이라 EPL만, 그리고 한 번 받은 팀은 다시 안 받아서 9월 뒤 이적이 안 들어왔음.
# 팀 ID는 af_players.json(af_sync가 만든 매칭)에 있으니 검색 없이 팀당 2회(/players/squads·/transfers), 7일마다.
# 그날 남은 요청에서 300회는 남김(서버가 요청 때 쓰는 몫). 데이터가 없는 팀부터.
AF_SQUAD_DAYS = 7

def fetch_squads_transfers_all(force=False, limit=None):
    import json
    if not API_FOOTBALL_KEY:
        return
    teams = json.load(open(f"{MODEL_DIR}/af_players.json", encoding="utf-8"))
    path = f"{MODEL_DIR}/team_extra.json"
    extra = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    team_info = json.load(open(f"{MODEL_DIR}/team_info.json", encoding="utf-8")) if os.path.exists(f"{MODEL_DIR}/team_info.json") else {}
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    age = lambda t: (pd.Timestamp(today) - pd.Timestamp((extra.get(t) or {}).get("updated") or "2000-01-01")).days
    todo = sorted([t for t in teams if force or age(t) >= AF_SQUAD_DAYS], key=lambda t: (bool((extra.get(t) or {}).get("transfers")), -age(t)))
    left = _af_remaining()
    budget = (max(0, left - 100) if limit else max(0, left - 300)) // 2   # 손으로 팀 수를 주면 100회만 남김
    todo = todo[:min(budget, limit or 999)]
    print(f"\n[스쿼드·이적] {len(todo)}팀 갱신(대기 {sum(1 for t in teams if age(t) >= AF_SQUAD_DAYS) - len(todo)}팀)")
    for t in todo:
        af_id = teams[t]["af_id"]
        entry = dict(extra.get(t) or {})
        try:
            raw = ((_af("/players/squads", team=af_id).get("response") or [{}])[0] or {}).get("players") or []
            squad = [{"name": p.get("name"), "position": SQUAD_POSITION_MAP.get(p.get("position"), p.get("position")),
                      "shirtNumber": p.get("number"), "photo": p.get("photo")} for p in raw]
            if squad:
                entry["squad"] = _clean_squad(squad, t, team_info)
            tf = _fetch_af_transfers(af_id)
            if tf:   # 빈 응답(한도 등)이면 기존 기록 유지
                entry["transfers"] = tf
        except Exception as e:
            print(f"  ⚠️ {t}: {e}"); continue
        entry["af_id"], entry["updated"] = af_id, today
        extra[t] = entry
    with open(path, "w", encoding="utf-8") as f:
        json.dump(extra, f, ensure_ascii=False, indent=2)
    print(f"  ✅ 스쿼드 있는 팀 {sum(1 for v in extra.values() if v.get('squad'))} · 이적 있는 팀 {sum(1 for v in extra.values() if v.get('transfers'))}")

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
    # 받지 못한 리그는 기존 일정 유지 — 예전엔 그 리그가 schedule.json에서 통째로 빠져 일정 탭·순위 예측에서 사라졌음
    prev = {}
    if os.path.exists(f"{MODEL_DIR}/schedule.json"):
        with open(f"{MODEL_DIR}/schedule.json", encoding='utf-8') as f:
            prev = json.load(f)
    schedule = {}
    for code, name in LEAGUES_V2.items():
        print(f"  [{name}] 일정 수집 중...")
        try:
            res = fd_get(
                f"{BASE_URL}/competitions/{code}/matches",
                headers=HEADERS,
                params={"season": CURRENT_SEASON}
            )
            ok = res.status_code == 200
            err = res.status_code
        except Exception as e:
            ok, err = False, e
        if not ok:
            print(f"  ❌ {name} 일정 오류: {err} — 기존 일정 유지")
            if code in prev:
                schedule[code] = prev[code]
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
        res = fd_get(f"{BASE_URL}/competitions/CL/matches", headers=HEADERS, params={"season": yr})
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

# UCL에 나오는 5대 리그 밖 팀(PSV·페예노르트·아약스 / 포르투·스포르팅·벤피카)의 자국 리그 — 무료 플랜 포함.
# 아직 예측 모델엔 안 씀(리그 수준 보정을 백테스트로 검증한 뒤 켤 예정) → 별도 파일에만 모아 둠
EXTRA_LEAGUES = {"DED": "에레디비시 (네덜란드)", "PPL": "프리메이라리가 (포르투갈)"}

def fetch_extra_leagues():
    """에레디비시·프리메이라리가 4시즌 경기 → extra_matches.csv(all_matches.csv와 같은 열, 같은 병합 규칙)"""
    print("\n[자국 리그(5대 리그 밖)] 수집 중...")
    path = f"{MODEL_DIR}/extra_matches.csv"
    fetched, failed = [], []
    for season in MATCH_SEASONS:
        for code in EXTRA_LEAGUES:
            df_s = fetch_matches(code, season)
            if df_s.empty:
                failed.append(f"{code} {season}")
            else:
                df_s['season'] = season
                fetched.append(df_s)
            time.sleep(6)
    if failed:
        print(f"  ⚠️ 수집 실패(기존 유지): {', '.join(failed)}")
    parts = fetched[:]
    if os.path.exists(path):
        old = pd.read_csv(path)
        old['date'] = pd.to_datetime(old['date'])
        parts.append(old)
    if not parts:
        return
    df = pd.concat(parts, ignore_index=True)
    df = df[[c for c in MATCH_COLUMNS if c in df.columns]].drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='first')
    df = df.sort_values(['date', 'league', 'home_team'], kind='mergesort').reset_index(drop=True)
    df['match_id'] = df['match_id'].astype('Int64')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"  ✅ extra_matches.csv: {len(df)}경기")

def main():
    """단계마다 실패를 따로 잡아서 하나가 깨져도 나머지 단계는 계속 돌고 받은 데이터는 저장됨(각 단계는 성공했을 때만 파일을 씀).
    예전엔 2026-10-01에 라리가 일정 요청 하나가 연결 오류로 실패하자 스크립트가 멈춰 그날 받은 경기·모델·일정이 전부 커밋되지 않았음.
    실패한 단계가 있으면 끝에 종료 코드 1 → 워크플로우는 커밋을 하고(if: !cancelled()) 실행 결과는 빨간색으로 남아 알아챌 수 있음"""
    import sys
    print("=== FotData 자동 업데이트 시작 ===")
    failed = []

    def step(name, fn, *args):
        try:
            return fn(*args)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ❌ [{name}] 실패 — 이 단계만 건너뛰고 계속: {type(e).__name__}: {e}")
            failed.append(name)
            print(f"::warning::{name} 실패: {type(e).__name__}: {str(e)[:200]}")
            return None

    def model_pipeline():
        # 1~2. 경기 데이터 수집 + 기존 CSV와 병합 → all_matches.csv
        df_total = collect_matches()
        # 3. 3시즌 혼합 스탯 (예측용) — 경기 수 구간별 가중치 + prestige 보정
        df_stats_current = calculate_blended_stats(df_total)
        df_stats_current.to_csv(f"{MODEL_DIR}/team_stats.csv", index=False, encoding='utf-8-sig')
        # 4~6. 피처 생성 + 시간순 검증 + 서빙 모델 학습/저장 (+ team_state.json, accuracy.json)
        return df_total, train_models(df_total)

    res = step("경기 수집·모델 학습", model_pipeline)
    df_total, accuracy_data = res if res else (None, None)

    # UCL 토너먼트
    step("UCL 토너먼트", fetch_ucl_tournament)

    # 전체 일정 (일정 탭용)
    schedule = step("전체 일정", fetch_full_schedule)

    # AI 예측 트랙레코드 (라이브 /predict를 호출하므로 반드시 위 모델 학습 이후,
    # 그리고 아직 이번 실행분 커밋을 push하기 전에 실행 — 그래야 "그 시점에 실제
    # 서빙 중이던 모델"의 예측을 기록하게 됨). 실패해도 다음 실행에서 재시도(경고만)
    if schedule:
        try:
            update_prediction_log(schedule)
        except Exception as e:
            print(f"⚠️ 예측 트랙레코드 갱신 실패(다음 실행에서 재시도): {e}")

    step("득점왕(API-Football)", fetch_top_scorers)
    try:
        fetch_scorers()
    except Exception as e:
        print(f"  ⚠️ 득점·도움 순위 실패(기존 유지): {e}")
    try:
        fetch_extra_leagues()
    except Exception as e:
        print(f"  ⚠️ 자국 리그 경기 수집 실패(기존 유지): {e}")

    # 로고 자동 업데이트
    step("로고", update_team_logos)

    # 팀 상세정보 (홈구장/스쿼드 등)
    step("팀 상세정보", fetch_team_info)

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
    try:   # 역대 우승·준우승(리그 페이지 시즌 탭) — 위키 문서 6장, 시즌이 끝나면 새 줄이 자동으로 들어옴
        fetch_league_history()
    except Exception as e:
        print(f"  ⚠️ 역대 우승 목록 갱신 실패(기존 유지): {e}")

    # API-Football Pro(2026-10-06): 끝난 경기 상세·결장·부상자·선수 프로필 — 키가 있을 때만(af_sync 안에서 확인)
    step("API-Football 경기 상세·부상자·선수", af_sync)
    step("대회 우승 팀(트로피 보강)", fetch_comp_winners)
    step("감독(API-Football·위키데이터)", fetch_coaches)
    step("스쿼드·이적(API-Football)", fetch_squads_transfers_all)
    step("선수 경력·트로피 미리 받기", af_profiles_sync)


    # (순위 예측은 main.py /predict/champion이 요청 때 현재 승점·남은 일정·모델 확률로 계산 — 2026-09-28부터
    #  예전 simulate_season/champion_predictions.json은 화면에 안 쓰여서 제거)

    if df_total is not None:
        print(f"\n🏆 업데이트 완료!")
        print(f"   데이터: {len(df_total)}경기")
        print(f"   서빙 모델(LR) 시간순 검증 정확도: {accuracy_data['logistic_regression']}%")
    if failed:
        print(f"\n⚠️ 실패한 단계 {len(failed)}개: {', '.join(failed)} — 나머지 결과는 저장됨(커밋은 워크플로우가 진행)")
        sys.exit(1)

SCORER_COMPS = ["PL", "PD", "BL1", "SA", "FL1", "CL"]

def fetch_scorers():
    """이번 시즌 득점·도움 순위(football-data.org /competitions/{리그}/scorers, 5대 리그 + UCL) → scorers.json.
    FOOTBALL_API_KEY 하나로 매일 받음(API-Football 무료 플랜은 24-25 시즌에 막혀 있어서 현재 시즌은 이쪽).
    리그별로 제대로 받은 것만 교체 — 403(플랜 제한)·429·빈 응답이면 그 리그는 기존 값 유지"""
    import json
    print("\n[득점·도움 순위] 수집 중...")
    if not API_KEY:
        print("  ⚠️ FOOTBALL_API_KEY가 없어 건너뜀")
        return
    path = f"{MODEL_DIR}/scorers.json"
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            out = json.load(f)
    changed = False
    for code in SCORER_COMPS:
        for attempt in range(2):
            try:
                r = fd_get(f"{BASE_URL}/competitions/{code}/scorers", headers=HEADERS, params={"limit": 100}, timeout=20)
            except Exception as e:
                print(f"  ⚠️ {code}: {e}")
                r = None
                break
            if r.status_code == 429 and attempt == 0:
                time.sleep(61)
                continue
            break
        time.sleep(6)
        if r is None or r.status_code != 200:
            print(f"  ⚠️ {code}: HTTP {getattr(r, 'status_code', '-')} (기존 유지){' — 무료 플랜에서 막힌 것 같음' if getattr(r, 'status_code', 0) == 403 else ''}")
            continue
        data = r.json()
        rows = []
        for sc in data.get("scorers") or []:
            pl, tm = sc.get("player") or {}, sc.get("team") or {}
            if not pl.get("name"):
                continue
            rows.append({"id": pl.get("id"), "name": pl.get("name"), "nationality": pl.get("nationality"),
                         "position": pl.get("section") or pl.get("position"), "shirt": pl.get("shirtNumber"),
                         "team": tm.get("name"), "team_logo": tm.get("crest"), "played": sc.get("playedMatches"),
                         "goals": sc.get("goals") or 0, "assists": sc.get("assists") or 0, "penalties": sc.get("penalties") or 0})
        if not rows:
            print(f"  ⚠️ {code}: 빈 응답 (기존 유지)")
            continue
        season = data.get("season") or {}
        out[code] = {"season": (season.get("startDate") or "")[:4], "matchday": season.get("currentMatchday"),
                     "updated": pd.Timestamp.utcnow().strftime("%Y-%m-%d"), "scorers": rows}
        changed = True
        print(f"  ✅ {code}: {len(rows)}명 (1위 {rows[0]['name']} {rows[0]['goals']}골)")
    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("  ✅ scorers.json 저장 완료")

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
    elif "--history-seasons" in sys.argv:
        fetch_history_seasons([int(a) for a in sys.argv[2:] if a.isdigit()] or None)
    elif "--history-players" in sys.argv:
        fetch_history_players([int(a) for a in sys.argv[2:] if a.isdigit()] or None, force="--force" in sys.argv)
    elif "--player-index" in sys.argv:
        build_player_season_index()
    elif "--af-profiles" in sys.argv:
        fetch_comp_winners()
        af_profiles_sync(next((int(a) for a in sys.argv[2:] if a.isdigit()), AF_PROFILE_BUDGET))
    elif "--coaches" in sys.argv:
        fetch_coaches(force="--force" in sys.argv)
    elif "--af-sync" in sys.argv:
        af_sync()
    elif "--league-history" in sys.argv:
        fetch_league_history()
    elif "--scorers-only" in sys.argv:
        fetch_scorers()
    elif "--extra-leagues" in sys.argv:
        fetch_extra_leagues()
    elif "--ucl-only" in sys.argv:
        fetch_ucl_tournament()
    elif "--transfers-only" in sys.argv:   # 스쿼드·이적 [--force] [팀 수]
        fetch_squads_transfers_all(force="--force" in sys.argv, limit=next((int(a) for a in sys.argv[2:] if a.isdigit()), None))
    else:
        main()