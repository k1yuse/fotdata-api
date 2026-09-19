# ── 자동 데이터 업데이트 스크립트 ──
import requests
import time
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

def fetch_matches(league_code, season):
    url = f"{BASE_URL}/competitions/{league_code}/matches"
    params = {"season": season, "status": "FINISHED"}
    name = LEAGUES_V2.get(league_code, league_code)
    print(f"  [{name}] 수집 중...")
    res = requests.get(url, headers=HEADERS, params=params)
    if res.status_code != 200:
        print(f"  ❌ 오류: {res.status_code}")
        return pd.DataFrame()
    matches = res.json().get("matches", [])
    print(f"  ✅ {len(matches)}경기")
    rows = []
    for m in matches:
        ft = m["score"]["fullTime"]
        rows.append({
            "match_id":   m["id"],
            "date":       m["utcDate"][:10],
            "league":     league_code,
            "home_team":  m["homeTeam"]["name"],
            "away_team":  m["awayTeam"]["name"],
            "home_goals": ft.get("home"),
            "away_goals": ft.get("away"),
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

def get_recent_form(df, team, before_date, n=5):
    """최근 N경기 승점 합"""
    team_matches = df[
        ((df['home_team']==team) | (df['away_team']==team)) &
        (df['date'] < before_date)
    ].tail(n)
    points = 0
    for _, row in team_matches.iterrows():
        if row['home_team'] == team:
            if row['result'] == 'H': points += 3
            elif row['result'] == 'D': points += 1
        else:
            if row['result'] == 'A': points += 3
            elif row['result'] == 'D': points += 1
    return points

def get_recent_goals(df, team, before_date, n=10):
    """최근 N경기 평균 득점, 실점"""
    team_matches = df[
        ((df['home_team']==team) | (df['away_team']==team)) &
        (df['date'] < before_date)
    ].tail(n)
    if len(team_matches) == 0:
        return 1.0, 1.0
    scored, conceded = 0, 0
    for _, row in team_matches.iterrows():
        if row['home_team'] == team:
            scored += row['home_goals']
            conceded += row['away_goals']
        else:
            scored += row['away_goals']
            conceded += row['home_goals']
    return scored / len(team_matches), conceded / len(team_matches)

def get_h2h_rate(df, home, away, before_date, n=10):
    """H2H 홈팀 승률"""
    h2h = df[
        ((df['home_team']==home) & (df['away_team']==away)) |
        ((df['home_team']==away) & (df['away_team']==home))
    ]
    h2h = h2h[h2h['date'] < before_date].tail(n)
    if len(h2h) == 0:
        return 0.33
    home_wins = len(h2h[((h2h['home_team']==home) & (h2h['result']=='H')) |
                        ((h2h['away_team']==home) & (h2h['result']=='A'))])
    return round(home_wins / len(h2h), 3)

def calculate_elo_ratings(df, k=20, home_advantage=70):
    """ELO 점수 계산 (시간 순서대로)"""
    elo = {}
    elo_history = []
    
    df_sorted = df.sort_values('date').reset_index(drop=True)
    
    for _, match in df_sorted.iterrows():
        home, away = match['home_team'], match['away_team']
        
        # 초기값 1500
        if home not in elo: elo[home] = 1500
        if away not in elo: elo[away] = 1500
        
        # 경기 전 ELO 저장
        elo_history.append({
            'date': match['date'],
            'home_team': home,
            'away_team': away,
            'home_elo_before': elo[home],
            'away_elo_before': elo[away],
        })
        
        # 기대 승률 (홈 어드밴티지 적용)
        home_elo_adj = elo[home] + home_advantage
        away_elo_adj = elo[away]
        expected_home = 1 / (1 + 10 ** ((away_elo_adj - home_elo_adj) / 400))
        
        # 실제 결과
        if match['result'] == 'H':
            actual_home = 1.0
        elif match['result'] == 'D':
            actual_home = 0.5
        else:
            actual_home = 0.0
        
        # ELO 업데이트
        change = k * (actual_home - expected_home)
        elo[home] += change
        elo[away] -= change
    
    return pd.DataFrame(elo_history), elo

def build_features(df, df_stats):
    """피처 생성 (ELO 포함)"""
    print("ELO 계산 중...")
    elo_df, final_elo = calculate_elo_ratings(df)
    
    # 빠른 조회를 위해 인덱스 설정
    elo_lookup = {}
    for _, row in elo_df.iterrows():
        key = (row['date'], row['home_team'], row['away_team'])
        elo_lookup[key] = (row['home_elo_before'], row['away_elo_before'])
    
    rows = []
    df_sorted = df.sort_values('date').reset_index(drop=True)
    
    for _, match in df_sorted.iterrows():
        home, away, date = match['home_team'], match['away_team'], match['date']
        
        # ELO
        home_elo, away_elo = elo_lookup.get((date, home, away), (1500, 1500))
        
        # 폼
        home_form = get_recent_form(df, home, date)
        away_form = get_recent_form(df, away, date)

        # 첫 경기는 데이터 없어서 건너뜀
        if len(df[(df['date'] < date) & ((df['home_team']==home) | (df['away_team']==home))]) < 5:
            continue
        if len(df[(df['date'] < date) & ((df['home_team']==away) | (df['away_team']==away))]) < 5:
            continue
        
        # 최근 평균 득실점
        home_avg_scored, home_avg_conceded = get_recent_goals(df, home, date)
        away_avg_scored, away_avg_conceded = get_recent_goals(df, away, date)
        
        # H2H
        h2h_rate = get_h2h_rate(df, home, away, date)
        
        # 팀 스탯
        h_stats = df_stats[df_stats['team']==home]
        a_stats = df_stats[df_stats['team']==away]
        if h_stats.empty or a_stats.empty:
            continue
        h = h_stats.iloc[0]
        a = a_stats.iloc[0]
        
        rows.append({
            'home_elo':          home_elo,
            'away_elo':          away_elo,
            'elo_diff':          home_elo - away_elo,
            'home_form':         home_form,
            'away_form':         away_form,
            'form_diff':         home_form - away_form,
            'home_avg_scored':   home_avg_scored,
            'away_avg_scored':   away_avg_scored,
            'home_avg_conceded': home_avg_conceded,
            'away_avg_conceded': away_avg_conceded,
            'home_attack':       h['attack_strength'],
            'away_attack':       a['attack_strength'],
            'home_defense':      h['defense_strength'],
            'away_defense':      a['defense_strength'],
            'home_win_rate':     h['win_rate'],
            'away_win_rate':     a['win_rate'],
            'win_rate_diff':     h['win_rate'] - a['win_rate'],
            'h2h_home_rate':     h2h_rate,
            'result':            match['result'],
        })
    return pd.DataFrame(rows)

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

    with open(f"{MODEL_DIR}/team_info.json", 'w', encoding='utf-8') as f:
        json.dump(team_info, f, ensure_ascii=False, indent=2)
    print(f"  ✅ team_info.json 저장 완료 ({len(team_info)}팀)")

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
            ft = m["score"]["fullTime"]
            rows.append({
                "date":       m["utcDate"],
                "matchday":   m.get("matchday"),
                "stage":      m.get("stage"),
                "home_team":  m["homeTeam"]["name"],
                "away_team":  m["awayTeam"]["name"],
                "home_goals": ft.get("home"),
                "away_goals": ft.get("away"),
                "status":     m["status"],
            })
        rows.sort(key=lambda r: r["date"])
        schedule[code] = rows
        print(f"  ✅ {name} {len(rows)}경기")
        time.sleep(6)

    with open(f"{MODEL_DIR}/schedule.json", 'w', encoding='utf-8') as f:
        json.dump(schedule, f, ensure_ascii=False, indent=2)
    print(f"  ✅ schedule.json 저장 완료")

def fetch_ucl_tournament():
    import json
    print("\n[UCL 토너먼트] 수집 중...")
    res = requests.get(
        f"{BASE_URL}/competitions/CL/matches",
        headers=HEADERS,
        params={"season": 2025}
    )
    if res.status_code != 200:
        print(f"  ❌ UCL 토너먼트 오류: {res.status_code}")
        return

    matches = res.json().get("matches", [])
    stages = {"PLAYOFFS": [], "LAST_16": [], "QUARTER_FINALS": [], "SEMI_FINALS": [], "FINAL": []}

    logo_path = f"{MODEL_DIR}/team_logos.json"
    logos = {}
    if os.path.exists(logo_path):
        with open(logo_path, 'r', encoding='utf-8') as f:
            logos = json.load(f)

    agg = {}
    for m in matches:
        stage = m.get("stage", "")
        if stage not in stages:
            continue
        home = m["homeTeam"].get("name")
        away = m["awayTeam"].get("name")
        if not home or not away:
            continue
        ft = m["score"]["fullTime"]
        status = m["status"]
        key = tuple(sorted([home, away]))
        if key not in agg:
            agg[key] = {"stage": stage, "team1": home, "team2": away, "team1_goals": 0, "team2_goals": 0, "legs": [], "status": "FINISHED"}
        if ft.get("home") is not None:
            hg, ag = ft["home"], ft["away"]
            if agg[key]["team1"] == home:
                agg[key]["team1_goals"] += hg
                agg[key]["team2_goals"] += ag
            else:
                agg[key]["team1_goals"] += ag
                agg[key]["team2_goals"] += hg
            agg[key]["legs"].append({"home_team": home, "away_team": away, "home_goals": hg, "away_goals": ag})
        if status in ["SCHEDULED", "TIMED"]:
            agg[key]["status"] = "UPCOMING"

    for key, v in agg.items():
        t1, t2 = v["team1"], v["team2"]
        t1g, t2g = v["team1_goals"], v["team2_goals"]
        winner = t1 if t1g > t2g else (t2 if t2g > t1g else None)
        stages[v["stage"]].append({
            "team1": t1, "team2": t2,
            "team1_goals": t1g, "team2_goals": t2g,
            "team1_logo": logos.get(t1, ""),
            "team2_logo": logos.get(t2, ""),
            "winner": winner, "status": v["status"], "legs": v["legs"]
        })

    stages = reconstruct_bracket_order(stages)

    with open(f"{MODEL_DIR}/ucl_tournament.json", 'w', encoding='utf-8') as f:
        json.dump(stages, f, ensure_ascii=False, indent=2)
    print(f"  ✅ UCL 토너먼트 저장 완료")

def main():
    print("=== FotData 자동 업데이트 시작 ===")

    # 1. 데이터 수집 (24-25 + 25-26 + 26-27)
    all_dfs = []
    for season in [2024, 2025, 2026]:
        print(f"\n[{season}-{season+1} 시즌]")
        for code in LEAGUES_V2:
            df_s = fetch_matches(code, season)
            if not df_s.empty:
                df_s['season'] = season
                all_dfs.append(df_s)
            time.sleep(6)

    df_total = pd.concat(all_dfs, ignore_index=True)
    df_total = df_total.drop_duplicates(
        subset=['date','home_team','away_team']
    ).sort_values('date').reset_index(drop=True)
    print(f"\n✅ 전체 데이터: {len(df_total)}경기")

    # 2. 전체 경기 저장 (H2H, 폼용)
    # 기존 데이터 불러오기
    existing_path = f"{MODEL_DIR}/all_matches.csv"
    if os.path.exists(existing_path):
        df_existing = pd.read_csv(existing_path)
        df_existing['date'] = pd.to_datetime(df_existing['date'])
        # 26-27 이전 데이터는 기존 것 유지, 26-27만 새로 교체
        df_old = df_existing[df_existing['date'] < '2026-08-01']
        df_new_2627 = df_total[df_total['date'] >= '2026-08-01']
        df_total = pd.concat([df_old, df_new_2627], ignore_index=True)
        df_total = df_total.drop_duplicates(subset=['date','home_team','away_team']).sort_values('date').reset_index(drop=True)
        print(f"✅ 기존 데이터 유지 + 26-27 업데이트: {len(df_total)}경기")

    df_total.to_csv(f"{MODEL_DIR}/all_matches.csv", index=False, encoding='utf-8-sig')

    # 3. 3시즌 혼합 스탯 (예측용) — 경기 수 구간별 가중치 + prestige 보정
    df_stats_current = calculate_blended_stats(df_total)
    df_stats_current.to_csv(f"{MODEL_DIR}/team_stats.csv", index=False, encoding='utf-8-sig')
    
    # 4. Feature 생성 (전체 데이터로 학습)
    df_stats_all = calculate_team_stats(df_total)
    print("\nFeature 생성 중...")
    df_features = build_features(df_total, df_stats_all)

    FEATURES = [
        'home_elo','away_elo','elo_diff',
        'home_form','away_form','form_diff',
        'home_avg_scored','away_avg_scored','home_avg_conceded','away_avg_conceded',
        'home_attack','away_attack','home_defense','away_defense',
        'home_win_rate','away_win_rate','win_rate_diff',
        'h2h_home_rate'
    ]

    X = df_features[FEATURES].dropna()
    y = df_features.loc[X.index, 'result']

    # 무한대 값 제거
    import numpy as np
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[X.index]

    # 이상치 확인
    print(f"학습 데이터: {len(X)}경기")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. 모델 학습
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=2000, random_state=42, C=0.1, solver='lbfgs')
    lr.fit(X_train_scaled, y_train)
    acc_lr = accuracy_score(y_test, lr.predict(X_test_scaled))
    print(f"✅ Logistic Regression: {acc_lr:.1%}")

    rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    print(f"✅ Random Forest: {acc_rf:.1%}")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    xgb = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.02,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                        random_state=42, eval_metric='mlogloss', verbosity=0)
    xgb.fit(X_train, y_train_enc)
    acc_xgb = accuracy_score(y_test, le.inverse_transform(xgb.predict(X_test)))
    print(f"✅ XGBoost: {acc_xgb:.1%}")

    # 6. 모델 저장
    joblib.dump(lr,     f"{MODEL_DIR}/logistic_regression.pkl")
    joblib.dump(rf,     f"{MODEL_DIR}/random_forest.pkl")
    joblib.dump(xgb,    f"{MODEL_DIR}/xgboost.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(le,     f"{MODEL_DIR}/label_encoder.pkl")

# UCL 토너먼트
    fetch_ucl_tournament()

    # 전체 일정 (일정 탭용)
    fetch_full_schedule()

    fetch_top_scorers()

    # 로고 자동 업데이트
    update_team_logos()

    # 팀 상세정보 (홈구장/스쿼드 등)
    fetch_team_info()

    # 우승 예측
    fetch_champion_predictions()

    # 정확도 저장
    import json as _json
    accuracy_data = {
        "logistic_regression": round(acc_lr * 100, 1),
        "random_forest": round(acc_rf * 100, 1),
        "xgboost": round(acc_xgb * 100, 1),
        "best": round(max(acc_lr, acc_rf, acc_xgb) * 100, 1),
        "total_matches": len(df_total),
        "training_matches": len(X),
        "updated_at": pd.Timestamp.now().isoformat(),
    }
    with open(f"{MODEL_DIR}/accuracy.json", 'w', encoding='utf-8') as f:
        _json.dump(accuracy_data, f, ensure_ascii=False, indent=2)
    print(f"✅ accuracy.json 저장 완료")
    
    print(f"\n🏆 업데이트 완료!")
    print(f"   데이터: {len(df_total)}경기")
    print(f"   최고 정확도: {max(acc_lr, acc_rf, acc_xgb):.1%}")

# 강등 위험 집계 인원수. FotData.html의 runSimulation()이 쓰는 releCount와 동일 기준으로
# 맞춰야 함 — 18팀 리그(분데스리가/리그앙)는 16위 강등 플레이오프 + 17·18위 직행강등이라
# "강등 확률"은 직행 2팀만 집계(PO 대상 16위는 프론트와 마찬가지로 제외). 20팀 리그는 하위 3팀.
RELEGATION_COUNT = {"BL1": 2, "FL1": 2}

def simulate_season(teams, df_stats, league_code, n_simulations=1000):
    """몬테카를로 시뮬레이션으로 리그 우승 예측"""
    import random

    rel_n = RELEGATION_COUNT.get(league_code, 3)
    win_counts = {team: 0 for team in teams}
    top4_counts = {team: 0 for team in teams}
    relegated_counts = {team: 0 for team in teams}
    
    for _ in range(n_simulations):
        # 시즌 일정 생성 (홈/원정 각 1번)
        points = {team: 0 for team in teams}
        
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                
                h = df_stats[df_stats['team'] == home]
                a = df_stats[df_stats['team'] == away]
                
                if h.empty or a.empty:
                    continue
                
                h = h.iloc[0]
                a = a.iloc[0]
                
                # ELO 기반 승률 계산 (prestige 반영)
                import random as _random
                h_prestige = h['prestige'] if 'prestige' in h.index and pd.notna(h['prestige']) else 0
                a_prestige = a['prestige'] if 'prestige' in a.index and pd.notna(a['prestige']) else 0
                home_elo = 1500 + (h['win_rate'] - 0.33) * 1000 + h_prestige + 70 + _random.gauss(0, 50)
                away_elo = 1500 + (a['win_rate'] - 0.33) * 1000 + a_prestige + _random.gauss(0, 50)
                
                exp_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
                exp_away = 1 / (1 + 10 ** ((home_elo - away_elo) / 400))
                
                # 무승부 확률 조정
                draw_prob = 0.25
                home_win_prob = exp_home * (1 - draw_prob)
                away_win_prob = exp_away * (1 - draw_prob)
                
                # 정규화
                total = home_win_prob + draw_prob + away_win_prob
                home_win_prob /= total
                draw_prob /= total
                away_win_prob /= total
                
                # 결과 결정
                rand = random.random()
                if rand < home_win_prob:
                    points[home] += 3
                elif rand < home_win_prob + draw_prob:
                    points[home] += 1
                    points[away] += 1
                else:
                    points[away] += 3
        
        # 순위 계산
        sorted_teams = sorted(points.items(), key=lambda x: x[1], reverse=True)
        
        # 우승
        win_counts[sorted_teams[0][0]] += 1
        
        # TOP 4
        for team, _ in sorted_teams[:4]:
            top4_counts[team] += 1
        
        # 강등 (리그별 직행 강등 인원수만큼 하위팀 집계)
        for team, _ in sorted_teams[-rel_n:]:
            relegated_counts[team] += 1
    
    results = []
    for team in teams:
        results.append({
            "team": team,
            "champion_prob": round(win_counts[team] / n_simulations * 100, 1),
            "top4_prob": round(top4_counts[team] / n_simulations * 100, 1),
            "relegation_prob": round(relegated_counts[team] / n_simulations * 100, 1),
        })
    
    return sorted(results, key=lambda x: x['champion_prob'], reverse=True)


def fetch_champion_predictions():
    """5대 리그 우승 예측 시뮬레이션"""
    import json
    print("\n[우승 예측] 시뮬레이션 중...")
    
    predictions = {}
    
    LEAGUE_TEAMS = {
        "PL":  "Premier League",
        "PD":  "LaLiga",
        "BL1": "Bundesliga",
        "SA":  "Serie A",
        "FL1": "Ligue 1",
    }
    
    df_all = pd.read_csv(f"{MODEL_DIR}/all_matches.csv")
    df_all['date'] = pd.to_datetime(df_all['date'])

    # 3시즌 혼합 + prestige 보정된 스탯을 미리 한 번만 계산 (5대 리그 전체)
    df_blended_all = calculate_blended_stats(df_all)

    for code, name in LEAGUE_TEAMS.items():
        print(f"  [{name}] 시뮬레이션 중...")
        
        # 26-27 시즌 해당 리그 팀만
        league_df = df_all[
            (df_all['league'] == code) &
            (df_all['date'] >= '2026-08-01')
        ]
        
        if league_df.empty:
            print(f"  ❌ {name} 데이터 없음")
            continue
        
        teams = list(set(league_df['home_team'].tolist() + league_df['away_team'].tolist()))
        
        # 미리 계산해둔 혼합 스탯에서 해당 리그 팀만 필터
        df_stats = df_blended_all[df_blended_all['team'].isin(teams)].reset_index(drop=True)
        league_teams = df_stats['team'].tolist()
        
        if len(league_teams) < 5:
            print(f"  ❌ {name} 팀 수 부족")
            continue
        
        results = simulate_season(league_teams, df_stats, code)
        predictions[code] = {
            "league": name,
            "teams": results
        }
        print(f"  ✅ {name} 완료 ({len(league_teams)}팀)")
    
    with open(f"{MODEL_DIR}/champion_predictions.json", 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"  ✅ champion_predictions.json 저장 완료")

def fetch_top_scorers():
    """리그별 득점왕/도움왕 데이터 수집"""
    import json
    print("\n[선수 스탯] 수집 중...")
    
    if not API_FOOTBALL_KEY:
        print("  ⚠️ API_FOOTBALL_KEY가 없어 건너뜀")
        return
    
    all_players = {"topscorers": {}, "topassists": {}}
    
    # 무료 플랜은 24-25 시즌만 가능
    SEASON = 2024
    
    for code, league_id in LEAGUE_IDS.items():
        league_name = LEAGUES_V2.get(code, code)
        
        # 무료 플랜은 EPL만 가능 (다른 리그는 유료)
        if code != "PL":
            print(f"  [{league_name}] 무료 플랜 미지원, 건너뜀")
            continue
        
        # 득점왕
        print(f"  [{league_name}] 득점왕 수집 중...")
        try:
            res = requests.get(
                f"{API_FOOTBALL_URL}/players/topscorers",
                headers=API_FOOTBALL_HEADERS,
                params={"league": league_id, "season": SEASON}
            )
            if res.status_code == 200:
                data = res.json()
                all_players["topscorers"][code] = data.get("response", [])
                print(f"    ✅ {len(data.get('response', []))}명")
            else:
                print(f"    ❌ 오류: {res.status_code}")
        except Exception as e:
            print(f"    ❌ 예외: {e}")
        
        time.sleep(2)
        
        # 도움왕
        print(f"  [{league_name}] 도움왕 수집 중...")
        try:
            res = requests.get(
                f"{API_FOOTBALL_URL}/players/topassists",
                headers=API_FOOTBALL_HEADERS,
                params={"league": league_id, "season": SEASON}
            )
            if res.status_code == 200:
                data = res.json()
                all_players["topassists"][code] = data.get("response", [])
                print(f"    ✅ {len(data.get('response', []))}명")
            else:
                print(f"    ❌ 오류: {res.status_code}")
        except Exception as e:
            print(f"    ❌ 예외: {e}")
        
        time.sleep(2)
    
    # 파일 저장
    with open(f"{MODEL_DIR}/players.json", 'w', encoding='utf-8') as f:
        json.dump(all_players, f, ensure_ascii=False, indent=2)
    print(f"  ✅ players.json 저장 완료")

if __name__ == "__main__":
    main()