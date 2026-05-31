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

    # 풋몹 기준 순서로 재정렬
    PO_ORDER = [
        ('Monaco', 'Paris'), ('Galatasaray', 'Juventus'),
        ('Benfica', 'Real Madrid'), ('Dortmund', 'Atalanta'),
        ('Qarab', 'Newcastle'), ('Brugge', 'Atlético'),
        ('Bodø', 'Internazionale'), ('Olympiakos', 'Leverkusen'),
    ]
    R16_ORDER = [
        ('Paris', 'Chelsea'), ('Galatasaray', 'Liverpool'),
        ('Real Madrid', 'Manchester City'), ('Atalanta', 'Bayern'),
        ('Newcastle', 'Barcelona'), ('Atlético', 'Tottenham'),
        ('Bodø', 'Sporting'), ('Leverkusen', 'Arsenal'),
    ]
    QF_ORDER = [
        ('Paris', 'Liverpool'), ('Real Madrid', 'Bayern'),
        ('Barcelona', 'Atlético'), ('Sporting', 'Arsenal'),
    ]

    def find_and_sort(stage_data, order):
        result = []
        for t1k, t2k in order:
            for m in stage_data:
                names = [m['team1'], m['team2']]
                if any(t1k in n for n in names) and any(t2k in n for n in names):
                    if t1k not in m['team1']:
                        m['team1'], m['team2'] = m['team2'], m['team1']
                        m['team1_goals'], m['team2_goals'] = m['team2_goals'], m['team1_goals']
                        m['team1_logo'], m['team2_logo'] = m['team2_logo'], m['team1_logo']
                    result.append(m)
                    break
        # 순서에 없는 새 팀(4강, 결승 등)은 그냥 뒤에 추가
        ordered_keys = set()
        for m in result:
            ordered_keys.add(tuple(sorted([m['team1'], m['team2']])))
        for m in stage_data:
            k = tuple(sorted([m['team1'], m['team2']]))
            if k not in ordered_keys:
                result.append(m)
        return result

    stages['PLAYOFFS'] = find_and_sort(stages['PLAYOFFS'], PO_ORDER)
    stages['LAST_16'] = find_and_sort(stages['LAST_16'], R16_ORDER)
    stages['QUARTER_FINALS'] = find_and_sort(stages['QUARTER_FINALS'], QF_ORDER)

    with open(f"{MODEL_DIR}/ucl_tournament.json", 'w', encoding='utf-8') as f:
        json.dump(stages, f, ensure_ascii=False, indent=2)
    print(f"  ✅ UCL 토너먼트 저장 완료")

def main():
    print("=== FotData 자동 업데이트 시작 ===")

    # 1. 데이터 수집 (23-24 + 24-25 + 25-26)
    all_dfs = []
    for season in [2023, 2024, 2025]:
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
        # 25-26 이전 데이터는 기존 것 유지, 25-26만 새로 교체
        df_old = df_existing[df_existing['date'] < '2025-08-01']
        df_new_2526 = df_total[df_total['date'] >= '2025-08-01']
        df_total = pd.concat([df_old, df_new_2526], ignore_index=True)
        df_total = df_total.drop_duplicates(subset=['date','home_team','away_team']).sort_values('date').reset_index(drop=True)
        print(f"✅ 기존 데이터 유지 + 25-26 업데이트: {len(df_total)}경기")

    df_total.to_csv(f"{MODEL_DIR}/all_matches.csv", index=False, encoding='utf-8-sig')

    # 3. 25-26 시즌 스탯 (예측용)
    df_2526 = df_total[df_total['date'] >= '2025-08-01']
    df_stats_2526 = calculate_team_stats(df_2526)
    df_stats_2526.to_csv(f"{MODEL_DIR}/team_stats.csv", index=False, encoding='utf-8-sig')

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
    
    fetch_top_scorers()

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