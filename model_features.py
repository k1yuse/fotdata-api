"""예측 모델 입력(피처) 규칙 — 새벽 학습(update_data.build_point_in_time_features)과
서버의 끝난 경기 실시간 반영(main._live_sync)이 같이 씀.

두 곳의 계산이 다르면 "학습한 입력 ≠ 예측 입력"이 되므로 이 파일 하나에만 둠.
서버(main.py)가 무거운 update_data(xgboost 등)를 불러오지 않도록 numpy만 쓰는 가벼운 파일로 분리(2026-10-10).
"""
import numpy as np

ELO_K = 20
ELO_HOME_ADVANTAGE = 70
FORM_N, GOALS_N, STATS_N, H2H_N = 5, 10, 38, 10
ELO_TRAIL_N = 20   # 예측 결과 화면 "파워 레이팅 추이" 그래프용으로 저장하는 최근 경기 수
RESULT_PTS = {'H': (3, 0), 'D': (1, 1), 'A': (0, 3)}   # 결과 → (홈 승점, 원정 승점)


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


def elo_change(elo_home, elo_away, result):
    """한 경기 뒤 홈 팀 ELO 변화량(원정은 같은 양만큼 반대로) — 홈 어드밴티지 포함 기대승률 대비 실제 결과"""
    expected_home = 1 / (1 + 10 ** ((elo_away - (elo_home + ELO_HOME_ADVANTAGE)) / 400))
    actual_home = {'H': 1.0, 'D': 0.5, 'A': 0.0}[result]
    return ELO_K * (actual_home - expected_home)
