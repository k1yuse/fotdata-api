"""API-Football 응답 → 화면용 형태 변환(2026-10-06 update_data.py에서 분리 — 서버 main.py도 경기 상세·라인업·선수 경력을
요청 때 바로 받아 같은 형태로 바꿔야 해서 둘이 같이 씀). 요청·저장은 하지 않음."""
import re

# ── 경기 상세(API-Football /fixtures?id=) → 화면용 형태 (2026-09-30) ──
# 경기 결과 창의 [요약 | 라인업 | 통계] 탭이 쓰는 형태. 수집(어느 경기를 언제 받을지)은 API-Football Pro 결제 후에 붙이고,
# 결과는 fotdata_model/match_details.json {"홈|원정|YYYY-MM-DD": 이 함수 결과}에 저장 → main.py /match/detail이 내려줌.
# /fixtures?id= 한 번이면 이벤트·라인업·팀 통계·선수별 스탯(평점)이 같이 옴(무료 키로 24-25 경기에서 형태 확인함).
AF_STAT_KEYS = [   # (API-Football 이름, 키, 화면 라벨, 단위, 좋은 쪽)
    ("Ball Possession", "possession", "점유율", "%", None),
    ("expected_goals", "xg", "기대 득점(xG)", "", "high"),
    ("Total Shots", "shots", "슈팅", "", "high"),
    ("Shots on Goal", "shots_on", "유효 슈팅", "", "high"),
    ("Shots insidebox", "shots_box", "박스 안 슈팅", "", "high"),
    ("Corner Kicks", "corners", "코너킥", "", "high"),
    ("Total passes", "passes", "패스", "", None),
    ("Passes %", "pass_acc", "패스 성공률", "%", "high"),
    ("Goalkeeper Saves", "saves", "선방", "", None),
    ("Fouls", "fouls", "파울", "", "low"),
    ("Offsides", "offsides", "오프사이드", "", "low"),
    ("Yellow Cards", "yellow", "경고", "", "low"),
    ("Red Cards", "red", "퇴장", "", "low"),
]

def af_match_detail(fx):
    """API-Football 경기 하나(/fixtures?id= 응답의 response[0]) → 화면용 dict(점수·팀 이름은 우리 일정에 이미 있어서 안 넣음)"""
    hid = fx["teams"]["home"]["id"]
    side = lambda tid: "home" if tid == hid else "away"
    num = lambda v: None if v in (None, "") else float(str(v).rstrip("%"))
    pinfo = {}   # 선수 ID → 평점·출전 시간·사진
    pstats = {"home": [], "away": []}   # 선수별 이 경기 기록(시즌 기록·선수 카드는 이걸 합산 — /players 시즌 기록은 다른 팀 기록이 섞여 와서 안 씀, 5.19)
    for tp in fx.get("players") or []:
        sd = side((tp.get("team") or {}).get("id"))
        for p in tp.get("players") or []:
            st = (p.get("statistics") or [{}])[0] or {}
            g = st.get("games") or {}
            pinfo[p["player"]["id"]] = {"rating": round(float(g["rating"]), 1) if g.get("rating") else None,
                                        "minutes": g.get("minutes"), "photo": p["player"].get("photo")}
            if not g.get("minutes"):
                continue
            gl, sh, ps, tk, du, dr, fo, cd, pn = (st.get(k) or {} for k in ("goals", "shots", "passes", "tackles", "duels", "dribbles", "fouls", "cards", "penalty"))
            pac = ps.get("accuracy")
            pstats[sd].append({"id": p["player"]["id"], "name": p["player"].get("name"), "photo": p["player"].get("photo"),
                               "num": g.get("number"), "pos": g.get("position"), "cap": bool(g.get("captain")), "sub": bool(g.get("substitute")),
                               "min": g["minutes"], "rt": float(g["rating"]) if g.get("rating") else None,
                               "g": gl.get("total") or 0, "a": gl.get("assists") or 0, "con": gl.get("conceded"), "sv": gl.get("saves"),
                               "sh": sh.get("total"), "sho": sh.get("on"), "ps": ps.get("total"), "kp": ps.get("key"),
                               "pac": int(pac) if str(pac or "").isdigit() else None,   # 경기 기록의 accuracy는 정확한 패스 "수"(시즌 기록은 %)
                               "tk": tk.get("total"), "bl": tk.get("blocks"), "in": tk.get("interceptions"), "du": du.get("total"), "dw": du.get("won"),
                               "da": dr.get("attempts"), "ds": dr.get("success"), "fc": fo.get("committed"), "fd": fo.get("drawn"),
                               "y": cd.get("yellow") or 0, "r": cd.get("red") or 0, "pns": pn.get("scored"), "pnm": pn.get("missed")})
    starters = {x["player"]["id"] for l in fx.get("lineups") or [] for x in l.get("startXI") or []}
    events, tally = [], {}
    def mark(pid, k, minute=None):
        """선수별 표시: 골·도움·카드는 횟수, 교체(off/on)는 그 분"""
        if not pid:
            return
        t = tally.setdefault(pid, {})
        t[k] = minute if k in ("off", "on") else t.get(k, 0) + 1
    for e in fx.get("events") or []:
        t, d = e.get("type"), e.get("detail") or ""
        kind = {"Goal": {"Own Goal": "own_goal", "Penalty": "pen_goal", "Missed Penalty": "missed_pen"}.get(d, "goal"),
                "Card": "red" if "Red" in d else "second_yellow" if "Second" in d else "yellow",
                "subst": "sub", "Var": "var"}.get(t, (t or "").lower())
        p, a = e.get("player") or {}, e.get("assist") or {}
        if kind == "sub" and p.get("id") not in starters and a.get("id") in starters:
            p, a = a, p   # 교체는 player = 나간 선수, assist = 들어온 선수가 원칙인데 반대로 오는 경우가 있어 선발 명단으로 바로잡음
        tm = e.get("time") or {}
        events.append({"min": tm.get("elapsed"), "extra": tm.get("extra"), "side": side(e["team"]["id"]), "type": kind,
                       "player": p.get("name"), "player_id": p.get("id"), "other": a.get("name"), "other_id": a.get("id"),
                       "detail": d, "comment": e.get("comments")})
        m = tm.get("elapsed")
        if kind in ("goal", "pen_goal"):
            mark(p.get("id"), "goals"); mark(a.get("id"), "assists")
        elif kind == "own_goal":
            mark(p.get("id"), "own_goals")
        elif kind in ("yellow", "second_yellow", "red"):
            mark(p.get("id"), kind)
        elif kind == "sub":
            mark(p.get("id"), "off", m); mark(a.get("id"), "on", m)
    def player(x):
        pl = x["player"]
        info = pinfo.get(pl["id"], {})
        return {"id": pl["id"], "name": pl.get("name"), "number": pl.get("number"), "pos": pl.get("pos"), "grid": pl.get("grid"),
                "rating": info.get("rating"), "minutes": info.get("minutes"), "photo": info.get("photo"), **tally.get(pl["id"], {})}
    lineups = {}
    for l in fx.get("lineups") or []:
        lineups[side(l["team"]["id"])] = {
            "formation": l.get("formation"), "coach": (l.get("coach") or {}).get("name"),
            "colors": (l.get("team") or {}).get("colors"),
            "start": [player(x) for x in l.get("startXI") or []],
            "subs": [player(x) for x in l.get("substitutes") or []]}
    stats, raw = [], {side(s["team"]["id"]): {x["type"]: x["value"] for x in s.get("statistics") or []} for s in fx.get("statistics") or []}
    for src, key, label, unit, better in AF_STAT_KEYS:
        hv, av = num(raw.get("home", {}).get(src)), num(raw.get("away", {}).get(src))
        if hv is None and av is None:
            continue
        stats.append({"key": key, "label": label, "unit": unit, "better": better, "home": hv or 0, "away": av or 0})
    rated = [(p["rating"], s, p) for s, l in lineups.items() for p in l["start"] + l["subs"] if p.get("rating")]
    best = max(rated, key=lambda r: r[0]) if rated else None
    st = (fx.get("fixture") or {}).get("status") or {}
    return {"source": "api-football", "fixture_id": (fx.get("fixture") or {}).get("id"),
            "status": {"short": st.get("short"), "elapsed": st.get("elapsed"), "extra": st.get("extra")},
            "referee": (fx.get("fixture") or {}).get("referee"), "venue": ((fx.get("fixture") or {}).get("venue") or {}).get("name"),
            "events": events, "lineups": lineups, "stats": stats, "pstats": pstats,
            "potm": {"id": best[2]["id"], "name": best[2]["name"], "side": best[1], "rating": best[0], "photo": best[2].get("photo")} if best else None}

# ── 선수 시즌 기록·부상자 → 화면용 형태 (2026-09-30, 결제 후 수집에서 사용) ──
# af_squads.json {"우리 팀 이름": {"season", "league", "updated", "players": [af_player_row…]}} → /team/squad (선수 카드·베스트 11·주요 선수)
# match_previews.json {"홈|원정|YYYY-MM-DD": {"lineups", "injuries": {"home": [...], "away": [...]}}} → /match/preview
AF_POS = {"Goalkeeper": "GK", "Defender": "DF", "Midfielder": "MF", "Attacker": "FW"}

def af_player_row(p, league_id=None):
    """/players 응답 한 명 → 프로필 + 그 리그 시즌 기록(여러 대회 기록이 오면 league_id 것)"""
    pl = p.get("player") or {}
    sts = p.get("statistics") or [{}]
    s = next((x for x in sts if league_id and (x.get("league") or {}).get("id") == league_id), sts[0]) or {}
    g, gl, sh, ps = s.get("games") or {}, s.get("goals") or {}, s.get("shots") or {}, s.get("passes") or {}
    tk, du, dr, fo, cd, pn = (s.get(k) or {} for k in ("tackles", "duels", "dribbles", "fouls", "cards", "penalty"))
    cm = lambda v: int(re.sub(r"\D", "", str(v))) if v and re.search(r"\d", str(v)) else None   # "182 cm"·"182" → 182
    # 목록·선수 카드용 전체 이름: "B. Saka" + firstname "Bukayo Ayoyinka Temidayo" → "Bukayo Saka"(이니셜이 맞는 단어). 줄인 이름이 아니면 그대로
    full, m = pl.get("name"), re.match(r"^(\w)\.\s+(.+)$", pl.get("name") or "")
    if m and pl.get("firstname"):
        toks = pl["firstname"].split()
        full = f"{next((t for t in toks if t[:1].upper() == m.group(1).upper()), toks[0])} {m.group(2)}"
    return {
        "id": pl.get("id"), "name": pl.get("name"), "full_name": full, "firstname": pl.get("firstname"), "lastname": pl.get("lastname"),
        "age": pl.get("age"), "birth_date": (pl.get("birth") or {}).get("date"), "birth_place": (pl.get("birth") or {}).get("place"),
        "nationality": pl.get("nationality"), "height": cm(pl.get("height")), "weight": cm(pl.get("weight")),
        "photo": pl.get("photo"), "injured": bool(pl.get("injured")),
        "pos": AF_POS.get(g.get("position")), "number": g.get("number"), "captain": bool(g.get("captain")),
        "apps": g.get("appearences") or 0, "starts": g.get("lineups") or 0, "minutes": g.get("minutes") or 0,
        "rating": round(float(g["rating"]), 2) if g.get("rating") else None,
        "goals": gl.get("total") or 0, "assists": gl.get("assists") or 0, "conceded": gl.get("conceded"), "saves": gl.get("saves"),
        "shots": sh.get("total"), "shots_on": sh.get("on"), "passes": ps.get("total"), "key_passes": ps.get("key"), "pass_acc": ps.get("accuracy"),
        "tackles": tk.get("total"), "interceptions": tk.get("interceptions"), "blocks": tk.get("blocks"),
        "duels": du.get("total"), "duels_won": du.get("won"), "dribbles": dr.get("attempts"), "dribbles_won": dr.get("success"),
        "fouls": fo.get("committed"), "fouled": fo.get("drawn"), "yellow": cd.get("yellow") or 0, "red": cd.get("red") or 0,
        "pen_scored": pn.get("scored"), "pen_missed": pn.get("missed"),
    }

AF_PCT_STATS = ["goals", "assists", "shots", "shots_on", "key_passes", "passes", "dribbles_won", "tackles", "interceptions", "blocks", "duels_won", "saves", "conceded"]

def add_percentiles(players, min_minutes=450):
    """선수 카드 순위 막대: 같은 리그·같은 포지션(GK/DF/MF/FW)에서 min_minutes 이상 뛴 선수 중 90분당 수치 백분위(0~100) → p["pct"].
    실점은 적을수록 좋아서 뒤집음. 표본이 5명 미만인 포지션은 계산 안 함(순위가 의미 없음)"""
    from collections import defaultdict
    groups = defaultdict(list)
    for p in players:
        if (p.get("minutes") or 0) >= min_minutes and p.get("pos"):
            groups[p["pos"]].append(p)
    for ps in groups.values():
        if len(ps) < 5:
            continue
        for k in AF_PCT_STATS:
            if all(p.get(k) is None for p in ps):
                continue
            per90 = [((p.get(k) or 0) / p["minutes"] * 90) for p in ps]
            for p, v in zip(ps, per90):
                below, eq = sum(x < v for x in per90), sum(x == v for x in per90)
                pc = round((below + 0.5 * eq) / len(per90) * 100)
                p.setdefault("pct", {})[k] = 100 - pc if k == "conceded" else pc
    return players

# 선수 카드 경력·트로피·부상 이력(/players/teams·/trophies·/sidelined, 선수당 요청 3번) → af_profiles.json {선수 ID: ...} → /player/profile
# 사카 샘플로 형태 확인(2026-09-30): 경력은 날짜 없이 "뛴 시즌 목록"만, 유스 팀(U18 등) 섞임 / 트로피엔 친선 대회(Emirates Cup 5회·Florida Cup·
# MLS All-Star)와 유스 대회(FA Youth Cup 등)가 섞여 있어 그대로 세면 부풀려짐 → 친선은 빼고 유스는 표시만
AF_FRIENDLY_CUPS = {"Emirates Cup", "Florida Cup", "MLS All-Star", "Trofeo Joan Gamper", "Club Friendlies", "Audi Cup", "Premier League Summer Series",
                    "International Champions Cup", "Trofeo Santiago Bernabéu", "Uhrencup", "Trofeo Teresa Herrera", "Franz Beckenbauer Supercup",
                    "Trofeo Luigi Berlusconi", "Trofeo Colombino", "Trofeo Carranza", "Friendlies", "Friendlies Clubs"}
AF_YOUTH_RE = re.compile(r"\bU\d{2}\b|\bYouth\b|Premier League 2|Primavera|Juvenil|Junior|Reserve", re.I)

def af_player_profile(teams, trophies, sidelined, nationality=None, current_season=None):
    cur = current_season or 2026
    career = []
    for x in teams or []:
        ss, tm = x.get("seasons") or [], x.get("team") or {}
        if not ss:
            continue
        name = tm.get("name") or ""
        career.append({"team": name, "team_id": tm.get("id"), "logo": tm.get("logo"), "from": min(ss), "to": None if max(ss) >= cur else max(ss),
                       "seasons": len(ss), "youth": bool(re.search(r"\bU\d{2}\b", name)),
                       "national": bool(nationality) and (name == nationality or name.startswith(nationality + " "))})
    career.sort(key=lambda c: (c["to"] is not None, -(c["to"] or 9999), -c["from"]))   # 지금 팀 → 최근에 떠난 팀 순
    tro = []
    for t in trophies or []:
        if t.get("league") in AF_FRIENDLY_CUPS:
            continue
        pl = {"Winner": "winner", "2nd Place": "runner_up"}.get(t.get("place"), t.get("place"))
        tro.append({"league": t.get("league"), "country": t.get("country"), "season": t.get("season"), "place": pl, "youth": bool(AF_YOUTH_RE.search(t.get("league") or ""))})
    # 같은 트로피가 시즌 없이 한 번 더 오는 경우가 있음(사카: FA Cup 우승 "2019/2020" + 시즌 없는 FA Cup 우승) → 시즌 있는 기록이 있으면 시즌 없는 쪽은 버림
    dated = {(t["league"], t["place"]) for t in tro if t["season"]}
    tro = [t for t in tro if t["season"] or (t["league"], t["place"]) not in dated]
    sd = [{"type": x.get("type"), "start": x.get("start"), "end": x.get("end")}
          for x in sorted(sidelined or [], key=lambda x: x.get("start") or "", reverse=True)[:10]]
    return {"career": career, "trophies": tro, "sidelined": sd}

def af_injury_rows(resp, team_id):
    """/injuries 응답 → 그 팀 결장자 [{id, name, photo, status: out(결장)|doubtful(출전 불투명), reason}]"""
    out = []
    for x in resp or []:
        if (x.get("team") or {}).get("id") != team_id:
            continue
        pl = x.get("player") or {}
        out.append({"id": pl.get("id"), "name": pl.get("name"), "photo": pl.get("photo"),
                    "status": "doubtful" if pl.get("type") == "Questionable" else "out", "reason": pl.get("reason")})
    return out



# ── 저장 파일 이름·시즌 기록 합산(2026-10-06 — 결제 후 수집) ──
def _ascii(s):
    import unicodedata
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()

def md_file(home, away, date):
    """경기 상세 파일 이름(fotdata_model/match_details/<이름>) — 날짜(일정의 UTC 날짜)_홈_원정"""
    slug = lambda t: re.sub(r"[^a-z0-9]+", "-", _ascii(t).lower()).strip("-")
    return f"{date[:10]}_{slug(home)}_{slug(away)}.json.gz"   # gzip(경기당 23KB → 4KB 안팎 — 시즌 끝나면 1,900경기라 저장소 무게 때문에)

def md_write(path, obj):
    import gzip, json
    with gzip.GzipFile(path, "wb", mtime=0) as f:   # mtime=0: 같은 내용이면 같은 파일(커밋에 괜히 안 잡히게)
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

def md_read(path):
    import gzip, json
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))

_SUM_KEYS = {"g": "goals", "a": "assists", "con": "conceded", "sv": "saves", "sh": "shots", "sho": "shots_on", "ps": "passes", "kp": "key_passes",
             "tk": "tackles", "bl": "blocks", "in": "interceptions", "du": "duels", "dw": "duels_won", "da": "dribbles", "ds": "dribbles_won",
             "fc": "fouls", "fd": "fouled", "y": "yellow", "r": "red", "pns": "pen_scored", "pnm": "pen_missed"}
_POS1 = {"G": "GK", "D": "DF", "M": "MF", "F": "FW"}

def af_season_players(details, team, profiles=None):
    """이 팀의 리그 경기 상세들(오래된 것 → 최근 순, 각 {home_team, away_team, date, home_goals, away_goals, detail})을 합산해
    선수 카드·베스트 11 형태(af_player_row와 같은 키 + matches 최근 경기)로. profiles = {선수 ID: 프로필(나이·키·국적·사진 등)}"""
    from collections import Counter
    acc = {}
    for m in details:
        home = m["home_team"] == team
        sd = "home" if home else "away"
        gf, ga = (m["home_goals"], m["away_goals"]) if home else (m["away_goals"], m["home_goals"])
        for x in ((m.get("detail") or m).get("pstats") or {}).get(sd) or []:
            a = acc.setdefault(x["id"], {"id": x["id"], "name": x.get("name"), "photo": x.get("photo"), "apps": 0, "starts": 0, "minutes": 0,
                                         "_rt": [], "_pos": Counter(), "_pac": 0, "_pst": 0, "matches": []})
            a["apps"] += 1; a["starts"] += 0 if x.get("sub") else 1; a["minutes"] += x.get("min") or 0
            if x.get("rt"): a["_rt"].append(x["rt"])
            if x.get("pos"): a["_pos"][x["pos"]] += 1
            if x.get("num") is not None: a["number"] = x["num"]
            a["captain"] = bool(x.get("cap"))
            for k, kk in _SUM_KEYS.items():
                if x.get(k) is not None:
                    a[kk] = (a.get(kk) or 0) + x[k]
            if x.get("pac") is not None and x.get("ps"):
                a["_pac"] += x["pac"]; a["_pst"] += x["ps"]
            a["matches"].append({"date": m["date"][:10], "opp": m["away_team"] if home else m["home_team"], "home": home, "gf": gf, "ga": ga,
                                 "minutes": x.get("min"), "goals": x.get("g") or 0, "assists": x.get("a") or 0, "yellow": x.get("y") or 0,
                                 "red": x.get("r") or 0, "rating": round(x["rt"], 1) if x.get("rt") else None})
    out = []
    profiles = profiles or {}
    for pid, a in acc.items():
        pr = profiles.get(pid) or profiles.get(str(pid)) or {}
        row = {**{k: pr.get(k) for k in ("name", "full_name", "firstname", "lastname", "age", "birth_date", "birth_place", "nationality",
                                         "height", "weight", "photo", "injured")}, **{k: v for k, v in a.items() if not k.startswith("_")}}
        row["name"] = pr.get("name") or a["name"]; row["photo"] = pr.get("photo") or a["photo"]
        row["full_name"] = pr.get("full_name") or row["name"]
        # 포지션: 시즌 프로필(Attacker 등)을 우선 — 경기 기록의 포지션은 라인업 줄 기준이라 4-2-3-1 측면 공격수(사카 등)가 미드필더로 잡힘
        row["pos"] = pr.get("pos") or (_POS1.get(a["_pos"].most_common(1)[0][0]) if a["_pos"] else None)
        row["rating"] = round(sum(a["_rt"]) / len(a["_rt"]), 2) if a["_rt"] else None
        row["pass_acc"] = round(a["_pac"] / a["_pst"] * 100) if a["_pst"] else None
        for k in ("goals", "assists", "yellow", "red"):
            row[k] = row.get(k) or 0
        row["matches"] = a["matches"][::-1][:10]
        out.append(row)
    # 이번 시즌 리그 경기를 안 뛴 선수도 선수 카드는 열리게(프로필만, 기록 0)
    for pid, pr in profiles.items():
        if int(pid) not in acc:
            out.append({**pr, "id": int(pid), "apps": 0, "starts": 0, "minutes": 0, "goals": 0, "assists": 0, "yellow": 0, "red": 0, "rating": None, "matches": []})
    return out

def af_profile_row(p):
    """/players 응답 한 명 → 프로필만(시즌 기록은 경기 상세 합산으로 — 이 응답의 statistics는 다른 팀 기록이 섞일 수 있어 안 씀)"""
    r = af_player_row(p)
    keep = ("id", "name", "full_name", "firstname", "lastname", "age", "birth_date", "birth_place", "nationality", "height", "weight", "photo", "injured", "pos")
    return {k: r.get(k) for k in keep}
