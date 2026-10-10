"""
경기별 링크 공유 썸네일(1200×630 JPEG) — main.py의 /og/match/{slug}.jpg가 부름 (2026-09-30)
카톡·페북·X 같은 링크 미리보기는 JS를 실행하지 않아서, 예측 결과를 보여주려면 서버가 이미지를 그려야 함.
디자인(2026-10-10, 앱 theme-a 톤): 밤하늘 남색 + 위쪽 라벤더 빛 + 오로라 와이어프레임 공(generate_bg_ball) + 원근 경기장 선.
브랜드(로고 공) = 오로라(하늘색 → 라벤더 → 복숭아), 홈 파랑 · 무 회색 · 원정 주황은 데이터에만. 아래 iris_* 도우미는
generate_og_image.py·generate_icons.py도 같이 씀.
속도(Render 무료 CPU는 로컬보다 20~40배 느림 — 2배 전체 캔버스로 그렸더니 카드 한 장에 1~4초라 미리보기 봇을 놓칠 수 있었음):
경기와 상관없는 배경(빛번짐·공·경기장·로고·워드마크)과 팀 색 링은 2배로 그려 줄인 것을 한 번만 만들어 재사용하고,
경기마다 바뀌는 건 1배 캔버스에 바로 — 테두리가 매끄러워야 하는 칩·막대만 작은 조각을 2배로 그려 줄여서 붙임.
폰트: fonts/FotDataCardSans-*.otf = Pretendard 1.3.9(OFL)에서 한글 2,350자 + 라틴만 추린 것
(OFL상 수정본은 원래 이름을 못 써서 이름만 바꿈 — fonts/OFL.txt). 여기 없는 글자는 두부(□)로 나오니 주의.
"""
import io
import math
import os
from functools import lru_cache

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import generate_bg_ball as ball

OUT_W, OUT_H = 1200, 630
S = 2
W, H = OUT_W * S, OUT_H * S
FONT_DIR = os.path.join(os.path.dirname(__file__), 'fonts')
BLUE, ORANGE, GREY = (88, 166, 255), (240, 136, 62), (142, 154, 179)
TEXT, MUTED = (238, 242, 248), (127, 138, 160)
INK2 = (185, 195, 212)
HOME_TX, AWAY_TX = (143, 193, 255), (255, 184, 137)   # 앱 theme-a 홈/원정 글자색(칩 글자 — 막대보다 밝게)
IRIS = ((156, 201, 255), (185, 168, 255), (255, 191, 152))   # 앱 --iris(#9cc9ff → #b9a8ff → #ffbf98)


def iris(t):
    """0~1 → 오로라 색(RGB)"""
    t = min(1.0, max(0.0, t)) * 2
    i = min(1, int(t))
    f, a, b = t - i, IRIS[i], IRIS[i + 1]
    return tuple(int(round(a[j] + (b[j] - a[j]) * f)) for j in range(3))


def iris_paint(mask, diag=False):
    """흑백 마스크(L) 모양대로 오로라를 칠한 RGBA. diag=True면 정사각형 대각선(로고 공 — SVG x1=0 y1=0 x2=1 y2=1),
    아니면 CSS 115°(글자·막대)"""
    w, h = mask.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if diag:
        t = (xx / max(1, w - 1) + yy / max(1, h - 1)) / 2
    else:
        t = (xx * 0.906 + yy * 0.423) / max(1.0, w * 0.906 + h * 0.423)
    t = np.clip(t, 0, 1) * 2
    lo = t < 1
    st = np.array(IRIS, dtype=np.float32)
    f = np.where(lo, t, t - 1)[..., None]
    rgb = np.where(lo[..., None], st[0] + (st[1] - st[0]) * f, st[1] + (st[2] - st[1]) * f)
    out = np.dstack([rgb, np.asarray(mask, dtype=np.float32)]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, 'RGBA')


LOGO_PENT = [(12, 7.7), (16.09, 10.67), (14.53, 15.48), (9.47, 15.48), (7.91, 10.67)]
LOGO_SPOKES = [((12, 7.7), (12, 3.8)), ((16.09, 10.67), (19.8, 9.47)), ((14.53, 15.48), (16.82, 18.63)),
               ((9.47, 15.48), (7.18, 18.63)), ((7.91, 10.67), (4.2, 9.47))]


def logo_mask(size, sw=1.4):
    """사이트 로고 공(viewBox 24: 원 r9 + 오각형 + 바깥 5선, 둥근 끝) 흑백 마스크 — 4배로 그려 줄임"""
    k4 = 4
    big = int(size * k4)
    m = Image.new('L', (big, big), 0)
    d = ImageDraw.Draw(m)
    k = big / 24
    w = max(2, int(round(sw * k)))
    P = lambda x, y: (x * k, y * k)
    r = 9 * k + w / 2
    d.ellipse([12 * k - r, 12 * k - r, 12 * k + r, 12 * k + r], outline=255, width=w)
    pent = [P(*p) for p in LOGO_PENT]
    d.line(pent + [pent[0]], fill=255, width=w, joint='curve')
    for a, b in LOGO_SPOKES:
        d.line([P(*a), P(*b)], fill=255, width=w)
    for x, y in pent + [P(*b) for _, b in LOGO_SPOKES]:
        d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=255)
    return m.resize((int(size), int(size)), Image.LANCZOS)


def draw_logo_iris(img, x, y, size, sw=1.4):
    img.alpha_composite(iris_paint(logo_mask(size, sw), diag=True), (int(x), int(y)))


def iris_text(img, xy, text, fnt, anchor='la'):
    """오로라 글자(마스크에 글자를 쓰고 그 범위만큼 그라데이션)"""
    m = Image.new('L', img.size, 0)
    ImageDraw.Draw(m).text(xy, text, font=fnt, fill=255, anchor=anchor)
    box = m.getbbox()
    if not box:
        return
    img.alpha_composite(iris_paint(m.crop(box)), box[:2])


@lru_cache(maxsize=48)
def font(px, weight='Bold'):
    """px = 실제 픽셀 크기(1배 캔버스엔 그대로, 2배로 그리는 조각엔 ×S로 넘김)"""
    return ImageFont.truetype(os.path.join(FONT_DIR, f'FotDataCardSans-{weight}.otf'), int(round(px)))


def _glow(img, cx, cy, r, rgb, a):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    k = np.clip(1 - d, 0, 1) ** 2 * a
    img[...] = img * (1 - k[..., None]) + np.array(rgb, dtype=np.float32) * k[..., None]


def _pitch_segments():
    """FotData.html pitchPerspectiveSegments()와 같은 원근 경기장(0~1 좌표, side: home/away/mid)"""
    L, Wd, segs = 52.5, 34, []
    def proj(x, z):
        t = (z + Wd) / (2 * Wd)
        return 0.5 + (x / L) * 0.46 * (0.7 + 0.3 * t), 0.1 + t * 0.8
    def side(x1, x2):
        m = (x1 + x2) / 2
        return 'home' if m < -0.5 else 'away' if m > 0.5 else 'mid'
    def line(x1, z1, x2, z2, s=None):
        a, b = proj(x1, z1), proj(x2, z2)
        segs.append((a[0], a[1], b[0], b[1], s or side(x1, x2)))
    def split(x1, z1, x2, z2, n=6):
        for k in range(n):
            line(x1 + (x2 - x1) * k / n, z1 + (z2 - z1) * k / n, x1 + (x2 - x1) * (k + 1) / n, z1 + (z2 - z1) * (k + 1) / n)
    def rect(x1, z1, x2, z2):
        split(x1, z1, x2, z1); split(x2, z1, x2, z2); split(x2, z2, x1, z2); split(x1, z2, x1, z1)
    rect(-L, -Wd, L, Wd)
    line(0, -Wd, 0, Wd, 'mid')
    for k in range(36):
        a0, a1 = k / 36 * math.pi * 2, (k + 1) / 36 * math.pi * 2
        line(9.15 * math.cos(a0), 9.15 * math.sin(a0), 9.15 * math.cos(a1), 9.15 * math.sin(a1), 'mid')
    for s in (-1, 1):
        rect(s * L, -20.16, s * (L - 16.5), 20.16)
        rect(s * L, -9.16, s * (L - 5.5), 9.16)
    return segs


def _draw_ball(img, cx, cy, size, alpha):
    segs, dots, rim = ball.primitives(n_edge_dots=700, n_face_dots=320)
    k = size / ball.SIZE
    layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    X = lambda x: cx + (x - ball.SIZE / 2) * k
    Y = lambda y: cy + (y - ball.SIZE / 2) * k
    tint = lambda x, y: iris(0.5 + ((x - ball.SIZE / 2) * 0.8 + (y - ball.SIZE / 2) * 0.6) / (1.8 * rim))   # 왼쪽 위 → 오른쪽 아래
    for x0, y0, x1, y1, o in sorted(segs, key=lambda s: s[4]):
        d.line([X(x0), Y(y0), X(x1), Y(y1)], fill=tint((x0 + x1) / 2, (y0 + y1) / 2) + (int(255 * o * alpha),), width=int(1.6 * S))
    for x, y, r, o in dots:
        rr = r * k * 1.5
        c = tint(x, y)
        d.ellipse([X(x) - rr, Y(y) - rr, X(x) + rr, Y(y) + rr], fill=tuple(v + (255 - v) // 4 for v in c) + (int(255 * o * alpha),))
    r = rim * k
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(196, 186, 255, int(190 * alpha)), width=int(2 * S))
    img.alpha_composite(layer)


# 팀 원형 배지 위치(1배 좌표)
HOME_X, AWAY_X, TEAM_Y, BADGE_R = 232, 968, 246, 88
RING_PAD = 44   # 링 바깥 빛 여유


@lru_cache(maxsize=1)
def _background():
    """경기와 상관없는 부분 전부(1배로 줄여서 캐시)"""
    img = np.empty((H, W, 3), dtype=np.float32)
    y = np.linspace(0, 1, H)[:, None, None]
    img[...] = np.array([7, 11, 22]) * (1 - y) + np.array([10, 15, 30]) * y
    _glow(img, W * 0.5, -H * 0.2, W * 0.55, (70, 66, 140), 0.55)       # 위쪽 라벤더(앱 배경과 같은 빛)
    _glow(img, W * 0.5, H * 0.4, W * 0.3, (40, 44, 90), 0.35)          # 가운데(스코어) 뒤
    _glow(img, HOME_X * S, TEAM_Y * S, W * 0.2, (30, 70, 140), 0.30)   # 홈 파랑
    _glow(img, AWAY_X * S, TEAM_Y * S, W * 0.2, (130, 70, 30), 0.24)   # 원정 주황
    base = Image.fromarray(img.clip(0, 255).astype(np.uint8)).convert('RGBA')
    _draw_ball(base, W * 0.5, H * 0.4, H * 0.95, 0.17)
    # 원근 경기장 선(예측 결과 머리와 같은 그림, 홈 쪽 파랑·원정 쪽 주황)
    layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = {'home': BLUE, 'away': ORANGE, 'mid': (200, 200, 240)}
    px, py, pw, ph = 40 * S, 96 * S, (OUT_W - 80) * S, 330 * S
    for u1, v1, u2, v2, s in _pitch_segments():
        d.line([px + u1 * pw, py + v1 * ph, px + u2 * pw, py + v2 * ph], fill=col[s] + (46 if s == 'mid' else 54,), width=int(1.6 * S))
    base.alpha_composite(layer)
    # 머리 왼쪽: 오로라 로고 공 + FotData(흰 글자 — 앱 theme-a 워드마크)
    draw_logo_iris(base, 54 * S, 38 * S, 42 * S)
    d = ImageDraw.Draw(base)
    d.text((106 * S, 59 * S), 'FotData', font=font(30 * S, 'Bold'), fill=TEXT, anchor='lm')
    return base.reduce(S)


@lru_cache(maxsize=4)
def _ring(color):
    """옅은 유리 원 + 팀 색 링 + 바깥 빛, 1배 조각. 예전엔 하얀 원(어두운 엠블럼이 보이게)이었는데 어두운 화면 위 스티커 같아서
    2026-10-10 유리 원으로 — 어두운 엠블럼은 _team이 모양을 따라 밝은 테두리 빛을 따로 넣음"""
    size = (BADGE_R + RING_PAD) * 2 * S
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c, r = size / 2, BADGE_R * S
    for i in range(14):
        rr = r + (6 + i * 2.4) * S
        d.ellipse([c - rr, c - rr, c + rr, c + rr], outline=color + (int(22 * (1 - i / 14)),), width=int(3 * S))
    d.ellipse([c - r, c - r, c + r, c + r], fill=(255, 255, 255, 20), outline=color + (200,), width=int(3 * S))
    return img.reduce(S)


def _crest_url(url):
    # 위키미디어 SVG는 Pillow가 못 읽어서 PNG 썸네일 주소로(…/thumb/a/ab/X.svg/256px-X.svg.png)
    if url and 'upload.wikimedia.org' in url and url.endswith('.svg') and '/thumb/' not in url:
        head, name = url.rsplit('/', 1)
        parts = head.split('/')
        i = parts.index('wikipedia') + 2
        return '/'.join(parts[:i] + ['thumb'] + parts[i:]) + f'/{name}/256px-{name}.png'
    return url


@lru_cache(maxsize=512)
def _crest_bytes(url):
    """로고 원본 바이트(압축된 PNG라 작음 — 디코딩한 이미지 대신 이걸 캐시). 실패하면 None(다음 요청 때 다시 시도하게 캐시 안 함)"""
    # 우리 고화질 로고(generate_logos_hd.py → logos/hd)는 저장소에 같이 있으니 디스크에서
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logos', 'hd', 'l', url.rsplit('/', 1)[-1])   # 큰 로고(400px)
    if '/logos/hd/' in url and os.path.exists(local):
        with open(local, 'rb') as f:
            return f.read()
    r = requests.get(_crest_url(url), timeout=4, headers={'User-Agent': 'FotData/1.0 (share card)'})
    r.raise_for_status()
    return r.content


def _crest(url):
    if not url:
        return None
    try:
        return Image.open(io.BytesIO(_crest_bytes(url))).convert('RGBA')
    except Exception:
        return None


def warm_crests(urls, workers=4):
    """서버 시작 때 로고를 미리 받아 둠 — 링크 미리보기 봇은 몇 초만 기다려서, 그때 로고를 받으면 늦음"""
    from concurrent.futures import ThreadPoolExecutor
    def get(u):
        try:
            _crest_bytes(u)
        except Exception:
            pass
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(get, [u for u in set(urls) if u]))


def _fit(d, text, size, weight, max_w):
    """max_w(px)를 넘으면 글자를 줄임"""
    while size > 14 and d.textlength(text, font=font(size, weight)) > max_w:
        size -= 1
    return font(size, weight)


def crest_luma(crest):
    """엠블럼 평균 밝기(0~1, 투명한 곳 빼고) — 앱 저장 이미지(FotData.html crestLuma)와 같은 기준"""
    a = np.asarray(crest.convert('RGBA'), dtype=np.float32)
    w = a[..., 3] / 255
    if w.sum() < 1:
        return 1.0
    y = (0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]) / 255
    return float((y * w).sum() / w.sum())


DARK_CREST = 0.42   # 이보다 어두운 엠블럼(토트넘·유벤투스·PSG 등)은 밝은 테두리 빛


def _crest_halo(c, pad=12):
    """엠블럼 모양을 따라 살짝 넓힌 밝은 빛(어두운 엠블럼이 남색 바탕에 묻히지 않게)"""
    m = Image.new('L', (c.width + pad * 2, c.height + pad * 2), 0)
    m.paste(c.split()[3], (pad, pad))
    m = m.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2.2))
    m = Image.fromarray((np.asarray(m, np.float32) * 0.85).astype(np.uint8))
    return Image.merge('RGBA', [Image.new('L', m.size, v) for v in (236, 241, 250)] + [m])


def _team(img, d, cx, color, name, crest):
    ring = _ring(color)
    img.alpha_composite(ring, (int(cx - ring.width / 2), int(TEAM_Y - ring.height / 2)))
    if crest is not None:
        box = int(BADGE_R * 1.3)
        c = crest.copy()
        c.thumbnail((box, box), Image.LANCZOS)
        if max(c.size) < box:   # 작은 원본은 키워서
            k = box / max(c.size)
            c = c.resize((max(1, int(c.width * k)), max(1, int(c.height * k))), Image.LANCZOS)
        if crest_luma(c) < DARK_CREST:
            img.alpha_composite(_crest_halo(c), (int(cx - c.width / 2 - 12), int(TEAM_Y - c.height / 2 - 12)))
        img.alpha_composite(c, (int(cx - c.width / 2), int(TEAM_Y - c.height / 2)))
    else:
        d.text((cx, TEAM_Y), ''.join(w[0] for w in name.split()[:2]).upper(), font=font(56), fill=TEXT, anchor='mm')
    d.text((cx, TEAM_Y + BADGE_R + 50), name, font=_fit(d, name, 34, 'Bold', 300), fill=TEXT, anchor='mm')


def _chip(img, cx, cy, tw, color):
    """예측 칩 = 앱 예측 배지(유리 알약 + 팀 색 안쪽 테두리 + 앞에 점) — 작은 조각을 2배로 그려 줄여서 붙임. 글자 시작 x를 돌려줌"""
    w, h = int(tw + 70), 50
    piece = Image.new('RGBA', (w * S, h * S), (0, 0, 0, 0))
    pd = ImageDraw.Draw(piece)
    pd.rounded_rectangle([1 * S, 1 * S, (w - 1) * S, (h - 1) * S], radius=24 * S, fill=(255, 255, 255, 22), outline=color + (120,), width=int(1.5 * S))
    pd.line([18 * S, 2 * S, (w - 18) * S, 2 * S], fill=(255, 255, 255, 60), width=int(1 * S))   # 위쪽 빛 한 줄(유리)
    pd.ellipse([22 * S, (h / 2 - 5) * S, 32 * S, (h / 2 + 5) * S], fill=color + (255,))
    x0 = int(cx - w / 2)
    img.alpha_composite(piece.reduce(S), (x0, int(cy - h / 2)))
    return x0 + 42


def _bar(img, x0, x1, y, h, pct, win):
    """확률 막대 = 앱 빅매치(홈 파랑 · 무 회색 · 원정 주황, 둥근 세 조각 사이 틈) — 2배 조각. 각 조각 가운데 x를 돌려줌"""
    w, gap = x1 - x0, 6
    piece = Image.new('RGBA', (w * S, h * S), (0, 0, 0, 0))
    pd = ImageDraw.Draw(piece)
    total = max(1, sum(pct))
    shown = [i for i in range(3) if pct[i] > 0]
    usable = w - gap * (len(shown) - 1)
    x, mids = 0.0, [x0 + w / 2] * 3
    for i in shown:
        seg = max(h, usable * pct[i] / total)
        col = (BLUE, (120, 130, 150), ORANGE)[i]
        pd.rounded_rectangle([x * S, 0, (x + seg) * S - 1, h * S - 1], radius=h / 2 * S, fill=col + (255,))
        mids[i] = x0 + x + seg / 2
        x += seg + gap
    img.alpha_composite(piece.reduce(S), (x0, y))
    return mids


def render(home_name, away_name, home_logo, away_logo, probs, score, prediction, meta=None, limited=False):
    """probs: (홈, 무, 원정) 0~1, score: "2-1", prediction: home_win/draw/away_win → JPEG bytes"""
    img = _background().copy()
    d = ImageDraw.Draw(img)
    d.text((OUT_W - 56, 59), meta or 'AI 경기 예측', font=font(21, 'SemiBold'), fill=MUTED, anchor='rm')

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:   # 미리 받아 두지 못한 로고면 두 개를 동시에
        hc, ac = ex.map(_crest, (home_logo, away_logo))
    _team(img, d, HOME_X, BLUE, home_name, hc)
    _team(img, d, AWAY_X, ORANGE, away_name, ac)

    # 가운데: 예상 스코어 + 예측 칩
    d.text((OUT_W / 2, 152), '예상 스코어', font=font(21, 'SemiBold'), fill=MUTED, anchor='mm')
    d.text((OUT_W / 2, 238), (score or '- -').replace('-', ' : '), font=font(112, 'Bold'), fill=TEXT, anchor='mm')
    color = BLUE if prediction == 'home_win' else ORANGE if prediction == 'away_win' else GREY
    tcol = HOME_TX if prediction == 'home_win' else AWAY_TX if prediction == 'away_win' else INK2
    label = f'{home_name} 승리 예측' if prediction == 'home_win' else f'{away_name} 승리 예측' if prediction == 'away_win' else '무승부 예측'
    fc = _fit(d, label, 24, 'Bold', 300)
    tx = _chip(img, OUT_W / 2, 338, d.textlength(label, font=fc), color)
    d.text((tx, 338), label, font=fc, fill=tcol, anchor='lm')

    # 아래: 앱 빅매치처럼 확률 숫자(위) + 이름(아래 작은 글자) + 둥근 세 조각 막대
    ph, pd_, pa = probs
    pct = [round(ph * 100), round(pd_ * 100), round(pa * 100)]
    win = {'home_win': 0, 'draw': 1, 'away_win': 2}.get(prediction, 0)
    x0, x1, by, bh = 72, OUT_W - 72, 520, 14
    mids = _bar(img, x0, x1, by, bh, pct, win)
    fl = font(19, 'SemiBold')
    labs = [f'{home_name} 승', '무승부', f'{away_name} 승']
    cols = [(120, 180, 255), (200, 206, 218), (255, 150, 90)]
    # 무승부 숫자는 무 조각 가운데 — 양옆 숫자와 겹치지 않게 가운데 쪽으로 밀어 넣음
    dx = min(max(mids[1], x0 + 230), x1 - 230)
    for i, (x, anc) in enumerate([(x0, 'l'), (dx, 'm'), (x1, 'r')]):
        num = f'{pct[i]}%'
        fp = font(52 if i == win else 40, 'Bold')
        d.text((x, by - 46), num, font=fp, fill=cols[i], anchor=anc + 's')
        d.text((x, by - 22), labs[i], font=fl, fill=TEXT if i == win else MUTED, anchor=anc + 's')

    foot = 'fotdata-official.com  ·  AI 예측은 참고용이에요'
    if limited:
        foot = '5대 리그 밖 팀은 UCL 기록만으로 계산한 참고용 예측  ·  fotdata-official.com'
    d.text((OUT_W / 2, 588), foot, font=font(18, 'SemiBold'), fill=(110, 120, 142), anchor='mm')

    buf = io.BytesIO()
    img.convert('RGB').save(buf, 'JPEG', quality=90, subsampling=0, optimize=True, progressive=True)   # PNG 290KB → ~110KB, 인코딩도 빠름
    return buf.getvalue()


if __name__ == '__main__':   # 로컬 확인용: python share_card.py [저장 경로] (기본: share_card_test.jpg)
    import json, sys
    base = os.path.join(os.path.dirname(__file__), 'fotdata_model')
    logos = {**json.load(open(os.path.join(base, 'team_logos.json'))), **json.load(open(os.path.join(base, 'team_logos_hd.json')))}
    png = render('Bayern', 'PSG', logos.get('FC Bayern München'), logos.get('Paris Saint-Germain FC'),
                 (0.52, 0.24, 0.24), '2-1', 'home_win', '챔피언스리그 · 10월 21일(수) 04:00')
    open(sys.argv[1] if len(sys.argv) > 1 else 'share_card_test.jpg', 'wb').write(png)
