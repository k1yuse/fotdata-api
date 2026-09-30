"""
경기별 링크 공유 썸네일(1200×630 JPEG) — main.py의 /og/match/{slug}.jpg가 부름 (2026-09-30)
카톡·페북·X 같은 링크 미리보기는 JS를 실행하지 않아서, 예측 결과를 보여주려면 서버가 이미지를 그려야 함.
디자인은 앱 공유 이미지(FotData.html shareResultCard)·예측 결과 머리와 같은 규칙:
진한 남색 + 빛번짐 + 와이어프레임 공(generate_bg_ball) + 원근 경기장 선, 홈 파랑 · 무 회색 · 원정 주황.
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
from PIL import Image, ImageDraw, ImageFont

import generate_bg_ball as ball

OUT_W, OUT_H = 1200, 630
S = 2
W, H = OUT_W * S, OUT_H * S
FONT_DIR = os.path.join(os.path.dirname(__file__), 'fonts')
BLUE, ORANGE, GREY = (88, 166, 255), (240, 136, 62), (139, 148, 158)
TEXT, MUTED = (230, 237, 243), (139, 148, 158)


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
    for x0, y0, x1, y1, o in sorted(segs, key=lambda s: s[4]):
        d.line([X(x0), Y(y0), X(x1), Y(y1)], fill=BLUE + (int(255 * o * alpha),), width=int(1.6 * S))
    for x, y, r, o in dots:
        rr = r * k * 1.5
        d.ellipse([X(x) - rr, Y(y) - rr, X(x) + rr, Y(y) + rr], fill=(142, 195, 255, int(255 * o * alpha)))
    r = rim * k
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(120, 184, 255, int(200 * alpha)), width=int(2 * S))
    img.alpha_composite(layer)


def _draw_logo(d, x, y, size):
    """사이트 로고 축구공(viewBox 24) — generate_og_image.draw_logo와 같은 좌표"""
    k = size / 24
    P = lambda px, py: (x + px * k, y + py * k)
    w = max(2, int(1.5 * k))
    d.ellipse([*P(3, 3), *P(21, 21)], outline=BLUE, width=w)
    pent = [P(12, 7.7), P(16.09, 10.67), P(14.53, 15.48), P(9.47, 15.48), P(7.91, 10.67)]
    d.line(pent + [pent[0]], fill=BLUE, width=w, joint='curve')
    for a, b in [((12, 7.7), (12, 3.8)), ((16.09, 10.67), (19.8, 9.47)), ((14.53, 15.48), (16.82, 18.63)),
                 ((9.47, 15.48), (7.18, 18.63)), ((7.91, 10.67), (4.2, 9.47))]:
        d.line([P(*a), P(*b)], fill=BLUE, width=w)


# 팀 원형 배지 위치(1배 좌표)
HOME_X, AWAY_X, TEAM_Y, BADGE_R = 232, 968, 246, 88
RING_PAD = 44   # 링 바깥 빛 여유


@lru_cache(maxsize=1)
def _background():
    """경기와 상관없는 부분 전부(1배로 줄여서 캐시)"""
    img = np.empty((H, W, 3), dtype=np.float32)
    y = np.linspace(0, 1, H)[:, None, None]
    img[...] = np.array([11, 15, 20]) * (1 - y) + np.array([15, 20, 27]) * y
    _glow(img, W * 0.5, H * 0.34, W * 0.36, (34, 70, 130), 0.42)       # 가운데(스코어) 뒤
    _glow(img, HOME_X * S, TEAM_Y * S, W * 0.2, (40, 90, 170), 0.38)   # 홈 파랑
    _glow(img, AWAY_X * S, TEAM_Y * S, W * 0.2, (150, 80, 30), 0.30)   # 원정 주황
    _glow(img, W * 0.02, H * 1.0, W * 0.3, (90, 60, 150), 0.22)
    base = Image.fromarray(img.clip(0, 255).astype(np.uint8)).convert('RGBA')
    _draw_ball(base, W * 0.5, H * 0.4, H * 0.95, 0.2)
    # 원근 경기장 선(예측 결과 머리와 같은 그림, 홈 쪽 파랑·원정 쪽 주황)
    layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = {'home': BLUE, 'away': ORANGE, 'mid': (143, 184, 255)}
    px, py, pw, ph = 40 * S, 96 * S, (OUT_W - 80) * S, 330 * S
    for u1, v1, u2, v2, s in _pitch_segments():
        d.line([px + u1 * pw, py + v1 * ph, px + u2 * pw, py + v2 * ph], fill=col[s] + (62,), width=int(1.6 * S))
    base.alpha_composite(layer)
    # 머리 왼쪽: 로고 + FotData
    d = ImageDraw.Draw(base)
    _draw_logo(d, 56 * S, 40 * S, 38 * S)
    fw = font(30 * S, 'Bold')
    d.text((104 * S, 59 * S), 'Fot', font=fw, fill=TEXT, anchor='lm')
    d.text((104 * S + d.textlength('Fot', font=fw), 59 * S), 'Data', font=fw, fill=BLUE, anchor='lm')
    return base.reduce(S)


@lru_cache(maxsize=4)
def _ring(color):
    """밝은 배지 + 팀 색 링 + 바깥 빛(랜딩 빅매치 카드와 같은 방식 — 남색·검정 엠블럼도 어두운 배경에서 보이게), 1배 조각"""
    size = (BADGE_R + RING_PAD) * 2 * S
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c, r = size / 2, BADGE_R * S
    for i in range(14):
        rr = r + (8 + i * 2.4) * S
        d.ellipse([c - rr, c - rr, c + rr, c + rr], outline=color + (int(26 * (1 - i / 14)),), width=int(3 * S))
    d.ellipse([c - r, c - r, c + r, c + r], fill=(236, 241, 247, 250), outline=color + (255,), width=int(5 * S))
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
        img.alpha_composite(c, (int(cx - c.width / 2), int(TEAM_Y - c.height / 2)))
    else:
        d.text((cx, TEAM_Y), ''.join(w[0] for w in name.split()[:2]).upper(), font=font(56), fill=(40, 48, 58), anchor='mm')
    d.text((cx, TEAM_Y + BADGE_R + 50), name, font=_fit(d, name, 34, 'Bold', 300), fill=TEXT, anchor='mm')


def _chip(img, cx, cy, tw, color):
    """예측 칩 배경(반투명 알약) — 작은 조각을 2배로 그려 줄여서 붙임"""
    w, h = int(tw + 48), 48
    piece = Image.new('RGBA', (w * S, h * S), (0, 0, 0, 0))
    ImageDraw.Draw(piece).rounded_rectangle([1 * S, 1 * S, (w - 1) * S, (h - 1) * S], radius=23 * S,
                                            fill=color + (40,), outline=color + (140,), width=int(1.5 * S))
    img.alpha_composite(piece.reduce(S), (int(cx - w / 2), int(cy - h / 2)))


def _bar(img, x0, x1, y, h, pct, win):
    """한 줄 확률 막대(홈 파랑 · 무 회색 · 원정 주황, 예측한 쪽만 진하게, 칸 사이 틈) — 2배 조각"""
    w = x1 - x0
    piece = Image.new('RGBA', (w * S, h * S), (0, 0, 0, 0))
    pd = ImageDraw.Draw(piece)
    pd.rectangle([0, 0, w * S, h * S], fill=(33, 38, 45, 255))
    total, x = max(1, sum(pct)), 0.0
    cuts = []
    for i, col in enumerate((BLUE, GREY, ORANGE)):
        seg = w * pct[i] / total
        if seg > 0:
            pd.rectangle([x * S, 0, (x + seg) * S, h * S], fill=tuple(int(c * (1 if i == win else 0.42) + 33 * (0 if i == win else 0.58)) for c in col) + (255,))
        x += seg
        cuts.append(x)
    for xx in cuts[:2]:
        pd.line([xx * S, 0, xx * S, h * S], fill=(13, 17, 23, 255), width=int(3 * S))
    mask = Image.new('L', piece.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w * S - 1, h * S - 1], radius=h / 2 * S, fill=255)
    piece.putalpha(mask)
    img.alpha_composite(piece.reduce(S), (x0, y))


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
    label = f'{home_name} 승리 예측' if prediction == 'home_win' else f'{away_name} 승리 예측' if prediction == 'away_win' else '무승부 예측'
    fc = _fit(d, label, 24, 'Bold', 300)
    _chip(img, OUT_W / 2, 336, d.textlength(label, font=fc), color)
    d.text((OUT_W / 2, 336), label, font=fc, fill=color, anchor='mm')

    # 아래: 확률 라벨 + 한 줄 막대
    ph, pd_, pa = probs
    pct = [round(ph * 100), round(pd_ * 100), round(pa * 100)]
    win = {'home_win': 0, 'draw': 1, 'away_win': 2}.get(prediction, 0)
    x0, x1, by, bh = 72, OUT_W - 72, 500, 22
    for i, (lab, col, anc, x) in enumerate([(f'{home_name} 승', BLUE, 'ls', x0), ('무승부', GREY, 'ms', OUT_W / 2), (f'{away_name} 승', ORANGE, 'rs', x1)]):
        on = i == win
        fl, fp, num = font(22, 'SemiBold'), font(34 if on else 28, 'Bold'), f'{pct[i]}%'
        if anc == 'ls':
            d.text((x, by - 16), num, font=fp, fill=col if on else MUTED, anchor='ls')
            d.text((x + d.textlength(num, font=fp) + 12, by - 16), lab, font=fl, fill=TEXT if on else MUTED, anchor='ls')
        elif anc == 'rs':
            d.text((x, by - 16), num, font=fp, fill=col if on else MUTED, anchor='rs')
            d.text((x - d.textlength(num, font=fp) - 12, by - 16), lab, font=fl, fill=TEXT if on else MUTED, anchor='rs')
        else:
            d.text((x, by - 16), f'{lab} {num}', font=font(24, 'Bold' if on else 'SemiBold'), fill=TEXT if on else MUTED, anchor='ms')
    _bar(img, x0, x1, by, bh, pct, win)

    foot = 'fotdata-api.vercel.app  ·  AI 예측은 참고용이에요'
    if limited:
        foot = '5대 리그 밖 팀은 UCL 기록만으로 계산한 참고용 예측  ·  fotdata-api.vercel.app'
    d.text((OUT_W / 2, 584), foot, font=font(18, 'SemiBold'), fill=(125, 133, 144), anchor='mm')

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
