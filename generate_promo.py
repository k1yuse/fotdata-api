"""
홍보 이미지(인스타그램·유튜브·네이버 카페) 생성 — python generate_promo.py  →  promo/*.png
  feed_brand.png     1080×1350  인스타 피드: 서비스 소개
  feed_bigmatch.png  1080×1350  인스타 피드: 이번 주 빅매치 AI 예측(서버 /bigmatch·/predict 실시간 값 — 매주 다시 실행)
  story.png          1080×1920  인스타 스토리(위·아래 UI 겹치는 자리는 비움)
  banner_wide.png    1920×1080  유튜브 썸네일·네이버 카페 대표 이미지
사이트·공유 썸네일과 같은 톤(진한 남색 + 파란 와이어프레임 공 + 축구공 로고). 글꼴은 저장소의 FotData Card Sans
(Pretendard OFL 축소판 — 재배포 가능, 한글 2,350자). 구단 엠블럼은 넣지 않음(홍보물에 상표 사용 위험 — 이름 글자만).
AI 예측 문구는 '참고용' 안내를 같이 넣고, 베팅·적중 보장처럼 읽히는 표현은 쓰지 않음.
"""
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

import generate_bg_ball as ball

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'promo')
API = 'https://fotdata-api.onrender.com'
SITE = 'fotdata-official.com'
S = 2                                        # 2배로 그려서 줄임(안티앨리어싱)
BLUE, ORANGE, TEXT, MUTED, DIM = (88, 166, 255), (240, 136, 62), (236, 241, 250), (142, 154, 179), (110, 118, 129)
GRAY = (110, 118, 129)
FONTS = {w: os.path.join(BASE, 'fonts', f'FotDataCardSans-{w}.otf') for w in ('Bold', 'SemiBold')}
LEAGUE_KO = {'PL': '프리미어리그', 'PD': '라리가', 'BL1': '분데스리가', 'SA': '세리에 A', 'FL1': '리그 1', 'CL': '챔피언스리그'}
_missing = set()


def font(size, w='Bold'):
    return ImageFont.truetype(FONTS[w], int(size * S))


def text(d, xy, s, size, fill, w='Bold', anchor='la'):
    f = font(size, w)
    for ch in s:   # 축소판 글꼴에 없는 글자 확인(□로 나옴)
        if ch.strip() and f.getmask(ch).getbbox() is None:
            _missing.add(ch)
    d.text((xy[0] * S, xy[1] * S), s, font=f, fill=fill, anchor=anchor)


def tlen(d, s, size, w='Bold'):
    return d.textlength(s, font=font(size, w)) / S


def background(w, h, glows):
    W, H = w * S, h * S
    y = np.linspace(0, 1, H)[:, None]
    base = np.array([8, 13, 26]) * (1 - y[..., None]) + np.array([12, 20, 36]) * y[..., None]
    img = np.broadcast_to(base, (H, W, 3)).astype(np.float32).copy()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for cx, cy, r, rgb, a in glows:
        dd = np.sqrt((xx - cx * W) ** 2 + (yy - cy * H) ** 2) / (r * max(W, H))
        k = np.clip(1 - dd, 0, 1) ** 2 * a
        img[...] = img * (1 - k[..., None]) + np.array(rgb) * k[..., None]
    return Image.fromarray(img.clip(0, 255).astype(np.uint8)).convert('RGBA')


def draw_ball(img, cx, cy, size, alpha=1.0):
    segs, dots, rim = ball.primitives(n_edge_dots=900, n_face_dots=420)
    k = size * S / ball.SIZE
    cx, cy = cx * S, cy * S
    layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    X = lambda x: cx + (x - ball.SIZE / 2) * k
    Y = lambda y: cy + (y - ball.SIZE / 2) * k
    for i in range(18):
        r = rim * k * (1.0 + i * 0.012)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=BLUE + (int(16 * alpha * (1 - i / 18)),), width=int(6 * S))
    for x0, y0, x1, y1, o in sorted(segs, key=lambda s: s[4]):
        d.line([X(x0), Y(y0), X(x1), Y(y1)], fill=BLUE + (int(255 * o * 0.95 * alpha),), width=int(2.4 * S))
    for x, y, r, o in dots:
        rr = r * k * 1.6
        d.ellipse([X(x) - rr, Y(y) - rr, X(x) + rr, Y(y) + rr], fill=(142, 195, 255, int(255 * o * alpha)))
    r = rim * k
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(120, 184, 255, int(200 * alpha)), width=int(3 * S))
    img.alpha_composite(layer)


def draw_logo(d, x, y, size):
    """사이트 로고 축구공(viewBox 24: 원 r9 + 오각형 + 바깥 5선) — generate_og_image.py와 같은 좌표"""
    k = size * S / 24
    P = lambda px, py: (x * S + px * k, y * S + py * k)
    w = max(2, int(1.4 * k))
    d.ellipse([*P(3, 3), *P(21, 21)], outline=BLUE, width=w)
    pent = [P(12, 7.7), P(16.09, 10.67), P(14.53, 15.48), P(9.47, 15.48), P(7.91, 10.67)]
    d.line(pent + [pent[0]], fill=BLUE, width=w, joint='curve')
    for a, b in [((12, 7.7), (12, 3.8)), ((16.09, 10.67), (19.8, 9.47)), ((14.53, 15.48), (16.82, 18.63)),
                 ((9.47, 15.48), (7.18, 18.63)), ((7.91, 10.67), (4.2, 9.47))]:
        d.line([P(*a), P(*b)], fill=BLUE, width=w)
    for px, py in pent + [P(12, 3.8), P(19.8, 9.47), P(16.82, 18.63), P(7.18, 18.63), P(4.2, 9.47)]:
        d.ellipse([px - w / 2, py - w / 2, px + w / 2, py + w / 2], fill=BLUE)


def wordmark(d, x, y, size, sub=True):
    draw_logo(d, x, y, size * 1.15)
    tx = x + size * 1.4
    text(d, (tx, y - size * 0.05), 'Fot', size, TEXT)
    text(d, (tx + tlen(d, 'Fot', size), y - size * 0.05), 'Data', size, BLUE)
    if sub:
        text(d, (tx + 3, y + size * 1.02), 'A I   F O O T B A L L', size * 0.27, MUTED, 'SemiBold')


def rrect(d, box, r, fill=None, outline=None, width=1):
    d.rounded_rectangle([v * S for v in box], radius=r * S, fill=fill, outline=outline, width=int(width * S))


def pill(d, x, y, s, size=26, fg=BLUE, bg=(17, 26, 46, 235), border=(56, 90, 140)):
    w = tlen(d, s, size, 'SemiBold') + size * 1.6
    h = size * 2.1
    rrect(d, (x, y, x + w, y + h), h / 2, fill=bg, outline=border, width=1.5)
    text(d, (x + w / 2, y + h / 2), s, size, fg, 'SemiBold', anchor='mm')
    return w


def league_chips(d, x, y, size=24, gap=12, max_w=None):
    cx, cy = x, y
    for name in ['EPL', '라리가', '분데스리가', '세리에A', '리그앙', 'UCL']:
        w = tlen(d, name, size, 'SemiBold') + size * 1.6
        if max_w and cx + w > x + max_w:
            cx, cy = x, cy + size * 2.1 + gap
        pill(d, cx, cy, name, size)
        cx += w + gap
    return cy + size * 2.1


def short_names():
    try:
        html = open(os.path.join(BASE, 'FotData.html'), encoding='utf-8').read()
        block = html[html.index('const SHORT_NAMES = {'):]
        return dict(re.findall(r"'([^']+)':\s*'([^']+)'", block[:block.index('};')]))
    except Exception:
        return {}


def short(name, table):
    if name in table:
        return table[name]
    t = re.sub(r'\b(FC|CF|AFC|BC|SC|AC|SK|FK|SSC|US|KV)\b|\b\d{4}\b', '', re.sub(r'^\d+\.\s*', '', name))
    return re.sub(r'\s+', ' ', t).strip() or name


def fetch_bigmatch():
    """이번 주 빅매치 + 모델 예측(서버가 자고 있으면 깨우느라 첫 요청이 느릴 수 있음)"""
    m = requests.get(f'{API}/bigmatch', timeout=90).json()['match']
    p = requests.post(f'{API}/predict', json={'home_team': m['home_team'], 'away_team': m['away_team']}, timeout=90).json()
    acc = requests.get(f'{API}/accuracy', timeout=60).json()
    tbl = short_names()
    k = datetime.fromisoformat(m['date'].replace('Z', '+00:00')).astimezone(timezone(timedelta(hours=9)))
    pr = p['probabilities']
    return {
        'home': short(m['home_team'], tbl), 'away': short(m['away_team'], tbl),
        'league': LEAGUE_KO.get(m['league'], m['league']),
        'when': f"{k.month}월 {k.day}일({'월화수목금토일'[k.weekday()]}) {k:%H:%M}",
        'p': (round(pr['home_win'] * 100), round(pr['draw'] * 100), round(pr['away_win'] * 100)),
        'score': (p.get('score_prediction') or {}).get('most_likely'),
        'pick': p['prediction'], 'acc': acc.get('logistic_regression'), 'n_test': acc.get('test_matches'),
    }


def match_card(img, d, x, y, w, m, scale=1.0):
    """빅매치 예측 카드(엠블럼 대신 홈 파랑·원정 주황 원) — 높이를 돌려줌"""
    s = scale
    h = 500 * s
    rrect(d, (x, y, x + w, y + h), 28 * s, fill=(17, 26, 46, 215), outline=(38, 51, 77), width=1.5)
    text(d, (x + w / 2, y + 46 * s), f"{m['league']} · {m['when']}", 24 * s, (121, 184, 255), 'SemiBold', anchor='mm')
    hp, dp, ap = m['p']
    cy = y + 170 * s
    for i, (name, pct, col) in enumerate([(m['home'], hp, BLUE), (m['away'], ap, ORANGE)]):
        cx = x + w * (0.22 if i == 0 else 0.78)
        r = 78 * s
        for g in range(10):   # 원 바깥 빛
            rr = r + g * 3 * s
            d.ellipse([(cx - rr) * S, (cy - rr) * S, (cx + rr) * S, (cy + rr) * S], outline=col + (int(40 * (1 - g / 10)),), width=int(3 * S))
        d.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S], fill=(10, 16, 32, 255), outline=col, width=int(5 * s * S))
        text(d, (cx, cy + 2 * s), f'{pct}%', 44 * s, col, anchor='mm')
        text(d, (cx, cy + r + 40 * s), name, 30 * s, TEXT, anchor='mm')
    text(d, (x + w / 2, cy - 18 * s), 'VS', 30 * s, MUTED, anchor='mm')
    text(d, (x + w / 2, cy + 24 * s), f'무 {dp}%', 22 * s, MUTED, 'SemiBold', anchor='mm')
    # 확률 막대
    bx, by, bw, bh = x + 40 * s, y + 330 * s, w - 80 * s, 16 * s
    cur = bx
    for pct, col in [(hp, BLUE), (dp, (71, 84, 112)), (ap, ORANGE)]:
        seg = bw * pct / max(1, hp + dp + ap)
        d.rectangle([cur * S, by * S, (cur + seg - 3 * s) * S, (by + bh) * S], fill=col)
        cur += seg
    pick = {'home_win': f"{m['home']} 승리 예측", 'away_win': f"{m['away']} 승리 예측"}.get(m['pick'], '무승부 예측')
    pc = BLUE if m['pick'] == 'home_win' else ORANGE if m['pick'] == 'away_win' else MUTED
    pw = tlen(d, pick, 26 * s) + 60 * s
    rrect(d, (x + w / 2 - pw / 2, y + 372 * s, x + w / 2 + pw / 2, y + 424 * s), 26 * s, fill=pc + (40,), outline=pc + (150,), width=1.5)
    text(d, (x + w / 2, y + 398 * s), pick, 26 * s, pc, anchor='mm')
    if m.get('score'):
        text(d, (x + w / 2, y + 458 * s), f"예상 스코어 {m['score'].replace('-', ' : ')}", 22 * s, MUTED, 'SemiBold', anchor='mm')
    return h


def save(img, w, h, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    img.resize((w, h), Image.LANCZOS).convert('RGB').save(path, optimize=True)
    print('✅', path, f'{w}×{h}')


def feed_brand():
    w, h = 1080, 1350
    img = background(w, h, [(0.85, 0.78, 0.55, (40, 90, 170), 0.6), (0.0, 0.0, 0.45, (90, 60, 150), 0.3)])
    draw_ball(img, 900, 1120, 760, 0.85)
    img = img.convert('RGB')
    d = ImageDraw.Draw(img, 'RGBA')   # RGB 위에 RGBA로 그려야 반투명이 섞임(RGBA 이미지에 그리면 픽셀을 그대로 덮어씀)
    L = 80
    wordmark(d, L, 90, 64)
    text(d, (L, 290), '축구를 데이터로', 92, TEXT)
    text(d, (L, 400), '예측하다', 92, BLUE)
    text(d, (L, 532), '유럽 5대 리그 + 챔피언스리그, AI가 먼저 계산해요', 30, MUTED, 'SemiBold')
    feats = [('경기 예측', '승·무·패 확률과 예상 스코어, 예측 근거까지'),
             ('순위 예측', '남은 시즌을 최대 1만 번 시뮬레이션'),
             ('AI 트랙레코드', '킥오프 전에 기록하고, 경기 후 그대로 채점')]
    y = 630
    for i, (t, sub) in enumerate(feats):
        rrect(d, (L, y, L + 620, y + 118), 22, fill=(17, 26, 46, 220), outline=(38, 51, 77), width=1.5)
        d.ellipse([(L + 28) * S, (y + 31) * S, (L + 84) * S, (y + 87) * S], fill=(88, 166, 255, 40), outline=BLUE, width=int(2 * S))
        text(d, (L + 56, y + 59), str(i + 1), 28, BLUE, anchor='mm')
        text(d, (L + 108, y + 26), t, 32, TEXT)
        text(d, (L + 108, y + 70), sub, 22, MUTED, 'SemiBold')
        y += 136
    league_chips(d, L, 1050, 23, gap=10, max_w=900)
    pill(d, L, 1210, f'무료 · 회원가입 없이  {SITE}', 26, fg=TEXT, bg=(88, 166, 255, 60), border=BLUE)
    save(img, w, h, 'feed_brand.png')


def feed_bigmatch(m):
    w, h = 1080, 1350
    img = background(w, h, [(0.5, 0.42, 0.6, (40, 90, 170), 0.55), (1.0, 1.0, 0.4, (90, 60, 150), 0.3)])
    draw_ball(img, 960, 160, 420, 0.5)
    img = img.convert('RGB')
    d = ImageDraw.Draw(img, 'RGBA')   # RGB 위에 RGBA로 그려야 반투명이 섞임(RGBA 이미지에 그리면 픽셀을 그대로 덮어씀)
    L = 70
    wordmark(d, L, 70, 44, sub=False)
    text(d, (L, 190), '이번 주 빅매치', 78, TEXT)
    text(d, (L, 290), 'AI의 예측은?', 78, BLUE)
    match_card(img, d, L, 430, w - 2 * L, m, 1.0)
    if m.get('acc'):
        text(d, (w / 2, 960), f"AI 모델 정확도 {m['acc']}% · 학습에 안 쓴 최근 {m['n_test']:,}경기로 검증", 24, MUTED, 'SemiBold', anchor='mm')
    text(d, (w / 2, 1060), '맞대결·최근 폼·순위 분석까지 전체 보기', 32, TEXT, anchor='mm')
    pw = tlen(d, SITE, 30) + 80
    rrect(d, (w / 2 - pw / 2, 1110, w / 2 + pw / 2, 1180), 35, fill=BLUE)
    text(d, (w / 2, 1145), SITE, 30, (10, 16, 32), anchor='mm')
    text(d, (w / 2, 1280), 'AI 예측은 통계 모델의 참고 정보이며 경기 결과를 보장하지 않아요', 20, DIM, 'SemiBold', anchor='mm')
    save(img, w, h, 'feed_bigmatch.png')


def story(m):
    w, h = 1080, 1920
    img = background(w, h, [(0.5, 0.35, 0.6, (40, 90, 170), 0.55), (0.0, 1.0, 0.45, (90, 60, 150), 0.3)])
    draw_ball(img, 540, 1700, 900, 0.4)
    img = img.convert('RGB')
    d = ImageDraw.Draw(img, 'RGBA')   # RGB 위에 RGBA로 그려야 반투명이 섞임(RGBA 이미지에 그리면 픽셀을 그대로 덮어씀)
    wm = 60 * 1.4 + tlen(d, 'FotData', 60)   # 로고 + 워드마크 폭(가운데 정렬)
    wordmark(d, w / 2 - wm / 2, 260, 60)
    text(d, (w / 2, 470), '축구를 데이터로', 84, TEXT, anchor='mm')
    text(d, (w / 2, 575), '예측하다', 84, BLUE, anchor='mm')
    text(d, (w / 2, 668), '이번 주 빅매치, AI의 예측', 32, MUTED, 'SemiBold', anchor='mm')
    match_card(img, d, 80, 730, w - 160, m, 1.0)
    rrect(d, (190, 1320, w - 190, 1412), 46, fill=BLUE)
    text(d, (w / 2, 1366), '프로필 링크에서 무료로 보기', 34, (10, 16, 32), anchor='mm')
    text(d, (w / 2, 1462), SITE, 28, MUTED, 'SemiBold', anchor='mm')
    text(d, (w / 2, 1508), 'AI 예측은 참고 정보예요', 20, DIM, 'SemiBold', anchor='mm')
    save(img, w, h, 'story.png')


def banner_wide(m):
    w, h = 1920, 1080
    img = background(w, h, [(0.75, 0.45, 0.45, (40, 90, 170), 0.6), (0.0, 1.0, 0.35, (90, 60, 150), 0.32)])
    draw_ball(img, 1500, 540, 1050, 0.55)
    img = img.convert('RGB')
    d = ImageDraw.Draw(img, 'RGBA')   # RGB 위에 RGBA로 그려야 반투명이 섞임(RGBA 이미지에 그리면 픽셀을 그대로 덮어씀)
    L = 130
    wordmark(d, L, 130, 70)
    text(d, (L, 330), '축구를 데이터로', 104, TEXT)
    text(d, (L, 455), '예측하다', 104, BLUE)
    text(d, (L, 610), '5대 리그 + 챔피언스리그 · AI 경기 예측 · 순위 예측 · 트랙레코드', 32, MUTED, 'SemiBold')
    league_chips(d, L, 690, 28)
    pill(d, L, 880, f'무료 · 회원가입 없이  {SITE}', 30, fg=TEXT, bg=(88, 166, 255, 60), border=BLUE)
    match_card(img, d, 1150, 300, 640, m, 1.0)
    save(img, w, h, 'banner_wide.png')


def main():
    m = fetch_bigmatch()
    print('빅매치:', m)
    feed_brand()
    feed_bigmatch(m)
    story(m)
    banner_wide(m)
    if _missing:
        print('⚠️ 글꼴에 없는 글자(□로 나옴):', ''.join(sorted(_missing)))


if __name__ == '__main__':
    sys.exit(main())
