"""
링크 공유 썸네일(og-image.png, 2400×1260 — 카톡·SNS 미리보기) — 2026-10-10 앱 theme-a 톤.
오른쪽에 랜딩 3D 연출과 같은 와이어프레임 축구공(generate_bg_ball.primitives)을 오로라 색으로 크게,
왼쪽엔 오로라 로고 공 + 흰 FotData 워드마크 + 헤드라인("예측하다"만 오로라) + 리그 칩(유리 알약 + 리그 아이콘).
글꼴은 사이트와 같은 Pretendard(fonts/FotDataCardSans — share_card.py와 공용), 오로라·로고 그리는 법도 share_card.py.
2배 크기로 그린 뒤 줄여서 안티앨리어싱. 문구를 바꾸면 다시 실행: python generate_og_image.py (+ 페이지 meta의 og-image ?v= 올리기)
"""
import numpy as np
from PIL import Image, ImageDraw

import generate_bg_ball as ball
from share_card import font as card_font, iris, iris_text, draw_logo_iris

OUT_W, OUT_H = 2400, 1260
S = 2                                   # 슈퍼샘플링 배율
W, H = OUT_W * S, OUT_H * S
TEXT, INK2, MUTED = (238, 242, 248), (185, 195, 212), (127, 138, 160)
font = lambda size, weight='Bold': card_font(size * S, weight)
LEAGUES = [('PL', 'EPL'), ('PD', '라리가'), ('BL1', '분데스리가'), ('SA', '세리에A'), ('FL1', '리그앙'), ('CL', 'UCL')]


def background():
    y = np.linspace(0, 1, H)[:, None]
    base = np.array([7, 11, 22]) * (1 - y[..., None]) + np.array([10, 15, 30]) * y[..., None]
    img = np.broadcast_to(base, (H, W, 3)).astype(np.float32).copy()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    def glow(cx, cy, r, rgb, a):
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
        k = np.clip(1 - d, 0, 1) ** 2 * a
        img[...] = img * (1 - k[..., None]) + np.array(rgb) * k[..., None]
    glow(W * 0.42, -H * 0.25, W * 0.6, (78, 74, 160), 0.6)      # 위쪽 라벤더(앱 배경과 같은 빛)
    glow(W * 0.735, H * 0.5, W * 0.36, (60, 62, 140), 0.5)      # 공 뒤
    glow(W * 0.92, H * 0.95, W * 0.25, (150, 100, 80), 0.25)    # 오른쪽 아래 복숭아
    glow(W * 0.0, H * 1.0, W * 0.3, (40, 70, 140), 0.25)        # 왼쪽 아래 하늘
    return Image.fromarray(img.clip(0, 255).astype(np.uint8)).convert('RGBA')


def draw_ball(img, cx, cy, size):
    segs, dots, rim = ball.primitives(n_edge_dots=900, n_face_dots=420)
    k = size / ball.SIZE
    layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    X = lambda x: cx + (x - ball.SIZE / 2) * k
    Y = lambda y: cy + (y - ball.SIZE / 2) * k
    tint = lambda x, y: iris(0.5 + ((x - ball.SIZE / 2) * 0.8 + (y - ball.SIZE / 2) * 0.6) / (1.7 * rim))
    for i in range(18):   # 테두리 빛번짐(여러 겹의 반투명 원)
        r = rim * k * (1.0 + i * 0.012)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(185, 168, 255, int(15 * (1 - i / 18))), width=int(6 * S))
    for x0, y0, x1, y1, o in sorted(segs, key=lambda s: s[4]):   # 뒷면부터
        d.line([X(x0), Y(y0), X(x1), Y(y1)], fill=tint((x0 + x1) / 2, (y0 + y1) / 2) + (int(255 * o * 0.95),), width=int(2.6 * S))
    for x, y, r, o in dots:
        rr = r * k * 1.6
        c = tint(x, y)
        d.ellipse([X(x) - rr, Y(y) - rr, X(x) + rr, Y(y) + rr], fill=tuple(v + (255 - v) // 4 for v in c) + (int(255 * o),))
    r = rim * k
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(200, 190, 255, 190), width=int(3 * S))
    img.alpha_composite(layer)


def chip(img, d, x, y, code, name):
    """리그 칩 = 앱 필터 칩(옅은 유리 알약 + 얇은 흰 테두리) + 리그 아이콘. 다음 칩 x를 돌려줌"""
    f, h, icon = font(34, 'SemiBold'), 76 * S, 38 * S
    tw = d.textlength(name, font=f)
    w = tw + icon + 86 * S
    piece = Image.new('RGBA', (int(w), int(h)), (0, 0, 0, 0))
    ImageDraw.Draw(piece).rounded_rectangle([0, 0, w - 1, h - 1], radius=h / 2, fill=(255, 255, 255, 14), outline=(255, 255, 255, 40), width=2 * S)
    img.alpha_composite(piece, (int(x), int(y)))
    try:
        ic = Image.open(f'logos/league/{code}.png').convert('RGBA')
        ic.thumbnail((icon, icon), Image.LANCZOS)
        img.alpha_composite(ic, (int(x + 30 * S + (icon - ic.width) / 2), int(y + (h - ic.height) / 2)))
    except OSError:
        pass
    d.text((x + 30 * S + icon + 14 * S, y + h / 2), name, font=f, fill=INK2, anchor='lm')
    return x + w + 16 * S


def main():
    img = background()
    draw_ball(img, W * 0.735, H * 0.5, H * 1.02)
    L = 150 * S
    # 로고 + 워드마크(흰 글자 — 앱 theme-a)
    draw_logo_iris(img, L - 6 * S, 146 * S, 126 * S)
    d = ImageDraw.Draw(img)
    d.text((L + 148 * S, 212 * S), 'FotData', font=font(104), fill=TEXT, anchor='ls')
    d.text((L + 152 * S, 262 * S), 'A I   F O O T B A L L', font=font(28, 'SemiBold'), fill=MUTED, anchor='ls')
    # 헤드라인(랜딩 첫 화면 문구) — 강조어만 오로라
    d.text((L, 540 * S), '축구를 데이터로', font=font(124), fill=TEXT, anchor='ls')
    iris_text(img, (L, 700 * S), '예측하다', font(124), anchor='ls')
    d = ImageDraw.Draw(img)
    d.text((L, 806 * S), '5대 리그 + 챔피언스리그 · AI 경기 예측 · 순위 예측 시뮬레이션', font=font(40, 'SemiBold'), fill=INK2, anchor='ls')
    for row, y in ((LEAGUES[:3], 862 * S), (LEAGUES[3:], 954 * S)):   # 두 줄(한 줄이면 오른쪽 공과 겹침)
        x = L
        for code, name in row:
            x = chip(img, d, x, y, code, name)
    d.text((L, 1130 * S), 'fotdata-official.com', font=font(34, 'SemiBold'), fill=MUTED, anchor='ls')
    img.resize((OUT_W, OUT_H), Image.LANCZOS).convert('RGB').save('og-image.png', optimize=True)
    print('og-image.png', OUT_W, 'x', OUT_H)


if __name__ == '__main__':
    main()
