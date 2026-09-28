"""
링크 공유 썸네일(og-image.png, 2400×1260 — 카톡·SNS 미리보기) 생성.
랜딩 3D 연출·메인 유리 테마와 같은 와이어프레임 축구공(generate_bg_ball.primitives)을 오른쪽에 크게,
왼쪽엔 새 축구공 로고 + FotData 워드마크 + 헤드라인. (예전 썸네일은 막대그래프 로고였음)
2배 크기로 그린 뒤 줄여서 안티앨리어싱. 문구를 바꾸면 다시 실행: python generate_og_image.py
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import generate_bg_ball as ball

OUT_W, OUT_H = 2400, 1260
S = 2                                   # 슈퍼샘플링 배율
W, H = OUT_W * S, OUT_H * S
FONT = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
BOLD, SEMI, REG = 6, 4, 0               # AppleSDGothicNeo.ttc 인덱스(Bold / SemiBold / Regular)
BLUE, TEXT, MUTED = (88, 166, 255), (230, 237, 243), (139, 148, 158)
font = lambda size, idx: ImageFont.truetype(FONT, size * S, index=idx)


def background():
    y = np.linspace(0, 1, H)[:, None]
    base = np.array([11, 15, 20]) * (1 - y[..., None]) + np.array([15, 20, 27]) * y[..., None]
    img = np.broadcast_to(base, (H, W, 3)).astype(np.float32).copy()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    def glow(cx, cy, r, rgb, a):
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
        k = np.clip(1 - d, 0, 1) ** 2 * a
        img[...] = img * (1 - k[..., None]) + np.array(rgb) * k[..., None]
    glow(W * 0.72, H * 0.45, W * 0.42, (40, 90, 170), 0.55)
    glow(W * 0.04, H * 1.0, W * 0.35, (90, 60, 150), 0.35)
    return Image.fromarray(img.clip(0, 255).astype(np.uint8)).convert('RGBA')


def draw_ball(img, cx, cy, size):
    segs, dots, rim = ball.primitives(n_edge_dots=900, n_face_dots=420)
    k = size / ball.SIZE
    layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    X = lambda x: cx + (x - ball.SIZE / 2) * k
    Y = lambda y: cy + (y - ball.SIZE / 2) * k
    # 테두리 빛번짐(여러 겹의 반투명 원)
    for i in range(18):
        r = rim * k * (1.0 + i * 0.012)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(88, 166, 255, int(16 * (1 - i / 18))), width=int(6 * S))
    for x0, y0, x1, y1, o in sorted(segs, key=lambda s: s[4]):   # 뒷면부터
        d.line([X(x0), Y(y0), X(x1), Y(y1)], fill=BLUE + (int(255 * o * 0.95),), width=int(2.6 * S))
    for x, y, r, o in dots:
        rr = r * k * 1.6
        d.ellipse([X(x) - rr, Y(y) - rr, X(x) + rr, Y(y) + rr], fill=(142, 195, 255, int(255 * o)))
    r = rim * k
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(120, 184, 255, 200), width=int(3 * S))
    img.alpha_composite(layer)


def draw_logo(d, x, y, size):
    # 사이트 로고 축구공(viewBox 24: 원 r9 + 오각형 + 바깥 5선)
    k = size / 24
    P = lambda px, py: (x + px * k, y + py * k)
    w = int(1.4 * k)
    d.ellipse([*P(3, 3), *P(21, 21)], outline=BLUE, width=w)
    pent = [P(12, 7.7), P(16.09, 10.67), P(14.53, 15.48), P(9.47, 15.48), P(7.91, 10.67)]
    d.line(pent + [pent[0]], fill=BLUE, width=w, joint='curve')
    for a, b in [((12, 7.7), (12, 3.8)), ((16.09, 10.67), (19.8, 9.47)), ((14.53, 15.48), (16.82, 18.63)),
                 ((9.47, 15.48), (7.18, 18.63)), ((7.91, 10.67), (4.2, 9.47))]:
        d.line([P(*a), P(*b)], fill=BLUE, width=w)
    for px, py in pent + [P(12, 3.8), P(19.8, 9.47), P(16.82, 18.63), P(7.18, 18.63), P(4.2, 9.47)]:
        d.ellipse([px - w / 2, py - w / 2, px + w / 2, py + w / 2], fill=BLUE)   # 둥근 선 끝


def main():
    img = background()
    draw_ball(img, W * 0.735, H * 0.5, H * 1.02)
    d = ImageDraw.Draw(img)
    L = 150 * S
    # 로고 + 워드마크
    draw_logo(d, L, 150 * S, 118 * S)
    f_word = font(104, BOLD)
    d.text((L + 146 * S, 150 * S + 2 * S), 'Fot', font=f_word, fill=TEXT)
    d.text((L + 146 * S + d.textlength('Fot', font=f_word), 150 * S + 2 * S), 'Data', font=f_word, fill=BLUE)
    d.text((L + 150 * S, 262 * S), 'A I   F O O T B A L L', font=font(30, SEMI), fill=MUTED)
    # 헤드라인
    f_h = font(118, BOLD)
    d.text((L, 430 * S), '축구를 데이터로', font=f_h, fill=TEXT)
    d.text((L, 580 * S), '예측하다', font=f_h, fill=BLUE)
    d.text((L, 770 * S), '5대 리그 + 챔피언스리그 · AI 경기 예측 · 순위 예측 시뮬레이션', font=font(40, REG), fill=MUTED)
    # 리그 태그
    x, y, f_p = L, 880 * S, font(34, SEMI)
    for name in ['EPL', '라리가', '분데스리가', '세리에A', '리그앙', 'UCL']:
        tw = d.textlength(name, font=f_p)
        d.rounded_rectangle([x, y, x + tw + 60 * S, y + 72 * S], radius=36 * S, fill=(22, 27, 34, 230), outline=(56, 90, 140), width=2 * S)
        d.text((x + 30 * S, y + 14 * S), name, font=f_p, fill=BLUE)
        x += tw + 60 * S + 18 * S
    d.text((L, 1110 * S), 'fotdata-api.vercel.app', font=font(34, REG), fill=(110, 118, 129))
    img.resize((OUT_W, OUT_H), Image.LANCZOS).convert('RGB').save('og-image.png', optimize=True)
    print('og-image.png', OUT_W, 'x', OUT_H)


if __name__ == '__main__':
    main()
