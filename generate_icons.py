"""
PWA/파비콘 아이콘(icon-192.png, icon-512.png) — 2026-10-10 앱 theme-a 톤.
밤하늘 남색 둥근 사각형(모서리 ~23%) + 위쪽 라벤더 빛 + 사이트 로고 공(원+오각형+5선)을 오로라(하늘색 → 라벤더 → 복숭아)로,
공 뒤엔 은은한 빛번짐. 로고·오로라 그리는 법은 share_card.py(logo_mask·iris_paint)와 같음.
다시 만들기: python generate_icons.py
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from share_card import iris_paint, logo_mask

OUT = '/Users/kim-yuseung/fotdata-api'


def make_icon(size, corner_ratio=0.229, ball_ratio=0.62, sw=1.55):
    S = 4                                   # 4배로 그려 줄임(둥근 모서리 매끈하게)
    big = size * S
    # 1) 배경: 위가 조금 밝은 남색 + 위쪽 라벤더 빛
    yy, xx = np.mgrid[0:big, 0:big].astype(np.float32) / big
    top, bot = np.array([24, 28, 56], np.float32), np.array([7, 10, 22], np.float32)
    rgb = top * (1 - yy[..., None]) + bot * yy[..., None]
    d = np.sqrt((xx - 0.5) ** 2 + (yy - 0.42) ** 2) / 0.62
    k = (np.clip(1 - d, 0, 1) ** 2 * 0.5)[..., None]
    rgb = rgb * (1 - k) + np.array([92, 84, 170], np.float32) * k
    bg = Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).convert('RGBA')
    # 2) 둥근 사각형 모양 + 위쪽 테두리 빛 한 줄(유리)
    shape = Image.new('L', (big, big), 0)
    ImageDraw.Draw(shape).rounded_rectangle([0, 0, big - 1, big - 1], radius=big * corner_ratio, fill=255)
    rim = Image.new('L', (big, big), 0)
    ImageDraw.Draw(rim).rounded_rectangle([0, 0, big - 1, big - 1], radius=big * corner_ratio, outline=255, width=max(2, big // 160))
    rim_fade = Image.fromarray((np.asarray(rim, np.float32) * np.clip(1 - yy * 2.2, 0, 1) * 0.28).astype(np.uint8))
    bg.alpha_composite(Image.merge('RGBA', [Image.new('L', (big, big), 255)] * 3 + [rim_fade]))
    # 3) 로고 공(오로라) + 뒤 빛번짐
    ball = int(big * ball_ratio / (18 / 24))   # 로고 viewBox 24 중 공(지름 18)이 ball_ratio가 되게
    off = (big - ball) // 2
    m = logo_mask(ball, sw)
    glow = Image.new('L', (big, big), 0)
    glow.paste(m, (off, off))
    glow = glow.filter(ImageFilter.GaussianBlur(big * 0.035))
    glow = Image.fromarray((np.asarray(glow, np.float32) * 0.55).astype(np.uint8))
    bg.alpha_composite(Image.merge('RGBA', [Image.new('L', (big, big), c) for c in (170, 160, 255)] + [glow]))
    bg.alpha_composite(iris_paint(m, diag=True), (off, off))
    bg.putalpha(shape)
    return bg.resize((size, size), Image.LANCZOS)


if __name__ == '__main__':
    for size, name in [(192, 'icon-192.png'), (512, 'icon-512.png')]:
        make_icon(size).save(f'{OUT}/{name}', optimize=True)
        print('저장:', name, size)
