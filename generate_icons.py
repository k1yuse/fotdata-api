"""
PWA/파비콘 아이콘(icon-192.png, icon-512.png)을 막대그래프 대신 축구공 마크로 재생성.
기존 파일과 동일하게: 파란(#58a6ff) 둥근 사각형 배경(모서리 반경 ~23%) + 안쪽에
사이트 전역에서 이미 쓰고 있는 축구공 SVG(원+오각형+5선, stroke-linecap/join round)를
같은 좌표 비율로 그대로 재현. 생성 후 삭제해도 되는 1회성 스크립트.
"""
from PIL import Image, ImageDraw
import math

BG = (13, 17, 23, 255)    # 배경: 사이트와 동일한 진한 남색
BALL = (88, 166, 255, 255)  # 축구공: 브랜드 블루

# 원본 SVG 좌표 (viewBox 0 0 24 24)
CIRCLE_CENTER = (12, 12)
CIRCLE_R = 9
PENTAGON = [(12, 7.7), (16.09, 10.67), (14.53, 15.48), (9.47, 15.48), (7.91, 10.67)]
LINES = [
    ((12, 7.7), (12, 3.8)),
    ((16.09, 10.67), (19.8, 9.47)),
    ((14.53, 15.48), (16.82, 18.63)),
    ((9.47, 15.48), (7.18, 18.63)),
    ((7.91, 10.67), (4.2, 9.47)),
]
STROKE_W_UNITS = 1.4  # SVG stroke-width


def make_icon(size, corner_ratio=0.229, ball_ratio=0.6):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 1) 배경: 둥근 사각형
    radius = size * corner_ratio
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=BG)

    # 2) 축구공: viewBox 좌표 -> 픽셀 좌표 변환
    scale = (ball_ratio * size) / (CIRCLE_R * 2)
    cx, cy = size / 2, size / 2

    def tx(pt):
        x, y = pt
        return (cx + (x - 12) * scale, cy + (y - 12) * scale)

    stroke_w = max(1, round(STROKE_W_UNITS * scale))
    cap_r = stroke_w / 2

    def round_line(p1, p2):
        draw.line([p1, p2], fill=BALL, width=stroke_w)
        for p in (p1, p2):
            draw.ellipse([p[0] - cap_r, p[1] - cap_r, p[0] + cap_r, p[1] + cap_r], fill=BALL)

    # 원 (외곽선)
    circle_bbox = [cx - CIRCLE_R * scale, cy - CIRCLE_R * scale, cx + CIRCLE_R * scale, cy + CIRCLE_R * scale]
    draw.ellipse(circle_bbox, outline=BALL, width=stroke_w)

    # 오각형 (외곽선, 둥근 모서리)
    pts = [tx(p) for p in PENTAGON]
    for i in range(len(pts)):
        round_line(pts[i], pts[(i + 1) % len(pts)])

    # 5개 외부 선
    for a, b in LINES:
        round_line(tx(a), tx(b))

    return img


for size, name in [(192, "icon-192.png"), (512, "icon-512.png")]:
    icon = make_icon(size)
    icon.save(f"/Users/kim-yuseung/fotdata-api/{name}")
    print(f"저장 완료: {name} ({size}x{size})")
