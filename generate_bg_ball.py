"""
메인 페이지(FotData.html) 유리 테마 배경용 와이어프레임 축구공 SVG(bg-ball.svg) 생성.
랜딩 3D 연출(landing-story.js)의 공과 같은 구조를 정지 이미지로 투영한 것 —
실시간 WebGL 대신 SVG 한 장이라 앱 페이지의 스크롤·배터리 부담이 없음.
  - 깎은 정이십면체(축구공) 90개 모서리를 구면 위 호로 그림
  - 카메라를 향한 앞면 선은 진하게, 뒷면은 흐리게(랜딩과 같은 입체감)
  - 모서리를 따라 흩뿌린 점 + 12개 오각형 면을 채운 점(클래식 축구공 무늬)
  - 원근을 고려한 윤곽 원(로고의 바깥 원)
같은 투명도 단계끼리 path 하나로 묶어서 파일을 작게 유지.
모양을 바꿀 땐 아래 상수(YAW/PITCH/SEED 등)만 고치고 다시 실행: python generate_bg_ball.py
"""
import math
import random

SIZE = 1000                 # viewBox
R = 2.2                     # 공 반지름(랜딩과 동일)
CAM_D = 8.6                 # 카메라 거리(랜딩 공 장면과 동일)
YAW, PITCH = math.radians(24), math.radians(-16)
BLUE = '#58a6ff'
LEVELS = 8                  # 투명도 단계 수
SEED = 7
OUT = 'bg-ball.svg'

random.seed(SEED)
f = (1 + 5 ** 0.5) / 2
V = [(0, 1, f), (0, 1, -f), (0, -1, f), (0, -1, -f), (1, f, 0), (1, -f, 0), (-1, f, 0), (-1, -f, 0),
     (f, 0, 1), (f, 0, -1), (-f, 0, 1), (-f, 0, -1)]
lerp = lambda a, b, t: tuple(a[k] + (b[k] - a[k]) * t for k in range(3))
norm = lambda p: math.sqrt(sum(x * x for x in p))
unit = lambda p: tuple(x / norm(p) for x in p)
dot = lambda a, b: sum(x * y for x, y in zip(a, b))
cross = lambda a, b: (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
nb = [[j for j in range(12) if j != i and abs(sum((V[i][k] - V[j][k]) ** 2 for k in range(3)) - 4) < 1e-6] for i in range(12)]

edges, pentas = [], []
for i in range(12):
    for j in nb[i]:
        if i < j:
            edges.append((lerp(V[i], V[j], 1 / 3), lerp(V[i], V[j], 2 / 3)))
for i in range(12):
    v, pts = V[i], [lerp(V[i], V[j], 1 / 3) for j in nb[i]]
    n, ref = unit(v), tuple(pts[0][k] - v[k] for k in range(3))
    def ang(p):
        q = tuple(p[k] - v[k] for k in range(3))
        return math.atan2(dot(cross(ref, q), n), dot(ref, q))
    pts.sort(key=ang)
    for k in range(5):
        edges.append((pts[k], pts[(k + 1) % 5]))
    pentas.append((v, pts))


def rotate(p):
    x, y, z = p
    x, z = x * math.cos(YAW) + z * math.sin(YAW), -x * math.sin(YAW) + z * math.cos(YAW)
    y, z = y * math.cos(PITCH) - z * math.sin(PITCH), y * math.sin(PITCH) + z * math.cos(PITCH)
    return (x, y, z)


# 투영: 카메라는 +z 쪽 CAM_D 거리. 윤곽 반지름 = F·R/√(d²−R²)이 SIZE의 42%가 되도록 F를 맞춤
RIM = SIZE * 0.42
F = RIM * math.sqrt(CAM_D ** 2 - R ** 2) / R


def project(p):
    s = F / (CAM_D - p[2])
    return (SIZE / 2 + p[0] * s, SIZE / 2 - p[1] * s)


def facing(p):
    # 구면 위 점 p의 바깥 법선과 카메라 방향의 내적 → 0.1(뒷면)~1(앞면)
    to_cam = unit((-p[0], -p[1], CAM_D - p[2]))
    t = max(0.0, min(1.0, (dot(unit(p), to_cam) + 0.15) / 0.5))
    return 0.1 + 0.9 * t * t * (3 - 2 * t)


def on_sphere(a, b, t):
    return tuple(x * R for x in unit(lerp(a, b, t)))


def primitives(n_edge_dots=520, n_face_dots=260, seed=SEED):
    """투영된 공 요소: (선분 [(x0,y0,x1,y1,불투명도)], 점 [(x,y,r,불투명도)], 윤곽 반지름) — SIZE 좌표계.
    generate_og_image.py(링크 공유 썸네일)도 같은 그림을 쓰려고 함수로 뺌"""
    rnd = random.Random(seed)
    segs, dots = [], []
    for a, b in edges:
        for k in range(10):
            p0, p1 = rotate(on_sphere(a, b, k / 10)), rotate(on_sphere(a, b, (k + 1) / 10))
            mid = tuple((p0[i] + p1[i]) / 2 for i in range(3))
            (x0, y0), (x1, y1) = project(p0), project(p1)
            segs.append((x0, y0, x1, y1, facing(mid)))
    for _ in range(n_edge_dots):   # 모서리를 따라 흩뿌린 점
        a, b = rnd.choice(edges)
        p = rotate(on_sphere(a, b, rnd.random()))
        x, y = project(p); dots.append((x, y, 1.9, facing(p)))
    for _ in range(n_face_dots):   # 오각형 면 채움
        c, pts = rnd.choice(pentas)
        k = rnd.randrange(5)
        u, w = rnd.random(), rnd.random()
        if u + w > 1:
            u, w = 1 - u, 1 - w
        q = tuple(c[j] + (pts[k][j] - c[j]) * u + (pts[(k + 1) % 5][j] - c[j]) * w for j in range(3))
        p = rotate(tuple(x * R * 0.995 for x in unit(q)))
        x, y = project(p); dots.append((x, y, 1.6, facing(p)))
    return segs, dots, RIM


def write_svg():
    segs_all, dots_all, _ = primitives()
    level = lambda o: min(LEVELS - 1, int(o * LEVELS))
    segs = [[] for _ in range(LEVELS)]
    dots = [[] for _ in range(LEVELS)]
    for x0, y0, x1, y1, o in segs_all:
        segs[level(o)].append(f'M{x0:.1f} {y0:.1f}L{x1:.1f} {y1:.1f}')
    for x, y, r, o in dots_all:
        dots[level(o)].append(f'M{x - r:.1f} {y:.1f}a{r} {r} 0 1 0 {2 * r} 0a{r} {r} 0 1 0 {-2 * r} 0')
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" width="{SIZE}" height="{SIZE}">',
           '<defs><radialGradient id="g" cx="50%" cy="50%" r="50%">'
           '<stop offset="0" stop-color="#58a6ff" stop-opacity="0.05"/>'
           '<stop offset="0.78" stop-color="#58a6ff" stop-opacity="0.05"/>'
           '<stop offset="0.88" stop-color="#58a6ff" stop-opacity="0.14"/>'
           '<stop offset="1" stop-color="#58a6ff" stop-opacity="0"/></radialGradient></defs>',
           f'<circle cx="{SIZE / 2}" cy="{SIZE / 2}" r="{RIM * 1.15:.1f}" fill="url(#g)"/>']
    for lv in range(LEVELS):   # 뒷면(흐린 단계)부터 그려서 앞면이 위에 오게
        op = (lv + 0.5) / LEVELS
        if segs[lv]:
            out.append(f'<path d="{"".join(segs[lv])}" stroke="{BLUE}" stroke-opacity="{op * 0.85:.2f}" stroke-width="2.2" stroke-linecap="round" fill="none"/>')
        if dots[lv]:
            out.append(f'<path d="{"".join(dots[lv])}" fill="#8ec3ff" fill-opacity="{op * 0.9:.2f}"/>')
    out.append(f'<circle cx="{SIZE / 2}" cy="{SIZE / 2}" r="{RIM:.1f}" fill="none" stroke="#78b8ff" stroke-opacity="0.7" stroke-width="2.4"/>')
    out.append('</svg>')
    open(OUT, 'w').write('\n'.join(out))
    print(OUT, sum(len(s) for s in out) // 1024, 'KB')


if __name__ == '__main__':
    write_svg()
