"""리그 아이콘(2026-10-06) — football-data 리그 로고에서 글자를 빼고 심볼만 잘라 logos/league/<코드>.png로.
풋몹처럼 리그 이름 옆엔 심볼만(글자 로고는 작은 칸에서 안 읽히고, 남색·검정 글자는 어두운 바탕에 묻혔음).
EPL 사자·챔스 별 공은 원본이 남색이라 흰색으로, 분데스리가는 아래 검정 글자를 빼고 빨간 사각형만.
다시 만들려면: python generate_league_logos.py"""
import io, os, requests
from PIL import Image

SRC = {"PL": "PL", "PD": "laliga", "BL1": "BL1", "SA": "c111", "FL1": "FL1", "CL": "CL"}
WHITE = {"PL", "CL"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logos", "league")
H = 128   # 화면 최대 64px의 2배

def alpha_rows(im):
    a = im.split()[3]
    w, h = im.size
    px = a.load()
    return [any(px[x, y] > 40 for x in range(w)) for y in range(h)]

def first_block(flags):
    """위에서부터 처음 나오는 내용 덩어리(빈 줄 3줄 이상 전까지) — 심볼 아래 글자를 떼어냄"""
    start = flags.index(True)
    end, gap = start, 0
    for i in range(start, len(flags)):
        if flags[i]:
            end, gap = i, 0
        else:
            gap += 1
            if gap >= 3:
                break
    return start, end + 1

def main():
    os.makedirs(OUT, exist_ok=True)
    for code, name in SRC.items():
        im = Image.open(io.BytesIO(requests.get(f"https://crests.football-data.org/{name}.png", timeout=20).content)).convert("RGBA")
        if im.width < 200:
            im = im.resize((200, round(im.height * 200 / im.width)), Image.LANCZOS)
        if code == "PL":   # 사자(왼쪽)만 — 글자 "Premier League"와는 빈 세로줄로 나뉨
            a = im.split()[3].load()
            cols = [any(a[x, y] > 40 for y in range(im.height)) for x in range(im.width)]
            x0 = cols.index(True); x1 = x0
            while x1 < im.width and cols[x1]:
                x1 += 1
            im = im.crop((x0, 0, x1, im.height))
        elif code == "BL1":   # 빨간 사각형(위)만
            px = im.load()
            red = [y for y in range(im.height) if sum(1 for x in range(im.width) if px[x, y][0] > 180 and px[x, y][1] < 80 and px[x, y][3] > 200) > im.width // 2]
            im = im.crop((0, red[0], im.width, red[-1] + 1))
        else:
            y0, y1 = first_block(alpha_rows(im))
            im = im.crop((0, y0, im.width, y1))
        im = im.crop(im.split()[3].getbbox())
        if code in WHITE:
            r, g, b, a = im.split()
            im = Image.merge("RGBA", (Image.new("L", im.size, 255),) * 3 + (a,))
        im = im.resize((max(1, round(im.width * H / im.height)), H), Image.LANCZOS)
        im.save(os.path.join(OUT, f"{code}.png"), optimize=True)
        print(code, im.size)

if __name__ == "__main__":
    main()
