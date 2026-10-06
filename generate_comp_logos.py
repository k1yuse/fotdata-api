"""컵대회·대륙/국가대표 대회 아이콘(2026-10-06) — API-Football 대회 로고를 내려받아 여백을 잘라 logos/comp/<API-Football 대회 ID>.png로.
선수 카드·구단 우승 기록의 트로피 줄과(앞으로) 리그 페이지 컵대회 탭이 씀. 한국어 이름·짧은 이름은 FotData.html COMP_INFO.
5대 리그·챔스는 logos/league/(심볼만 자른 것)를 그대로 씀. 다시 만들려면: python generate_comp_logos.py"""
import io, os, requests
from PIL import Image

COMPS = {
    45: "FA Cup", 48: "EFL Cup", 528: "Community Shield",
    143: "Copa del Rey", 556: "Supercopa de España",
    81: "DFB-Pokal", 529: "DFL-Supercup",
    137: "Coppa Italia", 547: "Supercoppa Italiana",
    66: "Coupe de France", 65: "Coupe de la Ligue", 526: "Trophée des Champions",
    3: "UEFA Europa League", 848: "UEFA Conference League", 531: "UEFA Super Cup",
    1168: "FIFA Intercontinental Cup",
    4: "UEFA Euro", 5: "UEFA Nations League", 9: "Copa América", 6: "Africa Cup of Nations", 7: "AFC Asian Cup",
    480: "Olympics",
}
# 남색·검정이라 어두운 바탕에 안 보이는 로고는 흰색으로(리그 아이콘의 EPL·챔스와 같은 처리)
WHITE = {143, 66, 556, 480}
# 월드컵(1)·클럽 월드컵(15)은 API-Football 로고가 "WORLD CUP" 글자 임시 이미지라 안 받음 → 화면에서 금색 트로피 아이콘
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logos", "comp")
H = 128

def main():
    os.makedirs(OUT, exist_ok=True)
    for cid, name in COMPS.items():
        r = requests.get(f"https://media.api-sports.io/football/leagues/{cid}.png", timeout=20)
        im = Image.open(io.BytesIO(r.content)).convert("RGBA")
        bb = im.split()[3].point(lambda a: 255 if a > 24 else 0).getbbox()
        if bb:
            im = im.crop(bb)
        s = H / max(im.width, im.height)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
        if cid in WHITE:
            im = Image.merge("RGBA", (Image.new("L", im.size, 255),) * 3 + (im.split()[3],))
        im.save(os.path.join(OUT, f"{cid}.png"), optimize=True)
        print(cid, name, im.size)

if __name__ == "__main__":
    main()
