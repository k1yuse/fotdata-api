"""
구단 로고 고화질·최신판 만들기 (2026-09-30) — 사이트 전체가 쓰는 로고를 football-data PNG(200px, 옛 버전 섞임) 대신
영문 위키백과 구단 문서 머리 인포박스의 **현재 엠블럼**(대부분 SVG 원본)을 400px 투명 WebP로 바꿔 저장.
  - 출력: logos/hd/<slug>.webp(160px — 목록·카드 등 사이트 대부분, 화면 최대 72px의 2배) + logos/hd/l/<slug>.webp(400px — 구단 둘러보기
    가운데 카드·공유 이미지처럼 크게 띄우는 곳만, 프론트 logoLarge()) + fotdata_model/team_logos_hd.json {팀 이름: 160px 주소}
    (처음엔 400px 하나만 썼더니 20~40px 칸에 40KB짜리가 내려가 첫 화면에서만 740KB 낭비 — PageSpeed 지적)
  - 서버(main.py _team_logos)가 team_logos.json 위에 덮어써서 /logos·순위표·빅매치·공유 썸네일이 모두 같은 로고를 씀
  - 실행: python generate_logos_hd.py  (키 불필요, 약 2~3분) — 승격팀이 생기거나 엠블럼이 바뀌면 다시 실행

왜 이 출처인가(2026-09-30 확인):
  - football-data SVG는 SVG가 있는 140팀 중 95팀이 PNG와 다른 디자인 버전, 74팀은 배경 사각형(토트넘은 불투명도 4.9% 숨은 사각형)
  - 위키미디어 공용(위키데이터 P154)은 자유 라이선스 로고만 있어 161팀 중 59팀뿐(큰 구단 엠블럼은 대부분 없음)
  - 위키백과 pageimages API는 저작권 있는 로고를 빼고 줌 → 인포박스 원문에서 파일 이름(`| image = Arsenal FC.svg`)을 직접 읽음
검수: 사용자가 지금 로고 ↔ 새 로고를 나란히 보고 확인함. 이상하게 잡힌 로고는 EXCLUDE(지금 로고 유지).
로고는 구단의 상표 — 구단을 알아보게 하는 용도(지금까지 football-data 로고와 같은 성격)로만 씀.
"""
import io
import json
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, 'logos', 'hd')
SITE = 'https://fotdata-api.vercel.app'
SIZE = 400                      # 큰 로고: 구단 둘러보기 200px × 2배 화면
SMALL = 160                     # 기본 로고: 화면 최대 72px × 2배 화면
UA = {'User-Agent': 'FotData/1.0 (https://fotdata-api.vercel.app; club crest refresh)'}
API = 'https://en.wikipedia.org/w/api.php'
# 사용자 검수(2026-09-30)에서 이상하게 잡힌 로고 → 지금(football-data) 로고 유지
EXCLUDE = {
    'Brighton & Hove Albion FC',   # "125 YEARS" 창단 기념판(한시적)
    'AFC Ajax',                    # 색이 빠진 버전
    'Hellas Verona FC',            # 남색 단색 버전
    'Luton Town FC',               # 흐린 남색 단색 버전
    'Sabah FK',                    # 분홍색 버전(리브랜딩 불확실)
    'Venezia FC',                  # 방패 대신 "V" 로고
    'FC Metz',                     # 글자+십자가만 있는 로고
    'Tottenham Hotspur FC',        # 남색 단색이라 사이트 어두운 배경에 묻힘(사용자 요청 — 예전 로고 유지)
}


def slug(t):
    s = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', '-', s).strip('-')


def get(url, **kw):
    for a in range(5):
        r = requests.get(url, headers=UA, timeout=30, **kw)
        if r.status_code != 429:
            return r
        time.sleep(float(r.headers.get('Retry-After') or 2 * (a + 1)))
    return r


def infobox_file(title):
    wt = get(API, params={'action': 'parse', 'page': title, 'prop': 'wikitext', 'section': 0, 'redirects': 1, 'format': 'json'}).json()
    txt = (wt.get('parse') or {}).get('wikitext', {}).get('*', '')
    for key in ('image', 'logo', 'crest'):
        m = re.search(r'\|\s*' + key + r'\s*=\s*(?:\[\[(?:File|Image):)?([^|\]\n<{}]+?\.(?:svg|png|gif|jpe?g))', txt, re.I)
        if m:
            return m.group(1).strip()
    return None


def corners_opaque(im):
    """네 모서리 중 칠해진 곳 수 — 3개 이상이면 배경 사각형(방패처럼 옆면이 가장자리에 길게 닿는 로고는 모서리가 비어 있음)"""
    w, h = im.size
    px = im.load()
    n = 0
    for cx, cy in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        patch = [px[min(max(cx + dx, 0), w - 1), min(max(cy + dy, 0), h - 1)][3] for dx in (-4, 0, 4) for dy in (-4, 0, 4)]
        n += sum(a > 20 for a in patch) >= 6
    return n


def trim_square(im, size=SIZE, pad=0.02):
    """투명 여백을 잘라내고 정사각형 가운데로(비율 유지) — 팀마다 로고 크기가 들쭉날쭉하지 않게"""
    im = im.convert('RGBA')
    bb = im.split()[3].point(lambda v: 255 if v > 8 else 0).getbbox()
    if bb:
        im = im.crop(bb)
    k = size * (1 - 2 * pad) / max(im.size)
    im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    cv = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    cv.alpha_composite(im, ((size - im.width) // 2, (size - im.height) // 2))
    return cv


def save_both(hd, name):
    os.makedirs(os.path.join(OUT_DIR, 'l'), exist_ok=True)
    hd.save(os.path.join(OUT_DIR, 'l', name + '.webp'), 'WEBP', quality=90, method=6)
    hd.resize((SMALL, SMALL), Image.LANCZOS).save(os.path.join(OUT_DIR, name + '.webp'), 'WEBP', quality=90, method=6)


def one(team, en_title):
    if team in EXCLUDE:
        return team, None, 'excluded'
    fn = infobox_file(en_title) if en_title else None
    if not fn:
        return team, None, 'no_file'
    q = get(API, params={'action': 'query', 'titles': f'File:{fn}', 'prop': 'imageinfo', 'iiprop': 'url|size', 'iiurlwidth': 800, 'format': 'json'}).json()['query']['pages']
    ii = (next(iter(q.values())).get('imageinfo') or [{}])[0]
    url = ii.get('thumburl') or ii.get('url')
    if not url:
        return team, None, 'no_url'
    try:
        im = Image.open(io.BytesIO(get(url).content)).convert('RGBA')
    except Exception as e:
        return team, None, f'bad_image: {e}'
    if max(im.size) < 300:
        return team, None, f'small {im.size}'
    hd = trim_square(im)
    if corners_opaque(hd) >= 3:
        return team, None, 'background'
    save_both(hd, slug(team))
    return team, f'{SITE}/logos/hd/{slug(team)}.webp', 'ok'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    logos = json.load(open(os.path.join(BASE, 'fotdata_model', 'team_logos.json'), encoding='utf-8'))
    wiki = json.load(open(os.path.join(BASE, 'fotdata_model', 'team_wiki.json'), encoding='utf-8'))
    only = [a for a in sys.argv[1:] if not a.startswith('--')]
    teams = [t for t in sorted(logos) if not only or t in only]
    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(lambda t: one(t, wiki.get(t, {}).get('en_title')), teams))
    path = os.path.join(BASE, 'fotdata_model', 'team_logos_hd.json')
    hd = json.load(open(path, encoding='utf-8')) if os.path.exists(path) and only else {}
    for team, url, status in res:
        if url:
            hd[team] = url
        elif team in hd and status == 'excluded':
            del hd[team]
    json.dump(dict(sorted(hd.items())), open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    from collections import Counter
    print(Counter(s.split(':')[0].split(' ')[0] for _, _, s in res))
    print('유지(지금 로고):', [(t, s) for t, u, s in res if not u])
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(OUT_DIR) for f in fs)
    print(f'✅ {len(hd)}팀 → logos/hd ({size / 1e6:.1f}MB)')


if __name__ == '__main__':
    main()
