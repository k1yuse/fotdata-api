// ── 랜딩 스크롤 3D 연출: "공이 데이터가 되고, 데이터가 예측이 된다" (2026-09-28) ──
// 장면(점들이 스크롤에 따라 모양을 바꿈):
//   0 공  : 로고와 같은 파란 선 축구공(깎은 정이십면체 90개 모서리) — 히어로 뒤, 스토리 끝, 마지막 CTA 뒤
//   1 구름: 공이 흩어진 데이터 점 구름 + 실제 학습 경기 수 카운터
//   2 경기장: 점들이 3D 경기장 라인으로 모이고 홈(파랑)/원정(주황) 중립 원형 — 다가오는 빅매치의 실제 예측 확률
//   3 막대: 점들이 빛줄기처럼 날아가 우승 확률 막대로 쌓임 — 실제 순위 예측 시뮬레이션 결과
// 구조: 점마다 네 장면의 목표 위치/색을 attribute로 미리 넣어두고, 셰이더가 (A→B, t)로 보간 — 수천 개 점도 GPU에서 가볍게 움직임.
// 문구는 전부 실제 HTML(검색 노출). 모션 최소화 설정·WebGL 미지원이면 고정 없이 문구+목록만 보여주는 정적 모드.
// 3D 연출 속 팀은 중립 원형(홈 파랑/원정 주황)으로만 표현하고, 구단 로고는 옆 HTML 카드(실제 데이터 표시)에만 씀.

const API = 'https://fotdata-api.onrender.com';
const THREE_URL = 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.min.js';
const LEAGUE_KO = { PL: 'EPL', PD: '라리가', BL1: '분데스리가', SA: '세리에A', FL1: '리그앙', CL: 'UCL' };
const SIM_LEAGUES = new Set(['PL', 'PD', 'BL1', 'SA', 'FL1']);
const COLORS = { blue: [0.345, 0.651, 1.0], orange: [0.941, 0.533, 0.243], gold: [0.941, 0.753, 0.251], line: [0.55, 0.72, 1.0], dim: [0.35, 0.5, 0.75], face: [0.5, 0.7, 1.0] };
const TOPK = 5;
const BALL_R = 2.2;

const story = document.getElementById('story');
// 공개 전 미리보기: ?story=1 로 접속했을 때만 켬(확인 후 기본 공개로 전환 예정)
const ENABLED = story && (new URLSearchParams(location.search).has('story') || story.dataset.enabled === '1');

// ── 표시 데이터 (API 실패 시 이 값으로 연출이 계속 돌아감) ──
const data = {
  total: 6024,
  match: { home: 'Manchester United FC', away: 'Tottenham Hotspur FC', p: [0.66, 0.21, 0.13], when: '', league: 'PL',
    homeLogo: 'https://crests.football-data.org/66.png', awayLogo: 'https://crests.football-data.org/73.png' },
  sim: { remaining: 330, top: [
    { team: 'Arsenal FC', prob: 0.47, logo: 'https://crests.football-data.org/57.png' }, { team: 'Manchester City FC', prob: 0.46, logo: 'https://crests.football-data.org/65.png' },
    { team: 'Liverpool FC', prob: 0.03, logo: 'https://crests.football-data.org/64.png' }, { team: 'Brighton & Hove Albion FC', prob: 0.02, logo: 'https://crests.football-data.org/397.png' },
    { team: 'Manchester United FC', prob: 0.01, logo: 'https://crests.football-data.org/66.png' }] },
};

const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const ease = t => t * t * (3 - 2 * t);
const shortName = t => t.replace(/ FC$| CF$| AFC$/, '').replace(/^FC /, '');
// 막대 라벨처럼 좁은 자리용 — 긴 구단명은 흔히 부르는 이름으로
const COMPACT = { 'Brighton & Hove Albion': 'Brighton', 'Wolverhampton Wanderers': 'Wolves', 'Tottenham Hotspur': 'Tottenham',
  'Nottingham Forest': "Nott'm Forest", 'Manchester United': 'Man United', 'Manchester City': 'Man City', 'Newcastle United': 'Newcastle' };
const compactName = t => { const s = shortName(t); return COMPACT[s] || s; };
const $ = id => document.getElementById(id);

if (!story) {
  // 섹션이 없는 페이지
} else if (!ENABLED) {
  story.remove();
} else {
  story.hidden = false;
  // 연출이 켜지면 기존 섹션의 회색 띠 배경을 걷어서 페이지 전체를 하나의 어두운 공간으로(회색 띠가 공을 가로로 자르던 문제)
  document.body.classList.add('story-on');
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const webgl = (() => { try { const c = document.createElement('canvas'); return !!(c.getContext('webgl2') || c.getContext('webgl')); } catch (e) { return false; } })();
  renderText();
  loadData();
  if (reduce || !webgl) {
    story.classList.add('static');
  } else {
    // 첫 화면(히어로)이 다 그려진 뒤 여유 있을 때 3D 라이브러리를 불러옴 — 첫 로딩·검색 점수 보호
    const start = () => import(THREE_URL).then(initScene).catch(() => story.classList.add('static'));
    if ('requestIdleCallback' in window) requestIdleCallback(start, { timeout: 1500 }); else setTimeout(start, 600);
  }
}

// ── 문구 채우기 (정적/애니메이션 공통) ──
function renderText() {
  const m = data.match;
  $('story-total').textContent = data.total.toLocaleString();
  $('story-count').textContent = data.total.toLocaleString();
  $('story-home').textContent = shortName(m.home);
  $('story-away').textContent = shortName(m.away);
  for (const [id, src] of [['story-home-logo', m.homeLogo], ['story-away-logo', m.awayLogo]]) {
    const img = $(id);
    if (src && img.getAttribute('src') !== src) { img.src = src; img.style.visibility = ''; }
    else if (!src) img.style.visibility = 'hidden';
  }
  $('story-when').textContent = m.when ? `다가오는 빅매치 · ${m.when}` : '다가오는 빅매치';
  const labels = ['story-bar-h', 'story-bar-d', 'story-bar-a'];
  m.p.forEach((v, i) => {
    const pct = Math.round(v * 100);
    $(labels[i] + '-val').textContent = pct + '%';
    $(labels[i]).dataset.target = pct;
    $(labels[i]).style.setProperty('--target', pct + '%');
  });
  $('story-home-label').textContent = shortName(m.home) + ' 승';
  $('story-away-label').textContent = shortName(m.away) + ' 승';
  $('story-remaining').textContent = data.sim.remaining;
  $('story-sim-list').innerHTML = data.sim.top.map((t, i) =>
    `<li><span class="rk${i === 0 ? ' first' : ''}">${i + 1}</span><span class="nm">${t.logo ? `<img src="${t.logo}" alt="" onerror="this.style.display='none'">` : ''}${shortName(t.team)}</span><span class="pc">${Math.round(t.prob * 100)}%</span></li>`).join('');
}

// ── 실제 데이터 불러오기 ──
async function loadData() {
  const get = u => fetch(API + u).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
  try { const a = await get('/accuracy'); if (a.total_matches) data.total = a.total_matches; } catch (e) {}
  renderText();
  onDataChange();
  try {
    // 빅매치는 경기예측 탭 배너와 같은 서버 규칙(/bigmatch: BIG_CLUBS끼리 가장 가까운 경기)으로 — 두 페이지가 항상 같은 경기
    const big = (await get('/bigmatch')).match;
    if (big) {
      const p = await fetch(API + '/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ home_team: big.home_team, away_team: big.away_team }) }).then(r => r.json());
      const d = new Date(big.date), wd = ['일', '월', '화', '수', '목', '금', '토'][d.getDay()];
      const hm = d.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', hour12: false });
      data.match = { home: big.home_team, away: big.away_team, league: big.league, homeLogo: big.home_logo, awayLogo: big.away_logo,
        p: [p.probabilities.home_win, p.probabilities.draw, p.probabilities.away_win],
        when: `${LEAGUE_KO[big.league] || big.league} · ${d.getMonth() + 1}.${d.getDate()} (${wd}) ${hm}` };
    }
  } catch (e) {}
  renderText();
  onDataChange();
  try {
    // 순위 예측과 같은 방식(현재 승점 + 남은 경기 × 모델 확률)으로 1,000번 시뮬레이션
    // 빅매치가 열리는 리그의 우승 경쟁으로 이어짐(UCL 빅매치면 EPL)
    const league = SIM_LEAGUES.has(data.match.league) ? data.match.league : 'PL';
    const champ = await get('/predict/champion/' + league);
    data.sim = { remaining: champ.remaining, top: simulateTitle(champ, 1000) };
  } catch (e) {}
  renderText();
  onDataChange();
}
let onDataChange = () => {};

function gauss() { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function simulateTitle(d, n) {
  const T = d.teams.length, idx = {}; d.teams.forEach((t, i) => { idx[t.team] = i; });
  const fx = d.fixtures.filter(f => idx[f.home] != null && idx[f.away] != null);
  const wins = new Float64Array(T), pts = new Float64Array(T), delta = new Float64Array(T), sigma = d.sigma ?? 0.15;
  for (let s = 0; s < n; s++) {
    for (let i = 0; i < T; i++) { pts[i] = d.teams[i].points * 1000 + d.teams[i].gd + Math.random() * 0.5; delta[i] = gauss() * sigma; }
    for (const f of fx) {
      const hi = idx[f.home], ai = idx[f.away], e = Math.exp(delta[hi] - delta[ai]);
      const h = f.p[0] * e, a = f.p[2] / e, r = Math.random() * (h + f.p[1] + a);
      if (r < h) pts[hi] += 3000; else if (r < h + f.p[1]) { pts[hi] += 1000; pts[ai] += 1000; } else pts[ai] += 3000;
    }
    let best = 0; for (let i = 1; i < T; i++) if (pts[i] > pts[best]) best = i;
    wins[best]++;
  }
  return d.teams.map((t, i) => ({ team: t.team, logo: t.logo, prob: wins[i] / n })).sort((a, b) => b.prob - a.prob).slice(0, TOPK);
}

// ── 장면 기하 ──
function ballGeometry() {
  // 정이십면체 꼭짓점 → 모서리를 1/3·2/3로 자른 점들로 깎은 정이십면체(축구공) 90개 모서리
  const f = (1 + Math.sqrt(5)) / 2;
  const V = [[0, 1, f], [0, 1, -f], [0, -1, f], [0, -1, -f], [1, f, 0], [1, -f, 0], [-1, f, 0], [-1, -f, 0], [f, 0, 1], [f, 0, -1], [-f, 0, 1], [-f, 0, -1]];
  const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
  const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const nb = V.map((v, i) => V.map((w, j) => j).filter(j => j !== i && Math.abs(d2(V[i], V[j]) - 4) < 1e-6));
  const edges = [], pentas = [];
  for (let i = 0; i < 12; i++) for (const j of nb[i]) if (i < j) edges.push([lerp(V[i], V[j], 1 / 3), lerp(V[i], V[j], 2 / 3)]);
  for (let i = 0; i < 12; i++) {
    const v = V[i], pts = nb[i].map(j => lerp(v, V[j], 1 / 3));
    // 꼭짓점 둘레로 각도 정렬해서 오각형으로 잇기
    const n = v.map(x => x / Math.hypot(...v));
    const ref = pts[0].map((x, k) => x - v[k]);
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const ang = p => { const q = p.map((x, k) => x - v[k]); return Math.atan2(dot(cross(ref, q), n), dot(ref, q)); };
    pts.sort((a, b) => ang(a) - ang(b));
    for (let k = 0; k < 5; k++) edges.push([pts[k], pts[(k + 1) % 5]]);
    pentas.push({ c: v, v: pts });
  }
  return { edges, pentas };   // 모서리 90개, 오각형 면 12개
}
const ballEdges = () => ballGeometry().edges;
const onSphere = (a, b, t, R) => { const p = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; const l = Math.hypot(...p); return [p[0] / l * R, p[1] / l * R, p[2] / l * R]; };

function pitchSegments() {
  // 105×68m 경기장을 가로 6.4로 축소, y=0 평면(XZ). [x1,z1,x2,z2] 선분 목록(원·호는 잘게 쪼갬)
  const s = 6.4 / 105, L = 52.5, W = 34, segs = [];
  const line = (x1, z1, x2, z2) => segs.push([x1 * s, z1 * s, x2 * s, z2 * s]);
  const rect = (x1, z1, x2, z2) => { line(x1, z1, x2, z1); line(x2, z1, x2, z2); line(x2, z2, x1, z2); line(x1, z2, x1, z1); };
  const arc = (cx, cz, r, a0, a1, n = 28) => { for (let k = 0; k < n; k++) { const t0 = a0 + (a1 - a0) * k / n, t1 = a0 + (a1 - a0) * (k + 1) / n; line(cx + r * Math.cos(t0), cz + r * Math.sin(t0), cx + r * Math.cos(t1), cz + r * Math.sin(t1)); } };
  rect(-L, -W, L, W); line(0, -W, 0, W); arc(0, 0, 9.15, 0, Math.PI * 2, 40);
  for (const sgn of [-1, 1]) {
    rect(sgn * L, -20.16, sgn * (L - 16.5), 20.16);
    rect(sgn * L, -9.16, sgn * (L - 5.5), 9.16);
    rect(sgn * L, -3.66, sgn * (L + 2.2), 3.66);
    const cx = sgn * (L - 11), a = Math.acos(5.5 / 9.15);
    if (sgn < 0) arc(cx, 0, 9.15, -a, a, 14); else arc(cx, 0, 9.15, Math.PI - a, Math.PI + a, 14);
  }
  return segs;
}

function buildTargets(N) {
  const P = [new Float32Array(N * 3), new Float32Array(N * 3), new Float32Array(N * 3), new Float32Array(N * 3)];
  const C = [new Float32Array(N * 3), new Float32Array(N * 3), new Float32Array(N * 3), new Float32Array(N * 3)];
  const seed = new Float32Array(N);
  const set = (arr, i, v) => { arr[i * 3] = v[0]; arr[i * 3 + 1] = v[1]; arr[i * 3 + 2] = v[2]; };
  const R = BALL_R, { edges, pentas } = ballGeometry(), nFace = Math.floor(N * 0.22);
  // 0 공: 90개 모서리(구면 위 호)를 따라 고르게 + 12개 오각형 면은 점으로 은은하게 채움(클래식 축구공 무늬)
  for (let i = 0; i < N; i++) {
    seed[i] = Math.random();
    if (i < nFace) {
      const pe = pentas[i % 12], k = Math.floor(Math.random() * 5), a = pe.v[k], b = pe.v[(k + 1) % 5];
      let u = Math.random(), w = Math.random(); if (u + w > 1) { u = 1 - u; w = 1 - w; }
      const q = [0, 1, 2].map(j => pe.c[j] + (a[j] - pe.c[j]) * u + (b[j] - pe.c[j]) * w);
      const l = Math.hypot(...q);
      set(P[0], i, [q[0] / l * R * 0.995, q[1] / l * R * 0.995, q[2] / l * R * 0.995]);
      set(C[0], i, COLORS.face);
    } else {
      const e = edges[i % edges.length], t = Math.random();
      set(P[0], i, onSphere(e[0], e[1], t, R));
      set(C[0], i, COLORS.blue);
    }
  }
  // 1 구름: 납작한 가우시안 은하
  for (let i = 0; i < N; i++) {
    const r = 0.7 + Math.abs(gauss()) * 1.15, th = Math.random() * Math.PI * 2;
    set(P[1], i, [Math.cos(th) * r * 1.3 + gauss() * 0.2, gauss() * 0.45, Math.sin(th) * r * 0.9 + gauss() * 0.2]);
    const k = Math.random();
    set(C[1], i, k < 0.82 ? COLORS.blue : k < 0.91 ? COLORS.line : COLORS.orange);
  }
  // 2 경기장: 라인 88% + 홈/원정 원형 6%씩
  const segs = pitchSegments(), lens = segs.map(g => Math.hypot(g[2] - g[0], g[3] - g[1])), total = lens.reduce((a, b) => a + b, 0);
  const nDisc = Math.floor(N * 0.06);
  for (let i = 0; i < N; i++) {
    if (i < nDisc * 2) {
      const home = i < nDisc, r = 0.42 * Math.sqrt(Math.random()), th = Math.random() * Math.PI * 2;
      set(P[2], i, [(home ? -1.55 : 1.55) + Math.cos(th) * r, 0.02, Math.sin(th) * r]);
      set(C[2], i, home ? COLORS.blue : COLORS.orange);
    } else {
      let u = Math.random() * total, k = 0; while (u > lens[k] && k < segs.length - 1) { u -= lens[k]; k++; }
      const g = segs[k], t = u / lens[k];
      set(P[2], i, [g[0] + (g[2] - g[0]) * t + gauss() * 0.008, 0, g[1] + (g[3] - g[1]) * t + gauss() * 0.008]);
      set(C[2], i, COLORS.line);
    }
  }
  fillBars(P[3], C[3], N);
  return { P, C, seed };
}

// 3 막대: 점 = 시뮬레이션 우승 횟수에 비례해 배분(막대 높이가 곧 우승 확률)
function barLayout() {
  const top = data.sim.top, max = Math.max(...top.map(t => t.prob), 0.01);
  const gap = 1.05, x0 = -gap * (top.length - 1) / 2, H = 2.7, base = -1.45;
  return top.map((t, i) => ({ team: t.team, logo: t.logo, prob: t.prob, x: x0 + gap * i, h: Math.max(0.06, H * t.prob / max), base }));
}
function fillBars(P, C, N) {
  const bars = barLayout(), sum = bars.reduce((a, b) => a + Math.max(b.prob, 0.004), 0);
  let i = 0;
  bars.forEach((b, k) => {
    const n = k === bars.length - 1 ? N - i : Math.round(N * Math.max(b.prob, 0.004) / sum);
    for (let j = 0; j < n && i < N; j++, i++) {
      P[i * 3] = b.x + (Math.random() - 0.5) * 0.46;
      P[i * 3 + 1] = b.base + Math.random() * b.h;
      P[i * 3 + 2] = (Math.random() - 0.5) * 0.46;
      const c = k === 0 ? COLORS.gold : COLORS.blue;
      C[i * 3] = c[0]; C[i * 3 + 1] = c[1]; C[i * 3 + 2] = c[2];
    }
  });
}

// ── 3D 장면 ──
function initScene(THREE) {
  const canvas = $('story-canvas');
  const mobile = window.innerWidth <= 768;
  const N = mobile ? 2600 : 6024;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, mobile ? 1.5 : 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
  const group = new THREE.Group(); scene.add(group);

  const { P, C, seed } = buildTargets(N);
  const geo = new THREE.BufferGeometry();
  P.forEach((p, k) => geo.setAttribute('p' + k, new THREE.BufferAttribute(p, 3)));
  C.forEach((c, k) => geo.setAttribute('c' + k, new THREE.BufferAttribute(c, 3)));
  geo.setAttribute('position', new THREE.BufferAttribute(P[0], 3));   // three.js 필수 attribute(경계 계산용)
  geo.setAttribute('seed', new THREE.BufferAttribute(seed, 1));
  const uniforms = { uA: { value: 0 }, uB: { value: 0 }, uT: { value: 0 }, uTime: { value: 0 }, uSize: { value: mobile ? 2.6 : 2.5 },
    uPR: { value: renderer.getPixelRatio() }, uOpacity: { value: 1 }, uBall: { value: 1 } };
  const mat = new THREE.ShaderMaterial({
    uniforms, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: `
      attribute vec3 p0; attribute vec3 p1; attribute vec3 p2; attribute vec3 p3;
      attribute vec3 c0; attribute vec3 c1; attribute vec3 c2; attribute vec3 c3;
      attribute float seed;
      uniform float uA, uB, uT, uTime, uSize, uPR, uBall;
      varying vec3 vColor;
      varying float vFace;
      vec3 pick(float i, vec3 a, vec3 b, vec3 c, vec3 d) { return i < 0.5 ? a : (i < 1.5 ? b : (i < 2.5 ? c : d)); }
      void main() {
        float t = clamp((uT - seed * 0.35) / 0.65, 0.0, 1.0);
        t = t * t * (3.0 - 2.0 * t);
        vec3 pos = mix(pick(uA, p0, p1, p2, p3), pick(uB, p0, p1, p2, p3), t);
        // 경기장 → 막대: 점들이 포물선을 그리며 날아가는 빛줄기
        if (abs(uA - 2.0) < 0.5 && abs(uB - 3.0) < 0.5) pos.y += sin(t * 3.14159) * (1.0 + seed * 1.8);
        pos += 0.012 * vec3(sin(uTime * 1.3 + seed * 40.0), cos(uTime * 1.1 + seed * 31.0), sin(uTime * 0.9 + seed * 17.0));
        vColor = mix(pick(uA, c0, c1, c2, c3), pick(uB, c0, c1, c2, c3), t);
        vec4 mv = modelViewMatrix * vec4(pos, 1.0);
        // 공 모양일 때: 카메라 쪽을 향한 앞면은 밝게, 뒷면은 흐리게 — 뒤쪽 무늬가 앞과 겹쳐 철망처럼 보이던 문제
        float facing = dot(normalize(normalMatrix * normalize(p0)), normalize(-mv.xyz));
        vFace = mix(1.0, 0.16 + 0.84 * smoothstep(-0.2, 0.35, facing), uBall);
        gl_PointSize = uSize * uPR * (8.0 / -mv.z);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      uniform float uOpacity;
      varying vec3 vColor;
      varying float vFace;
      void main() {
        float d = length(gl_PointCoord - 0.5);
        if (d > 0.5) discard;
        gl_FragColor = vec4(vColor, smoothstep(0.5, 0.05, d) * 0.85 * uOpacity * vFace);
      }`,
  });
  const points = new THREE.Points(geo, mat);
  points.frustumCulled = false;
  group.add(points);

  // 공 와이어(선) — 공 장면에서만 보임. 점과 같이 뒷면은 흐리게
  const wirePos = [];
  for (const e of ballEdges()) for (let k = 0; k < 10; k++) wirePos.push(...onSphere(e[0], e[1], k / 10, BALL_R), ...onSphere(e[0], e[1], (k + 1) / 10, BALL_R));
  const wireGeo = new THREE.BufferGeometry(); wireGeo.setAttribute('position', new THREE.Float32BufferAttribute(wirePos, 3));
  const wireMat = new THREE.ShaderMaterial({
    uniforms: { uOp: { value: 0.55 }, uColor: { value: new THREE.Color(0x58a6ff) } },
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: `
      varying float vF;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        float f = dot(normalize(normalMatrix * normalize(position)), normalize(-mv.xyz));
        vF = 0.1 + 0.9 * smoothstep(-0.15, 0.35, f);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      uniform float uOp; uniform vec3 uColor; varying float vF;
      void main() { gl_FragColor = vec4(uColor, uOp * vF); }`,
  });
  group.add(new THREE.LineSegments(wireGeo, wireMat));

  // 공 윤곽 원(로고의 바깥 원) + 은은한 빛번짐 — 항상 카메라를 향하고, 원근을 고려해 실제 구의 외곽선과 맞춤
  const ringPts = [];
  for (let k = 0; k <= 160; k++) { const a = k / 160 * Math.PI * 2; ringPts.push(Math.cos(a), Math.sin(a), 0); }
  const ringGeo = new THREE.BufferGeometry(); ringGeo.setAttribute('position', new THREE.Float32BufferAttribute(ringPts, 3));
  const ringMat = new THREE.LineBasicMaterial({ color: 0x78b8ff, transparent: true, opacity: 0.7, blending: THREE.AdditiveBlending, depthWrite: false });
  const ring = new THREE.Line(ringGeo, ringMat); scene.add(ring);
  const glowCanvas = document.createElement('canvas'); glowCanvas.width = glowCanvas.height = 256;
  const gctx = glowCanvas.getContext('2d'), grad = gctx.createRadialGradient(128, 128, 0, 128, 128, 128);
  // 가운데는 거의 비우고 테두리 바로 바깥만 은은하게(넓고 진하면 공 뒤에 회색 원판이 깔린 것처럼 보임)
  grad.addColorStop(0, 'rgba(88,166,255,0.05)'); grad.addColorStop(0.72, 'rgba(88,166,255,0.05)');
  grad.addColorStop(0.83, 'rgba(88,166,255,0.13)'); grad.addColorStop(1, 'rgba(88,166,255,0)');
  gctx.fillStyle = grad; gctx.fillRect(0, 0, 256, 256);
  const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(glowCanvas), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  scene.add(glow);
  function placeRim(w0) {
    // 구를 보는 원뿔의 접선 반지름: 중심을 지나는 평면에서 R·d/√(d²−R²)
    const d = camera.position.distanceTo(group.position);
    const r = BALL_R * d / Math.sqrt(Math.max(d * d - BALL_R * BALL_R, 0.01));
    ring.quaternion.copy(camera.quaternion); ring.scale.setScalar(r);
    glow.scale.setScalar(r * 2.3);
    ringMat.opacity = 0.62 * w0; glow.material.opacity = w0;
    ring.visible = glow.visible = w0 > 0.01;
  }

  const labelsWrap = $('story-bar-labels');
  let barLabelEls = [];
  function buildBarLabels() {
    labelsWrap.innerHTML = barLayout().map((b, i) =>
      `<div class="story-bar-label${i === 0 ? ' first' : ''}">${b.logo ? `<img src="${b.logo}" alt="" onerror="this.style.display='none'">` : ''}<b>${Math.round(b.prob * 100)}%</b><span>${compactName(b.team)}</span></div>`).join('');
    barLabelEls = [...labelsWrap.children];
  }

  // 실제 데이터가 늦게 오면 막대 목표 위치만 다시 채움
  onDataChange = () => {
    fillBars(geo.attributes.p3.array, geo.attributes.c3.array, N);
    geo.attributes.p3.needsUpdate = true; geo.attributes.c3.needsUpdate = true;
    buildBarLabels();
  };
  onDataChange();

  // 장면별 카메라 [위치, 바라보는 점]
  const CAM = [
    [[0, 0, 8.6], [0, 0, 0]],
    [[0, 1.2, 9.6], [0, 0, 0]],
    [[0, 6.8, 7.9], [0, -0.3, 0.35]],
    [[0, 0.9, 8.4], [0, 0.55, 0]],
  ];
  const camPos = new THREE.Vector3(), camLook = new THREE.Vector3(), tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3();


  const cta = document.querySelector('.cta-section');
  const state = { A: 0, B: 0, T: 0, opacity: 0.55, heroMode: true, p: 0, shift: 0, dy: 0, zoom: 1 };
  let W = 0, H = 0, camDist = 1, appliedShift = -1, appliedDy = 0;
  function resize() {
    W = window.innerWidth; H = window.innerHeight;
    renderer.setSize(W, H, false);
    camera.aspect = W / H;
    // 세로로 긴 화면(모바일)은 가로 시야가 좁아 장면이 넘치므로 카메라를 비율만큼 뒤로 뺌
    camDist = Math.max(1, 1 / camera.aspect);
    appliedShift = -1;   // 비율이 바뀌었으니 투영 행렬은 무조건 다시 계산
    applyShift(state.shift, state.dy);
    fitCta();
  }
  // 마지막 CTA 섹션을 "화면 − 상단 메뉴 − 푸터" 높이로 늘림: 맨 아래까지 내리면 위 섹션(지원하는 리그)은
  // 화면 밖으로 빠지고 공 + "지금 바로 시작하세요"만 보이게
  function fitCta() {
    if (!cta) return;
    const nav = document.querySelector('nav'), foot = document.querySelector('footer');
    cta.style.minHeight = Math.max(420, H - (nav ? nav.offsetHeight : 0) - (foot ? foot.offsetHeight : 0)) + 'px';
  }
  // 장면 위치: 히어로·CTA에선 가운데(헤드라인 뒤), 스토리 구간에선 글과 안 겹치게
  // 데스크톱은 오른쪽, 모바일은 아래로 — shift(0~1)로 부드럽게 이동
  // dy: 장면을 세로로 옮길 픽셀(CTA에선 섹션 중심을 따라 공이 같이 움직이게)
  function applyShift(k, dy = 0) {
    if (Math.abs(k - appliedShift) < 0.002 && Math.abs(dy - appliedDy) < 0.5 && appliedShift >= 0) return;
    appliedShift = k; appliedDy = dy;
    if (W > 768) camera.setViewOffset(W, H, -W * 0.2 * k, -dy, W, H); else camera.setViewOffset(W, H, 0, -H * 0.2 * k - dy, W, H);
    camera.updateProjectionMatrix();
  }
  resize();
  window.addEventListener('resize', resize);

  // ── 스크롤 → 장면 상태 ──
  const hero = document.querySelector('.hero');
  const steps = [...story.querySelectorAll('.story-step')];
  const count = $('story-count');
  // 스토리 구간 진행도 p(0~1)에 따른 장면: [시작 p, 끝 p, A, B]  (A=B면 머무름)
  const SEG = [[0, 0.10, 0, 0], [0.10, 0.30, 0, 1], [0.30, 0.40, 1, 1], [0.40, 0.55, 1, 2], [0.55, 0.64, 2, 2], [0.64, 0.80, 2, 3], [0.80, 0.87, 3, 3], [0.87, 0.95, 3, 0], [0.95, 1.0, 0, 0]];
  // 문구 단계별 보이는 구간
  const STEP_RANGE = [[0.08, 0.36], [0.40, 0.62], [0.66, 0.86], [0.90, 1.01]];
  const fade = (p, a, b, w = 0.035) => clamp(Math.min((p - a) / w, (b - p) / w));

  function readScroll() {
    const sr = story.getBoundingClientRect();
    if (sr.top > 0) {   // 히어로 구간: 공
      const k = clamp(1 - sr.top / H);
      Object.assign(state, { A: 0, B: 0, T: 0, heroMode: true, p: 0, opacity: 0.3 + 0.7 * k, shift: ease(k), dy: 0, zoom: 1 });
    } else if (sr.bottom > H * 0.7) {   // 스토리 구간
      const p = clamp(-sr.top / (sr.height - H));
      const s = SEG.find(g => p >= g[0] && p <= g[1]) || SEG[SEG.length - 1];
      Object.assign(state, { A: s[2], B: s[3], T: s[2] === s[3] ? 0 : ease(clamp((p - s[0]) / (s[1] - s[0]))), heroMode: false, p,
        // 스토리가 끝나 다음 섹션이 올라오면 글과 겹치기 전에 공이 먼저 사라짐
        opacity: clamp((sr.bottom - H * 0.7) / (H * 0.3)), shift: 1, dy: 0, zoom: 1 });
    } else {
      const cr = cta ? cta.getBoundingClientRect() : { top: 1e9, height: 0 };
      const c = clamp((H - cr.top) / (H * 0.42));   // 마지막 CTA: 구름 → 공으로 합쳐짐(화면 중간쯤에서 완성)
      // 공이 CTA 섹션 중심을 따라 움직여 글 뒤 정중앙에 자리 잡음
      Object.assign(state, { A: 1, B: 0, T: c, heroMode: false, p: 1, opacity: c > 0 ? 0.35 + 0.35 * c : 0, shift: 0,
        dy: clamp(cr.top + cr.height / 2 - H / 2, -H, H), zoom: W > 768 ? 1.22 : 0.92 });   // CTA 영역(메뉴~푸터)에 맞춤 — 데스크톱은 조금 작게, 세로 화면은 폭 기준이라 조금 크게
    }
    // 문구·카운터·막대 채우기
    steps.forEach((el, i) => { const o = fade(state.p, ...STEP_RANGE[i]); el.style.opacity = o; el.style.transform = `translateY(${(1 - o) * 18}px)`; el.style.pointerEvents = o > 0.5 ? 'auto' : 'none'; });
    const cp = ease(clamp((state.p - 0.10) / 0.20));
    count.textContent = Math.round(data.total * cp).toLocaleString();
    const bp = ease(clamp((state.p - 0.46) / 0.10));
    for (const id of ['story-bar-h', 'story-bar-d', 'story-bar-a']) { const el = $(id); el.style.width = (Number(el.dataset.target || 0) * bp) + '%'; }
  }

  // ── 렌더 루프 (보이지 않으면 멈춤) ──
  let spin = 0, last = performance.now(), running = false, opacity = 0;
  const clock0 = performance.now();
  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    readScroll();
    opacity += (state.opacity - opacity) * Math.min(1, dt * 6);
    canvas.style.opacity = opacity.toFixed(3);
    if (opacity < 0.01 && state.opacity === 0) { running = false; return; }   // 화면 밖: 루프 중단(스크롤 시 재개)

    uniforms.uA.value = state.A; uniforms.uB.value = state.B; uniforms.uT.value = state.T;
    uniforms.uTime.value = (now - clock0) / 1000;
    // 공·구름에선 천천히 회전, 경기장·막대에선 정면(가장 가까운 한 바퀴 지점으로 부드럽게 멈춤)
    const w = (s => (s === 0 || s === 1 ? 1 : 0));
    const spinning = w(state.A) * (1 - state.T) + w(state.B) * state.T;
    if (spinning > 0.5) spin += dt * 0.22;
    else { const target = Math.round(spin / (Math.PI * 2)) * Math.PI * 2; spin += (target - spin) * Math.min(1, dt * 3); }
    group.rotation.y = spin;
    group.rotation.x = state.heroMode ? 0.18 : 0.12 * spinning;
    const w0 = (state.A === 0 ? 1 - state.T : 0) + (state.B === 0 ? state.T : 0);   // 공 장면 비중
    // 선·윤곽은 점들이 거의 다 모였을 때 나타남(모이는 중에 선이 먼저 보이면 공이 찌그러져 보임)
    const formed = Math.pow(clamp((w0 - 0.55) / 0.45), 1.5);
    uniforms.uBall.value = w0;
    wireMat.uniforms.uOp.value = 0.5 * formed;
    applyShift(state.shift, state.dy);

    tmpA.set(...CAM[state.A][0]); tmpB.set(...CAM[state.B][0]); camPos.lerpVectors(tmpA, tmpB, state.T);
    tmpA.set(...CAM[state.A][1]); tmpB.set(...CAM[state.B][1]); camLook.lerpVectors(tmpA, tmpB, state.T);
    camera.position.copy(camPos).sub(camLook).multiplyScalar(camDist * state.zoom).add(camLook); camera.lookAt(camLook);
    group.updateMatrixWorld();
    placeRim(formed);
    renderer.render(scene, camera);

    // 막대 위 라벨: 3D 막대 꼭대기를 화면 좌표로 투영
    const barVis = (state.A === 3 ? 1 - state.T : 0) + (state.B === 3 ? state.T : 0);
    labelsWrap.style.opacity = clamp((barVis - 0.6) / 0.4).toFixed(3);
    if (barVis > 0.6) {
      barLayout().forEach((b, i) => {
        const el = barLabelEls[i]; if (!el) return;
        tmpA.set(b.x, b.base + b.h + 0.28, 0).applyMatrix4(group.matrixWorld).project(camera);
        el.style.transform = `translate(${(tmpA.x * 0.5 + 0.5) * W}px, ${(-tmpA.y * 0.5 + 0.5) * H}px) translate(-50%, -100%)`;
      });
    }
    requestAnimationFrame(frame);
  }
  const wake = () => { if (!running && !document.hidden) { running = true; last = performance.now(); requestAnimationFrame(frame); } };
  window.addEventListener('scroll', wake, { passive: true });
  document.addEventListener('visibilitychange', () => { if (document.hidden) running = false; else wake(); });
  // 인트로(골대 애니메이션)가 끝난 뒤 공이 이어받아 나타나도록
  const introGone = () => !document.getElementById('intro-overlay');
  const startWhenReady = () => { if (introGone()) { story.classList.add('live'); wake(); } else setTimeout(startWhenReady, 150); };
  startWhenReady();
}
