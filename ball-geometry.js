// ── 축구공(깎은 정이십면체) 기하 — 랜딩 3D 연출(landing-story.js)과 메인 배경 공(bg-ball.js)이 같이 씀 ──
// 두 곳의 공 모양이 항상 같도록 한 곳에서만 정의. (정지 이미지 bg-ball.svg는 generate_bg_ball.py가 같은 방식으로 생성)

export const BALL_R = 2.2;

// 정이십면체 꼭짓점 → 모서리를 1/3·2/3로 자른 점들로 깎은 정이십면체(축구공): 모서리 90개 + 오각형 면 12개
export function ballGeometry() {
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
  return { edges, pentas };
}

// 모서리 a→b의 t 지점을 반지름 R 구면 위로 올린 점(모서리가 공 표면을 따라 휘게)
export const onSphere = (a, b, t, R) => {
  const p = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const l = Math.hypot(...p);
  return [p[0] / l * R, p[1] / l * R, p[2] / l * R];
};

// 오각형 면 안의 무작위 점(구면 위, 살짝 안쪽)
export function pointInPenta(pe, R) {
  const k = Math.floor(Math.random() * 5), a = pe.v[k], b = pe.v[(k + 1) % 5];
  let u = Math.random(), w = Math.random(); if (u + w > 1) { u = 1 - u; w = 1 - w; }
  const q = [0, 1, 2].map(j => pe.c[j] + (a[j] - pe.c[j]) * u + (b[j] - pe.c[j]) * w);
  const l = Math.hypot(...q);
  return [q[0] / l * R * 0.995, q[1] / l * R * 0.995, q[2] / l * R * 0.995];
}

// 공 와이어 선분 좌표(모서리마다 10조각 호)
export function wirePositions(R) {
  const out = [];
  for (const e of ballGeometry().edges) for (let k = 0; k < 10; k++) out.push(...onSphere(e[0], e[1], k / 10, R), ...onSphere(e[0], e[1], (k + 1) / 10, R));
  return out;
}

// 앞면은 밝게·뒷면은 흐리게 하는 선 셰이더(구면 법선과 시선의 내적)
export const WIRE_VERTEX = `
  varying float vF;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float f = dot(normalize(normalMatrix * normalize(position)), normalize(-mv.xyz));
    vF = 0.1 + 0.9 * smoothstep(-0.15, 0.35, f);
    gl_Position = projectionMatrix * mv;
  }`;
export const WIRE_FRAGMENT = `
  uniform float uOp; uniform vec3 uColor; varying float vF;
  void main() { gl_FragColor = vec4(uColor, uOp * vF); }`;

// 원근을 고려한 구의 화면상 윤곽 반지름(구 중심을 지나는 평면 기준): R·d/√(d²−R²)
export const rimRadius = (R, d) => R * d / Math.sqrt(Math.max(d * d - R * R, 0.01));
