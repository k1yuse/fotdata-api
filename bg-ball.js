// ── 메인 페이지 유리 테마 배경: 천천히 도는 3D 와이어프레임 축구공 (2026-09-28) ──
// 랜딩 3D 연출의 공과 같은 모양(ball-geometry.js 공유)을 배경에 아주 흐리게. 앱 페이지는 오래 켜두는 곳이라 가볍게:
//   - 화면 전체가 아니라 공 영역 크기의 작은 캔버스, 점 ~1,800개(랜딩의 30%), 30fps 제한, 해상도 최대 1.5배
//   - 탭이 숨겨지면 멈춤, 모션 최소화 설정·WebGL 미지원이면 정지 SVG(bg-ball.svg) 그대로
//   - Three.js는 첫 화면 이후 여유 있을 때 불러오고, 준비되면 정지 SVG에서 자연스럽게 바뀜
import { BALL_R, ballGeometry, onSphere, pointInPenta, wirePositions, WIRE_VERTEX, WIRE_FRAGMENT, rimRadius } from './ball-geometry.js';

const THREE_URL = 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.min.js';
const root = document.documentElement;
const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const webgl = (() => { try { const c = document.createElement('canvas'); return !!(c.getContext('webgl2') || c.getContext('webgl')); } catch (e) { return false; } })();

if (root.classList.contains('theme-glass') && !reduce && webgl) {
  const start = () => import(THREE_URL).then(init).catch(() => {});
  if ('requestIdleCallback' in window) requestIdleCallback(start, { timeout: 2500 }); else setTimeout(start, 1200);
}

function init(THREE) {
  const canvas = document.createElement('canvas');
  canvas.className = 'bg-ball-canvas';
  canvas.setAttribute('aria-hidden', 'true');
  document.body.appendChild(canvas);

  const mobile = window.innerWidth <= 768;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'low-power' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 50);
  const D = 9.2;
  camera.position.set(0, 0, D);
  const group = new THREE.Group(); scene.add(group);

  // 점: 모서리 78% + 오각형 면 22% (랜딩 공과 같은 비율)
  const N = mobile ? 1200 : 1800, { edges, pentas } = ballGeometry();
  const pos = new Float32Array(N * 3), col = new Float32Array(N * 3);
  const nFace = Math.floor(N * 0.22);
  for (let i = 0; i < N; i++) {
    const p = i < nFace ? pointInPenta(pentas[i % 12], BALL_R) : (e => onSphere(e[0], e[1], Math.random(), BALL_R))(edges[i % edges.length]);
    pos.set(p, i * 3);
    col.set(i < nFace ? [0.5, 0.7, 1.0] : [0.345, 0.651, 1.0], i * 3);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const pointMat = new THREE.ShaderMaterial({
    uniforms: { uSize: { value: 2.4 }, uPR: { value: renderer.getPixelRatio() } },
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, vertexColors: true,
    vertexShader: `
      uniform float uSize, uPR; varying vec3 vColor; varying float vFace;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        float f = dot(normalize(normalMatrix * normalize(position)), normalize(-mv.xyz));
        vFace = 0.16 + 0.84 * smoothstep(-0.2, 0.35, f);
        vColor = color;
        gl_PointSize = uSize * uPR * (9.0 / -mv.z);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      varying vec3 vColor; varying float vFace;
      void main() {
        float d = length(gl_PointCoord - 0.5);
        if (d > 0.5) discard;
        gl_FragColor = vec4(vColor, smoothstep(0.5, 0.05, d) * 0.85 * vFace);
      }`,
  });
  const points = new THREE.Points(geo, pointMat); points.frustumCulled = false; group.add(points);

  const wireGeo = new THREE.BufferGeometry(); wireGeo.setAttribute('position', new THREE.Float32BufferAttribute(wirePositions(BALL_R), 3));
  group.add(new THREE.LineSegments(wireGeo, new THREE.ShaderMaterial({
    uniforms: { uOp: { value: 0.5 }, uColor: { value: new THREE.Color(0x58a6ff) } },
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, vertexShader: WIRE_VERTEX, fragmentShader: WIRE_FRAGMENT,
  })));

  // 윤곽 원(로고의 바깥 원) — 카메라 정면이라 회전과 무관, 원근 보정 반지름으로 실제 외곽에 맞춤
  const ringPts = [];
  for (let k = 0; k <= 160; k++) { const a = k / 160 * Math.PI * 2; ringPts.push(Math.cos(a), Math.sin(a), 0); }
  const ringGeo = new THREE.BufferGeometry(); ringGeo.setAttribute('position', new THREE.Float32BufferAttribute(ringPts, 3));
  const ring = new THREE.Line(ringGeo, new THREE.LineBasicMaterial({ color: 0x78b8ff, transparent: true, opacity: 0.6, blending: THREE.AdditiveBlending, depthWrite: false }));
  ring.scale.setScalar(rimRadius(BALL_R, D)); scene.add(ring);

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  resize();
  window.addEventListener('resize', resize);

  // 30fps 제한 렌더 루프 — 탭이 숨겨지면 멈춤
  let running = false, last = 0;
  const t0 = performance.now();
  function frame(now) {
    if (!running) return;
    requestAnimationFrame(frame);
    if (now - last < 1000 / 30) return;
    last = now;
    const t = (now - t0) / 1000;
    group.rotation.y = t * 0.16;
    group.rotation.x = -0.28 + Math.sin(t * 0.21) * 0.06;   // 살짝 기울어진 채 천천히 흔들림
    renderer.render(scene, camera);
  }
  const play = () => { if (!running && !document.hidden) { running = true; requestAnimationFrame(frame); } };
  document.addEventListener('visibilitychange', () => { if (document.hidden) running = false; else play(); });
  play();
  // 첫 프레임이 그려진 뒤 정지 SVG → 3D 캔버스로 교차 전환
  requestAnimationFrame(() => requestAnimationFrame(() => root.classList.add('bg3d-on')));
}
