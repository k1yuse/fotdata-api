const CACHE_NAME = 'fotdata-shell-v3';   // v2: 앱 주소 /FotData.html → /app(2026-10-03) · v3: API 저장(2026-10-08)
const API_CACHE = 'fotdata-api-v1';
const APP_SHELL = ['/', '/app', '/manifest.json', '/icon-192.png', '/icon-512.png'];
const API_ORIGIN = 'https://fotdata-api.onrender.com';
// 저장해 둔 API 응답을 먼저 보여주고 뒤에서 새로 받음(stale-while-revalidate) — Render 무료 서버가 재시작·준비 중일 때도
// 두 번째 방문부터는 화면이 바로 뜨게(2026-10-08). 서빙 데이터는 하루 한 번 바뀌므로 하루 넘은 저장분은 안 씀
const API_MAX_AGE = 20 * 3600 * 1000;
// 몇 분 단위로 바뀌는 것(진행 중 점수·확정 라인업)과 공유·캘린더는 저장하지 않음
const API_SKIP = ['/matches/live', '/match/preview', '/share/', '/og/', '/calendar/', '/players/search'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME && k !== API_CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

function apiFetchAndStore(request) {
  return fetch(request).then((res) => {
    if (res.ok && res.status === 200) {
      const clone = res.clone();
      clone.blob().then((body) => {
        const headers = new Headers(clone.headers);
        headers.set('sw-cached-at', String(Date.now()));
        caches.open(API_CACHE).then((cache) => cache.put(request, new Response(body, { status: 200, statusText: 'OK', headers })));
      }).catch(() => {});
    }
    return res;
  });
}

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  // 백엔드 API: 저장분이 있고 하루 안이면 바로 주고 뒤에서 새로 받음, 없으면 네트워크(실패하면 오래된 저장분이라도)
  if (url.origin === API_ORIGIN) {
    if (API_SKIP.some((p) => url.pathname.startsWith(p))) return;
    event.respondWith(caches.open(API_CACHE).then((cache) => cache.match(req).then((hit) => {
      const age = hit ? Date.now() - Number(hit.headers.get('sw-cached-at') || 0) : Infinity;
      if (hit && age < API_MAX_AGE) {
        event.waitUntil(apiFetchAndStore(req).catch(() => {}));
        return hit;
      }
      return apiFetchAndStore(req).catch(() => hit || Response.error());
    })));
    return;
  }
  if (url.origin !== self.location.origin) return;
  // 공유 미리보기(/m/…)·썸네일(/og/m/…)은 서버가 매번 만드는 것이라 캐시하지 않음
  if (url.pathname.startsWith('/m/') || url.pathname.startsWith('/og/') || url.pathname.startsWith('/cal/')) return;   // 캘린더 구독(/cal/…)도
  event.respondWith(
    fetch(req)
      .then((res) => {
        const resClone = res.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(req, resClone));
        return res;
      })
      .catch(() => caches.match(req))
  );
});
