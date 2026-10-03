const CACHE_NAME = 'fotdata-shell-v2';   // v2: 앱 주소 /FotData.html → /app(2026-10-03)
const APP_SHELL = ['/', '/app', '/manifest.json', '/icon-192.png', '/icon-512.png'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// 정적 앱 셸만 캐시하고, 백엔드 API(onrender.com) 요청은 항상 네트워크로 보내 최신 데이터를 유지한다.
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || url.origin !== self.location.origin) return;
  // 공유 미리보기(/m/…)·썸네일(/og/m/…)은 서버가 매번 만드는 것이라 캐시하지 않음
  if (url.pathname.startsWith('/m/') || url.pathname.startsWith('/og/') || url.pathname.startsWith('/cal/')) return;   // 캘린더 구독(/cal/…)도
  event.respondWith(
    fetch(event.request)
      .then((res) => {
        const resClone = res.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, resClone));
        return res;
      })
      .catch(() => caches.match(event.request))
  );
});
