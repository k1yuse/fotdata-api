// ── 방문 통계 (Umami Cloud, 2026-09-30) ──
// 쿠키를 안 쓰고 개인정보(IP·기기 식별자)도 저장하지 않는 방식. FotData.html과 landing.html(index.html)이 같이 불러옴.
// UMAMI_ID(공개값 — cloud.umami.is의 사이트 ID)가 비어 있거나 실제 배포 주소가 아니면(로컬 테스트) 아무것도 안 보냄.
// 페이지 쪽에선 window.fdTrack(이름, 데이터)로 이벤트, fdTrack('__view', 탭)으로 탭 화면 조회를 보냄 —
// 이 파일이 늦게 로드돼도 그 전 호출은 window.fdQ에 쌓였다가 Umami가 준비되면 한 번에 보냄.
(function () {
  var UMAMI_ID = 'dd80e172-7700-4498-a9f2-140e07f803f5';   // 2026-10-03 연결(fotdata.official 계정)
  var LIVE = /(^|\.)fotdata-official\.com$/.test(location.hostname);   // 2026-10-03 도메인 이전(옛 fotdata-api.vercel.app은 새 주소로 넘어감)
  var q = window.fdQ = window.fdQ || [];
  if (!UMAMI_ID || !LIVE) { window.fdTrack = function () {}; q.length = 0; return; }
  var ready = false;
  function send(name, data) {
    try {
      if (name === '__view') {
        // 앱은 탭을 바꿔도 주소가 안 바뀌어서, 탭마다 가상의 주소(/app/standings 등)로 조회를 남김
        window.umami.track(function (p) { return Object.assign({}, p, { url: '/app/' + data, title: 'FotData · ' + data }); });
      } else {
        window.umami.track(name, data);
      }
    } catch (e) {}
  }
  window.fdTrack = function (name, data) { if (ready) send(name, data); else q.push([name, data]); };
  var s = document.createElement('script');
  s.src = 'https://cloud.umami.is/script.js';
  s.defer = true;
  s.setAttribute('data-website-id', UMAMI_ID);
  s.onload = function () { ready = true; q.splice(0).forEach(function (e) { send(e[0], e[1]); }); };
  document.head.appendChild(s);
})();
