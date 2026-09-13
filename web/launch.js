// A file:// entry point can explain startup, but cannot start a native server.
if (location.protocol === 'file:') {
  document.body.innerHTML = `<main style="min-height:100vh;display:grid;place-items:center;padding:32px;height:auto">
    <section style="max-width:660px;padding:40px;border:1px solid #30473f;border-radius:16px;background:#0b1720">
      <p class="eyebrow">NEURONS · LOCAL APPLICATION</p>
      <h1 style="font-size:27px;line-height:1.5">관측실을 실행해 주세요.</h1>
      <p style="color:#acbec8;line-height:1.9">이 파일은 화면을 구성하는 파일입니다. 실제 뉴런 계산과 데이터 조회를 위해 Neurons 서버가 함께 실행되어야 합니다.</p>
      <p style="line-height:1.9">상위 폴더의 <strong style="color:#85ebc8">Run Neurons.cmd</strong>를 실행하면 서버가 시작되고 관측실이 자동으로 열립니다.</p>
      <a class="primary" style="display:inline-block;margin:16px 0;text-decoration:none" href="http://127.0.0.1:8878/">실행 중인 관측실 열기 ↗</a>
      <p class="caption">기본 주소 127.0.0.1:8878 · 연결되지 않으면 먼저 실행 파일을 열어 주세요.<br>소스만 받은 경우 setup.ps1을 먼저 실행합니다. 지도와 자극 실험은 LLM 없이 사용할 수 있습니다.</p>
    </section></main>`;
} else {
  const application = document.createElement('script');
  application.type = 'module';
  application.src = './app.js';
  document.body.appendChild(application);
}
