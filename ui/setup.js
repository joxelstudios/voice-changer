const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

let downloadsDone = { contentvec: false, demo: false };

async function runSetup() {
  const status = await invoke('check_setup');

  // Check VB-Cable
  const vbStatus = document.getElementById('status-vbcable');
  const vbMsg = document.getElementById('vbcable-msg');
  if (status.has_vb_cable) {
    vbStatus.textContent = 'Detected';
    vbStatus.className = 'step-status done';
  } else {
    vbStatus.textContent = 'Not found';
    vbStatus.className = 'step-status error';
    vbMsg.innerHTML = 'VB-Cable is needed for Discord/games to hear your voice. ' +
      '<a id="vb-link" href="#">Download VB-Cable</a> (free) then restart the app. ' +
      'You can skip this for now.';
    document.getElementById('vb-link').addEventListener('click', (e) => {
      e.preventDefault();
      // Open in default browser — Tauri webview won't navigate to external URLs
      invoke('open_url', { url: 'https://vb-audio.com/Cable/' }).catch(console.error);
    });
  }

  // Listen for progress events
  listen('download-progress', (event) => {
    const { name, progress } = event.payload;
    const barId = name === 'contentvec.onnx' ? 'progress-contentvec' : 'progress-demo';
    const statusId = name === 'contentvec.onnx' ? 'status-contentvec' : 'status-demo';
    document.getElementById(barId).style.width = `${(progress * 100).toFixed(1)}%`;
    const el = document.getElementById(statusId);
    el.textContent = `${(progress * 100).toFixed(0)}%`;
    el.className = 'step-status downloading';
  });

  listen('download-complete', (event) => {
    const name = event.payload;
    if (name === 'contentvec.onnx') {
      document.getElementById('status-contentvec').textContent = 'Done';
      document.getElementById('status-contentvec').className = 'step-status done';
      document.getElementById('progress-contentvec').style.width = '100%';
      downloadsDone.contentvec = true;
    } else {
      document.getElementById('status-demo').textContent = 'Done';
      document.getElementById('status-demo').className = 'step-status done';
      document.getElementById('progress-demo').style.width = '100%';
      downloadsDone.demo = true;
    }
    checkAllDone();
  });

  listen('download-error', (event) => {
    const { name, error } = event.payload;
    const statusId = name === 'contentvec.onnx' ? 'status-contentvec' : 'status-demo';
    const el = document.getElementById(statusId);
    el.textContent = 'Error';
    el.className = 'step-status error';
    console.error(`Download failed for ${name}: ${error}`);
  });

  // Start downloads for missing models
  if (status.has_contentvec) {
    document.getElementById('status-contentvec').textContent = 'Already downloaded';
    document.getElementById('status-contentvec').className = 'step-status done';
    document.getElementById('progress-contentvec').style.width = '100%';
    downloadsDone.contentvec = true;
  } else {
    invoke('download_model', { name: 'contentvec.onnx', url: status.contentvec_url });
  }

  if (status.has_demo_voice) {
    document.getElementById('status-demo').textContent = 'Already downloaded';
    document.getElementById('status-demo').className = 'step-status done';
    document.getElementById('progress-demo').style.width = '100%';
    downloadsDone.demo = true;
  } else {
    invoke('download_model', { name: 'demo-voice.onnx', url: status.demo_voice_url });
  }

  checkAllDone();
}

async function checkAllDone() {
  if (downloadsDone.contentvec && downloadsDone.demo) {
    // Create default preset and mark setup complete
    try {
      await invoke('create_default_preset');
      await invoke('mark_setup_complete');
    } catch (e) {
      console.error('Failed to finalize setup:', e);
    }

    const btn = document.getElementById('launch-btn');
    btn.classList.add('visible');
    btn.addEventListener('click', () => {
      document.location.href = 'index.html';
    });
  }
}

runSetup();
