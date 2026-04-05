const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

let engineRunning = false;

async function init() {
  try {
    const [inputs, outputs, virtualOut] = await Promise.all([
      invoke('get_input_devices'),
      invoke('get_output_devices'),
      invoke('get_virtual_output'),
    ]);

    const inputSelect = document.getElementById('input-device');
    const outputSelect = document.getElementById('output-device');

    inputs.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      inputSelect.appendChild(opt);
    });

    outputs.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      if (virtualOut && name === virtualOut) opt.selected = true;
      outputSelect.appendChild(opt);
    });
  } catch (e) {
    console.error('Init failed:', e);
  }

  // Listen for bypass changes from tray toggle (#5)
  listen('bypass-changed', (event) => {
    const status = document.getElementById('status');
    if (event.payload) {
      status.textContent = 'Bypassed';
      status.classList.remove('on');
    } else {
      status.textContent = engineRunning ? 'Running' : 'Stopped';
      if (engineRunning) status.classList.add('on');
    }
  });

  // Wire up event listeners
  document.getElementById('start-btn').addEventListener('click', toggleEngine);
  document.querySelectorAll('.effect-btn[data-effect]').forEach(btn => {
    btn.addEventListener('click', () => toggleEffect(btn));
  });
  document.getElementById('clear-btn').addEventListener('click', clearEffects);
}

async function toggleEngine() {
  const btn = document.getElementById('start-btn');
  const status = document.getElementById('status');

  if (engineRunning) {
    try {
      await invoke('stop_engine');
    } catch (e) {
      console.error('Failed to stop engine:', e);
    }
    engineRunning = false;
    btn.textContent = 'Start';
    btn.classList.remove('active');
    status.textContent = 'Stopped';
    status.classList.remove('on');
    // Clear effect UI state on stop (#6)
    document.querySelectorAll('.effect-btn.on').forEach(b => b.classList.remove('on'));
    document.getElementById('active-effects').innerHTML = '';
  } else {
    const input = document.getElementById('input-device').value;
    const output = document.getElementById('output-device').value;
    try {
      await invoke('start_engine', { inputDevice: input, outputDevice: output });
      engineRunning = true;
      btn.textContent = 'Stop';
      btn.classList.add('active');
      status.textContent = 'Running';
      status.classList.add('on');
    } catch (e) {
      console.error('Failed to start:', e);
    }
  }
}

async function toggleEffect(el) {
  if (!engineRunning) return;
  const effect = el.dataset.effect;
  if (!effect) return;

  el.classList.toggle('on');
  await rebuildEffects();
  await updateActiveEffects();
}

// Atomic rebuild: send the full desired effect list in one call (#10)
async function rebuildEffects() {
  const activeEffects = [];
  document.querySelectorAll('.effect-btn.on').forEach(btn => {
    const effect = btn.dataset.effect;
    if (effect) activeEffects.push(effect);
  });

  try {
    await invoke('set_effects', { effectTypes: activeEffects });
  } catch (e) {
    console.error('Failed to set effects:', e);
  }
}

async function clearEffects() {
  if (!engineRunning) return;
  try {
    await invoke('clear_effects');
  } catch (e) {
    console.error('Failed to clear effects:', e);
  }
  document.querySelectorAll('.effect-btn.on').forEach(b => b.classList.remove('on'));
  await updateActiveEffects();
}

// Safe DOM creation instead of innerHTML with template literals (#7)
async function updateActiveEffects() {
  try {
    const effects = await invoke('get_effects');
    const container = document.getElementById('active-effects');
    container.innerHTML = '';
    effects.forEach(([name]) => {
      const span = document.createElement('span');
      span.textContent = name;
      container.appendChild(span);
    });
  } catch (e) {
    // Engine might not be running
  }
}

init();
