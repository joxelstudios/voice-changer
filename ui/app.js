const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

let engineRunning = false;

async function init() {
  try {
    const [inputs, outputs, virtualOut, savedConfig] = await Promise.all([
      invoke('get_input_devices'),
      invoke('get_output_devices'),
      invoke('get_virtual_output'),
      invoke('get_saved_config'),
    ]);

    const inputSelect = document.getElementById('input-device');
    const outputSelect = document.getElementById('output-device');

    inputs.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      // Restore saved input device
      if (savedConfig.input_device && name === savedConfig.input_device) opt.selected = true;
      inputSelect.appendChild(opt);
    });

    outputs.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      // Restore saved output device, fallback to virtual output auto-detect
      if (savedConfig.output_device && name === savedConfig.output_device) {
        opt.selected = true;
      } else if (!savedConfig.output_device && virtualOut && name === virtualOut) {
        opt.selected = true;
      }
      outputSelect.appendChild(opt);
    });
  } catch (e) {
    console.error('Init failed:', e);
    document.getElementById('status').textContent = 'Init error: ' + e;
  }

  // Listen for bypass changes from tray toggle
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

  // Tabs
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });

  // AI controls
  document.getElementById('unload-btn').addEventListener('click', unloadVoice);
  document.getElementById('ai-pitch').addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    document.getElementById('ai-pitch-value').textContent = `${val >= 0 ? '+' : ''}${val} st`;
    invoke('set_ai_pitch', { semitones: val }).catch(console.error);
  });

  // Load presets
  await loadPresets();
}

// --- Tabs ---

function switchTab(tabName) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
  document.querySelector(`.tab[data-tab="${tabName}"]`).classList.add('active');
  document.getElementById(`tab-${tabName}`).classList.remove('hidden');
}

// --- Engine ---

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
    document.querySelectorAll('.effect-btn.on').forEach(b => b.classList.remove('on'));
    document.getElementById('active-effects').innerHTML = '';
  } else {
    const input = document.getElementById('input-device').value;
    const output = document.getElementById('output-device').value;

    if (!input || !output) {
      status.textContent = 'Error: select devices first';
      status.classList.remove('on');
      return;
    }

    try {
      await invoke('start_engine', { inputDevice: input, outputDevice: output });
      engineRunning = true;
      btn.textContent = 'Stop';
      btn.classList.add('active');
      status.textContent = 'Running';
      status.classList.add('on');
    } catch (e) {
      status.textContent = 'Error: ' + e;
      status.classList.remove('on');
      console.error('Failed to start:', e);
    }
  }
}

// --- DSP Effects ---

async function toggleEffect(el) {
  if (!engineRunning) return;
  const effect = el.dataset.effect;
  if (!effect) return;

  el.classList.toggle('on');
  await rebuildEffects();
  await updateActiveEffects();
}

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

// --- AI Voice ---

async function loadPresets() {
  try {
    const presets = await invoke('list_presets');
    const container = document.getElementById('preset-list');
    container.innerHTML = '';

    if (presets.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'preset-empty';
      empty.textContent = 'No presets found. Place RVC ONNX models in presets/ with .json configs.';
      container.appendChild(empty);
      return;
    }

    presets.forEach(preset => {
      const item = document.createElement('div');
      item.className = 'preset-item';
      item.addEventListener('click', () => selectPreset(preset.name, item));

      const name = document.createElement('span');
      name.textContent = preset.name;

      const pitch = document.createElement('small');
      pitch.textContent = preset.pitch_shift !== 0
        ? `${preset.pitch_shift >= 0 ? '+' : ''}${preset.pitch_shift} st`
        : '';

      item.appendChild(name);
      item.appendChild(pitch);
      container.appendChild(item);
    });
  } catch (e) {
    console.error('Failed to load presets:', e);
  }
}

async function selectPreset(name, element) {
  const aiStatus = document.getElementById('ai-status');
  aiStatus.textContent = 'Loading...';
  aiStatus.classList.remove('loaded');

  document.querySelectorAll('.preset-item.active').forEach(i => i.classList.remove('active'));

  try {
    await invoke('load_voice', { presetName: name });
    element.classList.add('active');
    aiStatus.textContent = `Loaded: ${name}`;
    aiStatus.classList.add('loaded');
    // Poll for AI errors after a few seconds to catch inference failures
    setTimeout(async () => {
      try {
        const err = await invoke('get_ai_error');
        if (err) {
          aiStatus.textContent = `AI Error: ${err}`;
          aiStatus.classList.remove('loaded');
        }
      } catch (_) {}
    }, 3000);
  } catch (e) {
    aiStatus.textContent = `Failed: ${e}`;
    console.error('Failed to load voice:', e);
  }
}

async function unloadVoice() {
  try {
    await invoke('unload_voice');
  } catch (e) {
    console.error('Failed to unload voice:', e);
  }
  document.querySelectorAll('.preset-item.active').forEach(i => i.classList.remove('active'));
  document.getElementById('ai-status').textContent = 'No voice loaded';
  document.getElementById('ai-status').classList.remove('loaded');
}

init();
