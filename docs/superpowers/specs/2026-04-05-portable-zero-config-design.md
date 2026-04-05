# Portable Zero-Config Voice Changer — Design Spec

## Problem

The voice changer app requires manual model downloads, preset creation, and audio device configuration. Users should be able to copy a single exe to a flash drive, run it on any Windows machine, and have everything work.

## Design

### First Launch Flow

1. User double-clicks `voice-changer.exe`
2. App checks for `data/` directory next to exe
3. If missing → open setup wizard window
4. Wizard downloads ContentVec model (~293MB) and a demo RVC voice (~55MB) with progress bar
5. Wizard checks for VB-Cable — if missing, shows message with download link (non-blocking)
6. Wizard creates default preset JSON
7. Wizard transitions to main app

If `data/` exists with models → skip wizard, launch main app directly.

### Directory Structure (Portable)

```
F:\
├── voice-changer.exe
└── data/
    ├── models/
    │   ├── contentvec.onnx    # ~293MB, downloaded from HuggingFace
    │   └── demo-voice.onnx    # ~55MB, downloaded from HuggingFace
    ├── presets/
    │   └── demo.json          # Auto-generated
    └── config.json            # Device selection, last preset, buffer size
```

All paths resolved relative to the exe location, not CWD.

### Model Downloads

| Model | Source | Size |
|-------|--------|------|
| ContentVec | `NaruseMioShirakana/MoeSS-SUBModel` → `vec-768-layer-9.onnx` | ~293MB |
| Demo voice | Public RVC ONNX model (permissive license) | ~55MB |

Downloads use HTTPS with progress reporting. Resumable if interrupted (check existing file size, use Range header).

### VB-Cable Detection

Enumerate audio output devices, search for "CABLE Input" or "VB-Audio" in device names. If not found:
- Show message: "VB-Cable virtual audio cable not detected"
- Button: "Download VB-Cable" → opens browser to `https://vb-audio.com/Cable/`
- Button: "Skip" → continue without virtual mic (app still works for audio preview)

### Config Persistence

`data/config.json`:
```json
{
  "input_device": "Microphone (Realtek)",
  "output_device": "CABLE Input (VB-Audio)",
  "last_preset": "demo",
  "sample_rate": 48000,
  "buffer_size": 512
}
```

- Loaded on startup, applied as defaults in device dropdowns
- Saved on app exit (or on device/preset change)
- Missing file → use sensible defaults (first input device, VB-Cable if found)

### Setup Wizard UI

Single-window wizard with steps:
1. "Downloading models..." — progress bar per model, total progress
2. "Checking audio setup..." — VB-Cable status with action buttons
3. "Ready!" — launch button

Implemented as a separate Tauri webview page (`setup.html`) that transitions to the main window on completion.

### Build & Distribution

- `cargo tauri build` on Windows produces the MSI/NSIS installer
- For portable mode: extract just the exe from the build output
- The exe self-creates `data/` on first run — no installer needed
- Single exe, no runtime dependencies (ONNX Runtime bundled via ort)
