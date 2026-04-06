mod config;
mod downloader;
mod setup;

use std::path::PathBuf;
use std::sync::Mutex;

use tauri::{
    menu::{MenuBuilder, MenuItemBuilder},
    tray::{MouseButton, TrayIconBuilder, TrayIconEvent},
    Emitter, Manager,
};

use audio_engine::{AudioEngine, EngineConfig, EngineState, list_input_devices, list_output_devices};
use dsp::{Echo, Effect, PitchShift, Reverb, RobotVoice};
use voice_conv::{VoiceConverter, VoiceConverterConfig, PresetManager, VoicePreset};

use config::AppConfig;

pub struct AppState {
    engine: Mutex<Option<EngineState>>,
    preset_manager: Mutex<PresetManager>,
    config: Mutex<AppConfig>,
    sample_rate: u32,
    data_dir: PathBuf,
}

/// Resolve the portable data directory relative to the exe location.
fn resolve_data_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
        .join("data")
}

fn lock_engine(state: &AppState) -> Result<std::sync::MutexGuard<'_, Option<EngineState>>, String> {
    state.engine.lock().map_err(|e| format!("Lock poisoned: {e}"))
}

fn build_effect(effect_type: &str, sample_rate: f32) -> Result<Box<dyn Effect>, String> {
    match effect_type {
        "pitch_up" => Ok(Box::new(PitchShift::new(2.0_f32.powf(5.0 / 12.0), sample_rate))),
        "pitch_down" => Ok(Box::new(PitchShift::new(2.0_f32.powf(-5.0 / 12.0), sample_rate))),
        "robot" => Ok(Box::new(RobotVoice::new(150.0, sample_rate))),
        "echo" => Ok(Box::new(Echo::new(200.0, 0.4, 0.3, sample_rate))),
        "reverb" => Ok(Box::new(Reverb::new(0.7, 0.3, sample_rate))),
        _ => Err(format!("Unknown effect: {effect_type}")),
    }
}

// --- Device Commands ---

#[tauri::command]
fn get_input_devices() -> Result<Vec<String>, String> {
    list_input_devices()
        .map(|devs| devs.into_iter().map(|d| d.name).collect())
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn get_output_devices() -> Result<Vec<String>, String> {
    list_output_devices()
        .map(|devs| devs.into_iter().map(|d| d.name).collect())
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn get_virtual_output() -> Option<String> {
    virtual_mic::find_virtual_output()
}

// --- Engine Commands ---

#[tauri::command]
fn start_engine(
    input_device: String,
    output_device: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let config = EngineConfig {
        input_device: input_device.clone(),
        output_device: output_device.clone(),
        sample_rate: state.sample_rate,
        buffer_size: 512,
    };

    let engine_state = AudioEngine::start(config).map_err(|e| e.to_string())?;
    *lock_engine(&state)? = Some(engine_state);

    // Save device selection to config
    if let Ok(mut cfg) = state.config.lock() {
        cfg.input_device = Some(input_device);
        cfg.output_device = Some(output_device);
        let _ = cfg.save(&state.data_dir.join("config.json"));
    }
    Ok(())
}

#[tauri::command]
fn stop_engine(state: tauri::State<'_, AppState>) -> Result<(), String> {
    *lock_engine(&state)? = None;
    Ok(())
}

#[tauri::command]
fn set_bypass(enabled: bool, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let guard = lock_engine(&state)?;
    if let Some(engine) = guard.as_ref() {
        engine.set_bypass(enabled);
    }
    Ok(())
}

// --- DSP Effect Commands ---

#[tauri::command]
fn add_effect(effect_type: String, state: tauri::State<'_, AppState>) -> Result<usize, String> {
    let sr = state.sample_rate as f32;
    let effect = build_effect(&effect_type, sr)?;
    let chain = {
        let guard = lock_engine(&state)?;
        let engine = guard.as_ref().ok_or("Engine not running")?;
        engine.effect_chain().clone()
    };
    let mut chain = chain.lock().map_err(|e| e.to_string())?;
    Ok(chain.add(effect))
}

#[tauri::command]
fn remove_effect(index: usize, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let chain = {
        let guard = lock_engine(&state)?;
        let engine = guard.as_ref().ok_or("Engine not running")?;
        engine.effect_chain().clone()
    };
    let mut chain = chain.lock().map_err(|e| e.to_string())?;
    chain.remove(index);
    Ok(())
}

#[tauri::command]
fn clear_effects(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let chain = {
        let guard = lock_engine(&state)?;
        let engine = guard.as_ref().ok_or("Engine not running")?;
        engine.effect_chain().clone()
    };
    let mut chain = chain.lock().map_err(|e| e.to_string())?;
    chain.clear();
    Ok(())
}

#[tauri::command]
fn set_effects(effect_types: Vec<String>, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let sr = state.sample_rate as f32;
    let effects: Result<Vec<Box<dyn Effect>>, String> = effect_types
        .iter()
        .map(|t| build_effect(t, sr))
        .collect();
    let effects = effects?;
    let chain = {
        let guard = lock_engine(&state)?;
        let engine = guard.as_ref().ok_or("Engine not running")?;
        engine.effect_chain().clone()
    };
    let mut chain = chain.lock().map_err(|e| e.to_string())?;
    chain.replace_all(effects);
    Ok(())
}

#[tauri::command]
fn get_effects(state: tauri::State<'_, AppState>) -> Result<Vec<(String, bool)>, String> {
    let chain = {
        let guard = lock_engine(&state)?;
        let engine = guard.as_ref().ok_or("Engine not running")?;
        engine.effect_chain().clone()
    };
    let chain = chain.lock().map_err(|e| e.to_string())?;
    Ok(chain.effect_names().into_iter().map(|(n, e)| (n.to_string(), e)).collect())
}

// --- AI Voice Commands ---

#[tauri::command]
fn list_presets(state: tauri::State<'_, AppState>) -> Result<Vec<VoicePreset>, String> {
    let mut manager = state.preset_manager.lock().map_err(|e| e.to_string())?;
    // Rescan in case presets were added (e.g., after setup wizard)
    let _ = manager.scan();
    Ok(manager.list().to_vec())
}

#[tauri::command]
fn load_voice(preset_name: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let manager = state.preset_manager.lock().map_err(|e| e.to_string())?;
    let preset = manager.get(&preset_name)
        .ok_or_else(|| format!("Preset not found: {preset_name}"))?
        .clone();

    // FIX BUG #3: Use the actual device sample rate, not hardcoded 48kHz
    let guard = lock_engine(&state)?;
    let engine = guard.as_ref().ok_or("Engine not running. Start the engine first.")?;
    let actual_sr = engine.actual_sample_rate();

    let content_model = state.data_dir.join("models").join("contentvec.onnx");
    let config = VoiceConverterConfig {
        content_model_path: content_model.to_string_lossy().to_string(),
        generator_model_path: preset.model_path.clone(),
        sample_rate: actual_sr,
        pitch_shift: preset.pitch_shift,
    };

    // Drop the engine lock before creating the converter (which takes time for warm-up)
    drop(guard);

    let converter = VoiceConverter::new(config).map_err(|e| e.to_string())?;

    let guard = lock_engine(&state)?;
    let engine = guard.as_ref().ok_or("Engine not running")?;
    engine.set_voice_converter(Some(converter));

    if let Ok(mut cfg) = state.config.lock() {
        cfg.last_preset = Some(preset_name);
        let _ = cfg.save(&state.data_dir.join("config.json"));
    }
    Ok(())
}

#[tauri::command]
fn unload_voice(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let guard = lock_engine(&state)?;
    if let Some(engine) = guard.as_ref() {
        engine.set_voice_converter(None);
    }
    Ok(())
}

#[tauri::command]
fn set_ai_pitch(semitones: f32, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let guard = lock_engine(&state)?;
    if let Some(engine) = guard.as_ref() {
        let vc = engine.voice_converter();
        if let Ok(mut vc_guard) = vc.lock() {
            if let Some(converter) = vc_guard.as_mut() {
                converter.set_pitch_shift(semitones);
            }
        }
    }
    Ok(())
}

#[tauri::command]
fn is_voice_loaded(state: tauri::State<'_, AppState>) -> Result<bool, String> {
    let guard = lock_engine(&state)?;
    if let Some(engine) = guard.as_ref() {
        Ok(engine.is_ai_active())
    } else {
        Ok(false)
    }
}

#[tauri::command]
fn save_preset(name: String, model_path: String, pitch_shift: f32, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let preset = VoicePreset { name, model_path, pitch_shift };
    let mut manager = state.preset_manager.lock().map_err(|e| e.to_string())?;
    manager.save_preset(&preset).map_err(|e| e.to_string())
}

// --- Setup Commands ---

#[tauri::command]
fn open_url(url: String) -> Result<(), String> {
    open::that(&url).map_err(|e| e.to_string())
}

#[tauri::command]
fn check_setup(state: tauri::State<'_, AppState>) -> Result<setup::SetupStatus, String> {
    Ok(setup::check_setup(&state.data_dir))
}

#[tauri::command]
fn download_model(app: tauri::AppHandle, name: String, url: String, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let dest = state.data_dir.join("models").join(&name);
    std::fs::create_dir_all(dest.parent().unwrap()).map_err(|e| e.to_string())?;

    let app_clone = app.clone();
    let name_clone = name.clone();
    std::thread::spawn(move || {
        match downloader::download_with_progress(&url, &dest, move |progress| {
            let _ = app_clone.emit("download-progress", serde_json::json!({
                "name": name_clone,
                "progress": progress,
            }));
        }) {
            Ok(_) => {
                let _ = app.emit("download-complete", &name);
            }
            Err(e) => {
                let _ = app.emit("download-error", serde_json::json!({
                    "name": name,
                    "error": e.to_string(),
                }));
            }
        }
    });
    Ok(())
}

#[tauri::command]
fn mark_setup_complete(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let version_file = state.data_dir.join(".setup_version");
    std::fs::write(&version_file, setup::SETUP_VERSION).map_err(|e| e.to_string())?;
    log::info!("Setup version {} written", setup::SETUP_VERSION);
    Ok(())
}

#[tauri::command]
fn create_default_preset(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let models_dir = state.data_dir.join("models");
    let demo_model = models_dir.join("demo-voice.onnx");

    if demo_model.exists() {
        let preset = VoicePreset {
            name: "Space Marine".to_string(),
            model_path: demo_model.to_string_lossy().to_string(),
            pitch_shift: 0.0,
        };
        let mut manager = state.preset_manager.lock().map_err(|e| e.to_string())?;
        manager.save_preset(&preset).map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
fn get_saved_config(state: tauri::State<'_, AppState>) -> Result<AppConfig, String> {
    let cfg = state.config.lock().map_err(|e| e.to_string())?;
    Ok(cfg.clone())
}

// --- App Entry ---

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::init();

    let data_dir = resolve_data_dir();
    log::info!("Data directory: {}", data_dir.display());

    // Ensure directories exist
    let _ = std::fs::create_dir_all(data_dir.join("models"));
    let _ = std::fs::create_dir_all(data_dir.join("presets"));

    // Load config
    let app_config = AppConfig::load(&data_dir.join("config.json"));

    // Load presets
    let preset_manager = PresetManager::new(&data_dir.join("presets"))
        .unwrap_or_else(|e| {
            log::warn!("Failed to load presets: {e}");
            PresetManager::new(std::path::Path::new(&data_dir.join("presets"))).unwrap()
        });

    // Check if setup is needed — determines which page to show
    let needs_setup = !setup::check_setup(&data_dir).ready;

    tauri::Builder::default()
        .manage(AppState {
            engine: Mutex::new(None),

            preset_manager: Mutex::new(preset_manager),
            config: Mutex::new(app_config),
            sample_rate: 48000,
            data_dir,
        })
        .setup(move |app| {
            // Tray
            let toggle = MenuItemBuilder::with_id("toggle", "Toggle Voice Changer").build(app)?;
            let quit = MenuItemBuilder::with_id("quit", "Quit").build(app)?;
            let menu = MenuBuilder::new(app)
                .item(&toggle)
                .separator()
                .item(&quit)
                .build()?;

            let _tray = TrayIconBuilder::new()
                .menu(&menu)
                .tooltip("Voice Changer")
                .on_menu_event(move |app, event| {
                    match event.id().as_ref() {
                        "toggle" => {
                            let state = app.state::<AppState>();
                            let new_bypass = {
                                let guard = state.engine.lock();
                                guard.ok().and_then(|g| {
                                    g.as_ref().map(|engine| {
                                        let val = !engine.is_bypassed();
                                        engine.set_bypass(val);
                                        val
                                    })
                                })
                            };
                            if let Some(bypass) = new_bypass {
                                let _ = app.emit("bypass-changed", bypass);
                            }
                        }
                        "quit" => app.exit(0),
                        _ => {}
                    }
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click { button: MouseButton::Left, .. } = event {
                        if let Some(window) = tray.app_handle().get_webview_window("main") {
                            if window.is_visible().unwrap_or(false) {
                                let _ = window.hide();
                            } else {
                                let _ = window.show();
                                let _ = window.set_focus();
                            }
                        }
                    }
                })
                .build(app)?;

            // Show setup wizard or main app based on setup status
            if needs_setup {
                let _window = tauri::WebviewWindowBuilder::new(
                    app,
                    "main",
                    tauri::WebviewUrl::App("setup.html".into()),
                )
                .title("Voice Changer — Setup")
                .inner_size(450.0, 400.0)
                .resizable(false)
                .build()?;
            } else {
                let _window = tauri::WebviewWindowBuilder::new(
                    app,
                    "main",
                    tauri::WebviewUrl::App("index.html".into()),
                )
                .title("Voice Changer")
                .inner_size(400.0, 500.0)
                .resizable(false)
                .build()?;
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_input_devices,
            get_output_devices,
            get_virtual_output,
            start_engine,
            stop_engine,
            set_bypass,
            add_effect,
            remove_effect,
            clear_effects,
            set_effects,
            get_effects,
            list_presets,
            load_voice,
            unload_voice,
            set_ai_pitch,
            is_voice_loaded,
            save_preset,
            check_setup,
            download_model,
            create_default_preset,
            mark_setup_complete,
            get_saved_config,
            open_url,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
