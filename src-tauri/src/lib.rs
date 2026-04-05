use std::sync::Mutex;

use tauri::{
    menu::{MenuBuilder, MenuItemBuilder},
    tray::{MouseButton, TrayIconBuilder, TrayIconEvent},
    Emitter, Manager,
};

use audio_engine::{AudioEngine, EngineConfig, EngineState, list_input_devices, list_output_devices};
use dsp::{Echo, Effect, PitchShift, Reverb, RobotVoice};

struct AppState {
    engine: Mutex<Option<EngineState>>,
    sample_rate: u32,
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
fn start_engine(
    input_device: String,
    output_device: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let config = EngineConfig {
        input_device,
        output_device,
        sample_rate: state.sample_rate,
        buffer_size: 512,
    };

    let engine_state = AudioEngine::start(config).map_err(|e| e.to_string())?;
    *lock_engine(&state)? = Some(engine_state);
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

#[tauri::command]
fn add_effect(effect_type: String, state: tauri::State<'_, AppState>) -> Result<usize, String> {
    let sr = state.sample_rate as f32;
    let effect = build_effect(&effect_type, sr)?;

    // Clone the Arc, then drop the engine lock before locking the chain
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

    // Build all effects before acquiring any locks
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
    Ok(chain
        .effect_names()
        .into_iter()
        .map(|(name, enabled)| (name.to_string(), enabled))
        .collect())
}

#[tauri::command]
fn get_virtual_output() -> Option<String> {
    virtual_mic::find_virtual_output()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::init();

    tauri::Builder::default()
        .manage(AppState {
            engine: Mutex::new(None),
            sample_rate: 48000,
        })
        .setup(|app| {
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
                        "quit" => {
                            app.exit(0);
                        }
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

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_input_devices,
            get_output_devices,
            start_engine,
            stop_engine,
            set_bypass,
            add_effect,
            remove_effect,
            clear_effects,
            set_effects,
            get_effects,
            get_virtual_output,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
