use std::sync::Mutex;

use tauri::{
    menu::{MenuBuilder, MenuItemBuilder},
    tray::TrayIconBuilder,
    Manager,
};

use audio_engine::{
    AudioEngine, EngineConfig, EngineState,
    list_input_devices, list_output_devices,
};
use dsp::{PitchShift, RobotVoice, Echo, Reverb};

struct AppState {
    engine: Mutex<Option<EngineState>>,
    sample_rate: u32,
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
    *state.engine.lock().unwrap() = Some(engine_state);
    Ok(())
}

#[tauri::command]
fn stop_engine(state: tauri::State<'_, AppState>) {
    *state.engine.lock().unwrap() = None;
}

#[tauri::command]
fn set_bypass(enabled: bool, state: tauri::State<'_, AppState>) {
    if let Some(engine) = state.engine.lock().unwrap().as_ref() {
        engine.set_bypass(enabled);
    }
}

#[tauri::command]
fn add_effect(effect_type: String, state: tauri::State<'_, AppState>) -> Result<usize, String> {
    let engine = state.engine.lock().unwrap();
    let engine = engine.as_ref().ok_or("Engine not running")?;
    let mut chain = engine.effect_chain().lock().unwrap();
    let sr = state.sample_rate as f32;

    let idx = match effect_type.as_str() {
        "pitch_up" => chain.add(Box::new(PitchShift::new(2.0_f32.powf(5.0 / 12.0), sr))),
        "pitch_down" => chain.add(Box::new(PitchShift::new(2.0_f32.powf(-5.0 / 12.0), sr))),
        "robot" => chain.add(Box::new(RobotVoice::new(150.0, sr))),
        "echo" => chain.add(Box::new(Echo::new(200.0, 0.4, 0.3, sr))),
        "reverb" => chain.add(Box::new(Reverb::new(0.7, 0.3, sr))),
        _ => return Err(format!("Unknown effect: {effect_type}")),
    };
    Ok(idx)
}

#[tauri::command]
fn remove_effect(index: usize, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let engine = state.engine.lock().unwrap();
    let engine = engine.as_ref().ok_or("Engine not running")?;
    let mut chain = engine.effect_chain().lock().unwrap();
    chain.remove(index);
    Ok(())
}

#[tauri::command]
fn clear_effects(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let engine = state.engine.lock().unwrap();
    let engine = engine.as_ref().ok_or("Engine not running")?;
    let mut chain = engine.effect_chain().lock().unwrap();
    chain.clear();
    Ok(())
}

#[tauri::command]
fn get_effects(state: tauri::State<'_, AppState>) -> Result<Vec<(String, bool)>, String> {
    let engine = state.engine.lock().unwrap();
    let engine = engine.as_ref().ok_or("Engine not running")?;
    let chain = engine.effect_chain().lock().unwrap();
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
            // Build tray menu
            let toggle = MenuItemBuilder::with_id("toggle", "Toggle Voice Changer")
                .build(app)?;
            let quit = MenuItemBuilder::with_id("quit", "Quit")
                .build(app)?;
            let menu = MenuBuilder::new(app)
                .item(&toggle)
                .separator()
                .item(&quit)
                .build()?;

            // Build tray icon
            let _tray = TrayIconBuilder::new()
                .menu(&menu)
                .tooltip("Voice Changer")
                .on_menu_event(move |app, event| {
                    match event.id().as_ref() {
                        "toggle" => {
                            let state = app.state::<AppState>();
                            let guard = state.engine.lock().unwrap();
                            if let Some(engine) = guard.as_ref() {
                                let new_bypass = !engine.is_bypassed();
                                engine.set_bypass(new_bypass);
                            }
                            drop(guard);
                        }
                        "quit" => {
                            app.exit(0);
                        }
                        _ => {}
                    }
                })
                .build(app)?;

            // Create the main window
            let _window = tauri::WebviewWindowBuilder::new(
                app,
                "main",
                tauri::WebviewUrl::App("index.html".into()),
            )
            .title("Voice Changer")
            .inner_size(400.0, 500.0)
            .resizable(false)
            .build()?;

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
            get_effects,
            get_virtual_output,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
