use std::path::Path;
use serde::Serialize;

/// Model download URLs
/// ContentVec vec-768-layer-12: the feature extractor that ozada/onnx_rvc models expect
pub const CONTENTVEC_URL: &str =
    "https://huggingface.co/ozada/onnx_rvc/resolve/main/vec-768-layer-12.onnx";
/// Space Marine voice (Dawn of War) - converted from GeorgeDr/Space_MarineRVC
pub const DEMO_VOICE_URL: &str =
    "https://github.com/joxelstudios/voice-changer/releases/download/v0.1.0/space_marine.onnx";

#[derive(Debug, Serialize)]
pub struct SetupStatus {
    pub ready: bool,
    pub has_contentvec: bool,
    pub has_demo_voice: bool,
    pub has_vb_cable: bool,
    pub contentvec_url: String,
    pub demo_voice_url: String,
}

fn file_ok(path: &Path, min_size: u64) -> bool {
    path.exists() && std::fs::metadata(path).map(|m| m.len() >= min_size).unwrap_or(false)
}

pub fn check_setup(data_dir: &Path) -> SetupStatus {
    let models_dir = data_dir.join("models");
    // Check models exist AND are reasonably sized (not wrong/truncated downloads)
    let has_contentvec = file_ok(&models_dir.join("contentvec.onnx"), 100_000_000); // >100MB
    let has_demo_voice = file_ok(&models_dir.join("demo-voice.onnx"), 50_000_000);  // >50MB
    let has_vb_cable = virtual_mic::find_virtual_output().is_some();

    SetupStatus {
        ready: has_contentvec && has_demo_voice,
        has_contentvec,
        has_demo_voice,
        has_vb_cable,
        contentvec_url: CONTENTVEC_URL.to_string(),
        demo_voice_url: DEMO_VOICE_URL.to_string(),
    }
}
