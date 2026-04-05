use std::path::Path;
use serde::Serialize;

/// Model download URLs
pub const CONTENTVEC_URL: &str =
    "https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/resolve/main/vec-768-layer-9.onnx";
// Placeholder — replace with an actual public RVC ONNX model URL
pub const DEMO_VOICE_URL: &str =
    "https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/resolve/main/vec-256-layer-9.onnx";

#[derive(Debug, Serialize)]
pub struct SetupStatus {
    pub ready: bool,
    pub has_contentvec: bool,
    pub has_demo_voice: bool,
    pub has_vb_cable: bool,
    pub contentvec_url: String,
    pub demo_voice_url: String,
}

pub fn check_setup(data_dir: &Path) -> SetupStatus {
    let models_dir = data_dir.join("models");
    let has_contentvec = models_dir.join("contentvec.onnx").exists();
    let has_demo_voice = models_dir.join("demo-voice.onnx").exists();
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
