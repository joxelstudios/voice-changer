use std::path::Path;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppConfig {
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    pub last_preset: Option<String>,
    pub sample_rate: Option<u32>,
    pub buffer_size: Option<u32>,
}

impl AppConfig {
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(path, json).map_err(|e| e.to_string())
    }
}
