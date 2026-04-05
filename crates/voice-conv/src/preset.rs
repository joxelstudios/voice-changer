use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// A voice preset: a named RVC model with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoicePreset {
    pub name: String,
    /// Path to the RVC generator ONNX model
    pub model_path: String,
    /// Default pitch shift in semitones
    pub pitch_shift: f32,
}

/// Manages voice presets from a directory.
pub struct PresetManager {
    presets_dir: PathBuf,
    presets: Vec<VoicePreset>,
}

impl PresetManager {
    pub fn new(presets_dir: &Path) -> Result<Self> {
        let mut manager = Self {
            presets_dir: presets_dir.to_path_buf(),
            presets: Vec::new(),
        };
        manager.scan()?;
        Ok(manager)
    }

    /// Scan the presets directory for .json preset files.
    pub fn scan(&mut self) -> Result<()> {
        self.presets.clear();

        if !self.presets_dir.exists() {
            log::info!("Presets directory does not exist: {:?}", self.presets_dir);
            return Ok(());
        }

        let entries = std::fs::read_dir(&self.presets_dir)
            .context("Failed to read presets directory")?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                match self.load_preset(&path) {
                    Ok(preset) => {
                        log::info!("Loaded preset: {}", preset.name);
                        self.presets.push(preset);
                    }
                    Err(e) => {
                        log::warn!("Failed to load preset {:?}: {e}", path);
                    }
                }
            }
        }

        log::info!("Loaded {} presets", self.presets.len());
        Ok(())
    }

    fn load_preset(&self, path: &Path) -> Result<VoicePreset> {
        let contents = std::fs::read_to_string(path)
            .context(format!("Failed to read preset file: {path:?}"))?;
        let preset: VoicePreset = serde_json::from_str(&contents)
            .context(format!("Failed to parse preset file: {path:?}"))?;
        Ok(preset)
    }

    /// Save a new preset to disk.
    pub fn save_preset(&mut self, preset: &VoicePreset) -> Result<()> {
        std::fs::create_dir_all(&self.presets_dir)?;
        let filename = format!("{}.json", preset.name.replace(' ', "_").to_lowercase());
        let path = self.presets_dir.join(filename);
        let json = serde_json::to_string_pretty(preset)?;
        std::fs::write(&path, json)?;
        self.presets.push(preset.clone());
        log::info!("Saved preset: {} -> {:?}", preset.name, path);
        Ok(())
    }

    pub fn list(&self) -> &[VoicePreset] {
        &self.presets
    }

    pub fn get(&self, name: &str) -> Option<&VoicePreset> {
        self.presets.iter().find(|p| p.name == name)
    }
}
