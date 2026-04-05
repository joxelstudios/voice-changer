use anyhow::{Context, Result};

use crate::content::ContentExtractor;
use crate::f0::{extract_f0, f0_to_mel_bins};
use crate::generator::RvcGenerator;
use crate::resample;

const CONTENT_SAMPLE_RATE: u32 = 16000;
const CONTENT_HOP_LENGTH: usize = 160; // 10ms at 16kHz

/// Configuration for the voice converter.
#[derive(Debug, Clone)]
pub struct VoiceConverterConfig {
    /// Path to ContentVec ONNX model (e.g., vec-768-layer-9.onnx)
    pub content_model_path: String,
    /// Path to RVC generator ONNX model
    pub generator_model_path: String,
    /// Input/output sample rate (typically 48000)
    pub sample_rate: u32,
    /// Pitch shift in semitones (0 = no shift)
    pub pitch_shift: f32,
}

/// Real-time voice converter using ContentVec + RVC ONNX pipeline.
pub struct VoiceConverter {
    content: ContentExtractor,
    generator: RvcGenerator,
    sample_rate: u32,
    pitch_shift: f32,
    // Overlap buffer for crossfading
    prev_tail: Vec<f32>,
    overlap_samples: usize,
}

impl VoiceConverter {
    pub fn new(config: VoiceConverterConfig) -> Result<Self> {
        let content = ContentExtractor::load(&config.content_model_path)
            .context("Failed to load content extractor")?;
        let generator = RvcGenerator::load(&config.generator_model_path)
            .context("Failed to load RVC generator")?;

        // 10ms overlap for crossfading between chunks
        let overlap_samples = (config.sample_rate as f32 * 0.01) as usize;

        log::info!(
            "Voice converter initialized (sr: {}, pitch: {:+} semitones)",
            config.sample_rate,
            config.pitch_shift
        );

        Ok(Self {
            content,
            generator,
            sample_rate: config.sample_rate,
            pitch_shift: config.pitch_shift,
            prev_tail: Vec::new(),
            overlap_samples,
        })
    }

    /// Process a chunk of audio through the voice conversion pipeline.
    ///
    /// Input: mono audio at `self.sample_rate`
    /// Output: converted audio at `self.sample_rate`
    pub fn process_chunk(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Resample to 16kHz for ContentVec
        let audio_16k = resample::resample(input, self.sample_rate, CONTENT_SAMPLE_RATE)?;

        // 2. Extract content features
        let features = self.content.extract(&audio_16k)?;
        let n_frames = features.shape()[0];

        if n_frames == 0 {
            return Ok(vec![0.0; input.len()]);
        }

        // 3. Extract F0 pitch contour
        let mut f0 = extract_f0(&audio_16k, CONTENT_SAMPLE_RATE, CONTENT_HOP_LENGTH);

        // Ensure f0 length matches feature frames
        f0.resize(n_frames, 0.0);

        // 4. Apply pitch shift
        if self.pitch_shift != 0.0 {
            let ratio = 2.0_f32.powf(self.pitch_shift / 12.0);
            for v in &mut f0 {
                if *v > 0.0 {
                    *v *= ratio;
                }
            }
        }

        // 5. Convert F0 to mel bins
        let f0_bins = f0_to_mel_bins(&f0);

        // 6. Run RVC generator
        let generated = self.generator.generate(&features, &f0_bins, &f0)?;

        // 7. Resample back to output sample rate
        // RVC generator outputs at the model's native rate (typically 32kHz or 40kHz)
        // For now assume generator outputs at the same rate we'll handle in post
        let output = if generated.len() != input.len() {
            // Simple length matching via linear interpolation
            let ratio = generated.len() as f64 / input.len() as f64;
            (0..input.len())
                .map(|i| {
                    let pos = i as f64 * ratio;
                    let idx = pos as usize;
                    let frac = pos - idx as f64;
                    let a = generated.get(idx).copied().unwrap_or(0.0);
                    let b = generated.get(idx + 1).copied().unwrap_or(a);
                    a * (1.0 - frac as f32) + b * frac as f32
                })
                .collect()
        } else {
            generated
        };

        // 8. Crossfade with previous chunk tail
        let result = self.crossfade(output);
        Ok(result)
    }

    /// Crossfade the new chunk with the tail of the previous chunk.
    fn crossfade(&mut self, mut output: Vec<f32>) -> Vec<f32> {
        if !self.prev_tail.is_empty() && !output.is_empty() {
            let fade_len = self.overlap_samples.min(self.prev_tail.len()).min(output.len());
            for i in 0..fade_len {
                let t = i as f32 / fade_len as f32;
                output[i] = self.prev_tail[self.prev_tail.len() - fade_len + i] * (1.0 - t)
                    + output[i] * t;
            }
        }

        // Store tail for next crossfade
        let tail_start = output.len().saturating_sub(self.overlap_samples);
        self.prev_tail = output[tail_start..].to_vec();

        output
    }

    pub fn set_pitch_shift(&mut self, semitones: f32) {
        self.pitch_shift = semitones;
    }

    pub fn pitch_shift(&self) -> f32 {
        self.pitch_shift
    }

    /// Reset internal state (call when switching voices or stopping)
    pub fn reset(&mut self) {
        self.prev_tail.clear();
    }
}
