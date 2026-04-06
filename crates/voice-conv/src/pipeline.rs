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
    /// Path to ContentVec ONNX model
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
    // SOLA overlap-add state
    prev_output: Vec<f32>,
    overlap_samples: usize,
    // Lookahead context buffer — past audio prepended to each chunk
    context_buffer: Vec<f32>,
    context_samples: usize,
}

impl VoiceConverter {
    pub fn new(config: VoiceConverterConfig) -> Result<Self> {
        let mut content = ContentExtractor::load(&config.content_model_path)
            .context("Failed to load content extractor")?;
        let mut generator = RvcGenerator::load(&config.generator_model_path)
            .context("Failed to load RVC generator")?;

        // 200ms overlap for SOLA crossfading (up from 10ms)
        let overlap_samples = (config.sample_rate as f32 * 0.2) as usize;
        // 500ms of past context prepended to each chunk for model quality
        let context_samples = (config.sample_rate as f32 * 0.5) as usize;

        // Warm-up inference with dummy data to trigger ONNX graph optimization
        // and kernel selection. First inference is always slow without this.
        log::info!("Running warm-up inference...");
        let warmup_16k = vec![0.0_f32; CONTENT_SAMPLE_RATE as usize / 10]; // 100ms
        if let Ok(features) = content.extract(&warmup_16k) {
            let n = features.shape()[0];
            if n > 0 {
                let f0 = vec![0.0_f32; n];
                let bins = vec![0_i64; n];
                let _ = generator.generate(&features, &bins, &f0);
            }
        }
        log::info!("Warm-up complete");

        log::info!(
            "Voice converter initialized (sr: {}, pitch: {:+} st, overlap: {}ms, context: {}ms)",
            config.sample_rate,
            config.pitch_shift,
            (overlap_samples as f32 / config.sample_rate as f32 * 1000.0) as u32,
            (context_samples as f32 / config.sample_rate as f32 * 1000.0) as u32,
        );

        Ok(Self {
            content,
            generator,
            sample_rate: config.sample_rate,
            pitch_shift: config.pitch_shift,
            prev_output: Vec::new(),
            overlap_samples,
            context_buffer: Vec::new(),
            context_samples,
        })
    }

    /// Process a chunk of audio through the voice conversion pipeline.
    pub fn process_chunk(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Build input with lookahead context
        let full_input = if self.context_buffer.is_empty() {
            input.to_vec()
        } else {
            let mut buf = self.context_buffer.clone();
            buf.extend_from_slice(input);
            buf
        };

        // Update context buffer with the latest audio
        let ctx_start = input.len().saturating_sub(self.context_samples);
        self.context_buffer = input[ctx_start..].to_vec();

        // 2. Resample to 16kHz for ContentVec
        let audio_16k = resample::resample(&full_input, self.sample_rate, CONTENT_SAMPLE_RATE)?;

        // 3. Run ContentVec and F0 extraction in parallel via scoped threads.
        // Split the borrow: content extractor goes to spawned thread, F0 runs on current.
        let content = &mut self.content;
        let (features_result, f0_raw) = std::thread::scope(|s| {
            let audio_ref = &audio_16k;

            let content_handle = s.spawn(move || {
                content.extract(audio_ref)
            });

            let f0 = extract_f0(audio_ref, CONTENT_SAMPLE_RATE, CONTENT_HOP_LENGTH);

            let features = content_handle.join().expect("ContentVec thread panicked");
            (features, f0)
        });

        let features = features_result?;
        let n_frames = features.shape()[0];

        if n_frames == 0 {
            return Ok(vec![0.0; input.len()]);
        }

        // 4. Align F0 length to feature frames
        let mut f0 = f0_raw;
        f0.resize(n_frames, 0.0);

        // 5. Apply pitch shift
        if self.pitch_shift != 0.0 {
            let ratio = 2.0_f32.powf(self.pitch_shift / 12.0);
            for v in &mut f0 {
                if *v > 0.0 {
                    *v *= ratio;
                }
            }
        }

        // 6. Convert F0 to mel bins
        let f0_bins = f0_to_mel_bins(&f0);

        // 7. Run RVC generator
        let generated = self.generator.generate(&features, &f0_bins, &f0)?;

        // 8. Trim the context prefix from the output (only keep the new audio portion)
        let context_16k_len = if !self.prev_output.is_empty() {
            // Approximate how many output samples correspond to the context
            let context_ratio = self.context_buffer.len() as f64 / full_input.len() as f64;
            (generated.len() as f64 * (1.0 - context_ratio / (1.0 + context_ratio))) as usize
        } else {
            0
        };
        let trimmed = if context_16k_len > 0 && context_16k_len < generated.len() {
            generated[generated.len() - (generated.len() - context_16k_len)..].to_vec()
        } else {
            generated
        };

        // 9. Length-match to input via linear interpolation
        let output = if trimmed.len() != input.len() {
            let ratio = trimmed.len() as f64 / input.len() as f64;
            (0..input.len())
                .map(|i| {
                    let pos = i as f64 * ratio;
                    let idx = pos as usize;
                    let frac = pos - idx as f64;
                    let a = trimmed.get(idx).copied().unwrap_or(0.0);
                    let b = trimmed.get(idx + 1).copied().unwrap_or(a);
                    a * (1.0 - frac as f32) + b * frac as f32
                })
                .collect()
        } else {
            trimmed
        };

        // 10. SOLA crossfade with previous chunk
        let result = self.sola_crossfade(output);
        Ok(result)
    }

    /// SOLA (Synchronous Overlap-Add) crossfade.
    /// Finds the best alignment within the overlap region to minimize
    /// discontinuity, then applies a Hann-windowed crossfade.
    fn sola_crossfade(&mut self, mut output: Vec<f32>) -> Vec<f32> {
        if self.prev_output.is_empty() || output.is_empty() {
            self.prev_output = output.clone();
            return output;
        }

        let overlap = self.overlap_samples.min(self.prev_output.len()).min(output.len());
        if overlap < 2 {
            self.prev_output = output.clone();
            return output;
        }

        // Find best offset within overlap using cross-correlation
        let search_range = (overlap / 4).max(1);
        let prev_tail = &self.prev_output[self.prev_output.len() - overlap..];
        let mut best_offset = 0;
        let mut best_corr = f32::NEG_INFINITY;

        for offset in 0..search_range {
            let mut corr = 0.0_f32;
            let len = overlap - offset;
            for i in 0..len {
                corr += prev_tail[offset + i] * output[i];
            }
            if corr > best_corr {
                best_corr = corr;
                best_offset = offset;
            }
        }

        // Apply Hann-windowed crossfade at best alignment
        let fade_len = overlap - best_offset;
        for i in 0..fade_len {
            let t = i as f32 / fade_len as f32;
            // Hann window for smooth crossfade
            let w = 0.5 * (1.0 - (std::f32::consts::PI * t).cos());
            output[i] = prev_tail[best_offset + i] * (1.0 - w) + output[i] * w;
        }

        self.prev_output = output.clone();
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
        self.prev_output.clear();
        self.context_buffer.clear();
    }
}
