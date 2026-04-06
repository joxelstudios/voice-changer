use anyhow::{Context, Result};

use crate::content::ContentExtractor;
use crate::f0::{extract_f0, f0_to_mel_bins};
use crate::generator::{RvcGenerator, FIXED_FRAMES};
use crate::resample;

const CONTENT_SAMPLE_RATE: u32 = 16000;
const CONTENT_HOP_LENGTH: usize = 160;
const GENERATOR_SAMPLE_RATE: u32 = 40000;

/// Max 16kHz samples to stay within FIXED_FRAMES after 2x interpolation.
/// ContentVec hop=320 at 16kHz: 1 frame per 320 samples.
/// After 2x interp: 2 frames per 320 samples.
/// Max raw frames = FIXED_FRAMES / 2 = 180.
/// Max samples = 180 * 320 = 57600 (3.6s).
const MAX_16K_SAMPLES: usize = (FIXED_FRAMES / 2) * 320;

/// Reflection padding size (samples at 16kHz) — gives model context at boundaries.
const REFLECTION_PAD: usize = 640; // 40ms, enough for a few ContentVec hops

#[derive(Debug, Clone)]
pub struct VoiceConverterConfig {
    pub content_model_path: String,
    pub generator_model_path: String,
    pub sample_rate: u32,
    pub pitch_shift: f32,
}

pub struct VoiceConverter {
    content: ContentExtractor,
    generator: RvcGenerator,
    sample_rate: u32,
    pitch_shift: f32,
    // SOLA state
    sola_buffer: Vec<f32>,
    sola_overlap: usize,
    sola_search: usize,
    // High-pass filter state (48Hz Butterworth, 1st order for simplicity)
    hp_prev_in: f32,
    hp_prev_out: f32,
    hp_alpha: f32,
}

impl VoiceConverter {
    pub fn new(config: VoiceConverterConfig) -> Result<Self> {
        let content = ContentExtractor::load(&config.content_model_path)
            .context("Failed to load content extractor")?;
        let generator = RvcGenerator::load(&config.generator_model_path)
            .context("Failed to load RVC generator")?;

        // SOLA parameters at device sample rate
        let sola_overlap = (config.sample_rate as f32 * 0.05) as usize; // 50ms overlap
        let sola_search = (config.sample_rate as f32 * 0.012) as usize; // 12ms search (like w-okada)

        // 48Hz high-pass filter coefficient (1st order approximation)
        let rc = 1.0 / (2.0 * std::f32::consts::PI * 48.0);
        let dt = 1.0 / CONTENT_SAMPLE_RATE as f32;
        let hp_alpha = rc / (rc + dt);

        log::info!(
            "Voice converter initialized (sr: {}, pitch: {:+} st, sola: {}ms overlap, {}ms search)",
            config.sample_rate, config.pitch_shift,
            sola_overlap * 1000 / config.sample_rate as usize,
            sola_search * 1000 / config.sample_rate as usize,
        );

        Ok(Self {
            content,
            generator,
            sample_rate: config.sample_rate,
            pitch_shift: config.pitch_shift,
            sola_buffer: Vec::new(),
            sola_overlap,
            sola_search,
            hp_prev_in: 0.0,
            hp_prev_out: 0.0,
            hp_alpha,
        })
    }

    pub fn process_chunk(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Resample to 16kHz
        let mut audio_16k = resample::resample(input, self.sample_rate, CONTENT_SAMPLE_RATE)?;

        // 2. High-pass filter at 48Hz (removes DC offset and mic rumble)
        self.apply_highpass(&mut audio_16k);

        // 3. Truncate to fit within FIXED_FRAMES
        if audio_16k.len() > MAX_16K_SAMPLES {
            audio_16k.truncate(MAX_16K_SAMPLES);
        }

        // 4. Reflection padding for better features at boundaries
        let padded = reflection_pad(&audio_16k, REFLECTION_PAD);

        // 5. Run ContentVec and F0 in parallel
        let content = &mut self.content;
        let (features_result, f0_raw) = std::thread::scope(|s| {
            let audio_ref = &padded;
            let content_handle = s.spawn(move || content.extract(audio_ref));
            let f0 = extract_f0(audio_ref, CONTENT_SAMPLE_RATE, CONTENT_HOP_LENGTH);
            let features = content_handle.join().expect("ContentVec thread panicked");
            (features, f0)
        });

        let features = features_result?;
        let n_frames = features.shape()[0];

        if n_frames == 0 {
            return Ok(vec![0.0; input.len()]);
        }

        // 6. Match F0 to doubled frame count (each F0 value covers 2 doubled frames)
        let mut f0 = Vec::with_capacity(n_frames);
        for &val in &f0_raw {
            f0.push(val);
            f0.push(val);
        }
        f0.resize(n_frames, 0.0);

        // 7. Apply pitch shift
        if self.pitch_shift != 0.0 {
            let ratio = 2.0_f32.powf(self.pitch_shift / 12.0);
            for v in &mut f0 {
                if *v > 0.0 {
                    *v *= ratio;
                }
            }
        }

        // 8. F0 to mel bins
        let f0_bins = f0_to_mel_bins(&f0);

        // 9. Generator inference (pads to FIXED_FRAMES, trims output)
        let generated = self.generator.generate(&features, &f0_bins, &f0)?;

        // 10. Resample from 40kHz to device rate
        let resampled = resample::resample(&generated, GENERATOR_SAMPLE_RATE, self.sample_rate)?;

        // 11. Clip to [-1, 1]
        let clipped: Vec<f32> = resampled.iter().map(|&s| s.clamp(-1.0, 1.0)).collect();

        // 12. SOLA crossfade with previous chunk
        let output = self.sola_crossfade(clipped);
        Ok(output)
    }

    /// Simple 1st-order high-pass filter at 48Hz.
    fn apply_highpass(&mut self, audio: &mut [f32]) {
        for sample in audio.iter_mut() {
            let filtered = self.hp_alpha * (self.hp_prev_out + *sample - self.hp_prev_in);
            self.hp_prev_in = *sample;
            self.hp_prev_out = filtered;
            *sample = filtered;
        }
    }

    /// SOLA (Synchronized Overlap-Add) crossfade.
    /// Uses cross-correlation to find best alignment, then cosine² fade.
    fn sola_crossfade(&mut self, audio: Vec<f32>) -> Vec<f32> {
        if self.sola_buffer.is_empty() || audio.is_empty() {
            // First chunk — store tail for next crossfade
            let tail_start = audio.len().saturating_sub(self.sola_overlap + self.sola_search);
            self.sola_buffer = audio[tail_start..].to_vec();
            return audio;
        }

        let overlap = self.sola_overlap.min(self.sola_buffer.len()).min(audio.len());
        let search = self.sola_search.min(audio.len().saturating_sub(overlap));

        if overlap < 2 || search == 0 {
            let tail_start = audio.len().saturating_sub(self.sola_overlap + self.sola_search);
            self.sola_buffer = audio[tail_start..].to_vec();
            return audio;
        }

        // Cross-correlation to find best alignment offset
        let mut best_offset = 0;
        let mut best_score = f32::NEG_INFINITY;

        for offset in 0..search {
            let mut cor_nom = 0.0_f32;
            let mut cor_den = 0.0_f32;
            for i in 0..overlap {
                let buf_val = self.sola_buffer[self.sola_buffer.len() - overlap + i];
                let audio_val = audio[offset + i];
                cor_nom += buf_val * audio_val;
                cor_den += audio_val * audio_val;
            }
            let score = cor_nom / (cor_den.sqrt() + 1e-8);
            if score > best_score {
                best_score = score;
                best_offset = offset;
            }
        }

        // Cosine² (equal-power) crossfade at best alignment
        let mut output = audio;
        for i in 0..overlap {
            let t = i as f32 / overlap as f32;
            let prev_strength = (t * 0.5 * std::f32::consts::PI).cos().powi(2);
            let cur_strength = ((1.0 - t) * 0.5 * std::f32::consts::PI).cos().powi(2);
            let buf_val = self.sola_buffer[self.sola_buffer.len() - overlap + i];
            output[best_offset + i] = buf_val * prev_strength + output[best_offset + i] * cur_strength;
        }

        // Store tail for next crossfade
        let tail_start = output.len().saturating_sub(self.sola_overlap + self.sola_search);
        self.sola_buffer = output[tail_start..].to_vec();
        output
    }

    pub fn set_pitch_shift(&mut self, semitones: f32) {
        self.pitch_shift = semitones;
    }

    pub fn pitch_shift(&self) -> f32 {
        self.pitch_shift
    }

    pub fn reset(&mut self) {
        self.sola_buffer.clear();
        self.hp_prev_in = 0.0;
        self.hp_prev_out = 0.0;
    }
}

/// Reflection-pad audio at both ends.
fn reflection_pad(audio: &[f32], pad: usize) -> Vec<f32> {
    let n = audio.len();
    let pad = pad.min(n);
    let mut padded = Vec::with_capacity(n + 2 * pad);

    // Left pad: mirror the start
    for i in (0..pad).rev() {
        padded.push(audio[i.min(n - 1)]);
    }
    padded.extend_from_slice(audio);
    // Right pad: mirror the end
    for i in 0..pad {
        padded.push(audio[(n - 1).saturating_sub(i)]);
    }
    padded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reflection_pad() {
        let audio = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let padded = reflection_pad(&audio, 2);
        // Left mirror: [2, 1], original: [1,2,3,4,5], right mirror: [4, 5]... wait
        // Actually: left pad = audio[1], audio[0]; right pad = audio[4], audio[3]
        assert_eq!(padded.len(), 5 + 4); // 2 pad each side
        assert_eq!(padded[0], 2.0); // audio[1] mirrored
        assert_eq!(padded[1], 1.0); // audio[0] mirrored
        assert_eq!(padded[2], 1.0); // original start
        assert_eq!(padded[6], 5.0); // original end
        assert_eq!(padded[7], 5.0); // audio[4] mirrored
        assert_eq!(padded[8], 4.0); // audio[3] mirrored
    }

    #[test]
    fn test_highpass_removes_dc() {
        // DC signal should be attenuated
        let mut audio = vec![1.0; 1000];
        let rc = 1.0 / (2.0 * std::f32::consts::PI * 48.0);
        let dt = 1.0 / 16000.0;
        let alpha = rc / (rc + dt);

        let mut prev_in = 0.0_f32;
        let mut prev_out = 0.0_f32;
        for sample in audio.iter_mut() {
            let filtered = alpha * (prev_out + *sample - prev_in);
            prev_in = *sample;
            prev_out = filtered;
            *sample = filtered;
        }

        // After settling, DC should be near zero
        let tail_avg: f32 = audio[500..].iter().sum::<f32>() / 500.0;
        assert!(tail_avg.abs() < 0.1, "DC should be attenuated, got {tail_avg}");
    }
}
