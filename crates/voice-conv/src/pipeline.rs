use anyhow::{Context, Result};

use crate::content::ContentExtractor;
use crate::f0::{extract_f0, f0_to_mel_bins};
use crate::generator::{RvcGenerator, FIXED_FRAMES};
use crate::resample;

const CONTENT_SAMPLE_RATE: u32 = 16000;
const CONTENT_HOP_LENGTH: usize = 160; // 10ms at 16kHz
/// RVC V2 models output at 40kHz
const GENERATOR_SAMPLE_RATE: u32 = 40000;

/// Max input samples at 16kHz that produces <= FIXED_FRAMES after ContentVec + repeat(2).
/// ContentVec produces 1 frame per 320 samples (hop=320 at 16kHz).
/// After repeat(2), frames double. So max raw frames = FIXED_FRAMES / 2 = 180.
/// Max 16kHz samples = 180 * 320 = 57600 (3.6 seconds).
const MAX_16K_SAMPLES: usize = (FIXED_FRAMES / 2) * 320;

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
    // Simple crossfade overlap
    prev_tail: Vec<f32>,
    overlap_samples: usize,
}

impl VoiceConverter {
    pub fn new(config: VoiceConverterConfig) -> Result<Self> {
        let content = ContentExtractor::load(&config.content_model_path)
            .context("Failed to load content extractor")?;
        let generator = RvcGenerator::load(&config.generator_model_path)
            .context("Failed to load RVC generator")?;

        let overlap_samples = (config.sample_rate as f32 * 0.05) as usize; // 50ms overlap

        log::info!(
            "Voice converter initialized (sr: {}, pitch: {:+} st)",
            config.sample_rate, config.pitch_shift,
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

    pub fn process_chunk(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        // 1. Resample to 16kHz for ContentVec
        let audio_16k = resample::resample(input, self.sample_rate, CONTENT_SAMPLE_RATE)?;

        // 2. Truncate to max length that fits FIXED_FRAMES after doubling
        let audio_16k = if audio_16k.len() > MAX_16K_SAMPLES {
            log::warn!("Truncating 16kHz audio from {} to {MAX_16K_SAMPLES} samples", audio_16k.len());
            audio_16k[..MAX_16K_SAMPLES].to_vec()
        } else {
            audio_16k
        };

        // 3. Run ContentVec and F0 extraction in parallel
        let content = &mut self.content;
        let (features_result, f0_raw) = std::thread::scope(|s| {
            let audio_ref = &audio_16k;
            let content_handle = s.spawn(move || content.extract(audio_ref));
            let f0 = extract_f0(audio_ref, CONTENT_SAMPLE_RATE, CONTENT_HOP_LENGTH);
            let features = content_handle.join().expect("ContentVec thread panicked");
            (features, f0)
        });

        let features = features_result?;
        let n_frames = features.shape()[0]; // Already doubled by ContentExtractor

        if n_frames == 0 {
            return Ok(vec![0.0; input.len()]);
        }

        // 4. Align F0 to doubled frame count
        // F0 was extracted at the original (non-doubled) rate, so we need to
        // repeat each F0 value to match the doubled frames
        let mut f0 = Vec::with_capacity(n_frames);
        for &val in &f0_raw {
            f0.push(val);
            f0.push(val); // duplicate to match frame doubling
        }
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

        // 7. Run RVC generator (pads to FIXED_FRAMES internally, trims output)
        let generated = self.generator.generate(&features, &f0_bins, &f0)?;

        // 8. Resample from generator's 40kHz to device sample rate
        let resampled = resample::resample(&generated, GENERATOR_SAMPLE_RATE, self.sample_rate)?;

        // 9. Length-match to input (should be close after proper resampling)
        let output = length_match(&resampled, input.len());

        // 10. Simple crossfade with previous chunk
        let result = self.crossfade(output);
        Ok(result)
    }

    fn crossfade(&mut self, mut output: Vec<f32>) -> Vec<f32> {
        if !self.prev_tail.is_empty() && !output.is_empty() {
            let fade_len = self.overlap_samples.min(self.prev_tail.len()).min(output.len());
            for i in 0..fade_len {
                let t = i as f32 / fade_len as f32;
                output[i] = self.prev_tail[self.prev_tail.len() - fade_len + i] * (1.0 - t)
                    + output[i] * t;
            }
        }
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

    pub fn reset(&mut self) {
        self.prev_tail.clear();
    }
}

fn length_match(audio: &[f32], target_len: usize) -> Vec<f32> {
    if audio.len() == target_len {
        return audio.to_vec();
    }
    let ratio = audio.len() as f64 / target_len.max(1) as f64;
    (0..target_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = pos as usize;
            let frac = (pos - idx as f64) as f32;
            let a = audio.get(idx).copied().unwrap_or(0.0);
            let b = audio.get(idx + 1).copied().unwrap_or(a);
            a * (1.0 - frac) + b * frac
        })
        .collect()
}
