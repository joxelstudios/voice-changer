/// DIO (Distributed Inline-filter Operation) F0 extraction.
/// Simplified implementation for real-time pitch detection.

const F0_FLOOR: f32 = 71.0;
const F0_CEIL: f32 = 800.0;

/// Extract F0 (fundamental frequency) contour from 16kHz audio.
/// Returns one F0 value per hop (hop_length samples).
/// Returns 0.0 for unvoiced frames.
pub fn extract_f0(audio: &[f32], sample_rate: u32, hop_length: usize) -> Vec<f32> {
    let sr = sample_rate as f32;
    let n_frames = audio.len() / hop_length;
    let mut f0 = vec![0.0_f32; n_frames];

    let min_period = (sr / F0_CEIL) as usize;
    let max_period = (sr / F0_FLOOR) as usize;

    for frame_idx in 0..n_frames {
        let center = frame_idx * hop_length + hop_length / 2;
        let window_size = max_period * 2;
        let start = center.saturating_sub(window_size / 2);
        let end = (start + window_size).min(audio.len());

        if end - start < max_period {
            continue;
        }

        let window = &audio[start..end];
        f0[frame_idx] = autocorrelation_f0(window, sr, min_period, max_period);
    }

    // Simple median filter to remove octave errors (window size 3)
    median_filter_3(&mut f0);

    f0
}

/// Autocorrelation-based pitch detection for a single frame.
fn autocorrelation_f0(frame: &[f32], sample_rate: f32, min_period: usize, max_period: usize) -> f32 {
    let n = frame.len();
    if n < max_period {
        return 0.0;
    }

    // Compute energy of the frame
    let energy: f32 = frame.iter().map(|s| s * s).sum();
    if energy < 1e-8 {
        return 0.0; // silence
    }

    let mut best_corr = 0.0_f32;
    let mut best_period = 0;

    for period in min_period..=max_period.min(n / 2) {
        // Normalized cross-correlation
        let mut corr = 0.0_f32;
        let mut energy_shifted = 0.0_f32;
        let len = n - period;

        for i in 0..len {
            corr += frame[i] * frame[i + period];
            energy_shifted += frame[i + period] * frame[i + period];
        }

        let norm = (energy * energy_shifted).sqrt();
        if norm > 1e-10 {
            let normalized = corr / norm;
            if normalized > best_corr {
                best_corr = normalized;
                best_period = period;
            }
        }
    }

    // Voicing threshold
    if best_corr < 0.3 || best_period == 0 {
        return 0.0;
    }

    sample_rate / best_period as f32
}

/// Convert F0 in Hz to mel-scale pitch bins (1-255) for RVC input.
/// Unvoiced frames (f0 == 0) map to 0.
/// Uses the same normalization as the RVC Python reference:
///   f0_mel_min = 1127 * ln(1 + 50/700)    ≈ 80.0
///   f0_mel_max = 1127 * ln(1 + 1100/700)  ≈ 1000.5
///   bin = (mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
pub fn f0_to_mel_bins(f0: &[f32]) -> Vec<i64> {
    let f0_mel_min: f32 = 1127.0 * (1.0_f32 + 50.0 / 700.0).ln();
    let f0_mel_max: f32 = 1127.0 * (1.0_f32 + 1100.0 / 700.0).ln();
    let mel_range = f0_mel_max - f0_mel_min;

    f0.iter()
        .map(|&freq| {
            if freq <= 0.0 {
                0_i64
            } else {
                let mel = 1127.0 * (1.0 + freq / 700.0).ln();
                let bin = ((mel - f0_mel_min) * 254.0 / mel_range + 1.0)
                    .round()
                    .clamp(1.0, 255.0) as i64;
                bin
            }
        })
        .collect()
}

fn median_filter_3(values: &mut [f32]) {
    if values.len() < 3 {
        return;
    }
    let original = values.to_vec();
    for i in 1..values.len() - 1 {
        let mut triple = [original[i - 1], original[i], original[i + 1]];
        triple.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values[i] = triple[1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_f0_sine_wave() {
        let sr = 16000;
        let freq = 200.0;
        let duration = 0.1; // 100ms
        let n_samples = (sr as f32 * duration) as usize;
        let audio: Vec<f32> = (0..n_samples)
            .map(|i| (std::f32::consts::TAU * freq * i as f32 / sr as f32).sin())
            .collect();

        let f0 = extract_f0(&audio, sr, 160);
        // Should detect approximately 200Hz
        let voiced: Vec<f32> = f0.iter().copied().filter(|&f| f > 0.0).collect();
        assert!(!voiced.is_empty(), "Should detect voiced frames");
        let avg_f0: f32 = voiced.iter().sum::<f32>() / voiced.len() as f32;
        assert!(
            (avg_f0 - freq).abs() < 30.0,
            "Expected ~{freq}Hz, got {avg_f0}Hz"
        );
    }

    #[test]
    fn test_f0_to_mel_bins() {
        let f0 = vec![0.0, 200.0, 440.0, 0.0];
        let bins = f0_to_mel_bins(&f0);
        assert_eq!(bins[0], 0); // unvoiced
        assert!(bins[1] > 0 && bins[1] < 256);
        assert!(bins[2] > bins[1]); // 440Hz should have higher mel bin than 200Hz
        assert_eq!(bins[3], 0); // unvoiced
    }
}
