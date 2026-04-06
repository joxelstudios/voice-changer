use anyhow::Result;
use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;

/// The ONNX model was traced with this exact frame count.
/// Attention layer reshapes are baked as constants — inference ONLY works with this value.
pub const FIXED_FRAMES: usize = 360;

/// RVC V2 Generator model.
pub struct RvcGenerator {
    session: Session,
    use_f0: bool,
}

impl RvcGenerator {
    pub fn load(model_path: &str) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {e}"))?
            .with_intra_threads(2)
            .map_err(|e| anyhow::anyhow!("Failed to set intra threads: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load RVC generator {model_path}: {e}"))?;

        let mut input_names: Vec<String> = Vec::new();
        for input in session.inputs() {
            log::info!("Generator input: name='{}'", input.name());
            input_names.push(input.name().to_string());
        }

        let use_f0 = input_names.iter().any(|n| n == "pitch");

        log::info!(
            "RVC generator loaded (f0: {}, inputs: {:?})",
            if use_f0 { "yes" } else { "no" },
            input_names,
        );
        Ok(Self { session, use_f0 })
    }

    /// Run voice conversion inference.
    /// Features are padded to FIXED_FRAMES, output is trimmed proportionally.
    pub fn generate(
        &mut self,
        features: &Array2<f32>,
        f0_bins: &[i64],
        f0_hz: &[f32],
    ) -> Result<Vec<f32>> {
        let actual_frames = features.shape()[0];
        let dim = features.shape()[1];

        if actual_frames == 0 {
            return Ok(Vec::new());
        }

        if actual_frames > FIXED_FRAMES {
            return Err(anyhow::anyhow!(
                "Frame count {actual_frames} exceeds FIXED_FRAMES {FIXED_FRAMES}. \
                 Input audio chunk is too long."
            ));
        }

        // Pad features, f0_bins, f0_hz to FIXED_FRAMES
        let (padded_phone, padded_bins, padded_hz) =
            pad_to_fixed_frames(features, f0_bins, f0_hz, FIXED_FRAMES);

        let mut inputs: Vec<(&str, ort::session::SessionInputValue<'_>)> = vec![
            ("phone", Value::from_array(([1_usize, FIXED_FRAMES, dim], padded_phone))
                .map_err(|e| anyhow::anyhow!("phone tensor: {e}"))?.into()),
            ("phone_lengths", Value::from_array(([1_usize], vec![actual_frames as i64]))
                .map_err(|e| anyhow::anyhow!("phone_lengths tensor: {e}"))?.into()),
            ("ds", Value::from_array(([1_usize], vec![0_i64]))
                .map_err(|e| anyhow::anyhow!("ds tensor: {e}"))?.into()),
        ];

        if self.use_f0 {
            inputs.push(("pitch", Value::from_array(([1_usize, FIXED_FRAMES], padded_bins))
                .map_err(|e| anyhow::anyhow!("pitch tensor: {e}"))?.into()));
            inputs.push(("pitchf", Value::from_array(([1_usize, FIXED_FRAMES], padded_hz))
                .map_err(|e| anyhow::anyhow!("pitchf tensor: {e}"))?.into()));
        }

        let outputs = self.session.run(inputs)
            .map_err(|e| anyhow::anyhow!("RVC generator inference failed: {e}"))?;

        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract generated audio: {e}"))?;

        let full_audio: Vec<f32> = data.to_vec();

        // Trim output proportionally to actual frame count
        let trim_len = (full_audio.len() as f64 * actual_frames as f64 / FIXED_FRAMES as f64) as usize;
        let trimmed = full_audio[..trim_len.min(full_audio.len())].to_vec();

        log::debug!(
            "RVC generator: {} actual frames (padded to {}), output {} -> {} samples",
            actual_frames, FIXED_FRAMES, full_audio.len(), trimmed.len()
        );
        Ok(trimmed)
    }

    pub fn uses_f0(&self) -> bool {
        self.use_f0
    }
}

/// Pad features, f0_bins, and f0_hz to exactly `target_frames`.
/// Features are zero-padded, f0 values are zero-padded (unvoiced).
pub fn pad_to_fixed_frames(
    features: &Array2<f32>,
    f0_bins: &[i64],
    f0_hz: &[f32],
    target_frames: usize,
) -> (Vec<f32>, Vec<i64>, Vec<f32>) {
    let actual = features.shape()[0];
    let dim = features.shape()[1];

    // Pad phone features: [actual, dim] → [target, dim] with zeros
    let mut padded_phone = Vec::with_capacity(target_frames * dim);
    padded_phone.extend(features.iter());
    padded_phone.resize(target_frames * dim, 0.0);

    // Pad f0 bins and hz with zeros (unvoiced)
    let mut padded_bins = f0_bins.to_vec();
    padded_bins.resize(target_frames, 0);

    let mut padded_hz = f0_hz.to_vec();
    padded_hz.resize(target_frames, 0.0);

    (padded_phone, padded_bins, padded_hz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_to_fixed_frames_shorter_input() {
        let features = Array2::from_shape_vec((20, 768), vec![1.0; 20 * 768]).unwrap();
        let bins = vec![50_i64; 20];
        let hz = vec![200.0_f32; 20];

        let (p_phone, p_bins, p_hz) = pad_to_fixed_frames(&features, &bins, &hz, FIXED_FRAMES);

        // Phone: 360 * 768 elements
        assert_eq!(p_phone.len(), FIXED_FRAMES * 768);
        // First 20*768 should be 1.0, rest should be 0.0
        assert_eq!(p_phone[0], 1.0);
        assert_eq!(p_phone[20 * 768 - 1], 1.0);
        assert_eq!(p_phone[20 * 768], 0.0);
        assert_eq!(p_phone[FIXED_FRAMES * 768 - 1], 0.0);

        // Bins: 360 elements
        assert_eq!(p_bins.len(), FIXED_FRAMES);
        assert_eq!(p_bins[0], 50);
        assert_eq!(p_bins[19], 50);
        assert_eq!(p_bins[20], 0); // padded with unvoiced

        // Hz: 360 elements
        assert_eq!(p_hz.len(), FIXED_FRAMES);
        assert_eq!(p_hz[0], 200.0);
        assert_eq!(p_hz[20], 0.0);
    }

    #[test]
    fn test_pad_to_fixed_frames_exact_size() {
        let features = Array2::from_shape_vec((FIXED_FRAMES, 768), vec![1.0; FIXED_FRAMES * 768]).unwrap();
        let bins = vec![50_i64; FIXED_FRAMES];
        let hz = vec![200.0_f32; FIXED_FRAMES];

        let (p_phone, p_bins, p_hz) = pad_to_fixed_frames(&features, &bins, &hz, FIXED_FRAMES);

        assert_eq!(p_phone.len(), FIXED_FRAMES * 768);
        assert!(p_phone.iter().all(|&v| v == 1.0)); // no padding added
        assert_eq!(p_bins.len(), FIXED_FRAMES);
        assert!(p_bins.iter().all(|&v| v == 50));
    }

    #[test]
    fn test_output_trim_proportional() {
        // Simulate: 20 actual frames padded to 360 → output 144000 samples
        // Trim to 20/360 * 144000 = 8000 samples
        let actual_frames = 20_usize;
        let full_output_len = 144000_usize; // 360 frames * 400 samples/frame at 40kHz
        let trim_len = (full_output_len as f64 * actual_frames as f64 / FIXED_FRAMES as f64) as usize;
        assert_eq!(trim_len, 8000);
    }

    #[test]
    fn test_output_trim_exact_frames() {
        // 360 actual frames → no trimming needed
        let actual_frames = FIXED_FRAMES;
        let full_output_len = 144000_usize;
        let trim_len = (full_output_len as f64 * actual_frames as f64 / FIXED_FRAMES as f64) as usize;
        assert_eq!(trim_len, full_output_len);
    }
}
