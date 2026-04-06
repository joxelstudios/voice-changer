use anyhow::Result;
use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;

/// ContentVec feature extractor (HuBERT-based).
/// Extracts 768-dim content features from 16kHz audio.
pub struct ContentExtractor {
    session: Session,
}

impl ContentExtractor {
    pub fn load(model_path: &str) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {e}"))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("Failed to set intra threads: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load ContentVec model {model_path}: {e}"))?;

        // Log actual input/output names for debugging
        for input in session.inputs() {
            log::info!("ContentVec input: name='{}'", input.name());
        }
        for output in session.outputs() {
            log::info!("ContentVec output: name='{}'", output.name());
        }

        log::info!("ContentVec model loaded from {model_path}");
        Ok(Self { session })
    }

    /// Extract content features from 16kHz mono audio.
    /// Returns [frames*2, 768] feature matrix (doubled frame rate for RVC generator).
    pub fn extract(&mut self, audio_16k: &[f32]) -> Result<Array2<f32>> {
        let n = audio_16k.len();

        // FIX: ContentVec expects 3D input [1, 1, N] (batch, channels, samples)
        let input_value = Value::from_array(([1_usize, 1_usize, n], audio_16k.to_vec()))
            .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {e}"))?;

        let input_sv: ort::session::SessionInputValue<'_> = input_value.into();
        let outputs = self.session.run(vec![("source", input_sv)])
            .map_err(|e| anyhow::anyhow!("ContentVec inference failed: {e}"))?;

        // Output shape: [1, 768, frames]
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract ContentVec output: {e}"))?;

        let dim = shape[1] as usize;  // 768
        let frames = shape[2] as usize;

        // FIX: Repeat frames 2x and transpose to [frames*2, 768]
        // Python reference: np.repeat(hubert, 2, axis=2).transpose(0, 2, 1)
        // Output is [1, 768, frames] → repeat on axis 2 → [1, 768, frames*2] → transpose → [1, frames*2, 768]
        let doubled_frames = frames * 2;
        let mut features = Vec::with_capacity(doubled_frames * dim);

        for f in 0..frames {
            // Each original frame is repeated twice
            for _repeat in 0..2 {
                for d in 0..dim {
                    // data is [1, 768, frames] in row-major: index = d * frames + f
                    features.push(data[d * frames + f]);
                }
            }
        }

        let result = Array2::from_shape_vec((doubled_frames, dim), features)?;
        log::debug!("ContentVec: {frames} frames → {doubled_frames} (doubled) x {dim} dims");
        Ok(result)
    }
}
