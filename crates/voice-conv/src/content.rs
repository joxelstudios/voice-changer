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

        // ContentVec expects 3D input [1, 1, N] (batch, channels, samples)
        let input_value = Value::from_array(([1_usize, 1_usize, n], audio_16k.to_vec()))
            .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {e}"))?;

        let input_sv: ort::session::SessionInputValue<'_> = input_value.into();
        let outputs = self.session.run(vec![("source", input_sv)])
            .map_err(|e| anyhow::anyhow!("ContentVec inference failed: {e}"))?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract ContentVec output: {e}"))?;

        // Output could be [1, frames, 768] or [1, 768, frames] depending on model.
        // Detect by checking which dimension is 768.
        let (frames, dim, is_features_last) = if shape.len() == 3 {
            let d1 = shape[1] as usize;
            let d2 = shape[2] as usize;
            if d2 == 768 || (d2 > d1 && d1 < 768) {
                // [1, frames, 768] — standard HuBERT output
                (d1, d2, true)
            } else {
                // [1, 768, frames] — some models use features-first
                (d2, d1, false)
            }
        } else {
            return Err(anyhow::anyhow!(
                "Unexpected ContentVec output rank: {} (shape: {:?})",
                shape.len(), &shape[..]
            ));
        };

        log::debug!(
            "ContentVec raw output: shape=[1, {}, {}], detected frames={} dim={} features_last={}",
            shape[1], shape[2], frames, dim, is_features_last
        );

        // Repeat each frame 2x to match RVC generator's expected hop rate.
        // Final output: [frames*2, 768]
        let doubled_frames = frames * 2;
        let mut features = Vec::with_capacity(doubled_frames * dim);

        for f in 0..frames {
            for _repeat in 0..2 {
                for d in 0..dim {
                    let val = if is_features_last {
                        // data layout [1, frames, 768]: index = f * dim + d
                        data[f * dim + d]
                    } else {
                        // data layout [1, 768, frames]: index = d * frames + f
                        data[d * frames + f]
                    };
                    features.push(val);
                }
            }
        }

        let result = Array2::from_shape_vec((doubled_frames, dim), features)?;
        log::debug!("ContentVec: {frames} raw frames → {doubled_frames} doubled x {dim} dims");
        Ok(result)
    }
}
