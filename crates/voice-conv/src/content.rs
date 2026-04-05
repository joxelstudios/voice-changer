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
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load ContentVec model {model_path}: {e}"))?;

        log::info!("ContentVec model loaded from {model_path}");
        Ok(Self { session })
    }

    /// Extract content features from 16kHz mono audio.
    pub fn extract(&mut self, audio_16k: &[f32]) -> Result<Array2<f32>> {
        let n = audio_16k.len();
        let input_value = Value::from_array(([1_usize, n], audio_16k.to_vec()))
            .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {e}"))?;

        let input_sv: ort::session::SessionInputValue<'_> = input_value.into();
        let outputs = self.session.run(vec![("source", input_sv)])
            .map_err(|e| anyhow::anyhow!("ContentVec inference failed: {e}"))?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract ContentVec output: {e}"))?;

        let frames = shape[1] as usize;
        let dim = shape[2] as usize;
        let features = Array2::from_shape_vec((frames, dim), data.to_vec())?;

        log::debug!("ContentVec: extracted {frames} frames x {dim} dims");
        Ok(features)
    }
}
