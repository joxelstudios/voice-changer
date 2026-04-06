use anyhow::Result;
use ort::session::Session;
use ort::value::Value;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

/// RVC V2 Generator model.
/// Takes ContentVec features + F0 pitch data → outputs converted audio.
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

        let use_f0 = session
            .inputs()
            .iter()
            .any(|input| input.name() == "pitch");

        // Log actual input names for debugging
        for input in session.inputs() {
            log::info!("Generator input: name='{}'", input.name());
        }
        log::info!(
            "RVC generator loaded from {model_path} (f0: {})",
            if use_f0 { "yes" } else { "no" }
        );
        Ok(Self { session, use_f0 })
    }

    /// Run voice conversion inference.
    pub fn generate(
        &mut self,
        features: &ndarray::Array2<f32>,
        f0_bins: &[i64],
        f0_hz: &[f32],
    ) -> Result<Vec<f32>> {
        let frames = features.shape()[0];
        let dim = features.shape()[1];

        let phone_data: Vec<f32> = features.iter().copied().collect();

        // FIX: Use standard normal N(0,1) distribution — RVC expects torch.randn() equivalent.
        // Was uniform [-1,1] which has wrong statistical properties for the neural vocoder.
        let mut rng = rand::thread_rng();
        let rnd_data: Vec<f32> = (0..192 * frames)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();

        let mut inputs: Vec<(&str, ort::session::SessionInputValue<'_>)> = vec![
            ("phone", Value::from_array(([1_usize, frames, dim], phone_data))
                .map_err(|e| anyhow::anyhow!("phone tensor: {e}"))?.into()),
            ("phone_lengths", Value::from_array(([1_usize], vec![frames as i64]))
                .map_err(|e| anyhow::anyhow!("phone_lengths tensor: {e}"))?.into()),
            ("ds", Value::from_array(([1_usize], vec![0_i64]))
                .map_err(|e| anyhow::anyhow!("ds tensor: {e}"))?.into()),
            ("rnd", Value::from_array(([1_usize, 192, frames], rnd_data))
                .map_err(|e| anyhow::anyhow!("rnd tensor: {e}"))?.into()),
        ];

        if self.use_f0 {
            inputs.push(("pitch", Value::from_array(([1_usize, frames], f0_bins.to_vec()))
                .map_err(|e| anyhow::anyhow!("pitch tensor: {e}"))?.into()));
            inputs.push(("pitchf", Value::from_array(([1_usize, frames], f0_hz.to_vec()))
                .map_err(|e| anyhow::anyhow!("pitchf tensor: {e}"))?.into()));
        }

        let outputs = self.session.run(inputs)
            .map_err(|e| anyhow::anyhow!("RVC generator inference failed: {e}"))?;

        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract generated audio: {e}"))?;

        let audio: Vec<f32> = data.to_vec();
        log::debug!("RVC generator: produced {} samples", audio.len());
        Ok(audio)
    }

    pub fn uses_f0(&self) -> bool {
        self.use_f0
    }
}
