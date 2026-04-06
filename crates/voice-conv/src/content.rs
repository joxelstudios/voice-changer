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

/// Detect output shape layout. Exported for testing.
pub fn detect_shape(shape: &[i64]) -> Option<(usize, usize, bool)> {
    if shape.len() != 3 {
        return None;
    }
    let d1 = shape[1] as usize;
    let d2 = shape[2] as usize;
    if d2 == 768 || (d2 > d1 && d1 < 768) {
        Some((d1, d2, true)) // [1, frames, 768]
    } else {
        Some((d2, d1, false)) // [1, 768, frames]
    }
}

/// Repeat features 2x along the frame axis. Exported for testing.
pub fn repeat_features(data: &[f32], frames: usize, dim: usize, is_features_last: bool) -> Vec<f32> {
    let doubled_frames = frames * 2;
    let mut features = Vec::with_capacity(doubled_frames * dim);

    for f in 0..frames {
        for _repeat in 0..2 {
            for d in 0..dim {
                let val = if is_features_last {
                    data[f * dim + d]
                } else {
                    data[d * frames + f]
                };
                features.push(val);
            }
        }
    }
    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_detection_features_last() {
        // [1, 10, 768] → frames=10, dim=768, features_last=true
        let (frames, dim, is_last) = detect_shape(&[1, 10, 768]).unwrap();
        assert_eq!(frames, 10);
        assert_eq!(dim, 768);
        assert!(is_last);
    }

    #[test]
    fn test_shape_detection_features_first() {
        // [1, 768, 10] → frames=10, dim=768, features_last=false
        let (frames, dim, is_last) = detect_shape(&[1, 768, 10]).unwrap();
        assert_eq!(frames, 10);
        assert_eq!(dim, 768);
        assert!(!is_last);
    }

    #[test]
    fn test_shape_detection_invalid_rank() {
        assert!(detect_shape(&[1, 768]).is_none());
        assert!(detect_shape(&[1, 2, 3, 4]).is_none());
    }

    #[test]
    fn test_repeat_features_doubles_frames() {
        // 3 frames, dim=2, features_last layout
        let data = vec![1.0, 2.0,  3.0, 4.0,  5.0, 6.0]; // [3, 2]
        let result = repeat_features(&data, 3, 2, true);

        assert_eq!(result.len(), 6 * 2); // 3*2 frames, 2 dim
        // Frame 0 repeated: [1,2], [1,2]
        assert_eq!(&result[0..2], &[1.0, 2.0]);
        assert_eq!(&result[2..4], &[1.0, 2.0]);
        // Frame 1 repeated: [3,4], [3,4]
        assert_eq!(&result[4..6], &[3.0, 4.0]);
        assert_eq!(&result[6..8], &[3.0, 4.0]);
        // Frame 2 repeated: [5,6], [5,6]
        assert_eq!(&result[8..10], &[5.0, 6.0]);
        assert_eq!(&result[10..12], &[5.0, 6.0]);
    }

    #[test]
    fn test_repeat_features_first_layout() {
        // 3 frames, dim=2, features_first layout: data is [1,2] = [dim, frames]
        // dim0: [1, 2, 3], dim1: [4, 5, 6]
        let data = vec![1.0, 2.0, 3.0,  4.0, 5.0, 6.0];
        let result = repeat_features(&data, 3, 2, false);

        assert_eq!(result.len(), 12);
        // Frame 0: dim0=1, dim1=4 → [1,4], [1,4]
        assert_eq!(&result[0..2], &[1.0, 4.0]);
        assert_eq!(&result[2..4], &[1.0, 4.0]);
        // Frame 1: dim0=2, dim1=5 → [2,5], [2,5]
        assert_eq!(&result[4..6], &[2.0, 5.0]);
        assert_eq!(&result[6..8], &[2.0, 5.0]);
    }
}
