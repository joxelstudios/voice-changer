use anyhow::Result;
use rubato::{FftFixedIn, Resampler};

/// Resample audio from one sample rate to another.
pub fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(input.to_vec());
    }

    let chunk_size = 1024;
    let mut resampler = FftFixedIn::<f32>::new(
        from_rate as usize,
        to_rate as usize,
        chunk_size,
        1, // sub_chunks
        1, // channels
    )?;

    let mut output = Vec::new();
    let mut pos = 0;

    while pos < input.len() {
        let end = (pos + chunk_size).min(input.len());
        let mut chunk = input[pos..end].to_vec();

        // Pad last chunk if needed
        if chunk.len() < chunk_size {
            chunk.resize(chunk_size, 0.0);
        }

        let result = resampler.process(&[&chunk], None)?;
        output.extend_from_slice(&result[0]);
        pos += chunk_size;
    }

    // Trim to expected length
    let expected_len = (input.len() as f64 * to_rate as f64 / from_rate as f64) as usize;
    output.truncate(expected_len);
    Ok(output)
}
