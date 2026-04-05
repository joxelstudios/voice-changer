use std::io::Write;
use std::path::Path;

/// Download a file from `url` to `dest` with progress callback.
/// Progress is reported as a float from 0.0 to 1.0.
pub fn download_with_progress(
    url: &str,
    dest: &Path,
    on_progress: impl Fn(f64) + Send + 'static,
) -> Result<(), String> {
    log::info!("Downloading {} -> {}", url, dest.display());

    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    let total_size = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);

    let mut reader = response.into_body().into_reader();
    let mut file = std::fs::File::create(dest)
        .map_err(|e| format!("Failed to create file {}: {e}", dest.display()))?;

    let mut downloaded: u64 = 0;
    let mut buf = vec![0u8; 64 * 1024]; // 64KB chunks
    let mut last_reported = 0.0_f64;

    loop {
        let n = std::io::Read::read(&mut reader, &mut buf)
            .map_err(|e| format!("Download read error: {e}"))?;
        if n == 0 {
            break;
        }

        file.write_all(&buf[..n])
            .map_err(|e| format!("File write error: {e}"))?;

        downloaded += n as u64;

        if total_size > 0 {
            let progress = downloaded as f64 / total_size as f64;
            // Report progress at most every 1%
            if progress - last_reported >= 0.01 {
                on_progress(progress);
                last_reported = progress;
            }
        }
    }

    on_progress(1.0);
    log::info!("Download complete: {} ({} bytes)", dest.display(), downloaded);
    Ok(())
}
