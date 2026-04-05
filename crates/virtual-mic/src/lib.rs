/// Lists virtual audio output devices available on this platform.
///
/// On Windows, looks for VB-Cable. On macOS, looks for BlackHole.
pub fn find_virtual_output() -> Option<String> {
    let devices = audio_engine::list_output_devices().ok()?;
    for device in &devices {
        let name_lower = device.name.to_lowercase();
        // Windows: VB-Cable
        if name_lower.contains("cable input") || name_lower.contains("vb-audio") {
            return Some(device.name.clone());
        }
        // macOS: BlackHole
        if name_lower.contains("blackhole") {
            return Some(device.name.clone());
        }
    }
    None
}
