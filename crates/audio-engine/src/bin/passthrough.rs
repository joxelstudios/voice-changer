use std::io::{self, Write};

use audio_engine::{AudioEngine, EngineConfig, list_input_devices, list_output_devices};

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("=== Voice Changer — Audio Passthrough Test ===\n");

    // List input devices
    let inputs = list_input_devices()?;
    println!("Input devices:");
    for (i, dev) in inputs.iter().enumerate() {
        println!("  [{i}] {}", dev.name);
    }

    print!("\nSelect input device: ");
    io::stdout().flush()?;
    let input_idx = read_index()?;
    let input_name = inputs
        .get(input_idx)
        .map(|d| d.name.clone())
        .ok_or_else(|| anyhow::anyhow!("Invalid input device index"))?;

    // List output devices
    let outputs = list_output_devices()?;
    println!("\nOutput devices:");
    for (i, dev) in outputs.iter().enumerate() {
        println!("  [{i}] {}", dev.name);
    }

    print!("\nSelect output device (pick VB-Cable/BlackHole): ");
    io::stdout().flush()?;
    let output_idx = read_index()?;
    let output_name = outputs
        .get(output_idx)
        .map(|d| d.name.clone())
        .ok_or_else(|| anyhow::anyhow!("Invalid output device index"))?;

    let config = EngineConfig {
        input_device: input_name.clone(),
        output_device: output_name.clone(),
        sample_rate: 48000,
        buffer_size: 512,
    };

    println!("\nStarting passthrough: {input_name} -> {output_name}");
    let state = AudioEngine::start(config)?;

    println!("Audio flowing! Press Enter to toggle bypass, Ctrl+C to quit.\n");

    loop {
        let mut buf = String::new();
        io::stdin().read_line(&mut buf)?;

        let new_bypass = !state.is_bypassed();
        state.set_bypass(new_bypass);
        println!(
            "Bypass: {}",
            if new_bypass { "ON (muted)" } else { "OFF (passing audio)" }
        );
    }
}

fn read_index() -> anyhow::Result<usize> {
    let mut buf = String::new();
    io::stdin().read_line(&mut buf)?;
    let idx: usize = buf.trim().parse()?;
    Ok(idx)
}
