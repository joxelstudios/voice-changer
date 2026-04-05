use std::io::{self, Write};

use audio_engine::{AudioEngine, EngineConfig, list_input_devices, list_output_devices};
use dsp::{PitchShift, RobotVoice, Echo, Reverb};

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("=== Voice Changer — Audio Test ===\n");

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

    let sample_rate = 48000;
    let config = EngineConfig {
        input_device: input_name.clone(),
        output_device: output_name.clone(),
        sample_rate,
        buffer_size: 512,
    };

    println!("\nStarting: {input_name} -> {output_name}");
    let state = AudioEngine::start(config)?;

    println!("Audio flowing!\n");
    println!("Commands:");
    println!("  p <semitones>  - Pitch shift (e.g. 'p 5' for 5 semitones up, 'p -7' for down)");
    println!("  r              - Toggle robot voice");
    println!("  e              - Toggle echo");
    println!("  v              - Toggle reverb");
    println!("  c              - Clear all effects");
    println!("  b              - Toggle bypass (mute)");
    println!("  l              - List active effects");
    println!("  q              - Quit\n");

    let sr = sample_rate as f32;

    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut buf = String::new();
        io::stdin().read_line(&mut buf)?;
        let cmd = buf.trim();

        if cmd.is_empty() {
            continue;
        }

        let mut chain = state.effect_chain().lock().unwrap();

        match cmd.split_whitespace().collect::<Vec<_>>().as_slice() {
            ["p", semitones] => {
                let semi: f32 = semitones.parse().unwrap_or(0.0);
                let shift = 2.0_f32.powf(semi / 12.0);
                chain.add(Box::new(PitchShift::new(shift, sr)));
                println!("Added pitch shift: {semi:+} semitones (ratio {shift:.3})");
            }
            ["r"] => {
                chain.add(Box::new(RobotVoice::new(150.0, sr)));
                println!("Added robot voice (150Hz carrier)");
            }
            ["e"] => {
                chain.add(Box::new(Echo::new(200.0, 0.4, 0.3, sr)));
                println!("Added echo (200ms delay, 0.4 feedback)");
            }
            ["v"] => {
                chain.add(Box::new(Reverb::new(0.7, 0.3, sr)));
                println!("Added reverb (0.7 decay, 0.3 mix)");
            }
            ["c"] => {
                chain.clear();
                println!("Cleared all effects");
            }
            ["b"] => {
                drop(chain);
                let new_bypass = !state.is_bypassed();
                state.set_bypass(new_bypass);
                println!(
                    "Bypass: {}",
                    if new_bypass { "ON (muted)" } else { "OFF" }
                );
            }
            ["l"] => {
                let names = chain.effect_names();
                if names.is_empty() {
                    println!("No effects active");
                } else {
                    println!("Active effects:");
                    for (i, (name, enabled)) in names.iter().enumerate() {
                        let status = if *enabled { "ON" } else { "OFF" };
                        println!("  [{i}] {name} [{status}]");
                    }
                }
            }
            ["q"] => {
                println!("Shutting down...");
                break;
            }
            _ => {
                println!("Unknown command: {cmd}");
            }
        }
    }

    Ok(())
}

fn read_index() -> anyhow::Result<usize> {
    let mut buf = String::new();
    io::stdin().read_line(&mut buf)?;
    let idx: usize = buf.trim().parse()?;
    Ok(idx)
}
