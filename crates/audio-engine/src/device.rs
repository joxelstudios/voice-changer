use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};

#[derive(Debug, Clone)]
pub struct AudioDevice {
    pub name: String,
    pub is_input: bool,
}

pub fn list_input_devices() -> Result<Vec<AudioDevice>> {
    let host = cpal::default_host();
    let devices = host
        .input_devices()
        .context("Failed to enumerate input devices")?;

    let mut result = Vec::new();
    for device in devices {
        if let Ok(name) = device.name() {
            result.push(AudioDevice {
                name,
                is_input: true,
            });
        }
    }
    Ok(result)
}

pub fn list_output_devices() -> Result<Vec<AudioDevice>> {
    let host = cpal::default_host();
    let devices = host
        .output_devices()
        .context("Failed to enumerate output devices")?;

    let mut result = Vec::new();
    for device in devices {
        if let Ok(name) = device.name() {
            result.push(AudioDevice {
                name,
                is_input: false,
            });
        }
    }
    Ok(result)
}

pub fn find_device_by_name(name: &str, input: bool) -> Result<cpal::Device> {
    let host = cpal::default_host();
    let devices = if input {
        host.input_devices().context("Failed to enumerate input devices")?
    } else {
        host.output_devices().context("Failed to enumerate output devices")?
    };

    for device in devices {
        if let Ok(dev_name) = device.name() {
            if dev_name == name {
                return Ok(device);
            }
        }
    }

    anyhow::bail!("Device not found: {name}")
}
