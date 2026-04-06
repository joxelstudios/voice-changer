use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::StreamConfig;
use dsp::EffectChain;
use ringbuf::traits::{Consumer, Producer};

use crate::device::find_device_by_name;
use crate::ring_buffer::AudioRingBuffer;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub input_device: String,
    pub output_device: String,
    pub sample_rate: u32,
    pub buffer_size: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            input_device: String::new(),
            output_device: String::new(),
            sample_rate: 48000,
            buffer_size: 512,
        }
    }
}

pub struct EngineState {
    bypass: Arc<AtomicBool>,
    effect_chain: Arc<Mutex<EffectChain>>,
    _input_stream: cpal::Stream,
    _output_stream: cpal::Stream,
}

// cpal::Stream contains platform-specific types that aren't Send/Sync on macOS.
// The streams are only held alive; shared state uses proper synchronization.
unsafe impl Send for EngineState {}
unsafe impl Sync for EngineState {}

pub struct AudioEngine;

impl AudioEngine {
    /// Start the audio pipeline with DSP effects.
    pub fn start(config: EngineConfig) -> Result<EngineState> {
        let input_dev = find_device_by_name(&config.input_device, true)
            .context("Input device lookup failed")?;
        let output_dev = find_device_by_name(&config.output_device, false)
            .context("Output device lookup failed")?;

        // Use each device's default config for maximum compatibility.
        // VB-Cable and other virtual devices often reject non-default configs.
        let input_config: StreamConfig = input_dev
            .default_input_config()
            .context("Failed to get default input config")?
            .into();
        let output_config: StreamConfig = output_dev
            .default_output_config()
            .context("Failed to get default output config")?
            .into();

        log::info!(
            "Input config: {}ch {}Hz, Output config: {}ch {}Hz",
            input_config.channels, input_config.sample_rate.0,
            output_config.channels, output_config.sample_rate.0,
        );

        let capacity = (input_config.sample_rate.0 as usize) / 5;
        let rb = AudioRingBuffer::new(capacity);
        let (mut producer, mut consumer) = rb.split();

        let bypass = Arc::new(AtomicBool::new(false));
        let effect_chain = Arc::new(Mutex::new(EffectChain::new()));

        let in_channels = input_config.channels as usize;
        let input_stream = input_dev
            .build_input_stream(
                &input_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if in_channels == 1 {
                        let written = producer.push_slice(data);
                        if written < data.len() {
                            log::warn!("Ring buffer overflow: dropped {} samples", data.len() - written);
                        }
                    } else {
                        // Downmix multi-channel to mono
                        for chunk in data.chunks(in_channels) {
                            let mono = chunk.iter().sum::<f32>() / in_channels as f32;
                            let _ = producer.push_slice(&[mono]);
                        }
                    }
                },
                |err| log::error!("Input stream error: {err}"),
                None,
            )
            .context("Failed to build input stream")?;

        let bypass_clone = bypass.clone();
        let chain_clone = effect_chain.clone();
        let out_channels = output_config.channels as usize;
        let output_stream = output_dev
            .build_output_stream(
                &output_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    if bypass_clone.load(Ordering::Relaxed) {
                        for sample in data.iter_mut() {
                            *sample = 0.0;
                        }
                        return;
                    }

                    let mono_frames = data.len() / out_channels;
                    let mut mono_buf = vec![0.0_f32; mono_frames];

                    // Pull mono from ring buffer
                    let read = consumer.pop_slice(&mut mono_buf);
                    for sample in &mut mono_buf[read..] {
                        *sample = 0.0;
                    }

                    // Apply effect chain (try_lock to never block the audio thread)
                    if let Ok(mut chain) = chain_clone.try_lock() {
                        chain.process(&mut mono_buf[..read]);
                    }

                    // Write mono to all output channels
                    if out_channels == 1 {
                        data[..mono_frames].copy_from_slice(&mono_buf);
                    } else {
                        for (i, &sample) in mono_buf.iter().enumerate() {
                            for ch in 0..out_channels {
                                data[i * out_channels + ch] = sample;
                            }
                        }
                    }
                },
                |err| log::error!("Output stream error: {err}"),
                None,
            )
            .context("Failed to build output stream")?;

        input_stream.play().context("Failed to start input stream")?;
        output_stream.play().context("Failed to start output stream")?;

        log::info!(
            "Audio engine started: {} ({}ch {}Hz) -> {} ({}ch {}Hz)",
            config.input_device, input_config.channels, input_config.sample_rate.0,
            config.output_device, output_config.channels, output_config.sample_rate.0,
        );

        Ok(EngineState {
            bypass,
            effect_chain,
            _input_stream: input_stream,
            _output_stream: output_stream,
        })
    }
}

impl EngineState {
    pub fn set_bypass(&self, enabled: bool) {
        self.bypass.store(enabled, Ordering::Relaxed);
        log::info!("Bypass {}", if enabled { "enabled" } else { "disabled" });
    }

    pub fn is_bypassed(&self) -> bool {
        self.bypass.load(Ordering::Relaxed)
    }

    pub fn effect_chain(&self) -> &Arc<Mutex<EffectChain>> {
        &self.effect_chain
    }
}
