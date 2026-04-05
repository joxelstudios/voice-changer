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

pub struct AudioEngine;

impl AudioEngine {
    /// Start the audio pipeline with DSP effects.
    ///
    /// Captures from `input_device`, pipes through a ring buffer,
    /// applies the effect chain, and outputs to `output_device`.
    pub fn start(config: EngineConfig) -> Result<EngineState> {
        let input_dev = find_device_by_name(&config.input_device, true)
            .context("Input device lookup failed")?;
        let output_dev = find_device_by_name(&config.output_device, false)
            .context("Output device lookup failed")?;

        let stream_config = StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(config.buffer_size),
        };

        // Ring buffer: ~200ms worth of audio
        let capacity = (config.sample_rate as usize) / 5;
        let rb = AudioRingBuffer::new(capacity);
        let (mut producer, mut consumer) = rb.split();

        let bypass = Arc::new(AtomicBool::new(false));
        let effect_chain = Arc::new(Mutex::new(EffectChain::new()));

        let input_stream = input_dev
            .build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let written = producer.push_slice(data);
                    if written < data.len() {
                        log::warn!(
                            "Ring buffer overflow: dropped {} samples",
                            data.len() - written
                        );
                    }
                },
                |err| {
                    log::error!("Input stream error: {err}");
                },
                None,
            )
            .context("Failed to build input stream")?;

        let bypass_clone = bypass.clone();
        let chain_clone = effect_chain.clone();
        let output_stream = output_dev
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    if bypass_clone.load(Ordering::Relaxed) {
                        for sample in data.iter_mut() {
                            *sample = 0.0;
                        }
                        return;
                    }

                    // Pull from ring buffer
                    let read = consumer.pop_slice(data);
                    for sample in &mut data[read..] {
                        *sample = 0.0;
                    }

                    // Apply effect chain (try_lock to never block the audio thread)
                    if let Ok(mut chain) = chain_clone.try_lock() {
                        chain.process(&mut data[..read]);
                    }
                },
                |err| {
                    log::error!("Output stream error: {err}");
                },
                None,
            )
            .context("Failed to build output stream")?;

        input_stream.play().context("Failed to start input stream")?;
        output_stream.play().context("Failed to start output stream")?;

        log::info!(
            "Audio engine started: {} -> {} @ {}Hz, buffer {}",
            config.input_device,
            config.output_device,
            config.sample_rate,
            config.buffer_size,
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

    /// Access the effect chain for adding/removing/toggling effects.
    /// This locks the mutex — do NOT call from the audio thread.
    pub fn effect_chain(&self) -> &Arc<Mutex<EffectChain>> {
        &self.effect_chain
    }
}
