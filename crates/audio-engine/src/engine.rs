use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::StreamConfig;
use dsp::EffectChain;
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use ringbuf::HeapRb;
use voice_conv::VoiceConverter;

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
    ai_active: Arc<AtomicBool>,
    effect_chain: Arc<Mutex<EffectChain>>,
    voice_converter: Arc<Mutex<Option<VoiceConverter>>>,
    actual_sample_rate: Arc<AtomicU32>,
    _input_stream: cpal::Stream,
    _output_stream: cpal::Stream,
    _ai_thread: Option<std::thread::JoinHandle<()>>,
    ai_thread_stop: Arc<AtomicBool>,
}

unsafe impl Send for EngineState {}
unsafe impl Sync for EngineState {}

pub struct AudioEngine;

impl AudioEngine {
    pub fn start(config: EngineConfig) -> Result<EngineState> {
        let input_dev = find_device_by_name(&config.input_device, true)
            .context("Input device lookup failed")?;
        let output_dev = find_device_by_name(&config.output_device, false)
            .context("Output device lookup failed")?;

        let input_config: StreamConfig = input_dev
            .default_input_config()
            .context("Failed to get default input config")?
            .into();
        let output_config: StreamConfig = output_dev
            .default_output_config()
            .context("Failed to get default output config")?
            .into();

        let sample_rate = input_config.sample_rate.0;

        log::info!(
            "Input config: {}ch {}Hz, Output config: {}ch {}Hz",
            input_config.channels, sample_rate,
            output_config.channels, output_config.sample_rate.0,
        );

        let capacity = (sample_rate as usize) / 2; // 500ms
        let rb_a = AudioRingBuffer::new(capacity);
        let (mut producer_a, mut consumer_a_dsp) = rb_a.split();

        let rb_ai_in = HeapRb::<f32>::new(capacity);
        let (mut producer_ai_in, mut consumer_ai_in) = rb_ai_in.split();

        let rb_ai_out = HeapRb::<f32>::new(capacity * 4); // large buffer for AI latency
        let (mut producer_ai_out, mut consumer_ai_out) = rb_ai_out.split();

        let bypass = Arc::new(AtomicBool::new(false));
        let ai_active = Arc::new(AtomicBool::new(false));
        let effect_chain = Arc::new(Mutex::new(EffectChain::new()));
        let voice_converter: Arc<Mutex<Option<VoiceConverter>>> = Arc::new(Mutex::new(None));
        let actual_sample_rate = Arc::new(AtomicU32::new(sample_rate));

        // --- Input stream ---
        let in_channels = input_config.channels as usize;
        let ai_active_input = ai_active.clone();
        let input_stream = input_dev
            .build_input_stream(
                &input_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let is_ai = ai_active_input.load(Ordering::Relaxed);
                    if in_channels == 1 {
                        if is_ai {
                            producer_ai_in.push_slice(data);
                        } else {
                            producer_a.push_slice(data);
                        }
                    } else {
                        for chunk in data.chunks(in_channels) {
                            let mono = chunk.iter().sum::<f32>() / in_channels as f32;
                            if is_ai {
                                let _ = producer_ai_in.push_slice(&[mono]);
                            } else {
                                let _ = producer_a.push_slice(&[mono]);
                            }
                        }
                    }
                },
                |err| log::error!("Input stream error: {err}"),
                None,
            )
            .context("Failed to build input stream")?;

        // --- AI processing thread ---
        // FIX BUG #1: Acquire lock BEFORE consuming input. If lock fails, don't consume.
        let ai_thread_stop = Arc::new(AtomicBool::new(false));
        let ai_stop_clone = ai_thread_stop.clone();
        let ai_active_thread = ai_active.clone();
        let vc_clone = voice_converter.clone();
        let chunk_size = (sample_rate as usize) / 10; // 100ms chunks

        let ai_thread = std::thread::Builder::new()
            .name("voice-ai".to_string())
            .spawn(move || {
                let mut input_buf = vec![0.0_f32; chunk_size];

                loop {
                    if ai_stop_clone.load(Ordering::Relaxed) {
                        break;
                    }

                    if !ai_active_thread.load(Ordering::Relaxed) {
                        std::thread::sleep(std::time::Duration::from_millis(10));
                        continue;
                    }

                    // Check enough data available
                    let available = consumer_ai_in.occupied_len();
                    if available < chunk_size {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                        continue;
                    }

                    // FIX: Acquire lock BEFORE consuming input samples
                    let mut guard = match vc_clone.lock() {
                        Ok(g) => g,
                        Err(_) => {
                            std::thread::sleep(std::time::Duration::from_millis(1));
                            continue;
                        }
                    };

                    // NOW consume input (lock is held, samples are safe)
                    let read = consumer_ai_in.pop_slice(&mut input_buf);
                    if read == 0 {
                        drop(guard);
                        continue;
                    }

                    if let Some(converter) = guard.as_mut() {
                        match converter.process_chunk(&input_buf[..read]) {
                            Ok(output) => {
                                let written = producer_ai_out.push_slice(&output);
                                if written < output.len() {
                                    log::warn!("AI output buffer overflow: wrote {written}/{}", output.len());
                                }
                            }
                            Err(e) => {
                                log::error!("Voice conversion error: {e}");
                                // Pass through on failure so user hears something
                                producer_ai_out.push_slice(&input_buf[..read]);
                            }
                        }
                    } else {
                        // Converter is None but AI active — pass through
                        producer_ai_out.push_slice(&input_buf[..read]);
                    }

                    drop(guard);
                }
                log::info!("AI processing thread stopped");
            })
            .context("Failed to spawn AI thread")?;

        // --- Output stream ---
        let bypass_clone = bypass.clone();
        let chain_clone = effect_chain.clone();
        let ai_active_output = ai_active.clone();
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

                    if ai_active_output.load(Ordering::Relaxed) {
                        let read = consumer_ai_out.pop_slice(&mut mono_buf);
                        for sample in &mut mono_buf[read..] {
                            *sample = 0.0;
                        }
                    } else {
                        let read = consumer_a_dsp.pop_slice(&mut mono_buf);
                        for sample in &mut mono_buf[read..] {
                            *sample = 0.0;
                        }
                        if let Ok(mut chain) = chain_clone.try_lock() {
                            chain.process(&mut mono_buf[..read]);
                        }
                    }

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
            config.input_device, input_config.channels, sample_rate,
            config.output_device, output_config.channels, output_config.sample_rate.0,
        );

        Ok(EngineState {
            bypass,
            ai_active,
            effect_chain,
            voice_converter,
            actual_sample_rate,
            _input_stream: input_stream,
            _output_stream: output_stream,
            _ai_thread: Some(ai_thread),
            ai_thread_stop,
        })
    }
}

impl EngineState {
    pub fn set_bypass(&self, enabled: bool) {
        self.bypass.store(enabled, Ordering::Relaxed);
    }

    pub fn is_bypassed(&self) -> bool {
        self.bypass.load(Ordering::Relaxed)
    }

    pub fn effect_chain(&self) -> &Arc<Mutex<EffectChain>> {
        &self.effect_chain
    }

    pub fn set_voice_converter(&self, converter: Option<VoiceConverter>) {
        if let Ok(mut guard) = self.voice_converter.lock() {
            let is_some = converter.is_some();
            *guard = converter;
            self.ai_active.store(is_some, Ordering::Relaxed);
            log::info!("AI mode: {}", if is_some { "enabled" } else { "disabled" });
        }
    }

    pub fn is_ai_active(&self) -> bool {
        self.ai_active.load(Ordering::Relaxed)
    }

    pub fn voice_converter(&self) -> &Arc<Mutex<Option<VoiceConverter>>> {
        &self.voice_converter
    }

    /// The actual sample rate negotiated with the audio device.
    pub fn actual_sample_rate(&self) -> u32 {
        self.actual_sample_rate.load(Ordering::Relaxed)
    }
}

impl Drop for EngineState {
    fn drop(&mut self) {
        self.ai_thread_stop.store(true, Ordering::Relaxed);
    }
}
