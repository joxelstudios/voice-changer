use crate::*;

const SAMPLE_RATE: f32 = 48000.0;

fn sine_wave(freq: f32, samples: usize, sample_rate: f32) -> Vec<f32> {
    (0..samples)
        .map(|i| (std::f32::consts::TAU * freq * i as f32 / sample_rate).sin())
        .collect()
}

#[test]
fn pitch_shift_preserves_energy() {
    let mut signal = sine_wave(440.0, 4800, SAMPLE_RATE);
    let energy_before: f32 = signal.iter().map(|s| s * s).sum();

    let mut effect = PitchShift::new(1.5, SAMPLE_RATE);
    effect.process(&mut signal);

    let energy_after: f32 = signal.iter().map(|s| s * s).sum();
    // Energy should be in the same ballpark (within 10x)
    assert!(energy_after > energy_before * 0.1, "Signal lost too much energy");
    assert!(energy_after < energy_before * 10.0, "Signal gained too much energy");
}

#[test]
fn robot_voice_modulates_signal() {
    let mut signal = sine_wave(440.0, 4800, SAMPLE_RATE);
    let original = signal.clone();

    let mut effect = RobotVoice::new(150.0, SAMPLE_RATE);
    effect.process(&mut signal);

    // Signal should be different from original
    let diff: f32 = signal
        .iter()
        .zip(original.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 0.0, "Robot voice should modify the signal");
}

#[test]
fn echo_adds_delayed_signal() {
    let mut signal = vec![0.0; 48000];
    // Impulse at sample 0
    signal[0] = 1.0;

    let mut effect = Echo::new(100.0, 0.5, 0.5, SAMPLE_RATE);
    effect.process(&mut signal);

    // Should see the original impulse and a delayed echo around sample 4800 (100ms)
    let echo_region = &signal[4700..4900];
    let max_echo = echo_region.iter().copied().fold(0.0_f32, f32::max);
    assert!(max_echo > 0.1, "Expected echo around 100ms delay, max was {max_echo}");
}

#[test]
fn reverb_produces_output() {
    let mut signal = sine_wave(440.0, 4800, SAMPLE_RATE);

    let mut effect = Reverb::new(0.7, 0.3, SAMPLE_RATE);
    effect.process(&mut signal);

    let has_nonzero = signal.iter().any(|s| s.abs() > 0.001);
    assert!(has_nonzero, "Reverb should produce output");
}

#[test]
fn effect_chain_applies_multiple() {
    let mut signal = sine_wave(440.0, 4800, SAMPLE_RATE);
    let original = signal.clone();

    let mut chain = EffectChain::new();
    chain.add(Box::new(RobotVoice::new(150.0, SAMPLE_RATE)));
    chain.add(Box::new(Echo::new(50.0, 0.3, 0.3, SAMPLE_RATE)));
    chain.process(&mut signal);

    let diff: f32 = signal
        .iter()
        .zip(original.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 0.0, "Chain should modify signal");
}

#[test]
fn effect_chain_disable_effect() {
    let mut signal1 = sine_wave(440.0, 4800, SAMPLE_RATE);
    let mut signal2 = signal1.clone();

    let mut chain = EffectChain::new();
    chain.add(Box::new(RobotVoice::new(150.0, SAMPLE_RATE)));
    chain.process(&mut signal1);

    // Disable and process — should be passthrough
    let mut chain2 = EffectChain::new();
    let idx2 = chain2.add(Box::new(RobotVoice::new(150.0, SAMPLE_RATE)));
    chain2.set_enabled(idx2, false);
    let original = signal2.clone();
    chain2.process(&mut signal2);

    assert_eq!(signal2, original, "Disabled effect should not modify signal");
}
