use crate::Effect;

/// Robot voice effect using ring modulation.
/// Multiplies the signal by a sine wave carrier frequency,
/// producing a metallic, robotic timbre.
pub struct RobotVoice {
    carrier_freq: f32,
    sample_rate: f32,
    phase: f32,
}

impl RobotVoice {
    pub fn new(carrier_freq: f32, sample_rate: f32) -> Self {
        Self {
            carrier_freq,
            sample_rate,
            phase: 0.0,
        }
    }

    pub fn set_carrier_freq(&mut self, freq: f32) {
        self.carrier_freq = freq.clamp(50.0, 500.0);
    }
}

impl Effect for RobotVoice {
    fn process(&mut self, buffer: &mut [f32]) {
        let phase_inc = self.carrier_freq / self.sample_rate;
        for sample in buffer.iter_mut() {
            let carrier = (self.phase * std::f32::consts::TAU).sin();
            *sample *= carrier;
            self.phase += phase_inc;
            if self.phase >= 1.0 {
                self.phase -= 1.0;
            }
        }
    }

    fn name(&self) -> &str {
        "Robot Voice"
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }
}
