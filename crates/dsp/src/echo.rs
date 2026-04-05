use crate::Effect;

/// Simple delay-line echo effect.
pub struct Echo {
    delay_samples: usize,
    feedback: f32,
    mix: f32,
    buffer: Vec<f32>,
    write_pos: usize,
}

impl Echo {
    /// Create a new echo effect.
    /// - `delay_ms`: delay time in milliseconds
    /// - `feedback`: how much of the delayed signal feeds back (0.0 - 0.9)
    /// - `mix`: wet/dry mix (0.0 = dry, 1.0 = fully wet)
    pub fn new(delay_ms: f32, feedback: f32, mix: f32, sample_rate: f32) -> Self {
        let delay_samples = (delay_ms / 1000.0 * sample_rate) as usize;
        Self {
            delay_samples,
            feedback: feedback.clamp(0.0, 0.9),
            mix: mix.clamp(0.0, 1.0),
            buffer: vec![0.0; delay_samples.max(1)],
            write_pos: 0,
        }
    }
}

impl Effect for Echo {
    fn process(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            let read_pos = (self.write_pos + self.buffer.len() - self.delay_samples) % self.buffer.len();
            let delayed = self.buffer[read_pos];

            // Write input + feedback into delay buffer
            self.buffer[self.write_pos] = *sample + delayed * self.feedback;
            self.write_pos = (self.write_pos + 1) % self.buffer.len();

            // Mix dry and wet
            *sample = *sample * (1.0 - self.mix) + delayed * self.mix;
        }
    }

    fn name(&self) -> &str {
        "Echo"
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        let delay_ms = self.delay_samples as f32 / sample_rate * 1000.0;
        *self = Self::new(delay_ms, self.feedback, self.mix, sample_rate);
    }
}
