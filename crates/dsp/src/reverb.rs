use crate::Effect;

/// Simple Schroeder reverb using parallel comb filters + series allpass filters.
pub struct Reverb {
    comb_filters: [CombFilter; 4],
    allpass_filters: [AllpassFilter; 2],
    mix: f32,
}

impl Reverb {
    pub fn new(decay: f32, mix: f32, sample_rate: f32) -> Self {
        let decay = decay.clamp(0.1, 0.99);
        let mix = mix.clamp(0.0, 1.0);

        // Comb filter delay times in ms (mutually prime-ish for diffusion)
        let comb_delays = [29.7, 37.1, 41.1, 43.7];
        let comb_filters = comb_delays.map(|ms| {
            let samples = (ms / 1000.0 * sample_rate) as usize;
            CombFilter::new(samples, decay)
        });

        // Allpass filter delay times
        let allpass_delays = [5.0, 1.7];
        let allpass_filters = allpass_delays.map(|ms| {
            let samples = (ms / 1000.0 * sample_rate) as usize;
            AllpassFilter::new(samples, 0.7)
        });

        Self {
            comb_filters,
            allpass_filters,
            mix,
        }
    }
}

impl Effect for Reverb {
    fn process(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            let dry = *sample;

            // Parallel comb filters summed
            let mut comb_sum = 0.0;
            for comb in &mut self.comb_filters {
                comb_sum += comb.process(dry);
            }
            comb_sum *= 0.25; // Normalize

            // Series allpass filters
            let mut wet = comb_sum;
            for allpass in &mut self.allpass_filters {
                wet = allpass.process(wet);
            }

            *sample = dry * (1.0 - self.mix) + wet * self.mix;
        }
    }

    fn name(&self) -> &str {
        "Reverb"
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        *self = Self::new(self.comb_filters[0].feedback, self.mix, sample_rate);
    }
}

struct CombFilter {
    buffer: Vec<f32>,
    pos: usize,
    feedback: f32,
}

impl CombFilter {
    fn new(delay_samples: usize, feedback: f32) -> Self {
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            pos: 0,
            feedback,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let delayed = self.buffer[self.pos];
        self.buffer[self.pos] = input + delayed * self.feedback;
        self.pos = (self.pos + 1) % self.buffer.len();
        delayed
    }
}

struct AllpassFilter {
    buffer: Vec<f32>,
    pos: usize,
    gain: f32,
}

impl AllpassFilter {
    fn new(delay_samples: usize, gain: f32) -> Self {
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            pos: 0,
            gain,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        let delayed = self.buffer[self.pos];
        let output = -input * self.gain + delayed;
        self.buffer[self.pos] = input + delayed * self.gain;
        self.pos = (self.pos + 1) % self.buffer.len();
        output
    }
}
