use crate::Effect;

/// Simple pitch shifter using granular synthesis with overlapping windows.
/// Not phase-vocoder quality, but low latency and good enough for real-time voice effects.
pub struct PitchShift {
    /// Pitch multiplier: >1.0 = higher, <1.0 = lower
    shift: f32,
    #[allow(dead_code)]
    sample_rate: f32,
    grain_size: usize,
    overlap: usize,
    input_buf: Vec<f32>,
    output_buf: Vec<f32>,
    input_pos: usize,
    read_pos: f32,
    window: Vec<f32>,
}

impl PitchShift {
    pub fn new(shift: f32, sample_rate: f32) -> Self {
        let grain_size = (sample_rate * 0.02) as usize; // 20ms grains
        let overlap = grain_size / 2;
        let window = hann_window(grain_size);

        Self {
            shift,
            sample_rate,
            grain_size,
            overlap,
            input_buf: vec![0.0; grain_size * 4],
            output_buf: vec![0.0; grain_size * 4],
            input_pos: 0,
            read_pos: 0.0,
            window,
        }
    }

    pub fn set_shift(&mut self, shift: f32) {
        self.shift = shift.clamp(0.25, 4.0);
    }
}

impl Effect for PitchShift {
    fn process(&mut self, buffer: &mut [f32]) {
        let buf_len = self.input_buf.len();

        for sample in buffer.iter_mut() {
            // Write input sample
            self.input_buf[self.input_pos % buf_len] = *sample;
            self.input_pos += 1;

            // Read at shifted rate with linear interpolation
            let read_idx = self.read_pos;
            let idx0 = read_idx as usize % buf_len;
            let idx1 = (idx0 + 1) % buf_len;
            let frac = read_idx - read_idx.floor();

            let interpolated =
                self.input_buf[idx0] * (1.0 - frac) + self.input_buf[idx1] * frac;

            // Apply window to reduce artifacts at grain boundaries
            let window_pos = self.input_pos % self.grain_size;
            let windowed = interpolated * self.window[window_pos];

            // Overlap-add into output
            let out_idx = self.input_pos % buf_len;
            self.output_buf[out_idx] = windowed;

            // Also add overlapping grain
            let overlap_idx = (self.input_pos + self.overlap) % buf_len;
            self.output_buf[overlap_idx] += windowed * 0.5;

            *sample = self.output_buf[self.input_pos.wrapping_sub(self.overlap) % buf_len];

            self.read_pos += self.shift;
            if self.read_pos >= buf_len as f32 {
                self.read_pos -= buf_len as f32;
            }
        }
    }

    fn name(&self) -> &str {
        "Pitch Shift"
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        *self = Self::new(self.shift, sample_rate);
    }
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let phase = std::f32::consts::PI * 2.0 * i as f32 / size as f32;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}
