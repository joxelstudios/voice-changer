mod pitch_shift;
mod robot;
mod echo;
mod reverb;
mod chain;
#[cfg(test)]
mod tests;

pub use pitch_shift::PitchShift;
pub use robot::RobotVoice;
pub use echo::Echo;
pub use reverb::Reverb;
pub use chain::EffectChain;

/// Trait for audio effects that process samples in-place.
pub trait Effect: Send {
    fn process(&mut self, buffer: &mut [f32]);
    fn name(&self) -> &str;
    fn set_sample_rate(&mut self, _sample_rate: f32) {}
}
