/// Trait for audio effects that process samples in-place.
pub trait Effect: Send {
    fn process(&mut self, buffer: &mut [f32]);
    fn name(&self) -> &str;
}
