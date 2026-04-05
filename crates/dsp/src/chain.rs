use crate::Effect;

/// A chain of effects applied sequentially.
/// Effects can be added, removed, and toggled at runtime.
pub struct EffectChain {
    effects: Vec<EffectSlot>,
}

struct EffectSlot {
    effect: Box<dyn Effect>,
    enabled: bool,
}

impl EffectChain {
    pub fn new() -> Self {
        Self {
            effects: Vec::new(),
        }
    }

    pub fn add(&mut self, effect: Box<dyn Effect>) -> usize {
        let idx = self.effects.len();
        self.effects.push(EffectSlot {
            effect,
            enabled: true,
        });
        idx
    }

    pub fn set_enabled(&mut self, index: usize, enabled: bool) {
        if let Some(slot) = self.effects.get_mut(index) {
            slot.enabled = enabled;
        }
    }

    pub fn remove(&mut self, index: usize) {
        if index < self.effects.len() {
            self.effects.remove(index);
        }
    }

    pub fn clear(&mut self) {
        self.effects.clear();
    }

    pub fn process(&mut self, buffer: &mut [f32]) {
        for slot in &mut self.effects {
            if slot.enabled {
                slot.effect.process(buffer);
            }
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        for slot in &mut self.effects {
            slot.effect.set_sample_rate(sample_rate);
        }
    }

    pub fn effect_names(&self) -> Vec<(&str, bool)> {
        self.effects
            .iter()
            .map(|slot| (slot.effect.name(), slot.enabled))
            .collect()
    }
}

impl Default for EffectChain {
    fn default() -> Self {
        Self::new()
    }
}
