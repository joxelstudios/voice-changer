use ringbuf::{HeapRb, traits::Split};

pub struct AudioRingBuffer {
    producer: ringbuf::HeapProd<f32>,
    consumer: ringbuf::HeapCons<f32>,
}

impl AudioRingBuffer {
    pub fn new(capacity: usize) -> Self {
        let rb = HeapRb::<f32>::new(capacity);
        let (producer, consumer) = rb.split();
        Self { producer, consumer }
    }

    pub fn split(self) -> (ringbuf::HeapProd<f32>, ringbuf::HeapCons<f32>) {
        (self.producer, self.consumer)
    }
}
