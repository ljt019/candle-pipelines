//! Logit Processing and Sampling
//!
//! Thin wrapper around Candle's LogitsProcessor to add min-p filtering.

use super::params::GenerationParams;
use candle_core::Tensor;
use candle_transformers::generation::{LogitsProcessor as CandleLogitsProcessor, Sampling};

pub struct LogitsProcessor {
    inner: CandleLogitsProcessor,
    min_p: Option<f32>,
    seed: u64,
    sampling: Sampling,
}

impl Clone for LogitsProcessor {
    fn clone(&self) -> Self {
        Self {
            inner: CandleLogitsProcessor::from_sampling(self.seed, self.sampling.clone()),
            min_p: self.min_p,
            seed: self.seed,
            sampling: self.sampling.clone(),
        }
    }
}

impl LogitsProcessor {
    pub fn new(seed: u64, sampling: Sampling, min_p: Option<f64>) -> Self {
        Self {
            inner: CandleLogitsProcessor::from_sampling(seed, sampling.clone()),
            min_p: min_p.map(|p| p as f32),
            seed,
            sampling,
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> candle_core::Result<u32> {
        let min_p = self.min_p;
        self.inner.sample_f(logits, |prs| {
            if let Some(min_p) = min_p {
                apply_min_p(prs, min_p);
            }
        })
    }
}

pub fn initialize_logits_processor(params: &GenerationParams, seed: u64) -> LogitsProcessor {
    LogitsProcessor::new(seed, params.sampling_strategy(), params.min_p)
}

fn apply_min_p(prs: &mut [f32], min_p: f32) {
    if min_p <= 0.0 || min_p >= 1.0 {
        return;
    }
    let max_prob = prs.iter().copied().fold(0.0f32, f32::max);
    let threshold = min_p * max_prob;
    for p in prs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_p_filters_low_probs() {
        let mut prs = vec![0.5, 0.3, 0.15, 0.05];
        apply_min_p(&mut prs, 0.5); // threshold = 0.25
        assert_eq!(prs, vec![0.5, 0.3, 0.0, 0.0]);
    }

    #[test]
    fn min_p_noop_when_zero() {
        let mut prs = vec![0.5, 0.3, 0.2];
        apply_min_p(&mut prs, 0.0);
        assert_eq!(prs, vec![0.5, 0.3, 0.2]);
    }
}
