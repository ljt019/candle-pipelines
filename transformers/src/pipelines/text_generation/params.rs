use candle_core::Tensor;
use candle_transformers::generation::{LogitsProcessor as CandleLogitsProcessor, Sampling};

pub use candle_transformers::utils::apply_repeat_penalty;

#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub temperature: f64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
    pub max_len: usize,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub min_p: Option<f64>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: rand::random(),
            max_len: 2048,
            top_p: Some(0.9),
            top_k: Some(40),
            min_p: None,
        }
    }
}

impl GenerationParams {
    pub(crate) fn sampling_strategy(&self) -> Sampling {
        if self.temperature <= 0.0 {
            return Sampling::ArgMax;
        }

        let temperature = self.temperature.max(1e-7);
        let top_k = self.top_k.unwrap_or(0);
        let top_p = self.top_p.unwrap_or(1.0);

        match (top_k > 0, top_p < 1.0) {
            (true, true) => Sampling::TopKThenTopP {
                k: top_k,
                p: top_p,
                temperature,
            },
            (true, false) => Sampling::TopK {
                k: top_k,
                temperature,
            },
            (false, true) => Sampling::TopP {
                p: top_p,
                temperature,
            },
            (false, false) => Sampling::All { temperature },
        }
    }
}

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
        apply_min_p(&mut prs, 0.5);
        assert_eq!(prs, vec![0.5, 0.3, 0.0, 0.0]);
    }

    #[test]
    fn min_p_noop_when_zero() {
        let mut prs = vec![0.5, 0.3, 0.2];
        apply_min_p(&mut prs, 0.0);
        assert_eq!(prs, vec![0.5, 0.3, 0.2]);
    }
}
