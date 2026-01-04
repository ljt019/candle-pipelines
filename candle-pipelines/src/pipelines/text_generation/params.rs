use candle_core::Tensor;
use candle_transformers::generation::{LogitsProcessor as CandleLogitsProcessor, Sampling};

use crate::error::{PipelineError, Result};
use crate::loaders::GenerationConfig;

pub use candle_transformers::utils::apply_repeat_penalty;

/// User overrides for generation parameters.
/// All fields are optional - only set fields will override model defaults.
#[derive(Debug, Clone, Default)]
pub struct GenerationOverrides {
    pub temperature: Option<f64>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<usize>,
    pub seed: Option<u64>,
    pub max_len: Option<usize>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub min_p: Option<f64>,
}

/// Resolved parameters controlling text generation sampling behavior.
/// All required fields are guaranteed to have values.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    /// Randomness of sampling. 0.0 = deterministic, higher = more random.
    pub temperature: f64,
    /// Penalty for repeating tokens. None = no penalty, >1.0 = less repetition.
    pub repeat_penalty: Option<f32>,
    /// Number of recent tokens to consider for repeat penalty.
    pub repeat_last_n: Option<usize>,
    /// Random seed for reproducible generation.
    pub seed: u64,
    /// Maximum tokens to generate per turn.
    pub max_len: usize,
    /// Nucleus sampling: only consider tokens with cumulative probability <= p.
    pub top_p: Option<f64>,
    /// Only consider the top k most likely tokens.
    pub top_k: Option<usize>,
    /// Filter tokens with probability < min_p * max_probability.
    pub min_p: Option<f64>,
}

impl GenerationParams {
    /// Resolve generation params from model config + user overrides.
    /// Returns error if required field is missing from both.
    pub fn resolve(config: &GenerationConfig, overrides: &GenerationOverrides) -> Result<Self> {
        let temperature = overrides
            .temperature
            .or(config.temperature)
            .ok_or_else(|| {
                PipelineError::Unexpected(
                    "Missing 'temperature': set via .temperature() or ensure model's generation_config.json has it".into()
                )
            })?;

        let repeat_penalty = overrides.repeat_penalty.or(config.repeat_penalty);
        let repeat_last_n = overrides.repeat_last_n.or(config.repeat_last_n);

        // These have sensible universal defaults - seed is random, max_len is context-dependent
        let seed = overrides.seed.unwrap_or_else(rand::random);
        let max_len = overrides.max_len.unwrap_or(2048);

        // Optional params - None is valid (means "don't apply this filter")
        let top_p = overrides.top_p.or(config.top_p);
        let top_k = overrides
            .top_k
            .or(config.top_k.map(|k| k as usize));
        let min_p = overrides.min_p.or(config.min_p);

        Ok(Self {
            temperature,
            repeat_penalty,
            repeat_last_n,
            seed,
            max_len,
            top_p,
            top_k,
            min_p,
        })
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
