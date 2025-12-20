//! Logit Processing and Sampling
//!
//! Delegates sampling to Candle's generation utilities while keeping adapters
//! for behaviors we support that Candle does not yet natively expose (e.g.,
//! min-p filtering).

use super::params::GenerationParams;
use candle_core::D;
use candle_core::{DType, Error, Tensor};
use candle_transformers::generation::{LogitsProcessor as CandleLogitsProcessor, Sampling};
use rand::{distr::Distribution, SeedableRng};

#[derive(Clone, Debug)]
pub struct LogitsProcessor {
    inner: CandleLogitsProcessor,
    sampling: Sampling,
    rng: rand::rngs::StdRng,
    min_p: Option<f64>,
}

impl LogitsProcessor {
    pub fn new(seed: u64, sampling: Sampling, min_p: Option<f64>) -> Self {
        Self {
            inner: CandleLogitsProcessor::from_sampling(seed, sampling.clone()),
            sampling,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            min_p,
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> candle_core::Result<u32> {
        self.sample_f(logits, |_| {})
    }

    pub fn sample_f(
        &mut self,
        logits: &Tensor,
        f: impl FnOnce(&mut [f32]),
    ) -> candle_core::Result<u32> {
        if self.min_p.is_none() {
            return self.inner.sample_f(logits, f);
        }

        self.sample_with_min_p(logits, f)
    }

    fn sample_with_min_p(
        &mut self,
        logits: &Tensor,
        f: impl FnOnce(&mut [f32]),
    ) -> candle_core::Result<u32> {
        let min_p = self.min_p.unwrap_or(0.0) as f32;
        let logits = logits.to_dtype(DType::F32)?;

        let prs = |temperature: f64| -> candle_core::Result<Vec<f32>> {
            let logits = (&logits / temperature)?;
            let prs = candle_nn::ops::softmax_last_dim(&logits)?;
            let mut prs = prs.to_vec1()?;
            f(&mut prs);
            Ok(prs)
        };

        let next_token = match &self.sampling {
            Sampling::ArgMax => logits.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?,
            Sampling::GumbelSoftmax { temperature } => {
                let sampled = candle_nn::sampling::gumbel_softmax(
                    &logits,
                    *temperature,
                    candle_core::D::Minus1,
                )?;
                sampled.to_scalar::<u32>()?
            }
            Sampling::All { temperature } => {
                let mut prs = prs(*temperature)?;
                apply_min_p(&mut prs, min_p);
                sample_multinomial(&prs, &mut self.rng)?
            }
            Sampling::TopP { p, temperature } => {
                let mut prs = prs(*temperature)?;
                apply_top_p(&mut prs, *p as f32);
                apply_min_p(&mut prs, min_p);
                sample_multinomial(&prs, &mut self.rng)?
            }
            Sampling::TopK { k, temperature } => {
                let mut prs = prs(*temperature)?;
                apply_top_k(&mut prs, *k);
                apply_min_p(&mut prs, min_p);
                sample_multinomial(&prs, &mut self.rng)?
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let mut prs = prs(*temperature)?;
                apply_top_k(&mut prs, *k);
                apply_top_p(&mut prs, *p as f32);
                apply_min_p(&mut prs, min_p);
                sample_multinomial(&prs, &mut self.rng)?
            }
        };

        Ok(next_token)
    }
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

fn apply_top_k(prs: &mut [f32], top_k: usize) {
    if top_k >= prs.len() {
        return;
    }

    let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
    argsort_indices.sort_unstable_by(|&i, &j| prs[j].total_cmp(&prs[i]));

    for index in argsort_indices.into_iter().skip(top_k) {
        prs[index] = 0.0;
    }
}

fn apply_top_p(prs: &mut [f32], top_p: f32) {
    if top_p <= 0.0 || top_p >= 1.0 {
        return;
    }

    let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
    argsort_indices.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));

    let mut cumsum = 0.0;
    for index in argsort_indices {
        if cumsum >= top_p {
            prs[index] = 0.0;
        } else {
            cumsum += prs[index];
        }
    }
}

fn sample_multinomial(prs: &Vec<f32>, rng: &mut rand::rngs::StdRng) -> candle_core::Result<u32> {
    let distr = rand::distr::weighted::WeightedIndex::new(prs).map_err(Error::wrap)?;
    let next_token = distr.sample(rng) as u32;
    Ok(next_token)
}

/// Initializes a LogitsProcessor based on sampling parameters.
pub fn initialize_logits_processor(params: &GenerationParams, seed: u64) -> LogitsProcessor {
    let sampling = params.sampling_strategy();
    LogitsProcessor::new(seed, sampling, params.min_p())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    fn minp_with_min_p_le_0_behaves_like_full_sampling() -> candle_core::Result<()> {
        let dev = candle_core::Device::Cpu;
        let logits = Tensor::from_vec(vec![0.0f32, 0.0, 0.0], 3, &dev)?;
        let mut proc = LogitsProcessor::new(0, Sampling::All { temperature: 1.0 }, None);
        let tok = proc.sample(&logits)?;
        assert!(tok < 3);
        Ok(())
    }

    #[test]
    fn minp_runs_after_top_filters() {
        let mut prs = vec![0.4f32, 0.35, 0.25];
        apply_top_k(&mut prs, 2);
        apply_top_p(&mut prs, 0.6);
        apply_min_p(&mut prs, 0.8);

        assert_eq!(prs, vec![0.4, 0.35, 0.0]);
    }

    #[test]
    fn top_k_zeros_out_non_top_values() {
        let mut prs = vec![0.1f32, 0.5, 0.2, 0.05, 0.15];
        apply_top_k(&mut prs, 3);

        assert_eq!(prs, vec![0.0, 0.5, 0.2, 0.0, 0.15]);
    }
}
