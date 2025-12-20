use candle_transformers::generation::Sampling;

/// Hugging Face style generation parameters. These mirror common generation
/// config keys so callers can pass through values directly from HF configs.
#[derive(Debug, Clone, Default)]
pub struct HfGenerationParams {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f32>,
    pub repeat_last_n: Option<usize>,
    pub seed: Option<u64>,
    pub max_new_tokens: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct SamplingParams {
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub min_p: Option<f64>,
}

/// Generation parameters for language models.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub temperature: f64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
    pub max_len: usize,
    pub sampling: SamplingParams,
}

impl GenerationParams {
    /// Create generation parameters from explicit values. This mirrors the
    /// previous constructor signature while internally routing through the
    /// new sampling configuration struct.
    pub fn new(
        temperature: f64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
        max_len: usize,
        top_p: f64,
        top_k: usize,
        min_p: f64,
    ) -> Self {
        Self {
            temperature,
            repeat_penalty,
            repeat_last_n,
            seed,
            max_len,
            sampling: SamplingParams {
                top_p: Some(top_p.clamp(0.0, 1.0)),
                top_k: Some(top_k).filter(|v| *v > 0),
                min_p: Some(min_p.clamp(0.0, 1.0)).filter(|v| *v > 0.0),
            },
        }
    }

    /// Merge Hugging Face style inputs with model defaults.
    pub fn from_hf_params(defaults: &GenerationParams, hf: HfGenerationParams) -> GenerationParams {
        let top_p = hf
            .top_p
            .or(defaults.sampling.top_p)
            .map(|p| p.clamp(0.0, 1.0));
        let min_p = hf
            .min_p
            .or(defaults.sampling.min_p)
            .map(|p| p.clamp(0.0, 1.0))
            .filter(|p| *p > 0.0);
        GenerationParams {
            temperature: hf.temperature.unwrap_or(defaults.temperature),
            repeat_penalty: hf.repetition_penalty.unwrap_or(defaults.repeat_penalty),
            repeat_last_n: hf.repeat_last_n.unwrap_or(defaults.repeat_last_n),
            seed: hf.seed.unwrap_or(defaults.seed),
            max_len: hf.max_new_tokens.unwrap_or(defaults.max_len),
            sampling: SamplingParams {
                top_p,
                top_k: hf.top_k.or(defaults.sampling.top_k),
                min_p,
            },
        }
    }

    pub fn sampling_strategy(&self) -> Sampling {
        if self.temperature <= 0.0 {
            return Sampling::ArgMax;
        }

        let temperature = self.temperature.max(1e-7);
        let top_k = self.sampling.top_k.unwrap_or(0);
        let top_p = self.sampling.top_p.unwrap_or(1.0);

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

    pub fn min_p(&self) -> Option<f64> {
        self.sampling.min_p
    }
}
