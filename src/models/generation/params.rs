use candle_transformers::generation::Sampling;

/// Generation parameters for language models.
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
    pub fn sampling_strategy(&self) -> Sampling {
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
