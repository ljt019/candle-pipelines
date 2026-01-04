use super::params::{apply_repeat_penalty, initialize_logits_processor, GenerationParams};
use candle_core::Result as CandleResult;
use crate::error::{PipelineError, Result};
use crate::models::capabilities::{ModelConfig, TextGenerationModel};
use crate::pipelines::stats::GenerationStats;
use candle_core::Tensor;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

pub struct BasePipeline<C: ModelConfig> {
    pub model: Arc<C::Model>,
    pub model_tokenizer: Tokenizer,
    pub cache: Arc<Mutex<<C::Model as TextGenerationModel>::Cache>>,
    pub gen_params: Arc<Mutex<GenerationParams>>,
    pub device: candle_core::Device,
    pub last_processed_tokens: Arc<Mutex<Vec<u32>>>,
}

impl<C: ModelConfig> BasePipeline<C>
where
    C::Model: Sync,
{
    pub fn new(
        model: Arc<C::Model>,
        gen_params: GenerationParams,
        device: candle_core::Device,
    ) -> Result<Self> {
        let model_tokenizer = model.get_tokenizer()?;
        let cache = model.new_cache();

        Ok(Self {
            model,
            model_tokenizer,
            cache: Arc::new(Mutex::new(cache)),
            gen_params: Arc::new(Mutex::new(gen_params)),
            device,
            last_processed_tokens: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn set_generation_params(&self, params: GenerationParams) {
        *self.gen_params.lock().unwrap() = params;
    }

    pub fn can_reuse_cache(&self, new_tokens: &[u32]) -> bool {
        new_tokens.starts_with(&self.last_processed_tokens.lock().unwrap())
    }

    pub fn completion_from_tokens_with_stats(
        &self,
        input_tokens: &[u32],
    ) -> Result<(String, GenerationStats)> {
        self.completion_from_tokens_with_prompt_stats(input_tokens, input_tokens.len())
    }

    pub fn completion_from_tokens_with_prompt_stats(
        &self,
        input_tokens: &[u32],
        prompt_token_count: usize,
    ) -> Result<(String, GenerationStats)> {
        let params = self.gen_params.lock().unwrap().clone();

        let mut logits_processor = initialize_logits_processor(&params, params.seed);

        let mut generated_tokens: Vec<u32> = Vec::with_capacity(params.max_len);
        let mut stats = GenerationStats::new();
        stats.set_prompt_tokens(prompt_token_count);

        // Process entire prompt at once to avoid candle dtype bug in Gemma3's
        // mask function when index_pos > 0 and seq_len > 1 (happens with chunking)
        let input = Tensor::new(input_tokens, &self.device)?.unsqueeze(0)?;
        let logits = {
            let mut cache = self.cache.lock().unwrap();
            self.model.forward(&input, &mut cache)
        }?;
        let last_logits = logits.squeeze(0)?;

        let mut next_token = logits_processor.sample(&last_logits)?;
        generated_tokens.push(next_token);
        stats.record_token();

        let eos_tokens = self.model.get_eos_tokens();
        if eos_tokens.is_empty() {
            return Err(PipelineError::Unexpected(
                "No EOS tokens configured for model. Cannot determine when to stop.".to_string(),
            ));
        }
        for _ in 0..params.max_len {
            if eos_tokens.contains(&next_token) {
                break;
            }

            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = {
                let mut cache = self.cache.lock().unwrap();
                self.model.forward(&input, &mut cache)
            }?;
            let logits = logits.squeeze(0)?;

            let logits = apply_repeat_penalty_if_needed(
                logits,
                &generated_tokens,
                params.repeat_penalty,
                params.repeat_last_n,
            )?;

            next_token = logits_processor.sample(&logits)?;
            generated_tokens.push(next_token);
            stats.record_token();
        }

        let eos_tokens = self.model.get_eos_tokens();
        let filtered_tokens: Vec<u32> = generated_tokens
            .into_iter()
            .filter(|&token| !eos_tokens.contains(&token))
            .collect();

        let generated_tokens_str = self
            .model_tokenizer
            .decode(&filtered_tokens, /*skip_special_tokens=*/ true)
            .expect("token decode failed");

        stats.finalize();

        Ok((generated_tokens_str, stats))
    }

    /// Returns a sync iterator that yields tokens as they're generated.
    pub fn token_iterator_with_prompt_count(
        &self,
        input_tokens: Vec<u32>,
        prompt_token_count: Option<usize>,
    ) -> (Arc<Mutex<GenerationStats>>, TokenIterator<C>)
    where
        C::Model: Send + Sync,
    {
        let stats = Arc::new(Mutex::new(GenerationStats::new()));
        stats
            .lock()
            .unwrap()
            .set_prompt_tokens(prompt_token_count.unwrap_or(input_tokens.len()));

        let iterator = TokenIterator::new(
            input_tokens,
            self.device.clone(),
            Arc::clone(&self.model),
            self.model_tokenizer.clone(),
            Arc::clone(&self.cache),
            Arc::clone(&self.gen_params),
            Arc::clone(&stats),
        );

        (stats, iterator)
    }
}

/// Sync iterator that generates tokens one at a time.
pub struct TokenIterator<C: ModelConfig> {
    // State
    initialized: bool,
    finished: bool,
    generated: Vec<u32>,
    next_token: u32,
    tokenizer: Tokenizer,
    last_decoded_len: usize,
    logits_processor: super::params::LogitsProcessor,

    // Config
    input_tokens: Vec<u32>,
    params: GenerationParams,
    eos_tokens: Vec<u32>,

    // Shared resources
    device: candle_core::Device,
    model: Arc<C::Model>,
    cache: Arc<Mutex<<C::Model as TextGenerationModel>::Cache>>,
    stats: Arc<Mutex<GenerationStats>>,
}

impl<C: ModelConfig> TokenIterator<C>
where
    C::Model: Send + Sync,
{
    fn new(
        input_tokens: Vec<u32>,
        device: candle_core::Device,
        model: Arc<C::Model>,
        tokenizer: Tokenizer,
        cache: Arc<Mutex<<C::Model as TextGenerationModel>::Cache>>,
        gen_params: Arc<Mutex<GenerationParams>>,
        stats: Arc<Mutex<GenerationStats>>,
    ) -> Self {
        let params = gen_params.lock().unwrap().clone();
        let logits_processor = initialize_logits_processor(&params, params.seed);
        let eos_tokens = model.get_eos_tokens();

        Self {
            initialized: false,
            finished: false,
            generated: Vec::with_capacity(params.max_len),
            next_token: 0,
            tokenizer,
            last_decoded_len: 0,
            logits_processor,
            input_tokens,
            params,
            eos_tokens,
            device,
            model,
            cache,
            stats,
        }
    }

    /// Decode accumulated tokens and return only the new characters since last decode.
    fn decode_incremental(&mut self) -> Result<Option<String>> {
        let decoded = self
            .tokenizer
            .decode(&self.generated, true)
            .map_err(|e| PipelineError::Tokenization(format!("Decode error: {}", e)))?;

        if decoded.len() > self.last_decoded_len {
            let new_text = decoded[self.last_decoded_len..].to_string();
            self.last_decoded_len = decoded.len();
            if !new_text.is_empty() {
                return Ok(Some(new_text));
            }
        }
        Ok(None)
    }

    fn initialize(&mut self) -> Result<Option<String>> {
        if self.eos_tokens.is_empty() {
            return Err(PipelineError::Unexpected(
                "No EOS tokens configured for model. Cannot determine when to stop.".to_string(),
            ));
        }

        // Process entire prompt at once
        let input = Tensor::new(self.input_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = {
            let mut cache = self.cache.lock().unwrap();
            self.model.forward(&input, &mut cache)
        }?;
        let last_logits = logits.squeeze(0)?;

        self.next_token = self.logits_processor.sample(&last_logits)?;
        self.generated.push(self.next_token);
        self.stats.lock().unwrap().record_token();
        self.initialized = true;

        // Yield first token if not EOS
        if self.eos_tokens.contains(&self.next_token) {
            self.finish();
            return Ok(None);
        }

        self.decode_incremental()
    }

    fn generate_next(&mut self) -> Result<Option<String>> {
        if self.eos_tokens.contains(&self.next_token) {
            self.finish();
            return Ok(None);
        }

        if self.generated.len() >= self.params.max_len {
            self.finish();
            return Ok(None);
        }

        let input = Tensor::new(&[self.next_token], &self.device)?.unsqueeze(0)?;
        let logits = {
            let mut cache = self.cache.lock().unwrap();
            self.model.forward(&input, &mut cache)
        }?;
        let logits = logits.squeeze(0)?;

        let logits = apply_repeat_penalty_if_needed(
            logits,
            &self.generated,
            self.params.repeat_penalty,
            self.params.repeat_last_n,
        )?;

        self.next_token = self.logits_processor.sample(&logits)?;
        self.generated.push(self.next_token);
        self.stats.lock().unwrap().record_token();

        if self.eos_tokens.contains(&self.next_token) {
            self.finish();
            return Ok(None);
        }

        self.decode_incremental()
    }

    fn finish(&mut self) {
        if !self.finished {
            self.finished = true;
            self.stats.lock().unwrap().finalize();
        }
    }
}

impl<C: ModelConfig> Iterator for TokenIterator<C>
where
    C::Model: Send + Sync,
{
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        // Initialize on first call
        if !self.initialized {
            match self.initialize() {
                Ok(Some(chunk)) => return Some(Ok(chunk)),
                Ok(None) => {
                    if self.finished {
                        return None;
                    }
                }
                Err(e) => {
                    self.finished = true;
                    return Some(Err(e));
                }
            }
        }

        // Generate tokens until we get one that produces output or we're done
        loop {
            if self.finished {
                return None;
            }

            match self.generate_next() {
                Ok(Some(chunk)) => return Some(Ok(chunk)),
                Ok(None) => {
                    if self.finished {
                        return None;
                    }
                    // Token didn't produce output, continue
                }
                Err(e) => {
                    self.finished = true;
                    return Some(Err(e));
                }
            }
        }
    }
}

/// Apply repeat penalty if configured, otherwise return logits unchanged.
fn apply_repeat_penalty_if_needed(
    logits: Tensor,
    generated_tokens: &[u32],
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,
) -> CandleResult<Tensor> {
    let Some(penalty) = repeat_penalty else {
        return Ok(logits);
    };

    if penalty <= 1.0 {
        return Ok(logits);
    }

    let last_n = repeat_last_n.unwrap_or(64);
    let start_at = generated_tokens.len().saturating_sub(last_n);
    let penalty_context = &generated_tokens[start_at..];

    if penalty_context.is_empty() {
        return Ok(logits);
    }

    apply_repeat_penalty(&logits, penalty, penalty_context)
}
