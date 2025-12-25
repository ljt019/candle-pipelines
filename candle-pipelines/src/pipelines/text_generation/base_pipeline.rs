use super::model::{LanguageModelContext, TextGenerationModel};
use super::params::{apply_repeat_penalty, initialize_logits_processor, GenerationParams};
use super::stats::GenerationStats;
use crate::error::{PipelineError, Result};
use candle_core::Tensor;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

pub struct BasePipeline<M: TextGenerationModel> {
    pub model: Arc<Mutex<M>>,
    pub model_tokenizer: Tokenizer,
    pub context: Arc<Mutex<M::Context>>,
    pub gen_params: Arc<Mutex<GenerationParams>>,
    pub device: candle_core::Device,
    pub last_processed_tokens: Arc<Mutex<Vec<u32>>>,
    pub last_generation_stats: std::sync::Arc<std::sync::Mutex<Option<GenerationStats>>>,
}

impl<M: TextGenerationModel> BasePipeline<M> {
    pub async fn new(
        model: M,
        gen_params: GenerationParams,
        device: candle_core::Device,
    ) -> Result<Self> {
        let model_tokenizer = model.get_tokenizer().await?;
        let context = model.new_context();

        let mut special_strings: std::collections::HashSet<String> = model_tokenizer
            .get_added_tokens_decoder()
            .values()
            .filter(|tok| tok.special)
            .map(|tok| tok.content.clone())
            .collect();

        special_strings.insert("<|im_start|>".to_string());
        special_strings.insert("<|im_end|>".to_string());
        special_strings.insert("<|im_sep|>".to_string());

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            model_tokenizer,
            context: Arc::new(Mutex::new(context)),
            gen_params: Arc::new(Mutex::new(gen_params)),
            device,
            last_processed_tokens: Arc::new(Mutex::new(Vec::new())),
            last_generation_stats: std::sync::Arc::new(std::sync::Mutex::new(None)),
        })
    }

    pub async fn context_position(&self) -> usize {
        self.context.lock().await.position()
    }

    pub fn last_generation_stats(&self) -> Option<GenerationStats> {
        self.last_generation_stats.lock().unwrap().clone()
    }

    pub async fn set_generation_params(&self, params: GenerationParams) {
        *self.gen_params.lock().await = params;
    }

    pub async fn can_reuse_cache(&self, new_tokens: &[u32]) -> bool {
        new_tokens.starts_with(&self.last_processed_tokens.lock().await)
    }

    pub async fn completion_from_tokens(&self, input_tokens: &[u32]) -> Result<String> {
        let (output, _) = self.completion_from_tokens_with_stats(input_tokens).await?;
        Ok(output)
    }

    pub async fn completion_from_tokens_with_stats(
        &self,
        input_tokens: &[u32],
    ) -> Result<(String, GenerationStats)> {
        self.completion_from_tokens_with_prompt_stats(input_tokens, input_tokens.len())
            .await
    }

    pub async fn completion_from_tokens_with_prompt_stats(
        &self,
        input_tokens: &[u32],
        prompt_token_count: usize,
    ) -> Result<(String, GenerationStats)> {
        const CHUNK_SIZE: usize = 64;

        let params = self.gen_params.lock().await.clone();

        let mut logits_processor = initialize_logits_processor(&params, params.seed);

        let mut generated_tokens: Vec<u32> = Vec::with_capacity(params.max_len);
        let mut stats = GenerationStats::new();
        stats.set_prompt_tokens(prompt_token_count);

        let mut idx = 0;
        let mut last_logits = None;
        while idx < input_tokens.len() {
            let end = usize::min(idx + CHUNK_SIZE, input_tokens.len());
            let chunk = &input_tokens[idx..end];

            let input = Tensor::new(chunk, &self.device)?.unsqueeze(0)?;
            let logits = {
                let mut ctx = self.context.lock().await;
                ctx.generate(&input)
            }?;
            last_logits = Some(logits.squeeze(0)?);
            idx = end;
        }

        let mut next_token = logits_processor.sample(&last_logits.expect("missing logits"))?;
        generated_tokens.push(next_token);
        stats.record_token();

        let eos_tokens = self.model.lock().await.get_eos_tokens();
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
                let mut ctx = self.context.lock().await;
                ctx.generate(&input)
            }?;
            let logits = logits.squeeze(0)?;

            let start_at = generated_tokens.len().saturating_sub(params.repeat_last_n);
            let penalty_context = &generated_tokens[start_at..];

            let logits = if params.repeat_penalty <= 1. || penalty_context.is_empty() {
                logits
            } else {
                apply_repeat_penalty(&logits, params.repeat_penalty, penalty_context)?
            };

            next_token = logits_processor.sample(&logits)?;
            generated_tokens.push(next_token);
            stats.record_token();
        }

        let eos_tokens = self.model.lock().await.get_eos_tokens();
        let filtered_tokens: Vec<u32> = generated_tokens
            .into_iter()
            .filter(|&token| !eos_tokens.contains(&token))
            .collect();

        let generated_tokens_str = self
            .model_tokenizer
            .decode(&filtered_tokens, /*skip_special_tokens=*/ true)
            .expect("token decode failed");

        stats.finalize();
        self.last_generation_stats
            .lock()
            .unwrap()
            .replace(stats.clone());

        Ok((generated_tokens_str, stats))
    }

    pub fn token_stream_with_prompt_count<'a>(
        &'a self,
        input_tokens: Vec<u32>,
        prompt_token_count: Option<usize>,
    ) -> (
        std::sync::Arc<std::sync::Mutex<GenerationStats>>,
        impl futures::Stream<Item = Result<String>> + Send + 'a,
    )
    where
        M: 'a + Send,
    {
        let device = self.device.clone();
        let model = std::sync::Arc::clone(&self.model);
        let tokenizer = self.model_tokenizer.clone();
        let context = std::sync::Arc::clone(&self.context);
        let gen_params = std::sync::Arc::clone(&self.gen_params);
        let last_stats = std::sync::Arc::clone(&self.last_generation_stats);

        let stats = std::sync::Arc::new(std::sync::Mutex::new(GenerationStats::new()));
        stats
            .lock()
            .unwrap()
            .set_prompt_tokens(prompt_token_count.unwrap_or(input_tokens.len()));

        let stats_clone = std::sync::Arc::clone(&stats);

        let stream = async_stream::try_stream! {
            let params = gen_params.lock().await.clone();
            let eos_tokens = model.lock().await.get_eos_tokens();
            if eos_tokens.is_empty() {
                Err(PipelineError::Unexpected(
                    "No EOS tokens configured for model. Cannot determine when to stop.".to_string(),
                ))?;
            }
            const CHUNK_SIZE: usize = 64;

            let mut logits_processor =
                initialize_logits_processor(&params, params.seed);

            let mut idx = 0;
            let mut last_logits = None;
            while idx < input_tokens.len() {
                let end = usize::min(idx + CHUNK_SIZE, input_tokens.len());
                let chunk = &input_tokens[idx..end];

                let input = Tensor::new(chunk, &device)?.unsqueeze(0)?;
                let logits = {
                    let mut ctx = context.lock().await;
                    ctx.generate(&input)
                }?;
                last_logits = Some(logits.squeeze(0)?);
                idx = end;
            }

            let mut generated: Vec<u32> = Vec::with_capacity(params.max_len);

            let mut dec_full = tokenizer.decode_stream(false);

            let mut next_token = logits_processor.sample(&last_logits.expect("missing logits"))?;
            generated.push(next_token);
            stats_clone.lock().unwrap().record_token();

            if !eos_tokens.contains(&next_token) {
                if let Some(chunk) =
                    dec_full.step(next_token).map_err(|e| PipelineError::Tokenization(format!("Failed to decode token {next_token}: {e}")))?
                {
                    yield chunk;
                }
            } else {
                let _ = dec_full
                    .step(next_token)
                    .map_err(|e| PipelineError::Tokenization(format!("Failed to decode token {next_token}: {e}")))?;
            }

            for _ in 0..params.max_len {
                if eos_tokens.contains(&next_token) {
                    break;
                }

                let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
                let logits = {
                    let mut ctx = context.lock().await;
                    ctx.generate(&input)
                }?;
                let logits = logits.squeeze(0)?;

                let start_at = generated.len().saturating_sub(params.repeat_last_n);
                let penalty_context = &generated[start_at..];

                let logits = if params.repeat_penalty <= 1. || penalty_context.is_empty() {
                    logits
                } else {
                    apply_repeat_penalty(&logits, params.repeat_penalty, penalty_context)?
                };

                next_token = logits_processor.sample(&logits)?;
                generated.push(next_token);
                stats_clone.lock().unwrap().record_token();

                if !eos_tokens.contains(&next_token) {
                    if let Some(chunk) = dec_full
                        .step(next_token)
                        .map_err(|e| PipelineError::Tokenization(format!("Failed to decode token {next_token}: {e}")))?
                    {
                        yield chunk;
                    }
                } else {
                    let _ = dec_full
                        .step(next_token)
                        .map_err(|e| PipelineError::Tokenization(format!("Failed to decode token {next_token}: {e}")))?;
                }
            }

            {
                let mut stats_guard = stats_clone.lock().unwrap();
                stats_guard.finalize();
                last_stats.lock().unwrap().replace(stats_guard.clone());
            }
        };

        (stats, stream)
    }
}
