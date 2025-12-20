use super::model::{LanguageModelContext, TextGenerationModel};
use crate::models::generation::{
    apply_repeat_penalty, initialize_logits_processor, GenerationParams,
};
use candle_core::Tensor;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

/// Base structure containing common fields for both pipeline types
pub struct BasePipeline<M: TextGenerationModel> {
    pub model: Arc<Mutex<M>>,
    pub model_tokenizer: Tokenizer,
    pub context: Arc<Mutex<M::Context>>,
    pub gen_params: Arc<Mutex<GenerationParams>>,
    pub device: candle_core::Device,
    pub last_processed_tokens: Arc<Mutex<Vec<u32>>>,
    pub special_strings: std::collections::HashSet<String>,
}

impl<M: TextGenerationModel> BasePipeline<M> {
    pub async fn new(
        model: M,
        gen_params: GenerationParams,
        device: candle_core::Device,
    ) -> anyhow::Result<Self> {
        let model_tokenizer = model.get_tokenizer().await?;
        let context = model.new_context();

        // Collect textual forms of special tokens for display filtering
        let mut special_strings: std::collections::HashSet<String> = model_tokenizer
            .get_added_tokens_decoder()
            .values()
            .filter(|tok| tok.special)
            .map(|tok| tok.content.clone())
            .collect();

        // Add standard special tokens that might not be in the decoder
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
            special_strings,
        })
    }

    /// Get the current position in the context (number of cached tokens)
    pub async fn context_position(&self) -> usize {
        self.context.lock().await.position()
    }

    pub async fn set_generation_params(&self, params: GenerationParams) {
        *self.gen_params.lock().await = params;
    }

    pub async fn can_reuse_cache(&self, new_tokens: &[u32]) -> bool {
        // Cache can be reused if the new prompt begins with the exact token
        // sequence that is already cached.
        new_tokens.starts_with(&self.last_processed_tokens.lock().await)
    }

    pub async fn completion_from_tokens(&self, input_tokens: &[u32]) -> anyhow::Result<String> {
        const CHUNK_SIZE: usize = 64; // Must be <= initial kv cache size

        let params = self.gen_params.lock().await.clone();

        let mut logits_processor = initialize_logits_processor(&params, params.seed);

        let mut generated_tokens: Vec<u32> = Vec::with_capacity(params.max_len);

        // Feed the initial prompt in manageable chunks to allow the KV cache to grow.
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

        // Safety: there is always at least one chunk, so last_logits is Some
        let mut next_token = logits_processor.sample(&last_logits.expect("missing logits"))?;
        generated_tokens.push(next_token);

        // Generate autoregressively
        let eos_tokens = self.model.lock().await.get_eos_tokens();
        if eos_tokens.is_empty() {
            anyhow::bail!("Model provided no EOS tokens; cannot run text generation");
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
        }

        // Filter out EOS tokens before decoding
        let eos_tokens = self.model.lock().await.get_eos_tokens();
        let filtered_tokens: Vec<u32> = generated_tokens
            .into_iter()
            .filter(|&token| !eos_tokens.contains(&token))
            .collect();

        // Fast multi-token decode in a single call instead of per-token concat
        let generated_tokens_str = self
            .model_tokenizer
            .decode(&filtered_tokens, /*skip_special_tokens=*/ true)
            .expect("token decode failed");

        Ok(generated_tokens_str)
    }

    /// Stream tokens from the model given input tokens.
    pub fn token_stream<'a>(
        &'a self,
        input_tokens: Vec<u32>,
    ) -> impl futures::Stream<Item = anyhow::Result<String>> + Send + 'a
    where
        M: 'a + Send,
    {
        // Capture everything the async generator needs by value
        let device = self.device.clone();
        let model = std::sync::Arc::clone(&self.model);
        let tokenizer = self.model_tokenizer.clone();
        let context = std::sync::Arc::clone(&self.context);
        let gen_params = std::sync::Arc::clone(&self.gen_params);

        async_stream::try_stream! {
            let params = gen_params.lock().await.clone();
            let eos_tokens = model.lock().await.get_eos_tokens();
            if eos_tokens.is_empty() {
                anyhow::bail!("Model provided no EOS tokens; cannot stream text generation");
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

            let mut next_token =
                logits_processor.sample(&last_logits.expect("missing logits"))?;
            generated.push(next_token);

            if !eos_tokens.contains(&next_token) {
                if let Some(chunk) = dec_full.step(next_token).map_err(|e| anyhow::anyhow!(e))? {
                    yield chunk;
                }
            } else {
                let _ = dec_full.step(next_token).map_err(|e| anyhow::anyhow!(e))?;
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

                if !eos_tokens.contains(&next_token) {
                    if let Some(chunk) = dec_full.step(next_token).map_err(|e| anyhow::anyhow!(e))? {
                        yield chunk;
                    }
                } else {
                    let _ = dec_full.step(next_token).map_err(|e| anyhow::anyhow!(e))?;
                }
            }
        }
    }
}
