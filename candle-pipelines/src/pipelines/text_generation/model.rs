use crate::error::Result;
use candle_core::{Device, Tensor};

use super::message::Message;
use super::tools::Tool;

/// Trait for KV cache types.
pub trait ModelCache: Send {
    fn reset(&mut self);
    fn current_seq_len(&self) -> usize;
}

#[allow(async_fn_in_trait)]
pub trait TextGenerationModel {
    type Options;
    type Cache: ModelCache + Send;

    /// Create a new model instance (sync, uses ureq for downloads).
    fn new(options: Self::Options, device: Device) -> Result<Self>
    where
        Self: Sized;

    /// Create a new model instance (async, uses reqwest for downloads).
    async fn new_async(options: Self::Options, device: Device) -> Result<Self>
    where
        Self: Sized;

    fn get_tokenizer(&self) -> Result<tokenizers::Tokenizer>;

    /// Apply chat template to messages. Tools are included in the prompt if
    /// the model supports tool calling and tools are provided.
    fn apply_chat_template(&self, messages: &[Message], tools: &[Tool]) -> Result<String>;

    fn get_eos_token(&self) -> Option<u32>;

    fn get_eos_tokens(&self) -> Vec<u32> {
        self.get_eos_token().into_iter().collect()
    }

    fn get_max_seq_len(&self) -> usize;

    /// Create a new empty KV cache for generation.
    fn new_cache(&self) -> Self::Cache;

    /// Run forward pass with external cache.
    fn forward(&self, input: &Tensor, cache: &mut Self::Cache) -> candle_core::Result<Tensor>;

    fn default_generation_params(
        &self,
    ) -> crate::pipelines::text_generation::params::GenerationParams {
        crate::pipelines::text_generation::params::GenerationParams::default()
    }
}

/// Marker trait for models that produce reasoning/thinking output.
pub trait Reasoning {}

/// Trait for models where reasoning can be toggled on/off.
pub trait ToggleableReasoning: Reasoning {
    /// Enable or disable reasoning mode.
    fn enable_reasoning(&self, enable: bool);
}
