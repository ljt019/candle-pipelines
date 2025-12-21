use crate::Message;
use crate::Result;
use candle_core::{Device, Tensor};

// Re-export tool-related types
pub use super::tools::{ErrorStrategy, IntoTool, Tool, ToolCalling};

/// Minimal interface required by the text-generation pipeline for a model context.
///
/// Both `Qwen3Model::Context` and `Gemma3Model::Context` already expose compatible
/// `generate` and `reset` methods, so we only need a thin trait wrapper that the
/// pipeline can work with generically.
pub trait LanguageModelContext: Send {
    /// Forward the input tokens through the model, returning the logits for the
    /// next token.
    fn generate(&mut self, input: &Tensor) -> candle_core::Result<Tensor>;

    /// Clear the internal state (kv-cache, position, etc.).
    fn reset(&mut self);

    /// Get the current position (number of cached tokens).
    fn position(&self) -> usize;

    /// Check if the cache is still valid for continuing from a given position.
    fn can_continue_from(&self, position: usize) -> bool;
}

#[allow(async_fn_in_trait)]
pub trait TextGenerationModel {
    /// Type used to configure model loading (e.g. which checkpoint size).
    type Options;
    /// The context type that will be returned by `new_context` and consumed by
    /// the pipeline. It must implement [`LanguageModelContext`] and be `Send`
    /// so that asynchronous streams capturing it can be moved across threads.
    type Context: LanguageModelContext + Send;

    async fn new(options: Self::Options, device: Device) -> Result<Self>
    where
        Self: Sized;

    async fn get_tokenizer(&self) -> Result<tokenizers::Tokenizer>;

    fn apply_chat_template(&self, messages: &[Message]) -> Result<String>;

    fn get_eos_token(&self) -> Option<u32>;

    /// Get all EOS token IDs for robust termination detection
    fn get_eos_tokens(&self) -> Vec<u32> {
        self.get_eos_token().into_iter().collect()
    }

    fn get_max_seq_len(&self) -> usize;

    fn new_context(&self) -> Self::Context;

    fn clear_context(&self, context: &mut Self::Context) -> Result<()>;

    /// Get the default generation parameters for this model.
    fn default_generation_params(
        &self,
    ) -> crate::pipelines::text_generation::params::GenerationParams {
        crate::pipelines::text_generation::params::GenerationParams::default()
    }
}

pub trait Reasoning {}

pub trait ToggleableReasoning {
    fn set_reasoning(&mut self, enable: bool) -> Result<()>;
}
