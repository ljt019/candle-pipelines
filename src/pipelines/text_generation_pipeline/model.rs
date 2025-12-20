use crate::Message;
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

    async fn new(options: Self::Options, device: Device) -> anyhow::Result<Self>
    where
        Self: Sized;

    async fn get_tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer>;

    fn apply_chat_template(&self, messages: &[Message]) -> anyhow::Result<String>;

    fn get_eos_token(&self) -> u32;

    /// Get all EOS token IDs for robust termination detection
    fn get_eos_tokens(&self) -> Vec<u32> {
        vec![self.get_eos_token()]
    }

    fn get_max_seq_len(&self) -> usize;

    fn new_context(&self) -> Self::Context;

    fn clear_context(&self, context: &mut Self::Context) -> anyhow::Result<()>;

    /// Get the default generation parameters for this model.
    /// Models can override this to provide model-specific defaults.
    fn default_generation_params(&self) -> crate::models::generation::GenerationParams {
        crate::models::generation::GenerationParams {
            temperature: 0.7,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 42,
            max_len: 1024,
            sampling: crate::models::generation::params::SamplingParams {
                top_p: Some(1.0),
                top_k: None,
                min_p: None,
            },
        }
    }
}

pub trait Reasoning {}

pub trait ToggleableReasoning {
    fn set_reasoning(&mut self, enable: bool) -> anyhow::Result<()>;
}
