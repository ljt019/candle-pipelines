use crate::error::Result;
use candle_core::{Device, Tensor};

use super::message::Message;

pub trait LanguageModelContext: Send {
    fn generate(&mut self, input: &Tensor) -> candle_core::Result<Tensor>;

    fn reset(&mut self);

    fn position(&self) -> usize;
}

#[allow(async_fn_in_trait)]
pub trait TextGenerationModel {
    type Options;
    type Context: LanguageModelContext + Send;

    async fn new(options: Self::Options, device: Device) -> Result<Self>
    where
        Self: Sized;

    async fn get_tokenizer(&self) -> Result<tokenizers::Tokenizer>;

    fn apply_chat_template(&self, messages: &[Message]) -> Result<String>;

    fn get_eos_token(&self) -> Option<u32>;

    fn get_eos_tokens(&self) -> Vec<u32> {
        self.get_eos_token().into_iter().collect()
    }

    fn get_max_seq_len(&self) -> usize;

    fn new_context(&self) -> Self::Context;

    fn clear_context(&self, context: &mut Self::Context);

    fn default_generation_params(
        &self,
    ) -> crate::pipelines::text_generation::params::GenerationParams {
        crate::pipelines::text_generation::params::GenerationParams::default()
    }
}

#[allow(dead_code)]
pub trait Reasoning {}

pub trait ToggleableReasoning {
    fn set_reasoning(&mut self, enable: bool);
}
