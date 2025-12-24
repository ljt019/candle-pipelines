#![allow(unused_assignments)]

use super::base_pipeline::BasePipeline;

use super::model::TextGenerationModel;
use super::model::{LanguageModelContext, ToggleableReasoning};
use super::params::GenerationParams;
use super::stats::GenerationStats;
use super::tools::{ErrorStrategy, Tool, ToolCalling};
use crate::error::Result;
use crate::error::{TokenizationError, ToolError};
use async_stream::try_stream;
use futures::StreamExt;
use regex::Regex;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub enum Input<'a> {
    Prompt(&'a str),
    Messages(&'a [super::message::Message]),
}

impl<'a> From<&'a str> for Input<'a> {
    fn from(s: &'a str) -> Self {
        Self::Prompt(s)
    }
}

impl<'a> From<&'a [super::message::Message]> for Input<'a> {
    fn from(m: &'a [super::message::Message]) -> Self {
        Self::Messages(m)
    }
}

impl<'a> From<&'a Vec<super::message::Message>> for Input<'a> {
    fn from(v: &'a Vec<super::message::Message>) -> Self {
        Self::Messages(v.as_slice())
    }
}

impl<'a> From<&'a String> for Input<'a> {
    fn from(s: &'a String) -> Self {
        Self::Prompt(s.as_str())
    }
}

pub struct TextGenerationPipeline<M: TextGenerationModel> {
    base: BasePipeline<M>,
    tool_error_strategy: ErrorStrategy,
}

impl<M: TextGenerationModel + Send> TextGenerationPipeline<M> {
    pub async fn new(
        model: M,
        gen_params: GenerationParams,
        device: candle_core::Device,
        tool_error_strategy: ErrorStrategy,
    ) -> Result<Self> {
        Ok(Self {
            base: BasePipeline::new(model, gen_params, device).await?,
            tool_error_strategy,
        })
    }

    pub fn tool_error_strategy(&self) -> &ErrorStrategy {
        &self.tool_error_strategy
    }

    pub async fn context_position(&self) -> usize {
        self.base.context_position().await
    }

    pub fn last_generation_stats(&self) -> Option<GenerationStats> {
        self.base.last_generation_stats()
    }

    pub async fn set_generation_params(&self, params: GenerationParams) {
        self.base.set_generation_params(params).await;
    }

    pub async fn max_context_length(&self) -> usize {
        self.base.model.lock().await.get_max_seq_len()
    }

    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        let tokens = self
            .base
            .model_tokenizer
            .encode(text, false)
            .map_err(|e| TokenizationError::encode_failed(text, e.to_string()))?;
        Ok(tokens.get_ids().len())
    }

    pub async fn clear_cache(&self) {
        self.base.context.lock().await.reset();
        self.base.last_processed_tokens.lock().await.clear();
    }

    pub async fn completion<'a>(&self, input: impl Into<Input<'a>>) -> Result<String> {
        match input.into() {
            Input::Prompt(p) => self.prompt_completion_internal(p).await,
            Input::Messages(m) => self.message_completion_internal(m).await,
        }
    }

    pub async fn completion_with_stats<'a>(
        &self,
        input: impl Into<Input<'a>>,
    ) -> Result<(String, GenerationStats)> {
        match input.into() {
            Input::Prompt(p) => self.prompt_completion_internal_with_stats(p).await,
            Input::Messages(m) => self.message_completion_internal_with_stats(m).await,
        }
    }

    pub async fn completion_batch(&self, prompts: &[&str]) -> Result<Vec<Result<String>>> {
        let mut outputs = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            outputs.push(self.prompt_completion_internal(prompt).await);
        }
        Ok(outputs)
    }

    pub async fn chat_batch(
        &self,
        conversations: &[&[super::message::Message]],
    ) -> Result<Vec<Result<String>>> {
        let mut outputs = Vec::with_capacity(conversations.len());
        for messages in conversations {
            self.base.context.lock().await.reset();
            self.base.last_processed_tokens.lock().await.clear();
            outputs.push(self.message_completion_internal(messages).await);
        }
        Ok(outputs)
    }

    async fn prompt_completion_internal(&self, prompt: &str) -> Result<String> {
        let (result, _) = self.prompt_completion_internal_with_stats(prompt).await?;
        Ok(result)
    }

    async fn prompt_completion_internal_with_stats(
        &self,
        prompt: &str,
    ) -> Result<(String, GenerationStats)> {
        self.base.context.lock().await.reset();

        let templated_prompt = self
            .base
            .model
            .lock()
            .await
            .apply_chat_template(&[super::message::Message::user(prompt)])?;

        let prompt_tokens = self
            .base
            .model_tokenizer
            .encode(templated_prompt.as_str(), true)
            .map_err(|e| TokenizationError::encode_failed(&templated_prompt, e.to_string()))?
            .get_ids()
            .to_vec();

        self.base
            .completion_from_tokens_with_stats(&prompt_tokens)
            .await
    }

    async fn message_completion_internal(
        &self,
        messages: &[super::message::Message],
    ) -> Result<String> {
        let (response, _) = self
            .message_completion_internal_with_stats(messages)
            .await?;

        Ok(response)
    }

    async fn message_completion_internal_with_stats(
        &self,
        messages: &[super::message::Message],
    ) -> Result<(String, GenerationStats)> {
        let templated_prompt = self.base.model.lock().await.apply_chat_template(messages)?;

        let new_tokens = self
            .base
            .model_tokenizer
            .encode(templated_prompt.as_str(), true)
            .map_err(|e| TokenizationError::encode_failed(&templated_prompt, e.to_string()))?
            .get_ids()
            .to_vec();

        let max_seq_len = self.base.model.lock().await.get_max_seq_len();
        let pending_tokens = new_tokens.len();

        if self.base.context.lock().await.position() + pending_tokens > max_seq_len {
            self.base.context.lock().await.reset();
            self.base.last_processed_tokens.lock().await.clear();
        } else if self.base.can_reuse_cache(&new_tokens).await {
            let prefix_len = self.base.last_processed_tokens.lock().await.len();
            let new_portion = &new_tokens[prefix_len..];
            let (response, stats) = self
                .base
                .completion_from_tokens_with_prompt_stats(new_portion, new_tokens.len())
                .await?;

            *self.base.last_processed_tokens.lock().await = new_tokens.clone();
            return Ok((response, stats));
        } else {
            self.base.context.lock().await.reset();
        }

        let (response, stats) = self
            .base
            .completion_from_tokens_with_prompt_stats(&new_tokens, new_tokens.len())
            .await?;

        *self.base.last_processed_tokens.lock().await = new_tokens;

        Ok((response, stats))
    }

    pub async fn completion_stream<'a>(
        &'a self,
        input: impl Into<Input<'a>>,
    ) -> Result<
        crate::pipelines::text_generation::streaming::CompletionStream<
            impl futures::Stream<Item = Result<String>> + Send + 'a,
        >,
    > {
        match input.into() {
            Input::Prompt(p) => {
                self.base.context.lock().await.reset();
                let templated = self
                    .base
                    .model
                    .lock()
                    .await
                    .apply_chat_template(&[super::message::Message::user(p)])?;
                let tokens = self
                    .base
                    .model_tokenizer
                    .encode(templated.as_str(), true)
                    .map_err(|e| TokenizationError::encode_failed(&templated, e.to_string()))?
                    .get_ids()
                    .to_vec();
                let prompt_tokens = tokens.len();
                Ok(self.completion_stream_from_tokens(tokens, prompt_tokens))
            }
            Input::Messages(m) => {
                let templated = self.base.model.lock().await.apply_chat_template(m)?;
                let new_tokens = self
                    .base
                    .model_tokenizer
                    .encode(templated.as_str(), true)
                    .map_err(|e| TokenizationError::encode_failed(&templated, e.to_string()))?
                    .get_ids()
                    .to_vec();

                let max_seq = self.base.model.lock().await.get_max_seq_len();
                if self.base.context.lock().await.position() + new_tokens.len() > max_seq {
                    self.base.context.lock().await.reset();
                    self.base.last_processed_tokens.lock().await.clear();
                } else if self.base.can_reuse_cache(&new_tokens).await {
                    let suffix =
                        new_tokens[self.base.last_processed_tokens.lock().await.len()..].to_vec();
                    *self.base.last_processed_tokens.lock().await = new_tokens;
                    let prompt_tokens = self.base.last_processed_tokens.lock().await.len();
                    return Ok(self.completion_stream_from_tokens(suffix, prompt_tokens));
                } else {
                    self.base.context.lock().await.reset();
                }

                *self.base.last_processed_tokens.lock().await = new_tokens.clone();
                let prompt_tokens = self.base.last_processed_tokens.lock().await.len();
                Ok(self.completion_stream_from_tokens(new_tokens, prompt_tokens))
            }
        }
    }

    fn completion_stream_from_tokens<'a>(
        &'a self,
        tokens: Vec<u32>,
        prompt_token_count: usize,
    ) -> crate::pipelines::text_generation::streaming::CompletionStream<
        impl futures::Stream<Item = Result<String>> + Send + 'a,
    >
    where
        M: Send + 'a,
    {
        let (stats, inner) = self
            .base
            .token_stream_with_prompt_count(tokens, Some(prompt_token_count));
        crate::pipelines::text_generation::streaming::CompletionStream::new(inner, stats)
    }
}

impl<M: TextGenerationModel + ToggleableReasoning> TextGenerationPipeline<M> {
    pub async fn set_reasoning(&self, enable: bool) -> Result<()> {
        self.base.model.lock().await.set_reasoning(enable)
    }
}

impl<M: TextGenerationModel + ToolCalling + Send> TextGenerationPipeline<M> {
    pub async fn unregister_tool(&self, name: &str) -> Result<()> {
        self.base.model.lock().await.unregister_tool(name)
    }

    pub async fn clear_tools(&self) -> Result<()> {
        self.base.model.lock().await.clear_tools()
    }

    pub async fn register_tools(&self, tools: Vec<Tool>) -> Result<()> {
        for tool in tools {
            self.base.model.lock().await.register_tool(tool)?;
        }
        Ok(())
    }

    pub async fn unregister_tools(&self, tools: Vec<Tool>) -> Result<()> {
        for tool in tools {
            self.base.model.lock().await.unregister_tool(&tool.name)?;
        }
        Ok(())
    }

    pub async fn registered_tools(&self) -> Vec<Tool> {
        self.base.model.lock().await.registered_tools()
    }

    async fn execute_tool_calls(
        &self,
        tool_calls: Vec<ToolCallInvocation>,
        tools: &[Tool],
    ) -> Result<Vec<String>> {
        let mut tool_responses = Vec::new();

        for call in tool_calls {
            let available_tools: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();
            let tool =
                tools
                    .iter()
                    .find(|t| t.name == call.name)
                    .ok_or_else(|| ToolError::NotFound {
                        name: call.name.clone(),
                        available: available_tools,
                    })?;

            let args = call.arguments.clone();
            let mut attempts = 0u32;

            loop {
                match tool.call(args.clone()).await {
                    Ok(result) => {
                        let trimmed_result = result.trim_end_matches('\n');
                        tool_responses.push(format!(
                            "<tool_result name=\"{}\">\n{}\n</tool_result>",
                            call.name, trimmed_result
                        ));
                        break;
                    }
                    Err(e) => {
                        attempts += 1;
                        if attempts >= tool.max_retries() {
                            match &self.tool_error_strategy {
                                ErrorStrategy::Fail => return Err(e),
                                ErrorStrategy::ReturnToModel => {
                                    let error_msg = format!("Error: {e}");
                                    let trimmed_error = error_msg.trim_end_matches('\n');
                                    tool_responses.push(format!(
                                        "<tool_result name=\"{}\">\n{}\n</tool_result>",
                                        call.name, trimmed_error
                                    ));
                                    break;
                                }
                            }
                        } else {
                            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        }
                    }
                }
            }
        }

        Ok(tool_responses)
    }

    pub async fn completion_with_tools<'a>(&self, input: impl Into<Input<'a>>) -> Result<String> {
        let tools = self.base.model.lock().await.registered_tools();
        if tools.is_empty() {
            return Err(ToolError::NoToolsRegistered.into());
        }

        let mut messages = match input.into() {
            Input::Prompt(p) => vec![super::message::Message::user(p)],
            Input::Messages(m) => m.to_vec(),
        };

        let mut full_response = String::new();

        loop {
            let templated = self
                .base
                .model
                .lock()
                .await
                .apply_chat_template(&messages)?;
            let new_tokens = self
                .base
                .model_tokenizer
                .encode(templated.as_str(), true)
                .map_err(|e| TokenizationError::encode_failed(&templated, e.to_string()))?
                .get_ids()
                .to_vec();

            let max_seq_len = self.base.model.lock().await.get_max_seq_len();
            let pending_tokens = new_tokens.len();

            let response =
                if self.base.context.lock().await.position() + pending_tokens > max_seq_len {
                    self.base.context.lock().await.reset();
                    self.base.last_processed_tokens.lock().await.clear();
                    self.base.completion_from_tokens(&new_tokens).await?
                } else if self.base.can_reuse_cache(&new_tokens).await {
                    let prefix_len = self.base.last_processed_tokens.lock().await.len();
                    let new_portion = &new_tokens[prefix_len..];
                    let res = self.base.completion_from_tokens(new_portion).await?;
                    *self.base.last_processed_tokens.lock().await = new_tokens;
                    res
                } else {
                    self.base.context.lock().await.reset();
                    let res = self.base.completion_from_tokens(&new_tokens).await?;
                    *self.base.last_processed_tokens.lock().await = new_tokens;
                    res
                };

            match Self::extract_tool_calls(&response) {
                Ok(tool_calls) if !tool_calls.is_empty() => {
                    full_response.push_str(&response);
                    full_response.push('\n');
                    messages.push(super::message::Message::assistant(&response));

                    let tool_responses = self.execute_tool_calls(tool_calls, &tools).await?;
                    let tool_response_text = tool_responses.join("\n");

                    full_response.push('\n');
                    full_response.push_str(&tool_response_text);
                    full_response.push('\n');

                    messages.push(super::message::Message::user(&tool_response_text));
                    continue;
                }
                _ => {
                    if !full_response.is_empty() {
                        full_response.push('\n');
                        full_response.push_str(&response);
                        return Ok(full_response);
                    } else {
                        return Ok(response);
                    }
                }
            }
        }
    }

    pub async fn completion_stream_with_tools<'a>(
        &'a self,
        input: impl Into<Input<'a>>,
    ) -> Result<
        crate::pipelines::text_generation::streaming::CompletionStream<
            impl futures::Stream<Item = Result<String>> + Send + 'a,
        >,
    > {
        let tools = self.base.model.lock().await.registered_tools();
        if tools.is_empty() {
            return Err(ToolError::NoToolsRegistered.into());
        }

        let initial_messages = match input.into() {
            Input::Prompt(p) => vec![super::message::Message::user(p)],
            Input::Messages(m) => m.to_vec(),
        };

        let stream_stats = std::sync::Arc::new(std::sync::Mutex::new(GenerationStats::new()));
        let stream_stats_inner = std::sync::Arc::clone(&stream_stats);

        let out_stream = try_stream! {
            let mut messages = initial_messages;
            let mut response_buffer = String::new();
            let mut needs_spacing = false;

            loop {
                if needs_spacing {
                    yield "\n".to_string();
                    needs_spacing = false;
                }

                {
                    let stream_stats = std::sync::Arc::clone(&stream_stats_inner);
                    let stream_inner = self.completion_stream(&messages[..]).await?;
                    futures::pin_mut!(stream_inner);

                    while let Some(chunk_res) = stream_inner.next().await {
                        let chunk = chunk_res?;
                        response_buffer.push_str(&chunk);
                        yield chunk;
                    }

                    if let Some(stats) = self.last_generation_stats() {
                        *stream_stats.lock().unwrap() = stats;
                    }
                }

                match Self::extract_tool_calls(&response_buffer) {
                    Ok(tool_calls) if !tool_calls.is_empty() => {
                        messages.push(super::message::Message::assistant(&response_buffer));
                        response_buffer.clear();

                        let tool_responses = self.execute_tool_calls(tool_calls, &tools).await?;
                        let tool_response_text = tool_responses.join("\n");

                        yield format!("\n\n{}\n", tool_response_text);

                        messages.push(super::message::Message::user(&tool_response_text));
                        needs_spacing = true;

                    }
                    _ => {
                        break;
                    }
                }
            }
        };
        Ok(
            crate::pipelines::text_generation::streaming::CompletionStream::new(
                out_stream,
                stream_stats,
            ),
        )
    }

    fn extract_tool_calls(text: &str) -> Result<Vec<ToolCallInvocation>> {
        let tool_regex = Regex::new(r"(?s)<tool_call>(.*?)</tool_call>")?;
        let mut tool_calls = Vec::new();

        for cap in tool_regex.captures_iter(text) {
            let json_str = cap.get(1).expect("expected capture group").as_str().trim();
            match serde_json::from_str::<RawToolCall>(json_str) {
                Ok(raw_call) => {
                    tool_calls.push(ToolCallInvocation {
                        name: raw_call.name,
                        arguments: raw_call
                            .arguments
                            .unwrap_or_else(|| serde_json::Value::Object(Default::default())),
                    });
                }
                Err(e) => {
                    eprintln!("Failed to parse tool call JSON: {e}");
                }
            }
        }

        Ok(tool_calls)
    }
}

#[derive(Deserialize)]
struct RawToolCall {
    name: String,
    #[serde(default)]
    arguments: Option<serde_json::Value>,
}

struct ToolCallInvocation {
    name: String,
    arguments: serde_json::Value,
}
