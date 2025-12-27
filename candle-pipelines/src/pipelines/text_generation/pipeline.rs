#![allow(unused_assignments)]

use std::future::Future;
use std::pin::Pin;

use super::base_pipeline::BasePipeline;
use super::message::Message;
use super::model::TextGenerationModel;
use super::model::{LanguageModelContext, Reasoning, ToggleableReasoning};
use super::params::GenerationParams;
use super::stats::GenerationStats;
use super::tools::{ErrorStrategy, Tool};
use crate::error::PipelineError;
use crate::error::Result;
use crate::models::{Gemma3, Qwen3};
use async_stream::try_stream;
use futures::{Stream, StreamExt};
use regex::Regex;
use serde::Deserialize;

// ============ Object-safe trait for runtime model switching ============

/// Boxed future type for trait object compatibility.
pub type BoxedFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Boxed stream type for trait object compatibility.
pub type BoxedStream<'a, T> = Pin<Box<dyn Stream<Item = T> + Send + 'a>>;

/// Object-safe text generation trait for runtime model switching.
///
/// ```rust,ignore
/// let pipeline: Box<dyn TextGeneration> = Box::new(builder.qwen3(...).build().await?);
///
/// pipeline.register_tool(my_tool);
/// let response = pipeline.completion(&messages).await?;
/// ```
pub trait TextGeneration: Send + Sync {
    /// Generate a complete response from messages.
    fn completion<'a>(&'a self, messages: &'a [Message]) -> BoxedFuture<'a, Result<String>>;

    /// Stream tokens as they're generated.
    fn completion_stream<'a>(
        &'a self,
        messages: &'a [Message],
    ) -> BoxedFuture<'a, Result<BoxedStream<'a, Result<String>>>>;

    /// Whether this model supports tool calling.
    fn supports_tools(&self) -> bool {
        false
    }

    /// Whether this model supports reasoning output (think tags).
    fn supports_reasoning(&self) -> bool {
        false
    }

    // ============ Tool methods (available on all pipelines) ============

    /// Register a tool for use during generation.
    fn register_tool(&self, tool: Tool);

    /// Remove a tool by name.
    fn unregister_tool(&self, name: &str);

    /// Remove all registered tools.
    fn clear_tools(&self);

    /// Returns all registered tools.
    fn registered_tools(&self) -> Vec<Tool>;

    /// Enable or disable tool usage.
    fn enable_tools(&self, enable: bool);

    /// Returns whether tools are enabled.
    fn tools_enabled(&self) -> bool;

    // ============ Reasoning methods ============

    #[doc(hidden)]
    fn as_toggleable_reasoning(&self) -> Option<&dyn ToggleableReasoning> {
        None
    }

    #[doc(hidden)]
    fn as_reasoning(&self) -> Option<&dyn Reasoning> {
        None
    }

    /// Clear the KV cache and reset generation state.
    fn clear_cache(&self) -> BoxedFuture<'_, ()>;
}

/// Trait alias for dynamic text generation pipelines.
///
/// Use with `TextGenerationExt` for `with_*` helper methods:
///
/// ```rust,ignore
/// use candle_pipelines::text_generation::{AnyTextGenerationPipeline, TextGenerationExt};
///
/// let pipeline: Box<dyn AnyTextGenerationPipeline> = Box::new(
///     TextGenerationPipelineBuilder::qwen3(...).build().await?
/// );
///
/// pipeline.with_tools(|tc| tc.register_tool(my_tool));
/// pipeline.completion(&messages).await?;
/// ```
pub trait AnyTextGenerationPipeline: TextGeneration + Send + Sync {}
impl<T: TextGeneration + Send + Sync> AnyTextGenerationPipeline for T {}

/// Extension trait for `with_*` helper methods for reasoning.
///
/// Works through Deref - `MutexGuard<Box<dyn AnyTextGenerationPipeline>>` works directly.
pub trait AnyTextGenerationPipelineExt {
    /// Execute closure with toggleable reasoning if supported.
    fn with_toggleable_reasoning(&self, f: impl FnOnce(&dyn ToggleableReasoning));
    /// Execute closure with reasoning if supported.
    fn with_reasoning(&self, f: impl FnOnce(&dyn Reasoning));
}

impl<T: TextGeneration + ?Sized> AnyTextGenerationPipelineExt for T {
    fn with_toggleable_reasoning(&self, f: impl FnOnce(&dyn ToggleableReasoning)) {
        if let Some(tr) = self.as_toggleable_reasoning() {
            f(tr);
        }
    }
    fn with_reasoning(&self, f: impl FnOnce(&dyn Reasoning)) {
        if let Some(r) = self.as_reasoning() {
            f(r);
        }
    }
}

// ============ Pipeline types ============

/// Input type for completion methods. Accepts prompts or message arrays.
#[derive(Debug, Clone)]
pub enum Input<'a> {
    /// A single prompt string.
    Prompt(&'a str),
    /// A conversation as a slice of messages.
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

/// Pipeline for generating text from prompts or conversations.
///
/// Created via [`TextGenerationPipelineBuilder`](super::TextGenerationPipelineBuilder).
///
/// # Example
///
/// ```rust,no_run
/// use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3Size, Message};
///
/// # async fn example() -> candle_pipelines::error::Result<()> {
/// let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
///     .build()
///     .await?;
///
/// // Simple prompt
/// let response = pipeline.completion("What is Rust?").await?;
///
/// // Multi-turn conversation
/// let messages = vec![
///     Message::system("You are a helpful assistant."),
///     Message::user("What is 2+2?"),
/// ];
/// let response = pipeline.completion(&messages).await?;
/// # Ok(())
/// # }
pub struct TextGenerationPipeline<M: TextGenerationModel> {
    pub(crate) base: BasePipeline<M>,
    tool_error_strategy: ErrorStrategy,
    tools: std::sync::RwLock<Vec<Tool>>,
    tools_enabled: std::sync::atomic::AtomicBool,
}

impl<M: TextGenerationModel + Send + Sync> TextGenerationPipeline<M> {
    pub(crate) async fn new(
        model: std::sync::Arc<M>,
        gen_params: GenerationParams,
        device: candle_core::Device,
        tool_error_strategy: ErrorStrategy,
    ) -> Result<Self> {
        Ok(Self {
            base: BasePipeline::new(model, gen_params, device).await?,
            tool_error_strategy,
            tools: std::sync::RwLock::new(Vec::new()),
            tools_enabled: std::sync::atomic::AtomicBool::new(true),
        })
    }

    // ============ Tool management methods ============

    /// Register a tool for use during generation.
    /// Tools are used automatically if the model supports tool calling.
    pub fn register_tool(&self, tool: Tool) {
        let mut tools = self.tools.write().unwrap();
        if let Some(pos) = tools.iter().position(|t| t.name() == tool.name()) {
            tools[pos] = tool;
        } else {
            tools.push(tool);
        }
    }

    /// Register multiple tools at once.
    pub async fn register_tools(&self, tools: Vec<Tool>) {
        for tool in tools {
            self.register_tool(tool);
        }
    }

    /// Remove a tool by name. No-op if not found.
    pub async fn unregister_tool(&self, name: &str) {
        let mut tools = self.tools.write().unwrap();
        if let Some(pos) = tools.iter().position(|t| t.name() == name) {
            tools.remove(pos);
        }
    }

    /// Remove multiple tools by name.
    pub async fn unregister_tools(&self, tools_to_remove: Vec<Tool>) {
        for tool in tools_to_remove {
            self.unregister_tool(&tool.name).await;
        }
    }

    /// Remove all registered tools.
    pub async fn clear_tools(&self) {
        self.tools.write().unwrap().clear();
    }

    /// Returns a list of all registered tools.
    pub async fn registered_tools(&self) -> Vec<Tool> {
        self.tools.read().unwrap().clone()
    }

    /// Enable or disable tool usage during generation.
    pub fn enable_tools(&self, enable: bool) {
        self.tools_enabled
            .store(enable, std::sync::atomic::Ordering::SeqCst);
    }

    /// Returns whether tools are currently enabled.
    pub fn tools_enabled(&self) -> bool {
        self.tools_enabled.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Returns tools to pass to model (empty if disabled).
    fn active_tools(&self) -> Vec<Tool> {
        if self.tools_enabled() {
            self.tools.read().unwrap().clone()
        } else {
            Vec::new()
        }
    }

    /// Returns the current tool error handling strategy.
    pub fn tool_error_strategy(&self) -> &ErrorStrategy {
        &self.tool_error_strategy
    }

    /// Returns stats from the last generation (tokens, timing).
    pub fn last_generation_stats(&self) -> Option<GenerationStats> {
        self.base.last_generation_stats()
    }

    /// Update generation parameters (temperature, top_p, etc.).
    pub async fn set_generation_params(&self, params: GenerationParams) {
        self.base.set_generation_params(params).await;
    }

    /// Returns the model's maximum context length in tokens.
    pub async fn max_context_length(&self) -> usize {
        self.base.model.get_max_seq_len()
    }

    /// Count tokens in text without generating.
    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        let tokens = self.base.model_tokenizer.encode(text, false).map_err(|e| {
            PipelineError::Tokenization(format!(
                "Tokenization failed on '{}...': {}",
                text.chars().take(50).collect::<String>(),
                e
            ))
        })?;
        Ok(tokens.get_ids().len())
    }

    /// Clear the KV cache.
    pub async fn clear_cache(&self) {
        self.base.context.lock().await.reset();
        self.base.last_processed_tokens.lock().await.clear();
    }

    // Basic completion - no tool checking. Used by per-model inherent methods.
    async fn completion_basic<'a>(&self, input: impl Into<Input<'a>>) -> Result<String> {
        match input.into() {
            Input::Prompt(p) => self.prompt_completion_internal(p).await,
            Input::Messages(m) => self.message_completion_internal(m).await,
        }
    }

    async fn completion_basic_with_stats<'a>(
        &self,
        input: impl Into<Input<'a>>,
    ) -> Result<(String, GenerationStats)> {
        match input.into() {
            Input::Prompt(p) => self.prompt_completion_internal_with_stats(p).await,
            Input::Messages(m) => self.message_completion_internal_with_stats(m).await,
        }
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

        let templated_prompt = self.base.model.apply_chat_template(
            &[super::message::Message::user(prompt)],
            &self.active_tools(),
        )?;

        let prompt_tokens = self
            .base
            .model_tokenizer
            .encode(templated_prompt.as_str(), true)
            .map_err(|e| {
                PipelineError::Tokenization(format!(
                    "Tokenization failed on '{}...': {}",
                    templated_prompt.chars().take(50).collect::<String>(),
                    e
                ))
            })?
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
        let templated_prompt = self
            .base
            .model
            .apply_chat_template(messages, &self.active_tools())?;

        let new_tokens = self
            .base
            .model_tokenizer
            .encode(templated_prompt.as_str(), true)
            .map_err(|e| {
                PipelineError::Tokenization(format!(
                    "Tokenization failed on '{}...': {}",
                    templated_prompt.chars().take(50).collect::<String>(),
                    e
                ))
            })?
            .get_ids()
            .to_vec();

        let max_seq_len = self.base.model.get_max_seq_len();
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

    // Basic streaming
    async fn completion_stream_basic<'a>(
        &'a self,
        input: impl Into<Input<'a>>,
    ) -> Result<
        crate::pipelines::text_generation::streaming::CompletionStream<
            impl futures::Stream<Item = Result<String>> + Send + 'a,
        >,
    > {
        let tools = self.active_tools();
        match input.into() {
            Input::Prompt(p) => {
                self.base.context.lock().await.reset();
                let templated = self
                    .base
                    .model
                    .apply_chat_template(&[super::message::Message::user(p)], &tools)?;
                let tokens = self
                    .base
                    .model_tokenizer
                    .encode(templated.as_str(), true)
                    .map_err(|e| {
                        PipelineError::Tokenization(format!(
                            "Tokenization failed on '{}...': {}",
                            templated.chars().take(50).collect::<String>(),
                            e
                        ))
                    })?
                    .get_ids()
                    .to_vec();
                let prompt_tokens = tokens.len();
                Ok(self.completion_stream_from_tokens(tokens, prompt_tokens))
            }
            Input::Messages(m) => {
                let templated = self.base.model.apply_chat_template(m, &tools)?;
                let new_tokens = self
                    .base
                    .model_tokenizer
                    .encode(templated.as_str(), true)
                    .map_err(|e| {
                        PipelineError::Tokenization(format!(
                            "Tokenization failed on '{}...': {}",
                            templated.chars().take(50).collect::<String>(),
                            e
                        ))
                    })?
                    .get_ids()
                    .to_vec();

                // Always reset context to avoid candle dtype bug in Gemma3's mask
                // when index_pos > 0 and seq_len > 1 (cache reuse triggers this)
                self.base.context.lock().await.reset();
                self.base.last_processed_tokens.lock().await.clear();

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

impl<M: TextGenerationModel + ToggleableReasoning + Sync> TextGenerationPipeline<M> {
    /// Enable or disable reasoning/thinking mode for models that support it.
    pub async fn enable_reasoning(&self, enable: bool) {
        self.base.model.enable_reasoning(enable)
    }
}

/// A tool execution result with name and raw content.
struct ToolResult {
    name: String,
    content: String,
}

impl ToolResult {
    /// Format for user output
    fn for_user(&self) -> String {
        format!(
            "<tool_result name=\"{}\">\n{}\n</tool_result>",
            self.name, self.content
        )
    }

    /// Format for model (raw content, template adds wrapping).
    fn for_model(&self) -> Message {
        Message::tool(&self.content)
    }
}

impl<M: TextGenerationModel + Send + Sync> TextGenerationPipeline<M> {
    async fn execute_tool_calls(
        &self,
        tool_calls: Vec<ToolCallInvocation>,
        tools: &[Tool],
    ) -> Result<Vec<ToolResult>> {
        let mut tool_results = Vec::new();

        for call in tool_calls {
            let available_tools: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();
            let tool = tools.iter().find(|t| t.name == call.name).ok_or_else(|| {
                PipelineError::Tool(format!(
                    "Tool '{}' not found. Registered tools: {}",
                    call.name,
                    available_tools.join(", ")
                ))
            })?;

            let args = call.arguments.clone();
            let mut attempts = 0u32;

            loop {
                match tool.call(args.clone()).await {
                    Ok(result) => {
                        tool_results.push(ToolResult {
                            name: call.name.clone(),
                            content: result.trim_end_matches('\n').to_string(),
                        });
                        break;
                    }
                    Err(e) => {
                        attempts += 1;
                        if attempts >= tool.max_retries() {
                            match &self.tool_error_strategy {
                                ErrorStrategy::Fail => return Err(e),
                                ErrorStrategy::ReturnToModel => {
                                    tool_results.push(ToolResult {
                                        name: call.name.clone(),
                                        content: format!("Error: {e}"),
                                    });
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

        Ok(tool_results)
    }

    /// Internal: tool-calling completion flow
    async fn completion_with_tools_internal(&self, messages: &[Message]) -> Result<String> {
        let tools = self.registered_tools().await;
        let mut messages = messages.to_vec();
        let mut full_response = String::new();

        loop {
            let templated = self.base.model.apply_chat_template(&messages, &tools)?;
            let new_tokens = self
                .base
                .model_tokenizer
                .encode(templated.as_str(), true)
                .map_err(|e| {
                    PipelineError::Tokenization(format!(
                        "Tokenization failed on '{}...': {}",
                        templated.chars().take(50).collect::<String>(),
                        e
                    ))
                })?
                .get_ids()
                .to_vec();

            let max_seq_len = self.base.model.get_max_seq_len();
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

                    let tool_results = self.execute_tool_calls(tool_calls, &tools).await?;

                    // Add wrapped results to user output
                    let user_output: Vec<String> =
                        tool_results.iter().map(|r| r.for_user()).collect();
                    full_response.push('\n');
                    full_response.push_str(&user_output.join("\n"));
                    full_response.push('\n');

                    // Add raw results as tool messages for model
                    for result in tool_results {
                        messages.push(result.for_model());
                    }
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

    /// Internal: tool-calling streaming flow
    async fn completion_stream_with_tools_internal<'a>(
        &'a self,
        initial_messages: Vec<Message>,
    ) -> Result<
        crate::pipelines::text_generation::streaming::CompletionStream<
            impl futures::Stream<Item = Result<String>> + Send + 'a,
        >,
    > {
        let tools = self.active_tools();

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
                    let stream_inner = self.completion_stream_basic(&messages[..]).await?;
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

                        let tool_results = self.execute_tool_calls(tool_calls, &tools).await?;

                        // Yield wrapped results for user output
                        let user_output: Vec<String> =
                            tool_results.iter().map(|r| r.for_user()).collect();
                        yield format!("\n\n{}\n", user_output.join("\n"));

                        // Add raw results as tool messages for model
                        for result in tool_results {
                            messages.push(result.for_model());
                        }
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
        let tool_regex =
            Regex::new(r"(?s)<tool_call>(.*?)</tool_call>").expect("hardcoded regex is valid");
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

// ============ TextGeneration trait impls ============

// ============ Per-model inherent methods ============

impl TextGenerationPipeline<Qwen3> {
    /// Generate a completion. Auto-uses tools if registered and enabled.
    pub async fn completion<'a>(&self, input: impl Into<Input<'a>>) -> Result<String> {
        let tools = self.registered_tools().await;
        if self.tools_enabled() && !tools.is_empty() {
            let messages = match input.into() {
                Input::Prompt(p) => vec![Message::user(p)],
                Input::Messages(m) => m.to_vec(),
            };
            self.completion_with_tools_internal(&messages).await
        } else {
            self.completion_basic(input).await
        }
    }

    /// Generate with stats. Auto-uses tools if registered and enabled.
    pub async fn completion_with_stats<'a>(
        &self,
        input: impl Into<Input<'a>>,
    ) -> Result<(String, GenerationStats)> {
        // Note: tool calling doesn't return stats, fallback to basic
        self.completion_basic_with_stats(input).await
    }

    /// Stream tokens. Auto-uses tools if registered and enabled.
    pub async fn completion_stream<'a>(
        &'a self,
        input: impl Into<Input<'a>>,
    ) -> Result<
        crate::pipelines::text_generation::streaming::CompletionStream<
            std::pin::Pin<Box<dyn futures::Stream<Item = Result<String>> + Send + 'a>>,
        >,
    > {
        let tools = self.registered_tools().await;
        if self.tools_enabled() && !tools.is_empty() {
            let messages = match input.into() {
                Input::Prompt(p) => vec![Message::user(p)],
                Input::Messages(m) => m.to_vec(),
            };
            let cs = self.completion_stream_with_tools_internal(messages).await?;
            Ok(cs.boxed())
        } else {
            let cs = self.completion_stream_basic(input).await?;
            Ok(cs.boxed())
        }
    }

    /// Generate completions for multiple prompts sequentially.
    pub async fn completion_batch(&self, prompts: &[&str]) -> Result<Vec<Result<String>>> {
        let mut outputs = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            outputs.push(self.completion(*prompt).await);
        }
        Ok(outputs)
    }
}

impl TextGenerationPipeline<Gemma3> {
    /// Generate a completion.
    pub async fn completion<'a>(&self, input: impl Into<Input<'a>>) -> Result<String> {
        self.completion_basic(input).await
    }

    /// Generate with stats.
    pub async fn completion_with_stats<'a>(
        &self,
        input: impl Into<Input<'a>>,
    ) -> Result<(String, GenerationStats)> {
        self.completion_basic_with_stats(input).await
    }

    /// Stream tokens.
    pub async fn completion_stream<'a>(
        &'a self,
        input: impl Into<Input<'a>>,
    ) -> Result<
        crate::pipelines::text_generation::streaming::CompletionStream<
            impl futures::Stream<Item = Result<String>> + Send + 'a,
        >,
    > {
        self.completion_stream_basic(input).await
    }

    /// Generate completions for multiple prompts sequentially.
    pub async fn completion_batch(&self, prompts: &[&str]) -> Result<Vec<Result<String>>> {
        let mut outputs = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            outputs.push(self.completion(*prompt).await);
        }
        Ok(outputs)
    }
}

// ============ TextGeneration trait impls (for dynamic dispatch) ============

impl TextGeneration for TextGenerationPipeline<Qwen3> {
    fn completion<'a>(&'a self, messages: &'a [Message]) -> BoxedFuture<'a, Result<String>> {
        Box::pin(async move { self.completion(messages).await })
    }

    fn completion_stream<'a>(
        &'a self,
        messages: &'a [Message],
    ) -> BoxedFuture<'a, Result<BoxedStream<'a, Result<String>>>> {
        Box::pin(async move {
            let stream = self.completion_stream(messages).await?;
            Ok(Box::pin(stream) as BoxedStream<'a, Result<String>>)
        })
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_reasoning(&self) -> bool {
        true
    }

    fn register_tool(&self, tool: Tool) {
        TextGenerationPipeline::register_tool(self, tool)
    }

    fn unregister_tool(&self, name: &str) {
        let mut tools = self.tools.write().unwrap();
        if let Some(pos) = tools.iter().position(|t| t.name() == name) {
            tools.remove(pos);
        }
    }

    fn clear_tools(&self) {
        self.tools.write().unwrap().clear();
    }

    fn registered_tools(&self) -> Vec<Tool> {
        self.tools.read().unwrap().clone()
    }

    fn enable_tools(&self, enable: bool) {
        TextGenerationPipeline::enable_tools(self, enable)
    }

    fn tools_enabled(&self) -> bool {
        TextGenerationPipeline::tools_enabled(self)
    }

    fn as_toggleable_reasoning(&self) -> Option<&dyn ToggleableReasoning> {
        Some(&*self.base.model)
    }

    fn as_reasoning(&self) -> Option<&dyn Reasoning> {
        Some(&*self.base.model)
    }

    fn clear_cache(&self) -> BoxedFuture<'_, ()> {
        Box::pin(async move { self.clear_cache().await })
    }
}

impl TextGeneration for TextGenerationPipeline<Gemma3> {
    fn completion<'a>(&'a self, messages: &'a [Message]) -> BoxedFuture<'a, Result<String>> {
        Box::pin(async move { self.completion(messages).await })
    }

    fn completion_stream<'a>(
        &'a self,
        messages: &'a [Message],
    ) -> BoxedFuture<'a, Result<BoxedStream<'a, Result<String>>>> {
        Box::pin(async move {
            let stream = self.completion_stream(messages).await?;
            Ok(Box::pin(stream) as BoxedStream<'a, Result<String>>)
        })
    }

    fn register_tool(&self, tool: Tool) {
        TextGenerationPipeline::register_tool(self, tool)
    }

    fn unregister_tool(&self, name: &str) {
        let mut tools = self.tools.write().unwrap();
        if let Some(pos) = tools.iter().position(|t| t.name() == name) {
            tools.remove(pos);
        }
    }

    fn clear_tools(&self) {
        self.tools.write().unwrap().clear();
    }

    fn registered_tools(&self) -> Vec<Tool> {
        self.tools.read().unwrap().clone()
    }

    fn enable_tools(&self, enable: bool) {
        TextGenerationPipeline::enable_tools(self, enable)
    }

    fn tools_enabled(&self) -> bool {
        TextGenerationPipeline::tools_enabled(self)
    }

    fn clear_cache(&self) -> BoxedFuture<'_, ()> {
        Box::pin(async move { self.clear_cache().await })
    }
}

// ============ Internal types ============

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

#[cfg(test)]
#[cfg(feature = "cuda")]
mod cache_tests {
    use super::LanguageModelContext;
    use crate::error::Result;
    use crate::text_generation::{Qwen3Size, TextGenerationPipelineBuilder};

    #[tokio::test]
    async fn multiple_pipelines_work_independently() -> Result<()> {
        let mut pipelines = Vec::new();
        for _ in 0..3 {
            let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
                .cuda(0)
                .temperature(0.7)
                .max_len(10)
                .build()
                .await?;
            pipelines.push(pipeline);
        }

        let _ = pipelines[0].completion("Hello").await?;
        assert!(pipelines[0].base.context.lock().await.position() > 0);

        for p in pipelines.iter().skip(1) {
            assert_eq!(p.base.context.lock().await.position(), 0);
        }

        Ok(())
    }
}
