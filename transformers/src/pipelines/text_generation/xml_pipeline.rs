use super::base_pipeline::BasePipeline;
use super::model::TextGenerationModel;
use super::model::{LanguageModelContext, ToggleableReasoning};
use super::params::GenerationParams;
use super::parser::{Event, XmlParser};
use super::pipeline::Input;
use super::tools::{ErrorStrategy, Tool, ToolCalling};
use crate::error::Result;
use crate::error::{TokenizationError, ToolError};
use async_stream::stream;
use futures::Stream;
use futures::StreamExt;
use regex::Regex;
use serde::Deserialize;

pub struct XmlTextGenerationPipeline<M: TextGenerationModel> {
    base: BasePipeline<M>,
    xml_parser: XmlParser,
    tool_error_strategy: ErrorStrategy,
}

impl<M: TextGenerationModel + Send> XmlTextGenerationPipeline<M> {
    pub async fn new(
        model: M,
        gen_params: GenerationParams,
        xml_parser: XmlParser,
        device: candle_core::Device,
        tool_error_strategy: ErrorStrategy,
    ) -> Result<Self> {
        Ok(Self {
            base: BasePipeline::new(model, gen_params, device).await?,
            xml_parser,
            tool_error_strategy,
        })
    }

    pub async fn context_position(&self) -> usize {
        self.base.context_position().await
    }

    pub async fn set_generation_params(&self, params: GenerationParams) {
        self.base.set_generation_params(params).await;
    }

    pub fn xml_parser(&self) -> &XmlParser {
        &self.xml_parser
    }

    pub async fn completion<'a>(&self, input: impl Into<Input<'a>>) -> Result<Vec<Event>> {
        let text = match input.into() {
            Input::Prompt(p) => self.prompt_completion_internal(p).await?,
            Input::Messages(m) => self.message_completion_internal(m).await?,
        };

        Ok(self.xml_parser.parse_complete(&text))
    }

    async fn prompt_completion_internal(&self, prompt: &str) -> Result<String> {
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

        self.base.completion_from_tokens(&prompt_tokens).await
    }

    async fn message_completion_internal(
        &self,
        messages: &[super::message::Message],
    ) -> Result<String> {
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
            let response = self.base.completion_from_tokens(new_portion).await?;

            *self.base.last_processed_tokens.lock().await = new_tokens;
            return Ok(response);
        } else {
            self.base.context.lock().await.reset();
        }

        let response = self.base.completion_from_tokens(&new_tokens).await?;

        *self.base.last_processed_tokens.lock().await = new_tokens;

        Ok(response)
    }

    pub async fn completion_stream<'a>(
        &'a self,
        input: impl Into<Input<'a>>,
    ) -> Result<
        crate::pipelines::text_generation::streaming::EventStream<
            impl Stream<Item = Event> + Send + 'a,
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

                Ok(self.event_stream_from_tokens(tokens))
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
                    return Ok(self.event_stream_from_tokens(suffix));
                } else {
                    self.base.context.lock().await.reset();
                }

                *self.base.last_processed_tokens.lock().await = new_tokens.clone();
                Ok(self.event_stream_from_tokens(new_tokens))
            }
        }
    }

    fn event_stream_from_tokens<'a>(
        &'a self,
        tokens: Vec<u32>,
    ) -> crate::pipelines::text_generation::streaming::EventStream<
        impl Stream<Item = Event> + Send + 'a,
    >
    where
        M: Send + 'a,
    {
        let prompt_tokens = tokens.len();
        let (_stream_stats, inner) = self
            .base
            .token_stream_with_prompt_count(tokens, Some(prompt_tokens));

        self.xml_parser.reset();
        let parser = self.xml_parser.clone();

        let event_stream = stream! {
            futures::pin_mut!(inner);
            while let Some(result) = inner.next().await {
                let token = result.expect("stream generation failed");
                let events = parser.parse_token(&token);
                for event in events {
                    yield event;
                }
            }

            let final_events = parser.flush();
            for event in final_events {
                yield event;
            }
        };

        crate::pipelines::text_generation::streaming::EventStream::new(event_stream)
    }
}

impl<M: TextGenerationModel + ToggleableReasoning> XmlTextGenerationPipeline<M> {
    pub async fn set_reasoning(&self, enable: bool) -> Result<()> {
        self.base.model.lock().await.set_reasoning(enable)
    }
}

impl<M: TextGenerationModel + ToolCalling + Send> XmlTextGenerationPipeline<M> {
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

    pub async fn completion_with_tools<'a>(
        &self,
        input: impl Into<Input<'a>>,
    ) -> Result<Vec<Event>> {
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
                        full_response.push_str(response.trim_start_matches('\n'));
                        return Ok(self.xml_parser.parse_complete(&full_response));
                    } else {
                        return Ok(self.xml_parser.parse_complete(&response));
                    }
                }
            }
        }
    }

    pub async fn completion_stream_with_tools<'a>(
        &'a self,
        input: impl Into<Input<'a>>,
    ) -> Result<
        crate::pipelines::text_generation::streaming::EventStream<
            impl Stream<Item = Event> + Send + 'a,
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

        let xml_parser = self.xml_parser.clone();

        let event_stream = stream! {
            let mut messages = initial_messages;
            let mut raw_buffer = String::new();

            loop {
                {
                    let templated = self
                        .base
                        .model
                        .lock()
                        .await
                        .apply_chat_template(&messages)
                        .expect("failed to apply chat template");
                    let new_tokens = self
                        .base
                        .model_tokenizer
                        .encode(templated.as_str(), true)
                        .expect("failed to encode")
                        .get_ids()
                        .to_vec();
                    let total_prompt_tokens = new_tokens.len();

                    let max_seq_len = self.base.model.lock().await.get_max_seq_len();
                    let pending_tokens = new_tokens.len();

                    let tokens_to_process = if self.base.context.lock().await.position() + pending_tokens > max_seq_len {
                        self.base.context.lock().await.reset();
                        self.base.last_processed_tokens.lock().await.clear();
                        new_tokens.clone()
                    } else if self.base.can_reuse_cache(&new_tokens).await {
                        let prefix_len = self.base.last_processed_tokens.lock().await.len();
                        let suffix = new_tokens[prefix_len..].to_vec();
                        *self.base.last_processed_tokens.lock().await = new_tokens;
                        suffix
                    } else {
                        self.base.context.lock().await.reset();
                        *self.base.last_processed_tokens.lock().await = new_tokens.clone();
                        new_tokens
                    };

                    let (_stream_stats, stream_inner) = self
                        .base
                        .token_stream_with_prompt_count(tokens_to_process, Some(total_prompt_tokens));
                    futures::pin_mut!(stream_inner);

                    while let Some(result) = stream_inner.next().await {
                        match result {
                            Ok(token) => {
                                raw_buffer.push_str(&token);

                                let events = xml_parser.parse_token(&token);
                                for event in events {
                                    yield event;
                                }
                            }
                            Err(_e) => {
                                break;
                            }
                        }
                    }

                    let final_events = xml_parser.flush();
                    for event in final_events {
                        yield event;
                    }
                }

                match Self::extract_tool_calls(&raw_buffer) {
                    Ok(tool_calls) if !tool_calls.is_empty() => {
                        messages.push(super::message::Message::assistant(&raw_buffer));
                        raw_buffer.clear();

                        let tool_responses = match self.execute_tool_calls(tool_calls, &tools).await {
                            Ok(responses) => responses,
                            Err(_e) => {
                                break;
                            }
                        };
                        let tool_response_text = tool_responses.join("\n");

                        let tool_events = xml_parser.parse_complete(&tool_response_text);
                        for event in tool_events {
                            yield event;
                        }

                        messages.push(super::message::Message::user(&tool_response_text));

                        xml_parser.reset();

                    }
                    _ => {
                        break;
                    }
                }
            }
        };

        Ok(crate::pipelines::text_generation::streaming::EventStream::new(event_stream))
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
