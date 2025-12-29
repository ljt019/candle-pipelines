//! Text generation pipeline for LLMs.
//!
//! Generate text from prompts or multi-turn conversations.
//! Supports streaming, tool calling, and configurable sampling parameters.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3Size};
//!
//! # fn main() -> candle_pipelines::error::Result<()> {
//! let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
//!     .build()?;
//!
//! let output = pipeline.run("Explain quantum computing briefly.")?;
//! println!("{}", output.text);
//! println!("Generated {} tokens in {:.2}s", output.stats.tokens_generated, output.stats.total_time.as_secs_f64());
//! # Ok(())
//! # }
//! ```
//!
//! # Multi-Turn Chat
//!
//! Use [`Message`] to build conversations:
//!
//! ```rust,no_run
//! # use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3Size, Message};
//! # fn main() -> candle_pipelines::error::Result<()> {
//! # let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B).build()?;
//! let mut messages = vec![
//!     Message::system("You are a helpful assistant."),
//!     Message::user("What is Rust?"),
//! ];
//!
//! let output = pipeline.run(&messages)?;
//!
//! // Continue the conversation
//! messages.push(Message::assistant(&output.text));
//! messages.push(Message::user("What makes it memory-safe?"));
//!
//! let followup = pipeline.run(&messages)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Generation Parameters
//!
//! Configure sampling via the builder:
//!
//! ```rust,no_run
//! # use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3Size};
//! # fn main() -> candle_pipelines::error::Result<()> {
//! let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
//!     .temperature(0.8)    // randomness (0.0 = deterministic)
//!     .top_k(50)           // sample from top 50 tokens
//!     .top_p(0.9)          // nucleus sampling
//!     .max_len(1024)       // max tokens to generate
//!     .repeat_penalty(1.1) // discourage repetition
//!     .build()?;
//! # Ok(())
//! # }
//! ```
//!
//! # Streaming
//!
//! Streaming is fully sync - no async runtime needed:
//!
//! ```rust,no_run
//! # use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3Size};
//! # fn main() -> candle_pipelines::error::Result<()> {
//! # let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B).build()?;
//! let mut stream = pipeline.run_iter("Write a poem about Rust.")?;
//!
//! for token in &mut stream {
//!     print!("{}", token?);
//! }
//! let stats = stream.stats();
//! # Ok(())
//! # }
//! ```
//!
//! # Tool Calling
//!
//! Let the model call your functions:
//!
//! ```rust,no_run
//! use candle_pipelines::text_generation::{tool, tools, TextGenerationPipelineBuilder, Qwen3Size};
//! use candle_pipelines::error::Result;
//!
//! #[tool]
//! /// Get current weather for a city.
//! fn get_weather(city: String) -> Result<String> {
//!     Ok(format!("Weather in {}: 72°F, sunny", city))
//! }
//!
//! # fn main() -> Result<()> {
//! let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
//!     .build()?;
//!
//! pipeline.register_tools(tools![get_weather]);
//!
//! // Tools are executed automatically when the model calls them
//! let output = pipeline.run("What's the weather in Tokyo?")?;
//! # Ok(())
//! # }
//! ```
//!
//! # XML Structured Output
//!
//! Parse XML tags in streaming output with [`XmlParser`]:
//!
//! ```rust,no_run
//! # use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3Size, TagParts, XmlParserBuilder};
//! # fn main() -> candle_pipelines::error::Result<()> {
//! let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
//!     .build()?;
//!
//! // Create parser for specific tags
//! let parser = XmlParserBuilder::new()
//!     .register_tag("think")
//!     .register_tag("answer")
//!     .build();
//!
//! // Wrap the token iterator with XML parsing
//! let stream = pipeline.run_iter("Solve 2+2. Think step by step.")?;
//! let event_iter = parser.wrap_iterator(stream);
//!
//! for event in event_iter {
//!     match (event.tag(), event.part()) {
//!         (Some("think"), TagParts::Content) => print!("[thinking] {}", event.get_content()),
//!         (Some("answer"), TagParts::Content) => print!("[answer] {}", event.get_content()),
//!         (None, TagParts::Content) => print!("{}", event.get_content()),
//!         _ => {}
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Supported Models
//!
//! | Model | Sizes | Builder Method |
//! |-------|-------|----------------|
//! | Qwen3 | `0.6B`, `1.7B`, `4B`, `8B`, `14B`, `32B` | [`TextGenerationPipelineBuilder::qwen3`] |
//! | Gemma3 | `1B`, `4B`, `12B`, `27B` | [`TextGenerationPipelineBuilder::gemma3`] |

// ============ Internal API ============

pub(crate) mod base_pipeline;
pub(crate) mod builder;
pub(crate) mod message;
pub(crate) mod model;
pub(crate) mod params;
pub(crate) mod parser;
pub(crate) mod pipeline;
pub(crate) mod stats;
pub(crate) mod streaming;
pub(crate) mod tools;

// For tool_macro
#[doc(hidden)]
pub use schemars;
#[doc(hidden)]
pub use tools::ToolFuture;

// ============ Public API ============

pub use crate::models::{Gemma3, Gemma3Size, Qwen3, Qwen3Size};
pub use builder::TextGenerationPipelineBuilder;
pub use candle_pipelines_macros::{tool, tools};
pub use message::Message;
pub use model::{Reasoning, ToggleableReasoning};
pub use params::GenerationParams;
pub use parser::{Event, EventIterator, EventStream, TagParts, XmlParser, XmlParserBuilder};
pub use pipeline::{
    AnyTextGenerationPipeline, AnyTextGenerationPipelineExt, BoxedIterator, Output,
    TextGeneration, TextGenerationPipeline,
};
pub use tools::{ErrorStrategy, Tool, ToolCalling};
