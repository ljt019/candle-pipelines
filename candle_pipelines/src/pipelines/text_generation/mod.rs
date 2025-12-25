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
//! # #[tokio::main]
//! # async fn main() -> candle_pipelines::error::Result<()> {
//! let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
//!     .build()
//!     .await?;
//!
//! let response = pipeline.completion("Explain quantum computing briefly.").await?;
//! println!("{}", response);
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
//! # #[tokio::main]
//! # async fn main() -> candle_pipelines::error::Result<()> {
//! # let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B).build().await?;
//! let mut messages = vec![
//!     Message::system("You are a helpful assistant."),
//!     Message::user("What is Rust?"),
//! ];
//!
//! let response = pipeline.completion(&messages).await?;
//!
//! // Continue the conversation
//! messages.push(Message::assistant(&response));
//! messages.push(Message::user("What makes it memory-safe?"));
//!
//! let followup_response = pipeline.completion(&messages).await?;
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
//! # #[tokio::main]
//! # async fn main() -> candle_pipelines::error::Result<()> {
//! let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
//!     .temperature(0.8)    // randomness (0.0 = deterministic)
//!     .top_k(50)           // sample from top 50 tokens
//!     .top_p(0.9)          // nucleus sampling
//!     .max_len(1024)       // max tokens to generate
//!     .repeat_penalty(1.1) // discourage repetition
//!     .build()
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Streaming
//!
//! Get tokens as they're generated:
//!
//! ```rust,no_run
//! # use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3Size};
//! # #[tokio::main]
//! # async fn main() -> candle_pipelines::error::Result<()> {
//! # let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B).build().await?;
//! let mut stream = pipeline.completion_stream("Write a poem about Rust.").await?;
//!
//! while let Some(token) = stream.next().await {
//!     print!("{}", token?);
//! }
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
//! # #[tokio::main]
//! # async fn main() -> Result<()> {
//! let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
//!     .build()
//!     .await?;
//!
//! pipeline.register_tools(tools![get_weather]).await;
//!
//! let response = pipeline
//!     .completion_with_tools("What's the weather in Tokyo?")
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! # XML Structured Output
//!
//! Parse XML tags in model output with [`XmlTextGenerationPipeline`]:
//!
//! ```rust,no_run
//! # use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3Size, TagParts};
//! # #[tokio::main]
//! # async fn main() -> candle_pipelines::error::Result<()> {
//! let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
//!     .build_xml(&["think", "answer"])  // tags to parse
//!     .await?;
//!
//! let events = pipeline.completion("Solve 2+2. Think step by step. Put your final answer in <answer></answer> tags.").await?;
//!
//! for event in events {
//!     match (event.tag(), event.part()) {
//!         (Some("think"), TagParts::Content) => print!("[thinking] {}", event.get_content()),
//!         (Some("answer"), TagParts::Content) => print!("[answer] {}", event.get_content()),
//!         // Regular content outside of any tags
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
pub(crate) mod xml_pipeline;

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
pub use params::GenerationParams;
pub use parser::TagParts;
pub use pipeline::TextGenerationPipeline;
pub use tools::{ErrorStrategy, Tool};
pub use xml_pipeline::XmlTextGenerationPipeline;
