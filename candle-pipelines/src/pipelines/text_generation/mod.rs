//! Text generation pipeline for LLMs.
//!
//! Generate text from prompts or multi-turn conversations.
//! Supports streaming, tool calling, and configurable sampling parameters.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Olmo3};
//!
//! # fn main() -> candle_pipelines::error::Result<()> {
//! let pipeline = TextGenerationPipelineBuilder::olmo3(Olmo3::Size7B)
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
//! # use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3, Message};
//! # fn main() -> candle_pipelines::error::Result<()> {
//! # let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3::Size0_6B).build()?;
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
//! # use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3};
//! # fn main() -> candle_pipelines::error::Result<()> {
//! let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3::Size0_6B)
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
//! # Token Iteration
//!
//! Iterate over tokens as they're generated:
//!
//! ```rust,no_run
//! # use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3};
//! # fn main() -> candle_pipelines::error::Result<()> {
//! # let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3::Size0_6B).build()?;
//! let mut tokens = pipeline.run_iter("Write a poem about Rust.")?;
//!
//! for token in &mut tokens {
//!     print!("{}", token?);
//! }
//! let stats = tokens.stats();
//! # Ok(())
//! # }
//! ```
//!
//! # Tool Calling
//!
//! Let the model call your functions:
//!
//! ```rust,no_run
//! use candle_pipelines::text_generation::{tool, tools, TextGenerationPipelineBuilder, Qwen3};
//! use candle_pipelines::error::Result;
//!
//! #[tool]
//! /// Get current weather for a city.
//! fn get_weather(city: String) -> Result<String> {
//!     Ok(format!("Weather in {}: 72°F, sunny", city))
//! }
//!
//! # fn main() -> Result<()> {
//! let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3::Size0_6B)
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
//! # use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3, Event, TagPart, XmlTag};
//! # fn main() -> candle_pipelines::error::Result<()> {
//! // Define which tags to parse using an enum
//! #[derive(Debug, Clone, PartialEq, XmlTag)]
//! enum Tags {
//!     #[tag("think")]
//!     Think,
//!     #[tag("answer")]
//!     Answer,
//! }
//!
//! let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3::Size0_6B)
//!     .build()?;
//!
//! // Create parser from tag enum
//! let parser = Tags::parser();
//!
//! // Wrap the token iterator with XML parsing
//! let tokens = pipeline.run_iter("Solve 2+2. Think step by step.")?;
//! let events = parser.parse_iter(tokens);
//!
//! for event in events {
//!     let event = event?; // Propagate errors
//!     match event {
//!         Event::Tag { tag: Tags::Think, part: TagPart::Content { text } } => print!("[thinking] {}", text),
//!         Event::Tag { tag: Tags::Answer, part: TagPart::Content { text } } => print!("[answer] {}", text),
//!         Event::Content { text } => print!("{}", text),
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
//! | Qwen3 | `Size0_6B`, `Size1_7B`, `Size4B`, `Size8B`, `Size14B`, `Size32B` | [`TextGenerationPipelineBuilder::qwen3`] |
//! | Gemma3 | `Size1B`, `Size4B`, `Size12B`, `Size27B` | [`TextGenerationPipelineBuilder::gemma3`] |
//! | Llama 3.2 | `Size1B`, `Size3B` | [`TextGenerationPipelineBuilder::llama3_2`] |
//! | OLMo-3 | `Size7B`, `Size32B` | [`TextGenerationPipelineBuilder::olmo3`] |

// ============ Internal API ============

pub(crate) mod base_pipeline;
pub(crate) mod builder;
pub(crate) mod message;
pub(crate) mod params;
pub(crate) mod pipeline;
pub(crate) mod streaming;
pub(crate) mod tools;
pub(crate) mod xml_parser;

// For tool_macro
#[doc(hidden)]
pub use schemars;
#[doc(hidden)]
pub use tools::ToolFuture;

// ============ Public API ============

// Model config enums - the simple API!
pub use crate::models::{Gemma3, Llama3_2, Olmo3, Qwen3};

pub use crate::pipelines::stats::GenerationStats;
pub use builder::TextGenerationPipelineBuilder;
pub use candle_pipelines_macros::{tool, tools, XmlTag};
pub use message::Message;
pub use params::GenerationParams;
pub use pipeline::{
    AnyTextGenerationPipeline, AnyTextGenerationPipelineExt, BoxedIterator, BoxedTokenIterator,
    Output, TextGeneration, TextGenerationPipeline, TokenIterator,
};
pub use tools::{ErrorStrategy, Tool};
pub use xml_parser::{Event, TagPart, XmlParser, XmlTag};
