//! Text generation pipeline for generating human-like text completions.
//!
//! This module provides functionality for generating text using large language models,
//! including both single completions and streaming outputs. It supports various generation
//! strategies, XML parsing for structured outputs, and tool calling capabilities.
//!
//! ## Main Types
//!
//! - [`TextGenerationPipeline`] - High-level interface for text generation
//! - [`XmlGenerationPipeline`] - Specialized pipeline for XML-structured generation
//! - [`TextGenerationPipelineBuilder`] - Builder pattern for pipeline configuration
//! - [`CompletionStream`] - Stream of generated tokens for real-time output
//! - [`GenerationParams`] - Parameters controlling generation behavior
//! - [`Tool`] - Trait for implementing function calling capabilities
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use anyhow::Result;
//! use transformers::pipelines::text_generation::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Create a text generation pipeline
//!     let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
//!         .temperature(0.7)
//!         .max_len(100)
//!         .build()
//!         .await?;
//!
//!     // Generate text completion
//!     let completion = pipeline.completion("Once upon a time").await?;
//!     println!("Generated: {}", completion);
//!
//!     // Stream generation in real-time
//!     let mut stream = pipeline.completion_stream("Tell me about Rust.").await?;
//!     while let Some(chunk) = stream.next().await {
//!         print!("{}", chunk?);
//!     }
//!     Ok(())
//! }
//! ```

pub mod base_pipeline;
pub mod builder;
pub mod model;
pub mod params;
pub mod parser;
pub mod pipeline;
pub mod stats;
pub mod streaming;
pub mod tools;
pub mod xml_pipeline;

pub use crate::models::{Gemma3Size, Qwen3Size};
pub use crate::tools;
pub use builder::TextGenerationPipelineBuilder;
pub use params::GenerationParams;
pub use pipeline::{Input, TextGenerationPipeline};
pub use stats::GenerationStats;
pub use streaming::{CompletionStream, EventStream};
pub use xml_pipeline::XmlGenerationPipeline;

// Re-export the procedural macro (functions as an item in Rust 2018+).
pub use crate::tool;

// Re-export `futures::StreamExt` so users iterating over streaming outputs
// get the `next`/`try_next` extension methods automatically when they
// glob-import this module.
pub use futures::StreamExt;
pub use futures::TryStreamExt;

// Re-export commonly used types and traits
pub use crate::{Message, MessageVecExt};

// Re-export Result type for convenience
pub use anyhow::Result;

// Re-export std::io::Write for flushing stdout in examples
pub use std::io::Write;

pub use parser::{Event, TagParts, XmlParser, XmlParserBuilder};
pub use tools::{ErrorStrategy, IntoTool, Tool, ToolCalling, ToolError};

#[macro_export]
macro_rules! tools {
    ($($tool:ident),+ $(,)?) => {
        vec![
            $(
                $tool::__tool()
            ),+
        ]
    };
}

// Note: No need to re-export tools macro since it's already defined above
