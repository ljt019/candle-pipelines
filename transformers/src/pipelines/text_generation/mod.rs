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

#[doc(hidden)]
pub use tools::ToolFuture; // For tool_macro

// ============ Public API ============

pub use crate::models::{Gemma3Size, Qwen3Size};
pub use builder::TextGenerationPipelineBuilder;
pub use message::Message;
pub use params::GenerationParams;
pub use parser::TagParts;
pub use pipeline::TextGenerationPipeline;
pub use tools::{tool, tools, ErrorStrategy, Tool};
