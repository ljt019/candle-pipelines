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

pub use crate::tool;

pub use futures::StreamExt;
pub use futures::TryStreamExt;

pub use crate::{Message, MessageVecExt};

pub use crate::Result;

pub use std::io::Write;

pub use parser::{Event, TagParts, XmlParser, XmlParserBuilder};
pub use tools::{ErrorStrategy, IntoTool, Tool, ToolCalling, ToolFuture};

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
