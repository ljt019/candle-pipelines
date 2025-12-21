//! # Transformers
//!
//! Rust pipelines for candle models. Like HuggingFace's `pipeline()` API.

pub mod error;
pub mod loaders;
pub mod message;
pub mod models;
pub mod pipelines;

pub use error::{Result, TransformersError};
pub use message::{Message, MessageVecExt, Role};
pub use models::{
    FillMaskModernBertModel, Gemma3Model, Gemma3Size, ModernBertSize, Qwen3Model, Qwen3Size,
    SentimentModernBertModel, ZeroShotModernBertModel,
};
pub use tool_macro::tool;
