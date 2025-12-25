//! Simple, intuitive pipelines for local LLM inference in Rust.
//!
//! Powered by [Candle](https://github.com/huggingface/candle), with an API inspired by Python's [Transformers](https://huggingface.co/docs/transformers).
//! Includes pipelines for text generation, classification, and masked language modeling.

#![deny(missing_docs)]

// ============ Internal API ============

pub(crate) mod loaders;
pub(crate) mod models;
pub(crate) mod pipelines;

// ============ Public API ============

pub mod error;

pub use pipelines::{fill_mask, sentiment, text_generation, zero_shot};
