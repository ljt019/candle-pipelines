//! Transformers provides a simple, intuitive interface for Rust developers to work with Language Models locally.
//!
//! Powered by the [Candle](https://github.com/huggingface/candle) crate, it offers an API inspired by Python's [Transformers](https://huggingface.co/docs/transformers)
//! but tailored for Rust developers. Includes pipelines for generation, classification, and masked language modeling.

// #![deny(missing_docs)]

// ============ Internal API ============

pub(crate) mod loaders;
pub(crate) mod models;
pub(crate) mod pipelines;

// ============ Public API ============

pub mod error;

pub use pipelines::{fill_mask, sentiment, text_generation, zero_shot};
