//! Fill-mask pipeline for predicting masked tokens in text.
//!
//! This module provides functionality for filling in masked tokens (typically `[MASK]`)
//! in text sequences using pre-trained language models. It's useful for text completion,
//! error correction, and exploring model behavior.
//!
//! ## Main Types
//!
//! - [`FillMaskPipeline`] - High-level interface for mask filling
//! - [`FillMaskPipelineBuilder`] - Builder pattern for pipeline configuration
//! - [`FillMaskModel`] - Trait for fill-mask model implementations
//! - [`ModernBertSize`] - Available model size options
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use anyhow::Result;
//! use transformers::pipelines::fill_mask::*;
//! use transformers::pipelines::utils::BasePipelineBuilder;
//!
//! fn main() -> Result<()> {
//!     // Create a fill-mask pipeline
//!     let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base)
//!         .build()?;
//!
//!     // Fill masked tokens
//!     let top = pipeline.predict("The capital of France is [MASK].")?;
//!     println!("Token: {} (score: {:.3})", top.word, top.score);
//!
//!     // Or get top-k candidates
//!     let top3 = pipeline.predict_top_k("The capital of France is [MASK].", 3)?;
//!     println!("Got {} candidates", top3.len());
//!     Ok(())
//! }
//! ```

pub mod builder;
pub mod model;
pub mod pipeline;

pub use builder::FillMaskPipelineBuilder;
pub use model::FillMaskModel;
pub use pipeline::FillMaskPipeline;

pub use crate::models::ModernBertSize;

pub use anyhow::Result;
