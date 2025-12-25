//! Sentiment analysis pipeline.
//!
//! Classify text as `positive`, `negative`, or `neutral`.
//! Returns both the predicted label and a confidence score.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use transformers::sentiment::{SentimentAnalysisPipelineBuilder, ModernBertSize};
//!
//! # fn main() -> transformers::error::Result<()> {
//! let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
//! let result = pipeline.predict("I absolutely love this product!")?;
//!
//! // sentiment: positive (confidence: 0.98)
//! println!("sentiment: {} (confidence: {:.2})", result.label, result.score);
//! # Ok(())
//! # }
//! ```
//!
//! # Batch Inference
//!
//! Analyze multiple texts efficiently:
//!
//! ```rust,no_run
//! # use transformers::sentiment::{SentimentAnalysisPipelineBuilder, ModernBertSize};
//! # fn main() -> transformers::error::Result<()> {
//! # let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
//! let reviews = &[
//!     "Best purchase I've ever made!",
//!     "Terrible quality, very disappointed.",
//!     "It's okay, nothing special.",
//! ];
//!
//! let results = pipeline.predict_batch(reviews)?;
//!
//! for (text, result) in reviews.iter().zip(results) {
//!     let r = result?;
//!     println!("{}: {} ({:.2})", text, r.label, r.score);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Supported Models
//!
//! For now only ModernBERT is supported, but I have plans to add more models in the future!
//!
//! | Model | Sizes | Builder Method |
//! |-------|-------|----------------|
//! | ModernBERT | `Base`, `Large` | [`SentimentAnalysisPipelineBuilder::modernbert`] |

// ============ Internal API ============

pub(crate) mod builder;
pub(crate) mod model;
pub(crate) mod pipeline;

// ============ Public API ============

pub use crate::models::ModernBertSize;
pub use builder::SentimentAnalysisPipelineBuilder;
pub use pipeline::{SentimentAnalysisPipeline, SentimentResult};

/// Only for generic annotations. Use [`SentimentAnalysisPipelineBuilder::modernbert`].
pub type SentimentModernBert = crate::models::modernbert::SentimentModernBertModel;
