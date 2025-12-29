//! Sentiment analysis pipeline.
//!
//! Classify text as `positive`, `negative`, or `neutral`.
//! Returns both the predicted label and a confidence score.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use candle_pipelines::sentiment::{SentimentAnalysisPipelineBuilder, ModernBertSize};
//!
//! # fn main() -> candle_pipelines::error::Result<()> {
//! let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
//!
//! // Single text - direct access
//! let output = pipeline.run("I absolutely love this product!")?;
//! println!("sentiment: {} (confidence: {:.2})", output.prediction.label, output.prediction.score);
//! # Ok(())
//! # }
//! ```
//!
//! # Batch Inference
//!
//! Analyze multiple texts at once (returns `BatchOutput`):
//!
//! ```rust,no_run
//! # use candle_pipelines::sentiment::{SentimentAnalysisPipelineBuilder, ModernBertSize};
//! # fn main() -> candle_pipelines::error::Result<()> {
//! # let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
//! let reviews = &[
//!     "Best purchase I've ever made!",
//!     "Terrible quality, very disappointed.",
//!     "It's okay, nothing special.",
//! ];
//!
//! let output = pipeline.run(reviews)?;
//!
//! for (text, pred) in reviews.iter().zip(output.predictions) {
//!     let p = pred?;
//!     println!("{}: {} ({:.2})", text, p.label, p.score);
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
pub use crate::pipelines::stats::PipelineStats;
pub use builder::SentimentAnalysisPipelineBuilder;
pub use pipeline::{BatchOutput, Output, Prediction, SentimentAnalysisPipeline, SentimentInput};

/// Only for generic annotations. Use [`SentimentAnalysisPipelineBuilder::modernbert`].
pub type SentimentModernBert = crate::models::modernbert::SentimentModernBertModel;
