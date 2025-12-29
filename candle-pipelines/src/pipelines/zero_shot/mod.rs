//! Zero-shot text classification pipeline.
//!
//! Classify text into categories you define at runtime, no training required.
//! Returns labels ranked by confidence score.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use candle_pipelines::zero_shot::{ZeroShotClassificationPipelineBuilder, ModernBertSize};
//!
//! # fn main() -> candle_pipelines::error::Result<()> {
//! let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
//! let labels = &["sports", "politics", "technology", "entertainment"];
//!
//! // Single text - direct access to predictions Vec
//! let output = pipeline.run("The team won the championship game!", labels)?;
//!
//! // sports: 0.87, entertainment: 0.08, politics: 0.03, technology: 0.02
//! for p in &output.predictions {
//!     println!("{}: {:.2}", p.label, p.score);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Single-Label vs Multi-Label
//!
//! **Single-label** (`run`): Scores sum to 1.0 - use when categories are mutually exclusive.
//!
//! **Multi-label** (`run_multi_label`): Independent probabilities - use when multiple labels can apply.
//!
//! ```rust,no_run
//! # use candle_pipelines::zero_shot::{ZeroShotClassificationPipelineBuilder, ModernBertSize};
//! # fn main() -> candle_pipelines::error::Result<()> {
//! # let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
//! let labels = &["urgent", "billing", "technical"];
//!
//! // This email could be both urgent AND billing-related
//! let output = pipeline.run_multi_label(
//!     "URGENT: Your payment failed, please update your card immediately!",
//!     labels,
//! )?;
//!
//! for p in &output.predictions {
//!     println!("{}: {:.2}", p.label, p.score);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Batch Inference
//!
//! Batch returns `BatchOutput` with nested Vec:
//!
//! ```rust,no_run
//! # use candle_pipelines::zero_shot::{ZeroShotClassificationPipelineBuilder, ModernBertSize};
//! # fn main() -> candle_pipelines::error::Result<()> {
//! # let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
//! let texts = &["Great goal by Messi!", "New iPhone announced", "Senate passes bill"];
//! let labels = &["sports", "tech", "politics"];
//!
//! let output = pipeline.run(texts, labels)?;
//!
//! for preds in output.predictions {
//!     let top = &preds?[0];
//!     println!("{}: {:.2}", top.label, top.score);
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
//! | ModernBERT | `Base`, `Large` | [`ZeroShotClassificationPipelineBuilder::modernbert`] |

// ============ Internal API ============

pub(crate) mod builder;
pub(crate) mod model;
pub(crate) mod pipeline;

// ============ Public API ============

pub use crate::models::ModernBertSize;
pub use crate::pipelines::stats::PipelineStats;
pub use builder::ZeroShotClassificationPipelineBuilder;
pub use pipeline::{BatchOutput, Output, Prediction, ZeroShotClassificationPipeline, ZeroShotInput};

/// Only for generic annotations. Use [`ZeroShotClassificationPipelineBuilder::modernbert`].
pub type ZeroShotModernBert = crate::models::modernbert::ZeroShotModernBertModel;
