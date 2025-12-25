//! Zero-shot text classification pipeline.
//!
//! Classify text into categories you define at runtime, no training required.
//! Returns labels ranked by confidence score.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use transformers::zero_shot::{ZeroShotClassificationPipelineBuilder, ModernBertSize};
//!
//! # fn main() -> transformers::error::Result<()> {
//! let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
//! let labels = &["sports", "politics", "technology", "entertainment"];
//!
//! let results = pipeline.classify("The team won the championship game!", labels)?;
//!
//! // sports: 0.87, entertainment: 0.08, politics: 0.03, technology: 0.02
//! for r in results {
//!     println!("{}: {:.2}", r.label, r.score);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Single-Label vs Multi-Label
//!
//! **Single-label** (`classify`): Scores sum to 1.0 - use when categories are mutually exclusive.
//!
//! **Multi-label** (`classify_multi_label`): Independent probabilities - use when multiple labels can apply.
//!
//! ```rust,no_run
//! # use transformers::zero_shot::{ZeroShotClassificationPipelineBuilder, ModernBertSize};
//! # fn main() -> transformers::error::Result<()> {
//! # let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
//! let labels = &["urgent", "billing", "technical"];
//!
//! // This email could be both urgent AND billing-related
//! let results = pipeline.classify_multi_label(
//!     "URGENT: Your payment failed, please update your card immediately!",
//!     labels,
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! # Batch Inference
//!
//! ```rust,no_run
//! # use transformers::zero_shot::{ZeroShotClassificationPipelineBuilder, ModernBertSize};
//! # fn main() -> transformers::error::Result<()> {
//! # let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
//! let texts = &["Great goal by Messi!", "New iPhone announced", "Senate passes bill"];
//! let labels = &["sports", "tech", "politics"];
//!
//! let results = pipeline.classify_batch(texts, labels)?;
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
pub use builder::ZeroShotClassificationPipelineBuilder;
pub use model::LabelScores;
pub use pipeline::ZeroShotClassificationPipeline;
