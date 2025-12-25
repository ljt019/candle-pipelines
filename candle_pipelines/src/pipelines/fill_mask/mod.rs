//! Masked language modeling pipeline.
//!
//! Fill-mask predicts the most likely word(s) for a `[MASK]` token in text.
//! Returns the predicted word and a confidence score.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use candle_pipelines::fill_mask::{FillMaskPipelineBuilder, ModernBertSize};
//!
//! # fn main() -> candle_pipelines::error::Result<()> {
//! let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
//! let prediction = pipeline.predict("The [MASK] of France is Paris.")?;
//!
//! // prediction: capital (confidence: 0.99)
//! println!("prediction: {} (confidence: {:.2})", prediction.word, prediction.score);
//! # Ok(())
//! # }
//! ```
//!
//! # Top-K Predictions
//!
//! Get multiple candidate words ranked by confidence:
//!
//! ```rust,no_run
//! # use candle_pipelines::fill_mask::{FillMaskPipelineBuilder, ModernBertSize};
//! # fn main() -> candle_pipelines::error::Result<()> {
//! # let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
//! let top_3 = pipeline.predict_top_k("I love my [MASK] car.", 3)?;
//!
//! for p in top_3 {
//!     println!("{}: {:.2}", p.word, p.score);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Batch Inference
//!
//! Process multiple texts efficiently:
//!
//! ```rust,no_run
//! # use candle_pipelines::fill_mask::{FillMaskPipelineBuilder, ModernBertSize};
//! # fn main() -> candle_pipelines::error::Result<()> {
//! # let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
//! let texts = &["The [MASK] is shining.", "She plays the [MASK] beautifully."];
//!
//! let results = pipeline.predict_batch(texts)?;
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
//! | ModernBERT | `Base`, `Large` | [`FillMaskPipelineBuilder::modernbert`] |

// ============ Internal API ============

pub(crate) mod builder;
pub(crate) mod model;
pub(crate) mod pipeline;

// ============ Public API ============

pub use crate::models::ModernBertSize;
pub use builder::FillMaskPipelineBuilder;
pub use pipeline::{FillMaskPipeline, FillMaskPrediction};

/// Only for generic annotations. Use [`FillMaskPipelineBuilder::modernbert`].
pub type FillMaskModernBert = crate::models::modernbert::FillMaskModernBertModel;
