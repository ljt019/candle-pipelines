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
//!
//! // Single text - direct access to prediction
//! let output = pipeline.run("The [MASK] of France is Paris.")?;
//! println!("{}: {:.2}", output.prediction.token, output.prediction.score);
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
//! let output = pipeline.run_top_k("I love my [MASK] car.", 3)?;
//!
//! for pred in &output.predictions {
//!     println!("{}: {:.2}", pred.token, pred.score);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Batch Inference
//!
//! Process multiple texts at once (returns `BatchOutput`):
//!
//! ```rust,no_run
//! # use candle_pipelines::fill_mask::{FillMaskPipelineBuilder, ModernBertSize};
//! # fn main() -> candle_pipelines::error::Result<()> {
//! # let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
//! let output = pipeline.run(&["The [MASK] is shining.", "She plays the [MASK] beautifully."])?;
//!
//! for pred in output.predictions {
//!     println!("{}", pred?.token);
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
//! | ModernBERT | `Base`, `Large` | [`FillMaskPipelineBuilder::modernbert`] |

// ============ Internal API ============

pub(crate) mod builder;
pub(crate) mod model;
pub(crate) mod pipeline;

// ============ Public API ============

pub use crate::models::ModernBertSize;
pub use crate::pipelines::stats::PipelineStats;
pub use builder::FillMaskPipelineBuilder;
pub use pipeline::{
    BatchOutput, BatchTopKOutput, FillMaskInput, FillMaskPipeline, Output, Prediction, TopKOutput,
};

/// Only for generic annotations. Use [`FillMaskPipelineBuilder::modernbert`].
pub type FillMaskModernBert = crate::models::modernbert::FillMaskModernBertModel;
