use super::model::FillMaskModel;
use crate::error::{PipelineError, Result};
use crate::pipelines::stats::PipelineStats;
use tokenizers::Tokenizer;

// ============ Output types ============

/// A single prediction from fill-mask inference.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// The predicted word/token.
    pub token: String,
    /// Confidence score (probability).
    pub score: f32,
}

/// Output from single-text `.run()`.
#[derive(Debug)]
pub struct Output {
    /// The prediction for the input text.
    pub prediction: Prediction,
    /// Execution statistics.
    pub stats: PipelineStats,
}

/// Output from batch `.run()`.
#[derive(Debug)]
pub struct BatchOutput {
    /// Predictions for each input text (may have individual failures).
    pub predictions: Vec<Result<Prediction>>,
    /// Execution statistics.
    pub stats: PipelineStats,
}

/// Output from single-text `.run_top_k()`.
#[derive(Debug)]
pub struct TopKOutput {
    /// Top k predictions for the input text.
    pub predictions: Vec<Prediction>,
    /// Execution statistics.
    pub stats: PipelineStats,
}

/// Output from batch `.run_top_k()`.
#[derive(Debug)]
pub struct BatchTopKOutput {
    /// Top k predictions for each input text (may have individual failures).
    pub predictions: Vec<Result<Vec<Prediction>>>,
    /// Execution statistics.
    pub stats: PipelineStats,
}

// ============ Input trait for type-based dispatch ============

/// Trait for fill-mask input that determines output type.
pub trait FillMaskInput<'a> {
    /// Output type for `.run()`.
    type RunOutput;
    /// Output type for `.run_top_k()`.
    type TopKOutput;

    #[doc(hidden)]
    fn into_texts(self) -> Vec<&'a str>;
    #[doc(hidden)]
    fn convert_run_output(predictions: Vec<Result<Prediction>>, stats: PipelineStats) -> Result<Self::RunOutput>;
    #[doc(hidden)]
    fn convert_top_k_output(predictions: Vec<Result<Vec<Prediction>>>, stats: PipelineStats) -> Result<Self::TopKOutput>;
}

impl<'a> FillMaskInput<'a> for &'a str {
    type RunOutput = Output;
    type TopKOutput = TopKOutput;

    fn into_texts(self) -> Vec<&'a str> {
        vec![self]
    }

    fn convert_run_output(mut predictions: Vec<Result<Prediction>>, stats: PipelineStats) -> Result<Self::RunOutput> {
        let prediction = predictions.pop()
            .ok_or_else(|| PipelineError::Unexpected("No predictions returned".into()))??;
        Ok(Output { prediction, stats })
    }

    fn convert_top_k_output(mut predictions: Vec<Result<Vec<Prediction>>>, stats: PipelineStats) -> Result<Self::TopKOutput> {
        let preds = predictions.pop()
            .ok_or_else(|| PipelineError::Unexpected("No predictions returned".into()))??;
        Ok(TopKOutput { predictions: preds, stats })
    }
}

impl<'a> FillMaskInput<'a> for &'a [&'a str] {
    type RunOutput = BatchOutput;
    type TopKOutput = BatchTopKOutput;

    fn into_texts(self) -> Vec<&'a str> {
        self.to_vec()
    }

    fn convert_run_output(predictions: Vec<Result<Prediction>>, stats: PipelineStats) -> Result<Self::RunOutput> {
        Ok(BatchOutput { predictions, stats })
    }

    fn convert_top_k_output(predictions: Vec<Result<Vec<Prediction>>>, stats: PipelineStats) -> Result<Self::TopKOutput> {
        Ok(BatchTopKOutput { predictions, stats })
    }
}

// Support fixed-size arrays
impl<'a, const N: usize> FillMaskInput<'a> for &'a [&'a str; N] {
    type RunOutput = BatchOutput;
    type TopKOutput = BatchTopKOutput;

    fn into_texts(self) -> Vec<&'a str> {
        self.as_slice().to_vec()
    }

    fn convert_run_output(predictions: Vec<Result<Prediction>>, stats: PipelineStats) -> Result<Self::RunOutput> {
        Ok(BatchOutput { predictions, stats })
    }

    fn convert_top_k_output(predictions: Vec<Result<Vec<Prediction>>>, stats: PipelineStats) -> Result<Self::TopKOutput> {
        Ok(BatchTopKOutput { predictions, stats })
    }
}

// ============ Pipeline ============

/// Pipeline for masked language modeling (fill-in-the-blank).
///
/// Predicts the most likely token(s) for a `[MASK]` placeholder in text.
///
/// Use [`FillMaskPipelineBuilder`](super::FillMaskPipelineBuilder) to construct.
///
/// # Examples
///
/// ```rust,no_run
/// # use candle_pipelines::fill_mask::{FillMaskPipelineBuilder, ModernBertSize};
/// # fn main() -> candle_pipelines::error::Result<()> {
/// let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
///
/// // Single text - returns Output with direct access
/// let output = pipeline.run("The capital of France is [MASK].")?;
/// println!("{}: {:.2}", output.prediction.token, output.prediction.score);
///
/// // Batch - returns BatchOutput with Vec
/// let output = pipeline.run(&["Paris is [MASK].", "London is [MASK]."])?;
/// for pred in output.predictions {
///     println!("{}", pred?.token);
/// }
/// # Ok(())
/// # }
/// ```
pub struct FillMaskPipeline<M: FillMaskModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: FillMaskModel> FillMaskPipeline<M> {
    /// Predict the single most likely token for each `[MASK]` position.
    ///
    /// - Single text input returns [`Output`] with direct `.prediction` access.
    /// - Batch input returns [`BatchOutput`] with `.predictions` Vec.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use candle_pipelines::fill_mask::{FillMaskPipelineBuilder, ModernBertSize};
    /// # fn main() -> candle_pipelines::error::Result<()> {
    /// # let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
    /// // Single - direct access
    /// let output = pipeline.run("The [MASK] sat on the mat.")?;
    /// println!("{}", output.prediction.token);
    ///
    /// // Batch - iterate results
    /// let output = pipeline.run(&["The [MASK] sat.", "A [MASK] barked."])?;
    /// for pred in output.predictions {
    ///     println!("{}", pred?.token);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn run<'a, I: FillMaskInput<'a>>(&self, input: I) -> Result<I::RunOutput> {
        let stats_builder = PipelineStats::start();
        let texts = input.into_texts();
        let item_count = texts.len();

        let results = self.model.predict_top_k_batch(&self.tokenizer, &texts, 1)?;

        let predictions: Vec<Result<Prediction>> = results
            .into_iter()
            .map(|result| {
                result.and_then(|mut preds| {
                    preds.pop().ok_or_else(|| {
                        PipelineError::Unexpected("Model returned no predictions".to_string())
                    }).map(|p| Prediction { token: p.word, score: p.score })
                })
            })
            .collect();

        I::convert_run_output(predictions, stats_builder.finish(item_count))
    }

    /// Predict the top `k` most likely tokens for each `[MASK]` position.
    ///
    /// - Single text input returns [`TopKOutput`] with direct `.predictions` Vec.
    /// - Batch input returns [`BatchTopKOutput`] with nested Vec.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use candle_pipelines::fill_mask::{FillMaskPipelineBuilder, ModernBertSize};
    /// # fn main() -> candle_pipelines::error::Result<()> {
    /// # let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
    /// // Single - direct Vec of predictions
    /// let output = pipeline.run_top_k("The [MASK] sat on the mat.", 5)?;
    /// for pred in &output.predictions {
    ///     println!("{}: {:.2}", pred.token, pred.score);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn run_top_k<'a, I: FillMaskInput<'a>>(&self, input: I, k: usize) -> Result<I::TopKOutput> {
        let stats_builder = PipelineStats::start();
        let texts = input.into_texts();
        let item_count = texts.len();

        let results = self.model.predict_top_k_batch(&self.tokenizer, &texts, k)?;

        let predictions: Vec<Result<Vec<Prediction>>> = results
            .into_iter()
            .map(|result| {
                result.map(|preds| {
                    preds
                        .into_iter()
                        .map(|p| Prediction { token: p.word, score: p.score })
                        .collect()
                })
            })
            .collect();

        I::convert_top_k_output(predictions, stats_builder.finish(item_count))
    }

    /// Returns the device (CPU/GPU) the model is running on.
    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
