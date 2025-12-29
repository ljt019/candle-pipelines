use super::model::ZeroShotClassificationModel;
use crate::error::{PipelineError, Result};
use crate::pipelines::stats::PipelineStats;
use tokenizers::Tokenizer;

// ============ Output types ============

/// A single classification result with label and confidence score.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// The predicted label.
    pub label: String,
    /// Confidence score (0.0 to 1.0).
    pub score: f32,
}

/// Output from single-text `.run()` or `.run_multi_label()`.
#[derive(Debug)]
pub struct Output {
    /// All labels with their scores, sorted by confidence.
    pub predictions: Vec<Prediction>,
    /// Execution statistics.
    pub stats: PipelineStats,
}

/// Output from batch `.run()` or `.run_multi_label()`.
#[derive(Debug)]
pub struct BatchOutput {
    /// Predictions for each input text (may have individual failures).
    pub predictions: Vec<Result<Vec<Prediction>>>,
    /// Execution statistics.
    pub stats: PipelineStats,
}

// ============ Input trait for type-based dispatch ============

/// Trait for zero-shot input that determines output type.
pub trait ZeroShotInput<'a> {
    /// Output type.
    type Output;

    #[doc(hidden)]
    fn into_texts(self) -> Vec<&'a str>;
    #[doc(hidden)]
    fn convert_output(
        predictions: Vec<Result<Vec<Prediction>>>,
        stats: PipelineStats,
    ) -> Result<Self::Output>;
}

impl<'a> ZeroShotInput<'a> for &'a str {
    type Output = Output;

    fn into_texts(self) -> Vec<&'a str> {
        vec![self]
    }

    fn convert_output(
        mut predictions: Vec<Result<Vec<Prediction>>>,
        stats: PipelineStats,
    ) -> Result<Self::Output> {
        let preds = predictions
            .pop()
            .ok_or_else(|| PipelineError::Unexpected("No predictions returned".into()))??;
        Ok(Output {
            predictions: preds,
            stats,
        })
    }
}

impl<'a> ZeroShotInput<'a> for &'a [&'a str] {
    type Output = BatchOutput;

    fn into_texts(self) -> Vec<&'a str> {
        self.to_vec()
    }

    fn convert_output(
        predictions: Vec<Result<Vec<Prediction>>>,
        stats: PipelineStats,
    ) -> Result<Self::Output> {
        Ok(BatchOutput { predictions, stats })
    }
}

impl<'a, const N: usize> ZeroShotInput<'a> for &'a [&'a str; N] {
    type Output = BatchOutput;

    fn into_texts(self) -> Vec<&'a str> {
        self.as_slice().to_vec()
    }

    fn convert_output(
        predictions: Vec<Result<Vec<Prediction>>>,
        stats: PipelineStats,
    ) -> Result<Self::Output> {
        Ok(BatchOutput { predictions, stats })
    }
}

// ============ Pipeline ============

/// Pipeline for zero-shot text classification.
///
/// Classify text into arbitrary categories without task-specific training.
/// Labels are provided at inference time.
///
/// Use [`ZeroShotClassificationPipelineBuilder`](super::ZeroShotClassificationPipelineBuilder) to construct.
///
/// # Examples
///
/// ```rust,no_run
/// # use candle_pipelines::zero_shot::{ZeroShotClassificationPipelineBuilder, ModernBertSize};
/// # fn main() -> candle_pipelines::error::Result<()> {
/// let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
/// let labels = &["sports", "politics", "technology"];
///
/// // Single text - direct access to predictions Vec
/// let output = pipeline.run("The team won the championship!", labels)?;
/// println!("{}: {:.2}", output.predictions[0].label, output.predictions[0].score);
///
/// // Batch - iterate results
/// let output = pipeline.run(&["Sports news", "Tech update"], labels)?;
/// for preds in output.predictions {
///     println!("{}", preds?[0].label);
/// }
/// # Ok(())
/// # }
/// ```
pub struct ZeroShotClassificationPipeline<M: ZeroShotClassificationModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: ZeroShotClassificationModel> ZeroShotClassificationPipeline<M> {
    /// Classify text into one of the candidate labels (single-label, scores sum to 1.0).
    ///
    /// - Single text input returns [`Output`] with direct `.predictions` access.
    /// - Batch input returns [`BatchOutput`] with nested Vec.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use candle_pipelines::zero_shot::{ZeroShotClassificationPipelineBuilder, ModernBertSize};
    /// # fn main() -> candle_pipelines::error::Result<()> {
    /// # let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
    /// let labels = &["sports", "politics", "technology"];
    ///
    /// // Single - direct access
    /// let output = pipeline.run("The team won!", labels)?;
    /// println!("{}", output.predictions[0].label);
    ///
    /// // Batch
    /// let output = pipeline.run(&["Sports news", "Tech update"], labels)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn run<'a, I: ZeroShotInput<'a>>(
        &self,
        input: I,
        candidate_labels: &[&str],
    ) -> Result<I::Output> {
        self.run_internal(input, candidate_labels, false)
    }

    /// Classify text with independent label probabilities (multi-label, scores don't sum to 1.0).
    ///
    /// Use this when a text can belong to multiple categories simultaneously.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use candle_pipelines::zero_shot::{ZeroShotClassificationPipelineBuilder, ModernBertSize};
    /// # fn main() -> candle_pipelines::error::Result<()> {
    /// # let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
    /// let labels = &["urgent", "billing", "technical"];
    ///
    /// // A support ticket might be both urgent AND technical
    /// let output = pipeline.run_multi_label("Critical server error!", labels)?;
    /// for pred in &output.predictions {
    ///     println!("{}: {:.2}", pred.label, pred.score);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn run_multi_label<'a, I: ZeroShotInput<'a>>(
        &self,
        input: I,
        candidate_labels: &[&str],
    ) -> Result<I::Output> {
        self.run_internal(input, candidate_labels, true)
    }

    fn run_internal<'a, I: ZeroShotInput<'a>>(
        &self,
        input: I,
        candidate_labels: &[&str],
        multi_label: bool,
    ) -> Result<I::Output> {
        let stats_builder = PipelineStats::start();
        let texts = input.into_texts();
        let item_count = texts.len();

        let results = if multi_label {
            self.model
                .predict_multi_label_batch(&self.tokenizer, &texts, candidate_labels)?
        } else {
            self.model
                .predict_batch(&self.tokenizer, &texts, candidate_labels)?
        };

        let predictions: Vec<Result<Vec<Prediction>>> = results
            .into_iter()
            .map(|result| {
                result.map(|entries| {
                    entries
                        .into_iter()
                        .map(|(label, score)| Prediction { label, score })
                        .collect()
                })
            })
            .collect();

        I::convert_output(predictions, stats_builder.finish(item_count))
    }

    /// Returns the device (CPU/GPU) the model is running on.
    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
