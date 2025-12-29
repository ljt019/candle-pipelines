use super::model::ZeroShotClassificationModel;
use crate::error::{PipelineError, Result};
use crate::pipelines::stats::PipelineStats;
use tokenizers::Tokenizer;

// ============ Output types ============

/// A label with confidence score.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Label name.
    pub label: String,
    /// Confidence score (0.0 to 1.0).
    pub score: f32,
}

/// Single-text output from `run()` or `run_multi_label()`.
#[derive(Debug)]
pub struct Output {
    /// All labels ranked by confidence.
    pub predictions: Vec<Prediction>,
    /// Execution statistics.
    pub stats: PipelineStats,
}

/// Single result in batch output.
#[derive(Debug)]
pub struct BatchResult {
    /// Input text.
    pub text: String,
    /// Predictions or error for this input.
    pub predictions: Result<Vec<Prediction>>,
}

/// Batch output from `run()` or `run_multi_label()`.
#[derive(Debug)]
pub struct BatchOutput {
    /// Results for each input.
    pub results: Vec<BatchResult>,
    /// Execution statistics.
    pub stats: PipelineStats,
}

// ============ Input trait for type-based dispatch ============

#[doc(hidden)]
pub trait ZeroShotInput<'a> {
    /// Output type.
    type Output;

    #[doc(hidden)]
    fn into_texts(self) -> Vec<&'a str>;
    #[doc(hidden)]
    fn convert_output(
        texts: Vec<&'a str>,
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
        _texts: Vec<&'a str>,
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
        texts: Vec<&'a str>,
        predictions: Vec<Result<Vec<Prediction>>>,
        stats: PipelineStats,
    ) -> Result<Self::Output> {
        let results = texts
            .into_iter()
            .zip(predictions)
            .map(|(text, predictions)| BatchResult {
                text: text.to_string(),
                predictions,
            })
            .collect();
        Ok(BatchOutput { results, stats })
    }
}

impl<'a, const N: usize> ZeroShotInput<'a> for &'a [&'a str; N] {
    type Output = BatchOutput;

    fn into_texts(self) -> Vec<&'a str> {
        self.as_slice().to_vec()
    }

    fn convert_output(
        texts: Vec<&'a str>,
        predictions: Vec<Result<Vec<Prediction>>>,
        stats: PipelineStats,
    ) -> Result<Self::Output> {
        let results = texts
            .into_iter()
            .zip(predictions)
            .map(|(text, predictions)| BatchResult {
                text: text.to_string(),
                predictions,
            })
            .collect();
        Ok(BatchOutput { results, stats })
    }
}

// ============ Pipeline ============

/// Classifies text into arbitrary categories without training.
///
/// Construct with [`ZeroShotClassificationPipelineBuilder`](super::ZeroShotClassificationPipelineBuilder).
///
/// # Examples
///
/// ```rust,no_run
/// # use candle_pipelines::zero_shot::{ZeroShotClassificationPipelineBuilder, ModernBertSize};
/// # fn main() -> candle_pipelines::error::Result<()> {
/// let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
/// let labels = &["sports", "politics", "technology"];
///
/// // Single text - direct access
/// let output = pipeline.run("The team won the championship!", labels)?;
/// println!("{}: {:.2}", output.predictions[0].label, output.predictions[0].score);
///
/// // Batch - results include input text
/// let output = pipeline.run(&["Sports news", "Tech update"], labels)?;
/// for r in output.results {
///     println!("{} → {}", r.text, r.predictions?[0].label);
/// }
/// # Ok(())
/// # }
/// ```
pub struct ZeroShotClassificationPipeline<M: ZeroShotClassificationModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: ZeroShotClassificationModel> ZeroShotClassificationPipeline<M> {
    /// Classify into one label (scores sum to 1.0).
    ///
    /// Single input → [`Output`], batch → [`BatchOutput`].
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use candle_pipelines::zero_shot::{ZeroShotClassificationPipelineBuilder, ModernBertSize};
    /// # fn main() -> candle_pipelines::error::Result<()> {
    /// # let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
    /// let labels = &["sports", "politics", "technology"];
    ///
    /// // Single
    /// let output = pipeline.run("The team won!", labels)?;
    /// println!("{}", output.predictions[0].label);
    ///
    /// // Batch
    /// let output = pipeline.run(&["Sports news", "Tech update"], labels)?;
    /// for r in output.results {
    ///     println!("{} → {}", r.text, r.predictions?[0].label);
    /// }
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

    /// Classify with independent probabilities (scores don't sum to 1.0).
    ///
    /// Use when text can match multiple categories.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use candle_pipelines::zero_shot::{ZeroShotClassificationPipelineBuilder, ModernBertSize};
    /// # fn main() -> candle_pipelines::error::Result<()> {
    /// # let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
    /// let labels = &["urgent", "billing", "technical"];
    ///
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

        I::convert_output(texts, predictions, stats_builder.finish(item_count))
    }

    /// Returns the device (CPU/GPU) the model is running on.
    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
