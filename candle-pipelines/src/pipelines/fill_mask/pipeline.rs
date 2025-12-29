use super::model::FillMaskModel;
use crate::error::{PipelineError, Result};
use crate::pipelines::stats::PipelineStats;
use tokenizers::Tokenizer;

// ============ Output types ============

/// A predicted token with confidence score.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// Predicted word/token.
    pub token: String,
    /// Confidence score (0.0 to 1.0).
    pub score: f32,
}

/// Single-text output from `run()`.
#[derive(Debug)]
pub struct Output {
    /// Predicted token.
    pub prediction: Prediction,
    /// Execution statistics.
    pub stats: PipelineStats,
}

/// Single result in batch output.
#[derive(Debug)]
pub struct BatchResult {
    /// Input text.
    pub text: String,
    /// Prediction or error for this input.
    pub prediction: Result<Prediction>,
}

/// Batch output from `run()`.
#[derive(Debug)]
pub struct BatchOutput {
    /// Results for each input.
    pub results: Vec<BatchResult>,
    /// Execution statistics.
    pub stats: PipelineStats,
}

/// Single-text output from `run_top_k()`.
#[derive(Debug)]
pub struct TopKOutput {
    /// Top k predictions.
    pub predictions: Vec<Prediction>,
    /// Execution statistics.
    pub stats: PipelineStats,
}

/// Single result in batch top-k output.
#[derive(Debug)]
pub struct BatchTopKResult {
    /// Input text.
    pub text: String,
    /// Top-k predictions or error for this input.
    pub predictions: Result<Vec<Prediction>>,
}

/// Batch output from `run_top_k()`.
#[derive(Debug)]
pub struct BatchTopKOutput {
    /// Results for each input.
    pub results: Vec<BatchTopKResult>,
    /// Execution statistics.
    pub stats: PipelineStats,
}

// ============ Input trait for type-based dispatch ============

#[doc(hidden)]
pub trait FillMaskInput<'a> {
    /// Output type for `.run()`.
    type RunOutput;
    /// Output type for `.run_top_k()`.
    type TopKOutput;

    #[doc(hidden)]
    fn into_texts(self) -> Vec<&'a str>;
    #[doc(hidden)]
    fn convert_run_output(
        texts: Vec<&'a str>,
        predictions: Vec<Result<Prediction>>,
        stats: PipelineStats,
    ) -> Result<Self::RunOutput>;
    #[doc(hidden)]
    fn convert_top_k_output(
        texts: Vec<&'a str>,
        predictions: Vec<Result<Vec<Prediction>>>,
        stats: PipelineStats,
    ) -> Result<Self::TopKOutput>;
}

impl<'a> FillMaskInput<'a> for &'a str {
    type RunOutput = Output;
    type TopKOutput = TopKOutput;

    fn into_texts(self) -> Vec<&'a str> {
        vec![self]
    }

    fn convert_run_output(
        _texts: Vec<&'a str>,
        mut predictions: Vec<Result<Prediction>>,
        stats: PipelineStats,
    ) -> Result<Self::RunOutput> {
        let prediction = predictions
            .pop()
            .ok_or_else(|| PipelineError::Unexpected("No predictions returned".into()))??;
        Ok(Output { prediction, stats })
    }

    fn convert_top_k_output(
        _texts: Vec<&'a str>,
        mut predictions: Vec<Result<Vec<Prediction>>>,
        stats: PipelineStats,
    ) -> Result<Self::TopKOutput> {
        let preds = predictions
            .pop()
            .ok_or_else(|| PipelineError::Unexpected("No predictions returned".into()))??;
        Ok(TopKOutput {
            predictions: preds,
            stats,
        })
    }
}

impl<'a> FillMaskInput<'a> for &'a [&'a str] {
    type RunOutput = BatchOutput;
    type TopKOutput = BatchTopKOutput;

    fn into_texts(self) -> Vec<&'a str> {
        self.to_vec()
    }

    fn convert_run_output(
        texts: Vec<&'a str>,
        predictions: Vec<Result<Prediction>>,
        stats: PipelineStats,
    ) -> Result<Self::RunOutput> {
        let results = texts
            .into_iter()
            .zip(predictions)
            .map(|(text, prediction)| BatchResult {
                text: text.to_string(),
                prediction,
            })
            .collect();
        Ok(BatchOutput { results, stats })
    }

    fn convert_top_k_output(
        texts: Vec<&'a str>,
        predictions: Vec<Result<Vec<Prediction>>>,
        stats: PipelineStats,
    ) -> Result<Self::TopKOutput> {
        let results = texts
            .into_iter()
            .zip(predictions)
            .map(|(text, predictions)| BatchTopKResult {
                text: text.to_string(),
                predictions,
            })
            .collect();
        Ok(BatchTopKOutput { results, stats })
    }
}

// Support fixed-size arrays
impl<'a, const N: usize> FillMaskInput<'a> for &'a [&'a str; N] {
    type RunOutput = BatchOutput;
    type TopKOutput = BatchTopKOutput;

    fn into_texts(self) -> Vec<&'a str> {
        self.as_slice().to_vec()
    }

    fn convert_run_output(
        texts: Vec<&'a str>,
        predictions: Vec<Result<Prediction>>,
        stats: PipelineStats,
    ) -> Result<Self::RunOutput> {
        let results = texts
            .into_iter()
            .zip(predictions)
            .map(|(text, prediction)| BatchResult {
                text: text.to_string(),
                prediction,
            })
            .collect();
        Ok(BatchOutput { results, stats })
    }

    fn convert_top_k_output(
        texts: Vec<&'a str>,
        predictions: Vec<Result<Vec<Prediction>>>,
        stats: PipelineStats,
    ) -> Result<Self::TopKOutput> {
        let results = texts
            .into_iter()
            .zip(predictions)
            .map(|(text, predictions)| BatchTopKResult {
                text: text.to_string(),
                predictions,
            })
            .collect();
        Ok(BatchTopKOutput { results, stats })
    }
}

// ============ Pipeline ============

/// Predicts tokens for `[MASK]` placeholders in text.
///
/// Construct with [`FillMaskPipelineBuilder`](super::FillMaskPipelineBuilder).
///
/// # Examples
///
/// ```rust,no_run
/// # use candle_pipelines::fill_mask::{FillMaskPipelineBuilder, ModernBertSize};
/// # fn main() -> candle_pipelines::error::Result<()> {
/// let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
///
/// // Single text - direct access
/// let output = pipeline.run("The capital of France is [MASK].")?;
/// println!("{}: {:.2}", output.prediction.token, output.prediction.score);
///
/// // Batch - results include input text
/// let output = pipeline.run(&["Paris is [MASK].", "London is [MASK]."])?;
/// for r in output.results {
///     println!("{} → {}", r.text, r.prediction?.token);
/// }
/// # Ok(())
/// # }
/// ```
pub struct FillMaskPipeline<M: FillMaskModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: FillMaskModel> FillMaskPipeline<M> {
    /// Predict most likely token for `[MASK]`.
    ///
    /// Single input → [`Output`], batch → [`BatchOutput`].
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use candle_pipelines::fill_mask::{FillMaskPipelineBuilder, ModernBertSize};
    /// # fn main() -> candle_pipelines::error::Result<()> {
    /// # let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
    /// // Single
    /// let output = pipeline.run("The [MASK] sat on the mat.")?;
    /// println!("{}", output.prediction.token);
    ///
    /// // Batch
    /// let output = pipeline.run(&["The [MASK] sat.", "A [MASK] barked."])?;
    /// for r in output.results {
    ///     println!("{} → {}", r.text, r.prediction?.token);
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
                    preds
                        .pop()
                        .ok_or_else(|| {
                            PipelineError::Unexpected("Model returned no predictions".to_string())
                        })
                        .map(|p| Prediction {
                            token: p.word,
                            score: p.score,
                        })
                })
            })
            .collect();

        I::convert_run_output(texts, predictions, stats_builder.finish(item_count))
    }

    /// Predict top `k` tokens for `[MASK]`.
    ///
    /// Single input → [`TopKOutput`], batch → [`BatchTopKOutput`].
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use candle_pipelines::fill_mask::{FillMaskPipelineBuilder, ModernBertSize};
    /// # fn main() -> candle_pipelines::error::Result<()> {
    /// # let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
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
                        .map(|p| Prediction {
                            token: p.word,
                            score: p.score,
                        })
                        .collect()
                })
            })
            .collect();

        I::convert_top_k_output(texts, predictions, stats_builder.finish(item_count))
    }

    /// Returns the device (CPU/GPU) the model is running on.
    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
