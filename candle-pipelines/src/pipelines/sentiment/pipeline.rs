use super::model::SentimentAnalysisModel;
use crate::error::{PipelineError, Result};
use crate::pipelines::stats::PipelineStats;
use tokenizers::Tokenizer;

// ============ Output types ============

/// A sentiment prediction with label and confidence score.
#[derive(Debug, Clone)]
pub struct Prediction {
    /// The predicted sentiment (e.g., "positive", "negative", "neutral").
    pub label: String,
    /// Confidence score (0.0 to 1.0).
    pub score: f32,
}

/// Single-text output from `run()`.
#[derive(Debug)]
pub struct Output {
    /// Sentiment prediction.
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

// ============ Input trait for type-based dispatch ============

#[doc(hidden)]
pub trait SentimentInput<'a> {
    /// Output type for `.run()`.
    type Output;

    #[doc(hidden)]
    fn into_texts(self) -> Vec<&'a str>;
    #[doc(hidden)]
    fn convert_output(
        texts: Vec<&'a str>,
        predictions: Vec<Result<Prediction>>,
        stats: PipelineStats,
    ) -> Result<Self::Output>;
}

impl<'a> SentimentInput<'a> for &'a str {
    type Output = Output;

    fn into_texts(self) -> Vec<&'a str> {
        vec![self]
    }

    fn convert_output(
        _texts: Vec<&'a str>,
        mut predictions: Vec<Result<Prediction>>,
        stats: PipelineStats,
    ) -> Result<Self::Output> {
        let prediction = predictions
            .pop()
            .ok_or_else(|| PipelineError::Unexpected("No predictions returned".into()))??;
        Ok(Output { prediction, stats })
    }
}

impl<'a> SentimentInput<'a> for &'a [&'a str] {
    type Output = BatchOutput;

    fn into_texts(self) -> Vec<&'a str> {
        self.to_vec()
    }

    fn convert_output(
        texts: Vec<&'a str>,
        predictions: Vec<Result<Prediction>>,
        stats: PipelineStats,
    ) -> Result<Self::Output> {
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
}

impl<'a, const N: usize> SentimentInput<'a> for &'a [&'a str; N] {
    type Output = BatchOutput;

    fn into_texts(self) -> Vec<&'a str> {
        self.as_slice().to_vec()
    }

    fn convert_output(
        texts: Vec<&'a str>,
        predictions: Vec<Result<Prediction>>,
        stats: PipelineStats,
    ) -> Result<Self::Output> {
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
}

// ============ Pipeline ============

/// Classifies text sentiment (positive, negative, neutral).
///
/// Construct with [`SentimentAnalysisPipelineBuilder`](super::SentimentAnalysisPipelineBuilder).
///
/// # Examples
///
/// ```rust,no_run
/// # use candle_pipelines::sentiment::{SentimentAnalysisPipelineBuilder, ModernBertSize};
/// # fn main() -> candle_pipelines::error::Result<()> {
/// let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
///
/// // Single text - direct access
/// let output = pipeline.run("I love this product!")?;
/// println!("{}: {:.2}", output.prediction.label, output.prediction.score);
///
/// // Batch - results include input text
/// let output = pipeline.run(&["Great!", "Terrible."])?;
/// for r in output.results {
///     println!("{} → {}", r.text, r.prediction?.label);
/// }
/// # Ok(())
/// # }
/// ```
pub struct SentimentAnalysisPipeline<M: SentimentAnalysisModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: SentimentAnalysisModel> SentimentAnalysisPipeline<M> {
    /// Analyze text sentiment.
    ///
    /// Single input → [`Output`], batch → [`BatchOutput`].
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use candle_pipelines::sentiment::{SentimentAnalysisPipelineBuilder, ModernBertSize};
    /// # fn main() -> candle_pipelines::error::Result<()> {
    /// # let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
    /// // Single
    /// let output = pipeline.run("I love this!")?;
    /// println!("{}", output.prediction.label);
    ///
    /// // Batch
    /// let output = pipeline.run(&["Great!", "Awful."])?;
    /// for r in output.results {
    ///     println!("{} → {}", r.text, r.prediction?.label);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn run<'a, I: SentimentInput<'a>>(&self, input: I) -> Result<I::Output> {
        let stats_builder = PipelineStats::start();
        let texts = input.into_texts();
        let item_count = texts.len();

        let results = self
            .model
            .predict_with_score_batch(&self.tokenizer, &texts)?;

        let predictions: Vec<Result<Prediction>> = results
            .into_iter()
            .map(|result| {
                result.map(|r| Prediction {
                    label: r.label,
                    score: r.score,
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
