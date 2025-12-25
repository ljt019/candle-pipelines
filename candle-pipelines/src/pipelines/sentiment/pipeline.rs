use super::model::SentimentAnalysisModel;
use crate::error::Result;
use tokenizers::Tokenizer;

/// A sentiment prediction with label and confidence score.
#[derive(Debug, Clone)]
pub struct SentimentResult {
    /// The predicted sentiment (e.g., "positive", "negative", "neutral").
    pub label: String,
    /// Confidence score (0.0 to 1.0).
    pub score: f32,
}

/// Pipeline for sentiment analysis.
///
/// Classifies text as positive, negative, or neutral with a confidence score.
///
/// Use [`SentimentAnalysisPipelineBuilder`](super::SentimentAnalysisPipelineBuilder) to construct.
///
/// # Examples
///
/// ```rust,no_run
/// # use candle_pipelines::sentiment::{SentimentAnalysisPipelineBuilder, ModernBertSize};
/// # fn main() -> candle_pipelines::error::Result<()> {
/// let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
///
/// let result = pipeline.predict("I love this product!")?;
/// println!("{}: {:.2}", result.label, result.score);
/// # Ok(())
/// # }
/// ```
pub struct SentimentAnalysisPipeline<M: SentimentAnalysisModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: SentimentAnalysisModel> SentimentAnalysisPipeline<M> {
    /// Predict the sentiment of text.
    pub fn predict(&self, text: &str) -> Result<SentimentResult> {
        self.model.predict_with_score(&self.tokenizer, text)
    }

    /// Predict sentiment for multiple texts.
    pub fn predict_batch(&self, texts: &[&str]) -> Result<Vec<Result<SentimentResult>>> {
        self.model.predict_with_score_batch(&self.tokenizer, texts)
    }

    /// Returns the device (CPU/GPU) the model is running on.
    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
