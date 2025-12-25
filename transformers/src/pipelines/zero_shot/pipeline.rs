use super::model::ZeroShotClassificationModel;
use crate::error::Result;
use tokenizers::Tokenizer;

/// A single classification result with label and confidence score.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// The predicted label.
    pub label: String,
    /// Confidence score (0.0 to 1.0).
    pub score: f32,
}

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
/// # use transformers::zero_shot::{ZeroShotClassificationPipelineBuilder, ModernBertSize};
/// # fn main() -> transformers::error::Result<()> {
/// let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
/// let labels = &["sports", "politics", "technology"];
///
/// let results = pipeline.classify("The team won the championship!", labels)?;
/// println!("{}: {:.2}", results[0].label, results[0].score);
/// # Ok(())
/// # }
/// ```
pub struct ZeroShotClassificationPipeline<M: ZeroShotClassificationModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: ZeroShotClassificationModel> ZeroShotClassificationPipeline<M> {
    /// Classify text into one of the candidate labels (single-label, scores sum to 1.0).
    pub fn classify(
        &self,
        text: &str,
        candidate_labels: &[&str],
    ) -> Result<Vec<ClassificationResult>> {
        let results = self
            .model
            .predict(&self.tokenizer, text, candidate_labels)?;
        Ok(results
            .into_iter()
            .map(|(label, score)| ClassificationResult { label, score })
            .collect())
    }

    /// Classify multiple texts (single-label mode).
    pub fn classify_batch(
        &self,
        texts: &[&str],
        candidate_labels: &[&str],
    ) -> Result<Vec<Result<Vec<ClassificationResult>>>> {
        let results = self
            .model
            .predict_batch(&self.tokenizer, texts, candidate_labels)?;

        Ok(results
            .into_iter()
            .map(|res| {
                res.map(|entries| {
                    entries
                        .into_iter()
                        .map(|(label, score)| ClassificationResult { label, score })
                        .collect()
                })
            })
            .collect())
    }

    /// Classify text with independent label probabilities (multi-label, scores don't sum to 1.0).
    pub fn classify_multi_label(
        &self,
        text: &str,
        candidate_labels: &[&str],
    ) -> Result<Vec<ClassificationResult>> {
        let results = self
            .model
            .predict_multi_label(&self.tokenizer, text, candidate_labels)?;
        Ok(results
            .into_iter()
            .map(|(label, score)| ClassificationResult { label, score })
            .collect())
    }

    /// Classify multiple texts (multi-label mode).
    pub fn classify_multi_label_batch(
        &self,
        texts: &[&str],
        candidate_labels: &[&str],
    ) -> Result<Vec<Result<Vec<ClassificationResult>>>> {
        let results =
            self.model
                .predict_multi_label_batch(&self.tokenizer, texts, candidate_labels)?;

        Ok(results
            .into_iter()
            .map(|res| {
                res.map(|entries| {
                    entries
                        .into_iter()
                        .map(|(label, score)| ClassificationResult { label, score })
                        .collect()
                })
            })
            .collect())
    }

    /// Returns the device (CPU/GPU) the model is running on.
    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
