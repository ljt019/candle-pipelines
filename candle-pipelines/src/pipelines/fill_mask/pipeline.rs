use super::model::FillMaskModel;
use crate::error::{PipelineError, Result};
use tokenizers::Tokenizer;

/// A single prediction from fill-mask inference.
#[derive(Debug, Clone)]
pub struct FillMaskPrediction {
    /// The predicted word/token.
    pub word: String,
    /// Confidence score (probability).
    pub score: f32,
}

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
/// let prediction = pipeline.predict("The capital of France is [MASK].")?;
/// println!("{}: {:.2}", prediction.word, prediction.score);
/// # Ok(())
/// # }
/// ```
pub struct FillMaskPipeline<M: FillMaskModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: FillMaskModel> FillMaskPipeline<M> {
    /// Predict the single most likely token for the `[MASK]` position.
    ///
    /// # Errors
    ///
    /// Returns an error if the input has no `[MASK]` token or tokenization fails.
    pub fn predict(&self, text: &str) -> Result<FillMaskPrediction> {
        let predictions = self.predict_top_k(text, 1)?;
        predictions
            .into_iter()
            .next()
            .ok_or_else(|| PipelineError::Unexpected("Model returned no predictions".to_string()))
    }

    /// Predict the most likely token for multiple texts.
    ///
    /// Each text must contain exactly one `[MASK]` token.
    pub fn predict_batch(&self, texts: &[&str]) -> Result<Vec<Result<FillMaskPrediction>>> {
        let batched = self.predict_top_k_batch(texts, 1)?;
        Ok(batched
            .into_iter()
            .map(|result| {
                result.and_then(|preds| {
                    preds.into_iter().next().ok_or_else(|| {
                        PipelineError::Unexpected("Model returned no predictions".to_string())
                    })
                })
            })
            .collect::<Vec<_>>())
    }

    /// Predict the top `k` most likely tokens for the `[MASK]` position.
    pub fn predict_top_k(&self, text: &str, k: usize) -> Result<Vec<FillMaskPrediction>> {
        self.model.predict_top_k(&self.tokenizer, text, k)
    }

    /// Predict the top `k` tokens for multiple texts.
    pub fn predict_top_k_batch(
        &self,
        texts: &[&str],
        k: usize,
    ) -> Result<Vec<Result<Vec<FillMaskPrediction>>>> {
        self.model.predict_top_k_batch(&self.tokenizer, texts, k)
    }

    /// Returns the device (CPU/GPU) the model is running on.
    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
