use super::model::FillMaskModel;
use crate::{Result, TransformersError};
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct FillMaskPrediction {
    pub word: String,
    pub score: f32,
}

pub struct FillMaskPipeline<M: FillMaskModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: FillMaskModel> FillMaskPipeline<M> {
    /// Return the top prediction for the masked token
    pub fn predict(&self, text: &str) -> Result<FillMaskPrediction> {
        let predictions = self.predict_top_k(text, 1)?;
        predictions
            .into_iter()
            .next()
            .ok_or_else(|| TransformersError::Generation("No predictions returned".to_string()))
    }

    /// Return the top prediction for each input in the batch.
    pub fn predict_batch(&self, texts: &[&str]) -> Result<Vec<Result<FillMaskPrediction>>> {
        let batched = self.predict_top_k_batch(texts, 1)?;
        Ok(batched
            .into_iter()
            .map(|result| {
                result.and_then(|preds| {
                    preds.into_iter().next().ok_or_else(|| {
                        TransformersError::Generation("No predictions returned".to_string())
                    })
                })
            })
            .collect::<Vec<_>>())
    }

    /// Return top-k predictions with scores for ranking/choice
    pub fn predict_top_k(&self, text: &str, k: usize) -> Result<Vec<FillMaskPrediction>> {
        self.model.predict_top_k(&self.tokenizer, text, k)
    }

    /// Return the top-k predictions for each input in the batch.
    pub fn predict_top_k_batch(
        &self,
        texts: &[&str],
        k: usize,
    ) -> Result<Vec<Result<Vec<FillMaskPrediction>>>> {
        self.model.predict_top_k_batch(&self.tokenizer, texts, k)
    }

    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
