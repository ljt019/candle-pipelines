use super::model::ZeroShotClassificationModel;
use crate::error::Result;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub label: String,
    pub score: f32,
}

pub struct ZeroShotClassificationPipeline<M: ZeroShotClassificationModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: ZeroShotClassificationModel> ZeroShotClassificationPipeline<M> {
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

    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
