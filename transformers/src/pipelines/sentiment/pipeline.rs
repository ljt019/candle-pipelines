use super::model::SentimentAnalysisModel;
use crate::Result;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct SentimentResult {
    pub label: String,
    pub score: f32,
}

pub struct SentimentAnalysisPipeline<M: SentimentAnalysisModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: SentimentAnalysisModel> SentimentAnalysisPipeline<M> {
    pub fn predict(&self, text: &str) -> Result<SentimentResult> {
        self.model.predict_with_score(&self.tokenizer, text)
    }

    pub fn predict_batch(&self, texts: &[&str]) -> Result<Vec<Result<SentimentResult>>> {
        self.model.predict_with_score_batch(&self.tokenizer, texts)
    }

    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
