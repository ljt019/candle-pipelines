use crate::Result;
use tokenizers::Tokenizer;

pub trait SentimentAnalysisModel {
    type Options: std::fmt::Debug + Clone;

    fn new(options: Self::Options, device: candle_core::Device) -> Result<Self>
    where
        Self: Sized;

    fn predict(&self, tokenizer: &Tokenizer, text: &str) -> Result<String>;

    /// Predict a batch of inputs, returning one result per item.
    fn predict_batch(&self, tokenizer: &Tokenizer, texts: &[&str]) -> Result<Vec<Result<String>>> {
        Ok(texts
            .iter()
            .map(|text| self.predict(tokenizer, text))
            .collect())
    }

    /// Predict sentiment and return both label + confidence score.
    ///
    /// Default implementation falls back to `predict` and assigns a score of 1.0.
    fn predict_with_score(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
    ) -> Result<super::pipeline::SentimentResult> {
        let label = self.predict(tokenizer, text)?;
        Ok(super::pipeline::SentimentResult { label, score: 1.0 })
    }

    /// Predict a batch of inputs, returning both label and score for each entry.
    fn predict_with_score_batch(
        &self,
        tokenizer: &Tokenizer,
        texts: &[&str],
    ) -> Result<Vec<Result<super::pipeline::SentimentResult>>> {
        Ok(texts
            .iter()
            .map(|text| self.predict_with_score(tokenizer, text))
            .collect())
    }

    fn get_tokenizer(options: Self::Options) -> Result<Tokenizer>;

    fn device(&self) -> &candle_core::Device;
}
