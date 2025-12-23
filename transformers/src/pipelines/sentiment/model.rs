use crate::error::Result;
use tokenizers::Tokenizer;

pub trait SentimentAnalysisModel {
    type Options: std::fmt::Debug + Clone;

    fn new(options: Self::Options, device: candle_core::Device) -> Result<Self>
    where
        Self: Sized;

    fn predict(&self, tokenizer: &Tokenizer, text: &str) -> Result<String>;

    fn predict_batch(&self, tokenizer: &Tokenizer, texts: &[&str]) -> Result<Vec<Result<String>>> {
        Ok(texts
            .iter()
            .map(|text| self.predict(tokenizer, text))
            .collect())
    }

    fn predict_with_score(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
    ) -> Result<super::pipeline::SentimentResult> {
        let label = self.predict(tokenizer, text)?;
        Ok(super::pipeline::SentimentResult { label, score: 1.0 })
    }

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
