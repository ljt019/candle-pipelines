use crate::error::Result;
use tokenizers::Tokenizer;

/// A vector of tuples each containing a label and a confidence score.
pub type LabelScores = Vec<(String, f32)>;

pub trait ZeroShotClassificationModel {
    type Options: std::fmt::Debug + Clone;

    fn new(options: Self::Options, device: candle_core::Device) -> Result<Self>
    where
        Self: Sized;

    fn predict(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> Result<LabelScores>;

    fn predict_batch(
        &self,
        tokenizer: &Tokenizer,
        texts: &[&str],
        candidate_labels: &[&str],
    ) -> Result<Vec<Result<LabelScores>>> {
        Ok(texts
            .iter()
            .map(|text| self.predict(tokenizer, text, candidate_labels))
            .collect())
    }

    fn predict_multi_label(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> Result<LabelScores>;

    fn predict_multi_label_batch(
        &self,
        tokenizer: &Tokenizer,
        texts: &[&str],
        candidate_labels: &[&str],
    ) -> Result<Vec<Result<LabelScores>>> {
        Ok(texts
            .iter()
            .map(|text| self.predict_multi_label(tokenizer, text, candidate_labels))
            .collect())
    }

    fn get_tokenizer(options: Self::Options) -> Result<Tokenizer>;

    fn device(&self) -> &candle_core::Device;
}
