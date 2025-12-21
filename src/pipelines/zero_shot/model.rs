use crate::Result;
use tokenizers::Tokenizer;

pub trait ZeroShotClassificationModel {
    type Options: std::fmt::Debug + Clone;

    fn new(options: Self::Options, device: candle_core::Device) -> Result<Self>
    where
        Self: Sized;

    /// Predict with normalized probabilities for single-label classification (probabilities sum to 1)
    fn predict(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> Result<Vec<(String, f32)>>;

    /// Predict a batch of inputs with normalized probabilities for single-label classification.
    fn predict_batch(
        &self,
        tokenizer: &Tokenizer,
        texts: &[&str],
        candidate_labels: &[&str],
    ) -> Result<Vec<Result<Vec<(String, f32)>>>> {
        Ok(texts
            .iter()
            .map(|text| self.predict(tokenizer, text, candidate_labels))
            .collect())
    }

    /// Predict with raw entailment probabilities for multi-label classification
    fn predict_multi_label(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> Result<Vec<(String, f32)>>;

    /// Predict a batch of inputs with raw entailment probabilities for multi-label classification.
    fn predict_multi_label_batch(
        &self,
        tokenizer: &Tokenizer,
        texts: &[&str],
        candidate_labels: &[&str],
    ) -> Result<Vec<Result<Vec<(String, f32)>>>> {
        Ok(texts
            .iter()
            .map(|text| self.predict_multi_label(tokenizer, text, candidate_labels))
            .collect())
    }

    fn get_tokenizer(options: Self::Options) -> Result<Tokenizer>;

    fn device(&self) -> &candle_core::Device;
}
