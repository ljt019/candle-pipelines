use crate::error::Result;
use tokenizers::Tokenizer;

/// Internal prediction type used by model trait.
#[derive(Debug, Clone)]
pub struct FillMaskPrediction {
    pub word: String,
    pub score: f32,
}

pub trait FillMaskModel {
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

    fn predict_top_k(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        k: usize,
    ) -> Result<Vec<FillMaskPrediction>> {
        if k == 0 {
            return Ok(vec![]);
        }
        let filled = self.predict(tokenizer, text)?;
        Ok(vec![FillMaskPrediction {
            word: filled.trim().to_string(),
            score: 1.0,
        }])
    }

    fn predict_top_k_batch(
        &self,
        tokenizer: &Tokenizer,
        texts: &[&str],
        k: usize,
    ) -> Result<Vec<Result<Vec<FillMaskPrediction>>>> {
        Ok(texts
            .iter()
            .map(|text| self.predict_top_k(tokenizer, text, k))
            .collect())
    }

    fn get_tokenizer(options: Self::Options) -> Result<Tokenizer>;

    fn device(&self) -> &candle_core::Device;
}
