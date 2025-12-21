//! ModernBERT model wrappers for fill-mask, sentiment, and zero-shot pipelines.
//!
//! Uses `candle_transformers::models::modernbert` for the underlying implementation.

use anyhow::{Error as E, Result as AnyhowResult};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::modernbert::{
    Config, ModernBertForMaskedLM as CandleModernBertForMaskedLM,
    ModernBertForSequenceClassification as CandleModernBertForSequenceClassification,
};
use hf_hub::{api::sync::Api, Repo, RepoType};
use serde::Deserialize;
use std::collections::HashMap;
use tokenizers::Tokenizer;

use crate::pipelines::fill_mask::pipeline::FillMaskPrediction;
use crate::pipelines::sentiment::pipeline::SentimentResult;

/// Available ModernBERT model sizes.
#[derive(Debug, Clone, Copy)]
pub enum ModernBertSize {
    Base,
    Large,
}

impl std::fmt::Display for ModernBertSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            ModernBertSize::Base => "modernbert-base",
            ModernBertSize::Large => "modernbert-large",
        };
        write!(f, "{name}")
    }
}

impl crate::pipelines::cache::ModelOptions for ModernBertSize {
    fn cache_key(&self) -> String {
        self.to_string()
    }
}

/// Fill-mask model using ModernBERT.
#[derive(Clone)]
pub struct FillMaskModernBertModel {
    model: CandleModernBertForMaskedLM,
    device: Device,
}

impl FillMaskModernBertModel {
    pub fn new(size: ModernBertSize, device: Device) -> AnyhowResult<Self> {
        let model_id = match size {
            ModernBertSize::Base => "answerdotai/ModernBERT-base",
            ModernBertSize::Large => "answerdotai/ModernBERT-large",
        };

        let (config, vb) = load_model_weights(model_id, &device)?;
        let model = CandleModernBertForMaskedLM::load(vb, &config)?;

        Ok(Self { model, device })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn predict(&self, tokenizer: &Tokenizer, text: &str) -> AnyhowResult<String> {
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| E::msg(format!("Tokenization error: {e}")))?;
        let mask_id = tokenizer.token_to_id("[MASK]").unwrap_or(103);
        let mask_index = encoding
            .get_ids()
            .iter()
            .position(|&id| id == mask_id)
            .ok_or_else(|| E::msg("No [MASK] token found in input"))?;

        let input_ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;
        let attention_mask =
            Tensor::new(encoding.get_attention_mask(), &self.device)?.unsqueeze(0)?;

        let logits = self.model.forward(&input_ids, &attention_mask)?;
        let logits = logits.squeeze(0)?.i((mask_index, ..))?;
        let probs = softmax(&logits, D::Minus1)?;
        let predicted = probs.argmax(D::Minus1)?.to_scalar::<u32>()?;

        let token_str = tokenizer
            .decode(&[predicted], true)
            .unwrap_or_default()
            .trim()
            .to_string();
        Ok(text.replace("[MASK]", &token_str))
    }

    pub fn get_tokenizer(size: ModernBertSize) -> AnyhowResult<Tokenizer> {
        let repo_id = match size {
            ModernBertSize::Base => "answerdotai/ModernBERT-base",
            ModernBertSize::Large => "answerdotai/ModernBERT-large",
        };
        load_tokenizer(repo_id)
    }
}

impl crate::pipelines::fill_mask::model::FillMaskModel for FillMaskModernBertModel {
    type Options = ModernBertSize;

    fn new(options: Self::Options, device: Device) -> anyhow::Result<Self> {
        FillMaskModernBertModel::new(options, device)
    }

    fn predict(&self, tokenizer: &Tokenizer, text: &str) -> AnyhowResult<String> {
        self.predict(tokenizer, text)
    }

    fn predict_top_k(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        k: usize,
    ) -> AnyhowResult<Vec<FillMaskPrediction>> {
        if k == 0 {
            return Ok(vec![]);
        }

        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| E::msg(format!("Tokenization error: {e}")))?;
        let mask_id = tokenizer.token_to_id("[MASK]").unwrap_or(103);
        let mask_index = encoding
            .get_ids()
            .iter()
            .position(|&id| id == mask_id)
            .ok_or_else(|| E::msg("No [MASK] token found in input"))?;

        let input_ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;
        let attention_mask =
            Tensor::new(encoding.get_attention_mask(), &self.device)?.unsqueeze(0)?;

        let logits = self.model.forward(&input_ids, &attention_mask)?;
        let logits = logits.squeeze(0)?.i((mask_index, ..))?;
        let probs = softmax(&logits, D::Minus1)?;
        let probs_vec = probs.to_vec1::<f32>()?;

        if probs_vec.is_empty() {
            return Ok(vec![]);
        }

        let mut idxs: Vec<usize> = (0..probs_vec.len()).collect();
        idxs.sort_by(|&i, &j| probs_vec[j].total_cmp(&probs_vec[i]));
        idxs.truncate(k.min(idxs.len()));

        let mut out = Vec::with_capacity(idxs.len());
        for idx in idxs {
            let token_str = tokenizer
                .decode(&[idx as u32], true)
                .unwrap_or_default()
                .trim()
                .to_string();
            if token_str.is_empty() {
                continue;
            }
            out.push(FillMaskPrediction {
                word: token_str,
                score: probs_vec[idx],
            });
        }

        Ok(out)
    }

    fn get_tokenizer(options: Self::Options) -> AnyhowResult<Tokenizer> {
        Self::get_tokenizer(options)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

/// Zero-shot classification model using ModernBERT.
#[derive(Clone)]
pub struct ZeroShotModernBertModel {
    model: CandleModernBertForSequenceClassification,
    device: Device,
    label2id: HashMap<String, u32>,
}

impl ZeroShotModernBertModel {
    pub fn new(size: ModernBertSize, device: Device) -> AnyhowResult<Self> {
        let model_id = match size {
            ModernBertSize::Base => "MoritzLaurer/ModernBERT-base-zeroshot-v2.0",
            ModernBertSize::Large => "MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
        };

        let (config, vb, label2id) = load_classifier_model(model_id, &device)?;
        let model = CandleModernBertForSequenceClassification::load(vb, &config)?;

        Ok(Self {
            model,
            device,
            label2id,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn predict(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> AnyhowResult<Vec<(String, f32)>> {
        self.predict_single_label(tokenizer, text, candidate_labels)
    }

    pub fn predict_single_label(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> AnyhowResult<Vec<(String, f32)>> {
        let mut results = self.predict_raw(tokenizer, text, candidate_labels)?;
        let sum: f32 = results.iter().map(|(_, p)| p).sum();
        if sum > 0.0 {
            for (_, p) in results.iter_mut() {
                *p /= sum;
            }
        }
        Ok(results)
    }

    pub fn predict_multi_label(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> AnyhowResult<Vec<(String, f32)>> {
        self.predict_raw(tokenizer, text, candidate_labels)
    }

    fn predict_raw(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> AnyhowResult<Vec<(String, f32)>> {
        if candidate_labels.is_empty() {
            return Ok(vec![]);
        }

        let entailment_id = *self
            .label2id
            .get("entailment")
            .ok_or_else(|| E::msg("Config missing 'entailment' in label2id"))?;

        let mut encodings = Vec::new();
        for &label in candidate_labels {
            let hypothesis = format!("This example is {label}.");
            let encoding = tokenizer
                .encode((text, hypothesis.as_str()), true)
                .map_err(|e| E::msg(format!("Tokenization error: {e}")))?;
            encodings.push(encoding);
        }

        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
        let pad_token_id = tokenizer
            .get_padding()
            .map(|p| p.pad_id)
            .or_else(|| tokenizer.token_to_id("<pad>"))
            .or_else(|| tokenizer.token_to_id("[PAD]"))
            .unwrap_or(0);

        let mut all_token_ids: Vec<u32> = Vec::new();
        let mut all_attention_masks: Vec<u32> = Vec::new();

        for encoding in encodings {
            let mut token_ids = encoding.get_ids().to_vec();
            let mut attention_mask = encoding.get_attention_mask().to_vec();
            token_ids.resize(max_len, pad_token_id);
            attention_mask.resize(max_len, 0);
            all_token_ids.extend(token_ids);
            all_attention_masks.extend(attention_mask);
        }

        let input_ids = Tensor::from_vec(
            all_token_ids,
            (candidate_labels.len(), max_len),
            &self.device,
        )?;
        let attention_mask = Tensor::from_vec(
            all_attention_masks,
            (candidate_labels.len(), max_len),
            &self.device,
        )?;

        let logits = self.model.forward(&input_ids, &attention_mask)?;
        let probabilities = softmax(&logits, D::Minus1)?;
        let entailment_probs = probabilities
            .i((.., entailment_id as usize))?
            .to_vec1::<f32>()?;

        let mut results: Vec<(String, f32)> = candidate_labels
            .iter()
            .map(|&l| l.to_string())
            .zip(entailment_probs)
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    pub fn get_tokenizer(size: ModernBertSize) -> AnyhowResult<Tokenizer> {
        let repo_id = match size {
            ModernBertSize::Base => "MoritzLaurer/ModernBERT-base-zeroshot-v2.0",
            ModernBertSize::Large => "MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
        };
        load_tokenizer(repo_id)
    }
}

impl crate::pipelines::zero_shot::model::ZeroShotClassificationModel for ZeroShotModernBertModel {
    type Options = ModernBertSize;

    fn new(options: Self::Options, device: Device) -> anyhow::Result<Self> {
        ZeroShotModernBertModel::new(options, device)
    }

    fn predict(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> AnyhowResult<Vec<(String, f32)>> {
        self.predict_single_label(tokenizer, text, candidate_labels)
    }

    fn predict_multi_label(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> AnyhowResult<Vec<(String, f32)>> {
        self.predict_raw(tokenizer, text, candidate_labels)
    }

    fn get_tokenizer(options: Self::Options) -> AnyhowResult<Tokenizer> {
        Self::get_tokenizer(options)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

/// Sentiment analysis model using ModernBERT.
#[derive(Clone)]
pub struct SentimentModernBertModel {
    model: CandleModernBertForSequenceClassification,
    device: Device,
    id2label: HashMap<String, String>,
}

impl SentimentModernBertModel {
    pub fn new(size: ModernBertSize, device: Device) -> AnyhowResult<Self> {
        let model_id = match size {
            ModernBertSize::Base => "clapAI/modernBERT-base-multilingual-sentiment",
            ModernBertSize::Large => "clapAI/modernBERT-large-multilingual-sentiment",
        };

        let (config, vb, id2label) = load_classifier_model_with_id2label(model_id, &device)?;
        let model = CandleModernBertForSequenceClassification::load(vb, &config)?;

        Ok(Self {
            model,
            device,
            id2label,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn predict(&self, tokenizer: &Tokenizer, text: &str) -> AnyhowResult<String> {
        let tokens = tokenizer
            .encode(text, true)
            .map_err(|e| E::msg(format!("Tokenization error: {e}")))?;

        let input_ids = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;
        let attention_mask =
            Tensor::new(tokens.get_attention_mask(), &self.device)?.unsqueeze(0)?;

        let logits = self.model.forward(&input_ids, &attention_mask)?;
        let pred_id = logits.argmax(D::Minus1)?.squeeze(0)?.to_scalar::<u32>()?;

        let label = self
            .id2label
            .get(&pred_id.to_string())
            .ok_or_else(|| E::msg(format!("Predicted ID '{pred_id}' not in id2label")))?
            .clone();

        Ok(label)
    }

    pub fn get_tokenizer(size: ModernBertSize) -> AnyhowResult<Tokenizer> {
        let repo_id = match size {
            ModernBertSize::Base => "clapAI/modernBERT-base-multilingual-sentiment",
            ModernBertSize::Large => "clapAI/modernBERT-large-multilingual-sentiment",
        };
        load_tokenizer(repo_id)
    }
}

impl crate::pipelines::sentiment::model::SentimentAnalysisModel for SentimentModernBertModel {
    type Options = ModernBertSize;

    fn new(options: Self::Options, device: Device) -> anyhow::Result<Self> {
        SentimentModernBertModel::new(options, device)
    }

    fn predict(&self, tokenizer: &Tokenizer, text: &str) -> AnyhowResult<String> {
        self.predict(tokenizer, text)
    }

    fn predict_with_score(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
    ) -> AnyhowResult<SentimentResult> {
        let tokens = tokenizer
            .encode(text, true)
            .map_err(|e| E::msg(format!("Tokenization error: {e}")))?;

        let input_ids = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;
        let attention_mask =
            Tensor::new(tokens.get_attention_mask(), &self.device)?.unsqueeze(0)?;

        let logits = self.model.forward(&input_ids, &attention_mask)?;
        let pred_id = logits.argmax(D::Minus1)?.squeeze(0)?.to_scalar::<u32>()?;

        let probs = softmax(&logits, D::Minus1)?;
        let probs_vec = probs.squeeze(0)?.to_vec1::<f32>()?;
        let score = probs_vec.get(pred_id as usize).copied().unwrap_or(0.0);

        let label = self
            .id2label
            .get(&pred_id.to_string())
            .ok_or_else(|| E::msg(format!("Predicted ID '{pred_id}' not in id2label")))?
            .clone();

        Ok(SentimentResult { label, score })
    }

    fn get_tokenizer(options: Self::Options) -> AnyhowResult<Tokenizer> {
        Self::get_tokenizer(options)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ============================================================================
// Helper functions for loading models
// ============================================================================

fn load_tokenizer(repo_id: &str) -> AnyhowResult<Tokenizer> {
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
    let tokenizer_path = repo.get("tokenizer.json")?;
    Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))
}

fn load_model_weights(
    repo_id: &str,
    device: &Device,
) -> AnyhowResult<(Config, VarBuilder<'static>)> {
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    let config_path = repo.get("config.json")?;
    let weights_path = repo
        .get("model.safetensors")
        .or_else(|_| repo.get("pytorch_model.bin"))?;

    let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;

    let vb = if weights_path.extension().is_some_and(|e| e == "safetensors") {
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? }
    } else {
        VarBuilder::from_pth(&weights_path, DType::F32, device)?
    };

    Ok((config, vb))
}

#[derive(Deserialize)]
struct ClassifierConfigJson {
    #[serde(default)]
    id2label: HashMap<String, String>,
    #[serde(default)]
    label2id: HashMap<String, u32>,
}

fn load_classifier_model(
    repo_id: &str,
    device: &Device,
) -> AnyhowResult<(Config, VarBuilder<'static>, HashMap<String, u32>)> {
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    let config_path = repo.get("config.json")?;
    let weights_path = repo
        .get("model.safetensors")
        .or_else(|_| repo.get("pytorch_model.bin"))?;

    let config_str = std::fs::read_to_string(&config_path)?;
    let config: Config = serde_json::from_str(&config_str)?;
    let class_cfg: ClassifierConfigJson = serde_json::from_str(&config_str)?;

    let vb = if weights_path.extension().is_some_and(|e| e == "safetensors") {
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? }
    } else {
        VarBuilder::from_pth(&weights_path, DType::F32, device)?
    };

    Ok((config, vb, class_cfg.label2id))
}

fn load_classifier_model_with_id2label(
    repo_id: &str,
    device: &Device,
) -> AnyhowResult<(Config, VarBuilder<'static>, HashMap<String, String>)> {
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    let config_path = repo.get("config.json")?;
    let weights_path = repo
        .get("model.safetensors")
        .or_else(|_| repo.get("pytorch_model.bin"))?;

    let config_str = std::fs::read_to_string(&config_path)?;
    let config: Config = serde_json::from_str(&config_str)?;
    let class_cfg: ClassifierConfigJson = serde_json::from_str(&config_str)?;

    let vb = if weights_path.extension().is_some_and(|e| e == "safetensors") {
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? }
    } else {
        VarBuilder::from_pth(&weights_path, DType::F32, device)?
    };

    Ok((config, vb, class_cfg.id2label))
}
