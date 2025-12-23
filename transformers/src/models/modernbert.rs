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

use crate::error::{GenerationError, ModelMetadataError, TokenizationError};
use crate::error::{Result, TransformersError};
use crate::pipelines::fill_mask::pipeline::FillMaskPrediction;
use crate::pipelines::sentiment::pipeline::SentimentResult;

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

#[derive(Clone)]
pub struct FillMaskModernBertModel {
    model: CandleModernBertForMaskedLM,
    device: Device,
}

impl FillMaskModernBertModel {
    pub fn new(size: ModernBertSize, device: Device) -> Result<Self> {
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

    pub fn predict(&self, tokenizer: &Tokenizer, text: &str) -> Result<String> {
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| TokenizationError::encode_failed(text, e.to_string()))?;
        let mask_id = tokenizer.token_to_id("[MASK]").unwrap_or(103);
        let mask_index = encoding
            .get_ids()
            .iter()
            .position(|&id| id == mask_id)
            .ok_or_else(|| {
                let preview: String = text.chars().take(50).collect();
                GenerationError::NoMaskToken {
                    input_preview: preview,
                }
            })?;

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

    pub fn get_tokenizer(size: ModernBertSize) -> Result<Tokenizer> {
        let repo_id = match size {
            ModernBertSize::Base => "answerdotai/ModernBERT-base",
            ModernBertSize::Large => "answerdotai/ModernBERT-large",
        };
        load_tokenizer(repo_id)
    }
}

impl crate::pipelines::fill_mask::model::FillMaskModel for FillMaskModernBertModel {
    type Options = ModernBertSize;

    fn new(options: Self::Options, device: Device) -> Result<Self> {
        FillMaskModernBertModel::new(options, device)
    }

    fn predict(&self, tokenizer: &Tokenizer, text: &str) -> Result<String> {
        self.predict(tokenizer, text)
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

        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| TokenizationError::encode_failed(text, e.to_string()))?;
        let mask_id = tokenizer.token_to_id("[MASK]").unwrap_or(103);
        let mask_index = encoding
            .get_ids()
            .iter()
            .position(|&id| id == mask_id)
            .ok_or_else(|| {
                let preview: String = text.chars().take(50).collect();
                GenerationError::NoMaskToken {
                    input_preview: preview,
                }
            })?;

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

    fn predict_top_k_batch(
        &self,
        tokenizer: &Tokenizer,
        texts: &[&str],
        k: usize,
    ) -> Result<Vec<Result<Vec<FillMaskPrediction>>>> {
        if texts.is_empty() || k == 0 {
            return Ok(texts.iter().map(|_| Ok(vec![])).collect());
        }

        let mask_id = tokenizer.token_to_id("[MASK]").unwrap_or(103);
        let pad_token_id = tokenizer
            .get_padding()
            .map(|p| p.pad_id)
            .or_else(|| tokenizer.token_to_id("<pad>"))
            .or_else(|| tokenizer.token_to_id("[PAD]"))
            .unwrap_or(0);

        let mut encodings = Vec::with_capacity(texts.len());
        let mut mask_indices = Vec::with_capacity(texts.len());
        let mut error_results: Vec<Option<TransformersError>> =
            (0..texts.len()).map(|_| None).collect();

        for (i, text) in texts.iter().enumerate() {
            match tokenizer.encode(*text, true) {
                Ok(encoding) => {
                    let mask_idx = encoding.get_ids().iter().position(|&id| id == mask_id);
                    match mask_idx {
                        Some(idx) => {
                            mask_indices.push(idx);
                            encodings.push(Some(encoding));
                        }
                        None => {
                            let preview: String = text.chars().take(50).collect();
                            error_results[i] = Some(
                                GenerationError::NoMaskToken {
                                    input_preview: preview,
                                }
                                .into(),
                            );
                            mask_indices.push(0);
                            encodings.push(None);
                        }
                    }
                }
                Err(e) => {
                    error_results[i] =
                        Some(TokenizationError::encode_failed(*text, e.to_string()).into());
                    mask_indices.push(0);
                    encodings.push(None);
                }
            }
        }

        let valid_indices: Vec<usize> = encodings
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.as_ref().map(|_| i))
            .collect();

        if valid_indices.is_empty() {
            return Ok(error_results
                .into_iter()
                .map(|e| Err(e.unwrap_or_else(|| GenerationError::NoPredictions.into())))
                .collect());
        }

        let valid_encodings: Vec<_> = valid_indices
            .iter()
            .map(|&i| encodings[i].as_ref().unwrap())
            .collect();
        let max_len = valid_encodings.iter().map(|e| e.len()).max().unwrap_or(0);

        let mut all_token_ids: Vec<u32> = Vec::new();
        let mut all_attention_masks: Vec<u32> = Vec::new();

        for encoding in &valid_encodings {
            let mut token_ids = encoding.get_ids().to_vec();
            let mut attention_mask = encoding.get_attention_mask().to_vec();
            token_ids.resize(max_len, pad_token_id);
            attention_mask.resize(max_len, 0);
            all_token_ids.extend(token_ids);
            all_attention_masks.extend(attention_mask);
        }

        let batch_size = valid_indices.len();
        let input_ids = Tensor::from_vec(all_token_ids, (batch_size, max_len), &self.device)?;
        let attention_mask =
            Tensor::from_vec(all_attention_masks, (batch_size, max_len), &self.device)?;

        let logits = self.model.forward(&input_ids, &attention_mask)?;

        let mut results: Vec<Result<Vec<FillMaskPrediction>>> = error_results
            .into_iter()
            .map(|e| match e {
                Some(err) => Err(err),
                None => Ok(vec![]),
            })
            .collect();

        for (batch_idx, &orig_idx) in valid_indices.iter().enumerate() {
            let mask_idx = mask_indices[orig_idx];
            let item_logits = logits.i((batch_idx, mask_idx, ..))?;
            let probs = softmax(&item_logits, D::Minus1)?;
            let probs_vec = probs.to_vec1::<f32>()?;

            if probs_vec.is_empty() {
                results[orig_idx] = Ok(vec![]);
                continue;
            }

            let mut idxs: Vec<usize> = (0..probs_vec.len()).collect();
            idxs.sort_by(|&i, &j| probs_vec[j].total_cmp(&probs_vec[i]));
            idxs.truncate(k.min(idxs.len()));

            let mut predictions = Vec::with_capacity(idxs.len());
            for idx in idxs {
                let token_str = tokenizer
                    .decode(&[idx as u32], true)
                    .unwrap_or_default()
                    .trim()
                    .to_string();
                if token_str.is_empty() {
                    continue;
                }
                predictions.push(FillMaskPrediction {
                    word: token_str,
                    score: probs_vec[idx],
                });
            }
            results[orig_idx] = Ok(predictions);
        }

        Ok(results)
    }

    fn get_tokenizer(options: Self::Options) -> Result<Tokenizer> {
        Self::get_tokenizer(options)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[derive(Clone)]
pub struct ZeroShotModernBertModel {
    model: CandleModernBertForSequenceClassification,
    device: Device,
    label2id: HashMap<String, u32>,
}

impl ZeroShotModernBertModel {
    pub fn new(size: ModernBertSize, device: Device) -> Result<Self> {
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
    ) -> Result<Vec<(String, f32)>> {
        self.predict_single_label(tokenizer, text, candidate_labels)
    }

    pub fn predict_single_label(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> Result<Vec<(String, f32)>> {
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
    ) -> Result<Vec<(String, f32)>> {
        self.predict_raw(tokenizer, text, candidate_labels)
    }

    fn predict_raw(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> Result<Vec<(String, f32)>> {
        if candidate_labels.is_empty() {
            return Ok(vec![]);
        }

        let available_labels: Vec<String> = self.label2id.keys().cloned().collect();
        let entailment_id =
            *self
                .label2id
                .get("entailment")
                .ok_or_else(|| ModelMetadataError::MissingLabel {
                    label: "entailment".into(),
                    available: available_labels,
                })?;

        let mut encodings = Vec::new();
        for &label in candidate_labels {
            let hypothesis = format!("This example is {label}.");
            let encoding = tokenizer
                .encode((text, hypothesis.as_str()), true)
                .map_err(|e| TokenizationError::encode_failed(text, e.to_string()))?;
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

    fn predict_raw_batch(
        &self,
        tokenizer: &Tokenizer,
        texts: &[&str],
        candidate_labels: &[&str],
    ) -> Result<Vec<Result<Vec<(String, f32)>>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        if candidate_labels.is_empty() {
            return Ok(texts.iter().map(|_| Ok(vec![])).collect());
        }

        let available_labels: Vec<String> = self.label2id.keys().cloned().collect();
        let entailment_id =
            *self
                .label2id
                .get("entailment")
                .ok_or_else(|| ModelMetadataError::MissingLabel {
                    label: "entailment".into(),
                    available: available_labels,
                })?;

        let pad_token_id = tokenizer
            .get_padding()
            .map(|p| p.pad_id)
            .or_else(|| tokenizer.token_to_id("<pad>"))
            .or_else(|| tokenizer.token_to_id("[PAD]"))
            .unwrap_or(0);

        let num_labels = candidate_labels.len();

        let mut all_encodings = Vec::with_capacity(texts.len() * num_labels);
        let mut error_results: Vec<Option<TransformersError>> =
            (0..texts.len()).map(|_| None).collect();

        for (text_idx, text) in texts.iter().enumerate() {
            let mut text_has_error = false;
            for &label in candidate_labels {
                if text_has_error {
                    all_encodings.push(None);
                    continue;
                }
                let hypothesis = format!("This example is {label}.");
                match tokenizer.encode((*text, hypothesis.as_str()), true) {
                    Ok(encoding) => all_encodings.push(Some(encoding)),
                    Err(e) => {
                        error_results[text_idx] =
                            Some(TokenizationError::encode_failed(*text, e.to_string()).into());
                        text_has_error = true;
                        all_encodings.push(None);
                    }
                }
            }
        }

        let valid_pair_indices: Vec<usize> = all_encodings
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.as_ref().map(|_| i))
            .collect();

        if valid_pair_indices.is_empty() {
            return Ok(error_results
                .into_iter()
                .map(|e| Err(e.unwrap_or_else(|| GenerationError::NoPredictions.into())))
                .collect());
        }

        let valid_encodings: Vec<_> = valid_pair_indices
            .iter()
            .map(|&i| all_encodings[i].as_ref().unwrap())
            .collect();
        let max_len = valid_encodings.iter().map(|e| e.len()).max().unwrap_or(0);

        let mut all_token_ids: Vec<u32> = Vec::new();
        let mut all_attention_masks: Vec<u32> = Vec::new();

        for encoding in &valid_encodings {
            let mut token_ids = encoding.get_ids().to_vec();
            let mut attention_mask = encoding.get_attention_mask().to_vec();
            token_ids.resize(max_len, pad_token_id);
            attention_mask.resize(max_len, 0);
            all_token_ids.extend(token_ids);
            all_attention_masks.extend(attention_mask);
        }

        let batch_size = valid_pair_indices.len();
        let input_ids = Tensor::from_vec(all_token_ids, (batch_size, max_len), &self.device)?;
        let attention_mask =
            Tensor::from_vec(all_attention_masks, (batch_size, max_len), &self.device)?;

        let logits = self.model.forward(&input_ids, &attention_mask)?;
        let probabilities = softmax(&logits, D::Minus1)?;
        let entailment_probs = probabilities
            .i((.., entailment_id as usize))?
            .to_vec1::<f32>()?;

        let mut results: Vec<Result<Vec<(String, f32)>>> = error_results
            .into_iter()
            .map(|e| match e {
                Some(err) => Err(err),
                None => Ok(vec![]),
            })
            .collect();

        let mut valid_idx_to_prob: std::collections::HashMap<usize, f32> =
            std::collections::HashMap::new();
        for (batch_idx, &pair_idx) in valid_pair_indices.iter().enumerate() {
            valid_idx_to_prob.insert(pair_idx, entailment_probs[batch_idx]);
        }

        for text_idx in 0..texts.len() {
            if results[text_idx].is_err() {
                continue;
            }

            let mut text_results: Vec<(String, f32)> = Vec::with_capacity(num_labels);
            let mut all_valid = true;

            for (label_idx, &label) in candidate_labels.iter().enumerate() {
                let pair_idx = text_idx * num_labels + label_idx;
                if let Some(&prob) = valid_idx_to_prob.get(&pair_idx) {
                    text_results.push((label.to_string(), prob));
                } else {
                    all_valid = false;
                    break;
                }
            }

            if all_valid {
                text_results
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                results[text_idx] = Ok(text_results);
            }
        }

        Ok(results)
    }

    pub fn get_tokenizer(size: ModernBertSize) -> Result<Tokenizer> {
        let repo_id = match size {
            ModernBertSize::Base => "MoritzLaurer/ModernBERT-base-zeroshot-v2.0",
            ModernBertSize::Large => "MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
        };
        load_tokenizer(repo_id)
    }
}

impl crate::pipelines::zero_shot::model::ZeroShotClassificationModel for ZeroShotModernBertModel {
    type Options = ModernBertSize;

    fn new(options: Self::Options, device: Device) -> Result<Self> {
        ZeroShotModernBertModel::new(options, device)
    }

    fn predict(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> Result<Vec<(String, f32)>> {
        self.predict_single_label(tokenizer, text, candidate_labels)
    }

    fn predict_batch(
        &self,
        tokenizer: &Tokenizer,
        texts: &[&str],
        candidate_labels: &[&str],
    ) -> Result<Vec<Result<Vec<(String, f32)>>>> {
        let raw_results = self.predict_raw_batch(tokenizer, texts, candidate_labels)?;

        Ok(raw_results
            .into_iter()
            .map(|result| {
                result.map(|mut scores| {
                    let sum: f32 = scores.iter().map(|(_, p)| p).sum();
                    if sum > 0.0 {
                        for (_, p) in scores.iter_mut() {
                            *p /= sum;
                        }
                    }
                    scores
                })
            })
            .collect())
    }

    fn predict_multi_label(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> Result<Vec<(String, f32)>> {
        self.predict_raw(tokenizer, text, candidate_labels)
    }

    fn predict_multi_label_batch(
        &self,
        tokenizer: &Tokenizer,
        texts: &[&str],
        candidate_labels: &[&str],
    ) -> Result<Vec<Result<Vec<(String, f32)>>>> {
        self.predict_raw_batch(tokenizer, texts, candidate_labels)
    }

    fn get_tokenizer(options: Self::Options) -> Result<Tokenizer> {
        Self::get_tokenizer(options)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

#[derive(Clone)]
pub struct SentimentModernBertModel {
    model: CandleModernBertForSequenceClassification,
    device: Device,
    id2label: HashMap<String, String>,
}

impl SentimentModernBertModel {
    pub fn new(size: ModernBertSize, device: Device) -> Result<Self> {
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

    pub fn predict(&self, tokenizer: &Tokenizer, text: &str) -> Result<String> {
        let tokens = tokenizer
            .encode(text, true)
            .map_err(|e| TokenizationError::encode_failed(text, e.to_string()))?;

        let input_ids = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;
        let attention_mask =
            Tensor::new(tokens.get_attention_mask(), &self.device)?.unsqueeze(0)?;

        let logits = self.model.forward(&input_ids, &attention_mask)?;
        let pred_id = logits.argmax(D::Minus1)?.squeeze(0)?.to_scalar::<u32>()?;

        let available_labels: Vec<String> = self.id2label.keys().cloned().collect();
        let label = self
            .id2label
            .get(&pred_id.to_string())
            .ok_or_else(|| GenerationError::UnknownLabelId {
                id: pred_id as i64,
                available: available_labels,
            })?
            .clone();

        Ok(label)
    }

    pub fn get_tokenizer(size: ModernBertSize) -> Result<Tokenizer> {
        let repo_id = match size {
            ModernBertSize::Base => "clapAI/modernBERT-base-multilingual-sentiment",
            ModernBertSize::Large => "clapAI/modernBERT-large-multilingual-sentiment",
        };
        load_tokenizer(repo_id)
    }
}

impl crate::pipelines::sentiment::model::SentimentAnalysisModel for SentimentModernBertModel {
    type Options = ModernBertSize;

    fn new(options: Self::Options, device: Device) -> Result<Self> {
        SentimentModernBertModel::new(options, device)
    }

    fn predict(&self, tokenizer: &Tokenizer, text: &str) -> Result<String> {
        self.predict(tokenizer, text)
    }

    fn predict_with_score(&self, tokenizer: &Tokenizer, text: &str) -> Result<SentimentResult> {
        let tokens = tokenizer
            .encode(text, true)
            .map_err(|e| TokenizationError::encode_failed(text, e.to_string()))?;

        let input_ids = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;
        let attention_mask =
            Tensor::new(tokens.get_attention_mask(), &self.device)?.unsqueeze(0)?;

        let logits = self.model.forward(&input_ids, &attention_mask)?;
        let pred_id = logits.argmax(D::Minus1)?.squeeze(0)?.to_scalar::<u32>()?;

        let probs = softmax(&logits, D::Minus1)?;
        let probs_vec = probs.squeeze(0)?.to_vec1::<f32>()?;
        let score = probs_vec.get(pred_id as usize).copied().unwrap_or(0.0);

        let available_labels: Vec<String> = self.id2label.keys().cloned().collect();
        let label = self
            .id2label
            .get(&pred_id.to_string())
            .ok_or_else(|| GenerationError::UnknownLabelId {
                id: pred_id as i64,
                available: available_labels,
            })?
            .clone();

        Ok(SentimentResult { label, score })
    }

    fn predict_with_score_batch(
        &self,
        tokenizer: &Tokenizer,
        texts: &[&str],
    ) -> Result<Vec<Result<SentimentResult>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let pad_token_id = tokenizer
            .get_padding()
            .map(|p| p.pad_id)
            .or_else(|| tokenizer.token_to_id("<pad>"))
            .or_else(|| tokenizer.token_to_id("[PAD]"))
            .unwrap_or(0);

        let mut encodings = Vec::with_capacity(texts.len());
        let mut error_results: Vec<Option<TransformersError>> =
            (0..texts.len()).map(|_| None).collect();

        for (i, text) in texts.iter().enumerate() {
            match tokenizer.encode(*text, true) {
                Ok(encoding) => encodings.push(Some(encoding)),
                Err(e) => {
                    error_results[i] =
                        Some(TokenizationError::encode_failed(*text, e.to_string()).into());
                    encodings.push(None);
                }
            }
        }

        let valid_indices: Vec<usize> = encodings
            .iter()
            .enumerate()
            .filter_map(|(i, e)| e.as_ref().map(|_| i))
            .collect();

        if valid_indices.is_empty() {
            return Ok(error_results
                .into_iter()
                .map(|e| Err(e.unwrap_or_else(|| GenerationError::NoPredictions.into())))
                .collect());
        }

        let valid_encodings: Vec<_> = valid_indices
            .iter()
            .map(|&i| encodings[i].as_ref().unwrap())
            .collect();
        let max_len = valid_encodings.iter().map(|e| e.len()).max().unwrap_or(0);

        let mut all_token_ids: Vec<u32> = Vec::new();
        let mut all_attention_masks: Vec<u32> = Vec::new();

        for encoding in &valid_encodings {
            let mut token_ids = encoding.get_ids().to_vec();
            let mut attention_mask = encoding.get_attention_mask().to_vec();
            token_ids.resize(max_len, pad_token_id);
            attention_mask.resize(max_len, 0);
            all_token_ids.extend(token_ids);
            all_attention_masks.extend(attention_mask);
        }

        let batch_size = valid_indices.len();
        let input_ids = Tensor::from_vec(all_token_ids, (batch_size, max_len), &self.device)?;
        let attention_mask =
            Tensor::from_vec(all_attention_masks, (batch_size, max_len), &self.device)?;

        let logits = self.model.forward(&input_ids, &attention_mask)?;
        let probs = softmax(&logits, D::Minus1)?;
        let pred_ids = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
        let probs_2d = probs.to_vec2::<f32>()?;

        let mut results: Vec<Result<SentimentResult>> = error_results
            .into_iter()
            .map(|e| match e {
                Some(err) => Err(err),
                None => Ok(SentimentResult {
                    label: String::new(),
                    score: 0.0,
                }),
            })
            .collect();

        for (batch_idx, &orig_idx) in valid_indices.iter().enumerate() {
            let pred_id = pred_ids[batch_idx];
            let score = probs_2d[batch_idx]
                .get(pred_id as usize)
                .copied()
                .unwrap_or(0.0);

            let available_labels: Vec<String> = self.id2label.keys().cloned().collect();
            match self.id2label.get(&pred_id.to_string()) {
                Some(label) => {
                    results[orig_idx] = Ok(SentimentResult {
                        label: label.clone(),
                        score,
                    });
                }
                None => {
                    results[orig_idx] = Err(GenerationError::UnknownLabelId {
                        id: pred_id as i64,
                        available: available_labels,
                    }
                    .into());
                }
            }
        }

        Ok(results)
    }

    fn get_tokenizer(options: Self::Options) -> Result<Tokenizer> {
        Self::get_tokenizer(options)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

fn load_tokenizer(repo_id: &str) -> Result<Tokenizer> {
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
    let tokenizer_path = repo.get("tokenizer.json")?;
    let path_str = tokenizer_path.display().to_string();
    Tokenizer::from_file(&tokenizer_path).map_err(|e| {
        TokenizationError::LoadFailed {
            path: path_str,
            reason: e.to_string(),
        }
        .into()
    })
}

fn load_model_weights(repo_id: &str, device: &Device) -> Result<(Config, VarBuilder<'static>)> {
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

fn patch_config_num_labels(config: &mut Config, num_labels: usize) {
    use candle_transformers::models::modernbert::{ClassifierConfig, ClassifierPooling};

    if config.classifier_config.is_none()
        || config
            .classifier_config
            .as_ref()
            .map(|c| c.id2label.len())
            .unwrap_or(0)
            != num_labels
    {
        let id2label: HashMap<String, String> = (0..num_labels)
            .map(|i| (i.to_string(), format!("label_{i}")))
            .collect();
        let label2id: HashMap<String, String> = id2label
            .iter()
            .map(|(k, v)| (v.clone(), k.clone()))
            .collect();

        config.classifier_config = Some(ClassifierConfig {
            id2label,
            label2id,
            classifier_pooling: ClassifierPooling::default(),
        });
    }
}

fn load_classifier_model(
    repo_id: &str,
    device: &Device,
) -> Result<(Config, VarBuilder<'static>, HashMap<String, u32>)> {
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    let config_path = repo.get("config.json")?;
    let weights_path = repo
        .get("model.safetensors")
        .or_else(|_| repo.get("pytorch_model.bin"))?;

    let config_str = std::fs::read_to_string(&config_path)?;
    let mut config: Config = serde_json::from_str(&config_str)?;
    let class_cfg: ClassifierConfigJson = serde_json::from_str(&config_str)?;

    let num_labels = class_cfg.label2id.len().max(class_cfg.id2label.len());
    patch_config_num_labels(&mut config, num_labels);

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
) -> Result<(Config, VarBuilder<'static>, HashMap<String, String>)> {
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    let config_path = repo.get("config.json")?;
    let weights_path = repo
        .get("model.safetensors")
        .or_else(|_| repo.get("pytorch_model.bin"))?;

    let config_str = std::fs::read_to_string(&config_path)?;
    let mut config: Config = serde_json::from_str(&config_str)?;
    let class_cfg: ClassifierConfigJson = serde_json::from_str(&config_str)?;

    let num_labels = class_cfg.label2id.len().max(class_cfg.id2label.len());
    patch_config_num_labels(&mut config, num_labels);

    let vb = if weights_path.extension().is_some_and(|e| e == "safetensors") {
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? }
    } else {
        VarBuilder::from_pth(&weights_path, DType::F32, device)?
    };

    Ok((config, vb, class_cfg.id2label))
}
