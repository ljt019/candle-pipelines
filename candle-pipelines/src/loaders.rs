use serde::Deserialize;
use std::time::Duration;

use crate::error::{PipelineError, Result};

#[derive(Clone)]
pub struct GenerationConfig {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u64>,
    pub min_p: Option<f64>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<usize>,
    pub eos_token_ids: Vec<u64>,
}
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct HfLoader {
    pub repo: String,
    pub filename: String,
}

impl HfLoader {
    pub fn new(repo: &str, filename: &str) -> Self {
        Self {
            repo: repo.into(),
            filename: filename.into(),
        }
    }

    /// Synchronously load file from HuggingFace Hub (uses ureq).
    pub fn load(&self) -> Result<PathBuf> {
        let hf_api = hf_hub::api::sync::ApiBuilder::new()
            .build()
            .map_err(|e| {
                PipelineError::Download(format!("Failed to initialize HuggingFace API: {e}"))
            })?;
        let hf_api = hf_api.model(self.repo.clone());

        let max_retries = 3;
        let mut attempts = 0u32;

        for attempt in 0..max_retries {
            match hf_api.get(&self.filename) {
                Ok(path) => return Ok(path),
                Err(e) => {
                    let error_msg = e.to_string();
                    attempts = attempt + 1;
                    if error_msg.contains("Lock acquisition failed") && attempt < max_retries - 1 {
                        let wait_time = Duration::from_millis(100 * (1 << attempt));
                        std::thread::sleep(wait_time);
                        continue;
                    }
                    return Err(PipelineError::Download(format!(
                        "Failed to download '{}' from '{}': {}",
                        self.filename, self.repo, error_msg
                    )));
                }
            }
        }

        Err(PipelineError::Download(format!(
            "Download timed out for '{}' from '{}' after {} attempt(s)",
            self.filename, self.repo, attempts
        )))
    }

    /// Asynchronously load file from HuggingFace Hub (uses reqwest).
    pub async fn load_async(&self) -> Result<PathBuf> {
        let hf_api = hf_hub::api::tokio::ApiBuilder::new()
            .with_chunk_size(None)
            .build()
            .map_err(|e| {
                PipelineError::Download(format!("Failed to initialize HuggingFace API: {e}"))
            })?;
        let hf_api = hf_api.model(self.repo.clone());

        let max_retries = 3;
        let mut attempts = 0u32;

        for attempt in 0..max_retries {
            match hf_api.get(self.filename.as_str()).await {
                Ok(path) => return Ok(path),
                Err(e) => {
                    let error_msg = e.to_string();
                    attempts = attempt + 1;
                    if error_msg.contains("Lock acquisition failed") && attempt < max_retries - 1 {
                        let wait_time = Duration::from_millis(100 * (1 << attempt));
                        tokio::time::sleep(wait_time).await;
                        continue;
                    }
                    return Err(PipelineError::Download(format!(
                        "Failed to download '{}' from '{}': {}",
                        self.filename, self.repo, error_msg
                    )));
                }
            }
        }

        Err(PipelineError::Download(format!(
            "Download timed out for '{}' from '{}' after {} attempt(s)",
            self.filename, self.repo, attempts
        )))
    }
}

#[derive(Clone)]
pub struct TokenizerLoader {
    pub tokenizer_file_loader: HfLoader,
}

impl TokenizerLoader {
    pub fn new(repo: &str, filename: &str) -> Self {
        let tokenizer_file_loader = HfLoader::new(repo, filename);

        Self {
            tokenizer_file_loader,
        }
    }

    pub fn load(&self) -> Result<Tokenizer> {
        let tokenizer_file_path = self.tokenizer_file_loader.load()?;
        let path_str = tokenizer_file_path.display().to_string();

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_file_path).map_err(|e| {
            PipelineError::Tokenization(format!(
                "Failed to load tokenizer from '{}': {}",
                path_str, e
            ))
        })?;

        Ok(tokenizer)
    }
}

pub struct GenerationConfigLoader {
    pub generation_config_file_loader: HfLoader,
}

#[derive(Deserialize)]
struct RawGenerationConfig {
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    min_p: Option<f64>,
    #[serde(alias = "repetition_penalty", alias = "repeat_penalty")]
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,
    #[serde(alias = "eos_token_id", alias = "eos_token_ids")]
    eos_token_ids: Option<serde_json::Value>,
}

impl GenerationConfigLoader {
    pub fn new(repo: &str, filename: &str) -> Self {
        let generation_config_file_loader = HfLoader::new(repo, filename);

        Self {
            generation_config_file_loader,
        }
    }

    fn parse_config(path: std::path::PathBuf) -> Result<GenerationConfig> {
        let generation_config_content = std::fs::read_to_string(path)?;

        let raw: RawGenerationConfig = serde_json::from_str(&generation_config_content)?;

        let eos_token_ids = match raw.eos_token_ids {
            Some(serde_json::Value::Number(n)) => {
                vec![n.as_u64().ok_or_else(|| {
                    PipelineError::Unexpected(format!(
                        "Invalid eos_token_id: expected unsigned integer, got {n}"
                    ))
                })?]
            }
            Some(serde_json::Value::Array(arr)) => arr
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    v.as_u64().ok_or_else(|| {
                        PipelineError::Unexpected(format!(
                            "Invalid eos_token_ids[{i}]: expected unsigned integer, got {v}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            _ => Vec::new(),
        };

        Ok(GenerationConfig {
            temperature: raw.temperature,
            top_p: raw.top_p,
            top_k: raw.top_k,
            min_p: raw.min_p,
            repeat_penalty: raw.repeat_penalty,
            repeat_last_n: raw.repeat_last_n,
            eos_token_ids,
        })
    }

    pub fn load(&self) -> Result<GenerationConfig> {
        let path = self.generation_config_file_loader.load()?;
        Self::parse_config(path)
    }

    pub async fn load_async(&self) -> Result<GenerationConfig> {
        let path = self.generation_config_file_loader.load_async().await?;
        Self::parse_config(path)
    }
}

#[derive(Clone)]
pub struct GgufModelLoader {
    pub model_file_loader: HfLoader,
}

impl GgufModelLoader {
    pub fn new(model_repo: &str, model_filename: &str) -> Self {
        let model_file_loader = HfLoader::new(model_repo, model_filename);

        Self { model_file_loader }
    }

    fn parse_gguf(
        model_file_path: std::path::PathBuf,
    ) -> Result<(std::fs::File, candle_core::quantized::gguf_file::Content)> {
        let mut file = std::fs::File::open(&model_file_path)?;
        let file_content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| e.with_path(model_file_path))?;

        Ok((file, file_content))
    }

    pub fn load(&self) -> Result<(std::fs::File, candle_core::quantized::gguf_file::Content)> {
        let model_file_path = self.model_file_loader.load()?;
        Self::parse_gguf(model_file_path)
    }

    pub async fn load_async(
        &self,
    ) -> Result<(std::fs::File, candle_core::quantized::gguf_file::Content)> {
        let model_file_path = self.model_file_loader.load_async().await?;
        Self::parse_gguf(model_file_path)
    }
}
