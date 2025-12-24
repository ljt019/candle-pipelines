use thiserror::Error;

#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum DownloadError {
    #[error("Failed to download '{file}' from '{repo}': {reason}")]
    Failed {
        repo: String,
        file: String,
        reason: String,
    },

    #[error("Download timed out for '{file}' from '{repo}' after {attempts} attempt(s)")]
    Timeout {
        repo: String,
        file: String,
        attempts: u32,
    },

    #[error("Failed to initialize HuggingFace API: {reason}")]
    ApiInit { reason: String },
}

#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum ModelMetadataError {
    #[error("Missing required metadata key '{key}' for {model_type} model. Available: {}", format_keys(.available))]
    MissingKey {
        key: String,
        model_type: String,
        available: Vec<String>,
    },

    #[error("Invalid value for '{key}': expected {expected}, got {actual}")]
    InvalidValue {
        key: String,
        expected: String,
        actual: String,
    },

    #[error("Missing '{label}' in label2id mapping. Available: {}", .available.join(", "))]
    MissingLabel {
        label: String,
        available: Vec<String>,
    },

    #[error("Missing 'eos_token_ids' in generation config for {model}. Cannot determine when to stop generation.")]
    MissingEosTokens { model: String },
}

fn format_keys(keys: &[String]) -> String {
    if keys.len() <= 5 {
        keys.join(", ")
    } else {
        format!("{}, ... ({} more)", keys[..5].join(", "), keys.len() - 5)
    }
}

#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum ChatTemplateError {
    #[error("Missing 'chat_template' in tokenizer config for {model}")]
    MissingTemplate { model: String },

    #[error("Failed to parse chat template for {model}: {reason}")]
    ParseFailed { model: String, reason: String },

    #[error("Failed to render template for {model} ({message_count} messages): {reason}")]
    RenderFailed {
        model: String,
        message_count: usize,
        reason: String,
    },
}

#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum TokenizationError {
    #[error("Failed to load tokenizer from '{path}': {reason}")]
    LoadFailed { path: String, reason: String },

    #[error("Tokenization failed on '{input_preview}': {reason}")]
    EncodeFailed {
        input_preview: String,
        reason: String,
    },

    #[error("Failed to decode token {token_id}: {reason}")]
    DecodeFailed { token_id: u32, reason: String },
}

impl TokenizationError {
    pub(crate) fn encode_failed(input: &str, reason: impl Into<String>) -> Self {
        let preview: String = input.chars().take(50).collect();
        Self::EncodeFailed {
            input_preview: preview,
            reason: reason.into(),
        }
    }
}

#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum GenerationError {
    #[error("Reached max_len ({max_len} tokens) after generating {generated} tokens. Increase max_len or shorten prompt.")]
    MaxTokensReached { max_len: usize, generated: usize },

    #[error("No EOS tokens configured for model. Cannot determine when to stop.")]
    NoEosTokens,

    #[error("No [MASK] token in input '{input_preview}'. Fill-mask requires exactly one [MASK].")]
    NoMaskToken { input_preview: String },

    #[error("Model returned no predictions")]
    NoPredictions,

    #[error("Predicted label ID {id} not in id2label. Available: {}", .available.join(", "))]
    UnknownLabelId { id: i64, available: Vec<String> },

    #[error("Batch item {index} failed: {reason}")]
    BatchItemFailed { index: usize, reason: String },
}

#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum ToolError {
    #[error("Tool '{name}' not found. Registered tools: {}", .available.join(", "))]
    NotFound {
        name: String,
        available: Vec<String>,
    },

    #[error("No tools registered. Call register_tools() before completion_with_tools().")]
    NoToolsRegistered,

    #[error("Tool '{name}' failed after {attempts} attempt(s): {reason}")]
    ExecutionFailed {
        name: String,
        attempts: u32,
        reason: String,
    },

    #[error("Invalid parameters for '{name}': {reason}")]
    InvalidParams { name: String, reason: String },

    #[error("Schema error for '{name}': {reason}")]
    SchemaError { name: String, reason: String },
}

#[derive(Error, Debug, Clone)]
#[non_exhaustive]
pub enum DeviceError {
    #[error("Failed to init CUDA device {index}: {reason}. Try DeviceRequest::Cpu as fallback.")]
    CudaInitFailed { index: usize, reason: String },
}

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum TransformersError {
    #[error(transparent)]
    Download(#[from] DownloadError),

    #[error(transparent)]
    ModelMetadata(#[from] ModelMetadataError),

    #[error(transparent)]
    ChatTemplate(#[from] ChatTemplateError),

    #[error(transparent)]
    Tokenization(#[from] TokenizationError),

    #[error(transparent)]
    Generation(#[from] GenerationError),

    #[error(transparent)]
    Tool(#[from] ToolError),

    #[error(transparent)]
    Device(#[from] DeviceError),

    #[error("Candle error: {0}")]
    Candle(String),

    #[error("IO error: {0}")]
    Io(String),

    #[error("JSON error: {0}")]
    SerdeJson(String),

    #[error("JSON schema error: {0}")]
    JsonSchema(String),

    #[error("JSON parse error: {0}")]
    JsonParse(String),

    #[error("Invalid generation parameters: {0}")]
    InvalidParams(String),
}

pub type Result<T> = std::result::Result<T, TransformersError>;

impl From<candle_core::Error> for TransformersError {
    fn from(value: candle_core::Error) -> Self {
        TransformersError::Candle(value.to_string())
    }
}

impl From<std::io::Error> for TransformersError {
    fn from(value: std::io::Error) -> Self {
        TransformersError::Io(value.to_string())
    }
}

impl From<serde_json::Error> for TransformersError {
    fn from(value: serde_json::Error) -> Self {
        TransformersError::SerdeJson(value.to_string())
    }
}

impl From<hf_hub::api::sync::ApiError> for TransformersError {
    fn from(value: hf_hub::api::sync::ApiError) -> Self {
        DownloadError::Failed {
            repo: "unknown".into(),
            file: "unknown".into(),
            reason: value.to_string(),
        }
        .into()
    }
}

impl From<regex::Error> for TransformersError {
    fn from(value: regex::Error) -> Self {
        GenerationError::BatchItemFailed {
            index: 0,
            reason: value.to_string(),
        }
        .into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_failed_includes_context() {
        let err = DownloadError::Failed {
            repo: "unsloth/Qwen3-0.6B-GGUF".into(),
            file: "qwen3-0.6b-q4_k_m.gguf".into(),
            reason: "connection reset".into(),
        };
        let msg = err.to_string();

        assert!(msg.contains("unsloth/Qwen3-0.6B-GGUF"));
        assert!(msg.contains("qwen3-0.6b-q4_k_m.gguf"));
        assert!(msg.contains("connection reset"));
    }

    #[test]
    fn download_timeout_shows_attempts() {
        let err = DownloadError::Timeout {
            repo: "org/model".into(),
            file: "model.gguf".into(),
            attempts: 3,
        };
        let msg = err.to_string();

        assert!(msg.contains("3 attempt"));
    }

    #[test]
    fn missing_key_shows_available() {
        let err = ModelMetadataError::MissingKey {
            key: "qwen3.block_count".into(),
            model_type: "Qwen3".into(),
            available: vec!["qwen3.vocab_size".into(), "qwen3.hidden_size".into()],
        };
        let msg = err.to_string();

        assert!(msg.contains("qwen3.block_count"));
        assert!(msg.contains("Qwen3"));
        assert!(msg.contains("qwen3.vocab_size"));
    }

    #[test]
    fn missing_label_shows_available() {
        let err = ModelMetadataError::MissingLabel {
            label: "entailment".into(),
            available: vec!["positive".into(), "negative".into()],
        };
        let msg = err.to_string();

        assert!(msg.contains("entailment"));
        assert!(msg.contains("positive"));
    }

    #[test]
    fn encode_failed_truncates_long_input() {
        let long_input = "a".repeat(200);
        let err = TokenizationError::encode_failed(&long_input, "invalid utf-8");

        match err {
            TokenizationError::EncodeFailed { input_preview, .. } => {
                assert_eq!(input_preview.len(), 50);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn load_failed_includes_path() {
        let err = TokenizationError::LoadFailed {
            path: "/path/to/tokenizer.json".into(),
            reason: "file not found".into(),
        };
        let msg = err.to_string();

        assert!(msg.contains("/path/to/tokenizer.json"));
        assert!(msg.contains("file not found"));
    }

    #[test]
    fn max_tokens_shows_limits() {
        let err = GenerationError::MaxTokensReached {
            max_len: 2048,
            generated: 2048,
        };
        let msg = err.to_string();

        assert!(msg.contains("2048"));
        assert!(msg.contains("max_len"));
    }

    #[test]
    fn no_mask_shows_input() {
        let err = GenerationError::NoMaskToken {
            input_preview: "The quick brown fox".into(),
        };
        let msg = err.to_string();

        assert!(msg.contains("The quick brown fox"));
        assert!(msg.contains("[MASK]"));
    }

    #[test]
    fn tool_not_found_shows_available() {
        let err = ToolError::NotFound {
            name: "get_weather".into(),
            available: vec!["search".into(), "calculate".into()],
        };
        let msg = err.to_string();

        assert!(msg.contains("get_weather"));
        assert!(msg.contains("search"));
    }

    #[test]
    fn tool_execution_shows_attempts() {
        let err = ToolError::ExecutionFailed {
            name: "api_call".into(),
            attempts: 3,
            reason: "timeout".into(),
        };
        let msg = err.to_string();

        assert!(msg.contains("api_call"));
        assert!(msg.contains("3 attempt"));
    }

    #[test]
    fn no_tools_registered_suggests_fix() {
        let err = ToolError::NoToolsRegistered;
        assert!(err.to_string().contains("register_tools()"));
    }

    #[test]
    fn cuda_init_suggests_fallback() {
        let err = DeviceError::CudaInitFailed {
            index: 0,
            reason: "CUDA driver not found".into(),
        };
        let msg = err.to_string();

        assert!(msg.contains("device 0"));
        assert!(msg.contains("DeviceRequest::Cpu"));
    }

    #[test]
    fn download_error_converts() {
        let err: TransformersError = DownloadError::Failed {
            repo: "org/model".into(),
            file: "weights.gguf".into(),
            reason: "404".into(),
        }
        .into();

        assert!(matches!(err, TransformersError::Download(_)));
        assert!(err.to_string().contains("org/model"));
    }

    #[test]
    fn tool_error_converts() {
        let err: TransformersError = ToolError::NoToolsRegistered.into();
        assert!(matches!(err, TransformersError::Tool(_)));
    }

    #[test]
    fn match_requires_wildcard() {
        let err = DownloadError::Failed {
            repo: "x".into(),
            file: "y".into(),
            reason: "z".into(),
        };

        let msg = match err {
            DownloadError::Failed { reason, .. } => reason,
            DownloadError::Timeout { .. } => "timeout".into(),
            _ => "other".into(),
        };

        assert_eq!(msg, "z");
    }
}
