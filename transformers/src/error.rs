//! The error types for this crate.
//!
//! Most functions return [`Result<T>`] which uses `TransformersError` as the error type.

use serde::Serialize;
use thiserror::Error;

/// Errors that can occur during a file download operation.
///
/// This enum represents potential failure scenarios in a file download context,
/// including general failures, timeouts, and initialization errors.
#[derive(Error, Debug, Clone, Serialize)]
#[non_exhaustive]
pub enum DownloadError {
    /// Failed to download a file from a given HF repository.
    #[error("Failed to download '{file}' from '{repo}': {reason}")]
    Failed {
        /// The repository from which the file was being downloaded.
        repo: String,
        /// The file that failed to download.
        file: String,
        /// The reason why the download failed.
        reason: String,
    },

    /// A download operation timed out.
    ///
    /// This error occurs when the download operation exceeds the retry limit.
    #[error("Download timed out for '{file}' from '{repo}' after {attempts} attempt(s)")]
    Timeout {
        /// The repository from which the download was attempted.
        repo: String,
        /// The file that could not be downloaded before the timeout.
        file: String,
        /// The number of attempts made before timing out.
        attempts: u32,
    },

    /// Failed to initialize the HuggingFace API.
    #[error("Failed to initialize HuggingFace API: {reason}")]
    ApiInit {
        /// The reason HF returned as to why the API initialization failed.
        reason: String,
    },
}

/// Errors that can occur when loading model metadata.
///
/// This enum represents potential failure scenarios in model metadata loading,
/// including missing keys, invalid values, and missing labels.
#[derive(Error, Debug, Clone, Serialize)]
#[non_exhaustive]
pub enum ModelMetadataError {
    /// Missing required metadata key for a model.
    #[error("Missing required metadata key '{key}' for {model_type} model. Available: {}", format_keys(.available))]
    MissingKey {
        /// The key that is missing from the model metadata.
        key: String,
        /// The type of model that is missing the key.
        model_type: String,
        /// The available keys for the model.
        available: Vec<String>,
    },

    /// Invalid value for a metadata key.
    #[error("Invalid value for '{key}': expected {expected}, got {actual}")]
    InvalidValue {
        /// The key that has an invalid value.
        key: String,
        /// The expected value for the key.
        expected: String,
        /// The actual value for the key.
        actual: String,
    },

    /// Missing label in label2id mapping.
    #[error("Missing '{label}' in label2id mapping. Available: {}", .available.join(", "))]
    MissingLabel {
        /// The label that is missing from the label2id mapping.
        label: String,
        /// The available labels for the model.
        available: Vec<String>,
    },

    /// Missing EOS token IDs in generation config.
    #[error("Missing 'eos_token_ids' in generation config for {model}. Cannot determine when to stop generation.")]
    MissingEosTokens {
        /// The model that is missing the EOS token IDs.
        model: String,
    },
}

fn format_keys(keys: &[String]) -> String {
    if keys.len() <= 5 {
        keys.join(", ")
    } else {
        format!("{}, ... ({} more)", keys[..5].join(", "), keys.len() - 5)
    }
}

/// Errors that can occur when loading chat template.
///
/// This enum represents potential failure scenarios in chat template loading,
/// including missing template, parsing failures, and rendering failures.
#[derive(Error, Debug, Clone, Serialize)]
#[non_exhaustive]
pub enum ChatTemplateError {
    /// Missing chat template in tokenizer config.
    #[error("Missing 'chat_template' in tokenizer config for {model}")]
    MissingTemplate {
        /// The model that is missing the chat template.
        model: String,
    },

    /// Failed to parse chat template.
    #[error("Failed to parse chat template for {model}: {reason}")]
    ParseFailed {
        /// The model that failed to parse the chat template.
        model: String,
        /// The reason why the chat template failed to parse.
        reason: String,
    },

    /// Failed to render chat template.
    #[error("Failed to render template for {model} ({message_count} messages): {reason}")]
    RenderFailed {
        /// The model that failed to render the chat template.
        model: String,
        /// The number of messages in the chat template.
        message_count: usize,
        /// The reason why the chat template failed to render.
        reason: String,
    },
}

/// Errors that can occur when tokenizing input text.
///
/// This enum represents potential failure scenarios in tokenization,
/// including loading failures, encoding failures, and decoding failures.
#[derive(Error, Debug, Clone, Serialize)]
#[non_exhaustive]
pub enum TokenizationError {
    /// Failed to load tokenizer from a given path.
    #[error("Failed to load tokenizer from '{path}': {reason}")]
    LoadFailed {
        /// The path that failed to load the tokenizer.
        path: String,
        /// The reason why the tokenizer failed to load.
        reason: String,
    },

    /// Tokenization failed on a given input.
    #[error("Tokenization failed on '{input_preview}': {reason}")]
    EncodeFailed {
        /// The preview of the input text that failed to encode.
        input_preview: String,
        /// The reason why the tokenization failed to encode.
        reason: String,
    },

    /// Failed to decode a token.
    #[error("Failed to decode token {token_id}: {reason}")]
    DecodeFailed {
        /// The ID of the token that failed to decode.
        token_id: u32,
        /// The reason why the token failed to decode.
        reason: String,
    },
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

/// Errors that can occur when generating text.
///
/// This enum represents potential failure scenarios in text generation,
/// including max tokens reached, no EOS tokens, no mask token, no predictions,
/// unknown label ID, and batch item failed.
#[derive(Error, Debug, Clone, Serialize)]
#[non_exhaustive]
pub enum GenerationError {
    /// Reached max_len after trying to generate requested tokens.
    #[error("Reached max_len ({max_len} tokens) after generating {generated} tokens. Increase max_len or shorten prompt.")]
    MaxTokensReached {
        /// The maximum length allowed.
        max_len: usize,
        /// The number of tokens generated.
        generated: usize,
    },

    /// No EOS tokens configured for model.
    #[error("No EOS tokens configured for model. Cannot determine when to stop.")]
    NoEosTokens,

    /// No [MASK] token in input.
    #[error("No [MASK] token in input '{input_preview}'. Fill-mask requires exactly one [MASK].")]
    NoMaskToken {
        /// The preview of the input text that is missing the [MASK] token.
        input_preview: String,
    },

    /// Model returned no predictions.
    #[error("Model returned no predictions")]
    NoPredictions,

    /// Predicted label ID not in id2label.
    #[error("Predicted label ID {id} not in id2label. Available: {}", .available.join(", "))]
    UnknownLabelId {
        /// The predicted label ID that is not in the id2label mapping.
        id: i64,
        /// The available labels for the model.
        available: Vec<String>,
    },

    /// Batch item failed.
    #[error("Batch item {index} failed: {reason}")]
    BatchItemFailed {
        /// The index of the batch item that failed.
        index: usize,
        /// The reason why the batch item failed.
        reason: String,
    },
}

/// Errors that can occur when using tools.
///
/// This enum represents potential failure scenarios in tool usage,
/// including tool not found, no tools registered, tool execution failed,
/// invalid parameters, and schema error.
#[derive(Error, Debug, Clone, Serialize)]
#[non_exhaustive]
pub enum ToolError {
    /// Tool not found.
    #[error("Tool '{name}' not found. Registered tools: {}", .available.join(", "))]
    NotFound {
        /// The name of the tool that was not found.
        name: String,
        /// The available tools.
        available: Vec<String>,
    },

    /// No tools registered.
    #[error("No tools registered. Call register_tools() before completion_with_tools().")]
    NoToolsRegistered,

    /// Tool execution failed.
    #[error("Tool '{name}' failed after {attempts} attempt(s): {reason}")]
    ExecutionFailed {
        /// The name of the tool that failed to execute.
        name: String,
        /// The number of attempts made before the tool execution failed.
        attempts: u32,
        /// The reason why the tool failed to execute.
        reason: String,
    },

    /// Invalid parameters for a tool.
    #[error("Invalid parameters for '{name}': {reason}")]
    InvalidParams {
        /// The name of the tool that has invalid parameters.
        name: String,
        /// The reason why the tool has invalid parameters.
        reason: String,
    },

    /// Schema error for a tool.
    #[error("Schema error for '{name}': {reason}")]
    SchemaError {
        /// The name of the tool that has a schema error.
        name: String,
        /// The reason why the tool has a schema error.
        reason: String,
    },
}

/// Errors that can occur when initializing a CUDA device.
///
/// This enum represents potential failure scenarios in CUDA device initialization,
/// including CUDA driver not found and CUDA device index out of bounds.
#[derive(Error, Debug, Clone, Serialize)]
#[non_exhaustive]
pub enum DeviceError {
    /// Failed to init cuda device.
    #[error("Failed to init CUDA device {index}: {reason}. Try DeviceRequest::Cpu as fallback.")]
    CudaInitFailed {
        /// The index of the CUDA device that failed to initialize.
        index: usize,
        /// The reason why the CUDA device failed to initialize.
        reason: String,
    },
}

/// The unified error type for all crate errors.
///
/// This enum wraps domain-specific errors ([`DownloadError`], [`GenerationError`], etc.)
/// and errors from external crates (Candle, IO, JSON). Use `?` to propagate or match
/// on variants for granular handling.
#[derive(Error, Debug, Serialize)]
#[non_exhaustive]
pub enum TransformersError {
    /// Errors that can occur during a file download operation.
    #[error(transparent)]
    Download(#[from] DownloadError),

    /// Errors that can occur when loading model metadata.
    #[error(transparent)]
    ModelMetadata(#[from] ModelMetadataError),

    /// Errors that can occur when loading chat template.
    #[error(transparent)]
    ChatTemplate(#[from] ChatTemplateError),

    /// Errors that can occur when tokenizing input text.
    #[error(transparent)]
    Tokenization(#[from] TokenizationError),

    /// Errors that can occur when generating text.
    #[error(transparent)]
    Generation(#[from] GenerationError),

    /// Errors that can occur when using tools.
    #[error(transparent)]
    Tool(#[from] ToolError),

    /// Errors that can occur when initializing a CUDA device.
    #[error(transparent)]
    Device(#[from] DeviceError),

    /// Errors that can occur when using Candle.
    #[error("Candle error: {0}")]
    Candle(String),

    /// Errors that can occur when using IO.
    #[error("IO error: {0}")]
    Io(String),

    /// Errors that can occur when using JSON.
    #[error("JSON error: {0}")]
    SerdeJson(String),

    /// JSON schema error.
    #[error("JSON schema error: {0}")]
    JsonSchema(String),

    /// JSON parse error.
    #[error("JSON parse error: {0}")]
    JsonParse(String),

    /// Invalid generation parameters.
    #[error("Invalid generation parameters: {0}")]
    InvalidParams(String),
}

/// A [`Result`](std::result::Result) alias using [`TransformersError`] as the error type.
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
}
