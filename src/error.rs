use crate::pipelines::text_generation::tools::ToolError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TransformersError {
    // Model loading
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Invalid model format: {0}")]
    ModelFormat(String),

    #[error("Model metadata missing: {0}")]
    ModelMetadata(String),

    // Tokenization
    #[error("Tokenizer not found: {0}")]
    TokenizerNotFound(String),

    #[error("Tokenization failed: {0}")]
    Tokenization(String),

    // Generation
    #[error("Generation failed: {0}")]
    Generation(String),

    #[error("Max tokens exceeded")]
    MaxTokens,

    #[error("Invalid generation parameters: {0}")]
    InvalidParams(String),

    // Chat/Template
    #[error("Chat template error: {0}")]
    ChatTemplate(String),

    // Tools
    #[error("Tool error: {0}")]
    Tool(#[from] ToolError),

    // Network/Download
    #[error("Download failed: {0}")]
    Download(String),

    // Device
    #[error("Device error: {0}")]
    Device(String),

    // JSON mode
    #[error("JSON schema error: {0}")]
    JsonSchema(String),

    #[error("JSON parse error: {0}")]
    JsonParse(String),

    // Pass-through from dependencies
    #[error(transparent)]
    Candle(#[from] candle_core::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, TransformersError>;

impl From<hf_hub::api::sync::ApiError> for TransformersError {
    fn from(value: hf_hub::api::sync::ApiError) -> Self {
        TransformersError::Download(value.to_string())
    }
}

impl From<regex::Error> for TransformersError {
    fn from(value: regex::Error) -> Self {
        TransformersError::Generation(value.to_string())
    }
}
