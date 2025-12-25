//! Error types for this crate.
//!
//! All fallible operations return [`Result<T>`] which uses [`PipelineError`] as the error type.

use thiserror::Error;

/// A [`Result`](std::result::Result) alias using [`PipelineError`] as the error type.
pub type Result<T> = std::result::Result<T, PipelineError>;

/// The unified error type for all crate errors.
///
/// # Example
///
/// ```rust,no_run
/// use candle_pipelines::error::PipelineError;
///
/// fn handle_error(e: PipelineError) {
///     match &e {
///         PipelineError::Download(_) => {
///             // Network issue - retry with backoff
///         }
///         PipelineError::Device(_) => {
///             // GPU unavailable - fall back to CPU
///         }
///         PipelineError::Tokenization(_) => {
///             // Bad input - fix and retry
///         }
///         PipelineError::Tool(_) => {
///             // Tool misconfigured - fix tool setup
///         }
///         PipelineError::Unexpected(_) => {
///             // Internal error - report bug
///             eprintln!("Internal error: {e}");
///         }
///         _ => {
///             // Future error variants
///         }
///     }
/// }
/// ```
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum PipelineError {
    /// Network or download failure. Retry may help.
    #[error("{0}")]
    Download(String),

    /// Tokenization failure. Check input text.
    #[error("{0}")]
    Tokenization(String),

    /// Tool configuration or execution failure. Fix tool setup.
    #[error("{0}")]
    Tool(String),

    /// Device initialization failure. Fall back to CPU.
    #[error("{0}")]
    Device(String),

    /// Internal error. Report if seen.
    #[error("{0}")]
    Unexpected(String),
}

impl From<hf_hub::api::sync::ApiError> for PipelineError {
    fn from(value: hf_hub::api::sync::ApiError) -> Self {
        PipelineError::Download(format!("HuggingFace API error: {}", value))
    }
}

impl From<candle_core::Error> for PipelineError {
    fn from(value: candle_core::Error) -> Self {
        PipelineError::Unexpected(value.to_string())
    }
}

impl From<std::io::Error> for PipelineError {
    fn from(value: std::io::Error) -> Self {
        PipelineError::Unexpected(value.to_string())
    }
}

impl From<serde_json::Error> for PipelineError {
    fn from(value: serde_json::Error) -> Self {
        PipelineError::Unexpected(value.to_string())
    }
}
