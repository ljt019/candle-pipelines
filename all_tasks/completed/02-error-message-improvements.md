# Error Message Improvements

## Summary
Enhance error messages with actionable context. Users should understand what went wrong and how to fix it without reading source code.

## Motivation
Current errors are bare-bones:
- `"Generation failed: {0}"` - what generation? what failed?
- `"Model not found: {0}"` - where did we look? is it a typo?
- `"Tokenization failed: {0}"` - what input caused it?

Users end up in source code trying to understand failures.

---

## Error Inventory

### Download Errors (4 sites)

| Location | Current Code | Context Available |
|----------|--------------|-------------------|
| `src/loaders.rs:76` | `.map_err(\|e\| TransformersError::Download(e.to_string()))?` | `self.repo`, `self.filename` |
| `src/loaders.rs:93` | `TransformersError::Download(error_msg)` | `self.repo`, `self.filename`, `attempt` count, `max_retries` |
| `src/loaders.rs:96` | `TransformersError::Download(error_msg)` | `self.repo`, `self.filename`, underlying error |
| `src/loaders.rs:103` | `TransformersError::Download("unknown failure")` | `self.repo`, `self.filename` |

### ModelMetadata Errors (10 sites)

| Location | Current Message | Context Available |
|----------|-----------------|-------------------|
| `src/loaders.rs:167` | `"Invalid EOS token ID"` | actual value received |
| `src/loaders.rs:173` | `"Invalid EOS token ID in array"` | actual value, array index |
| `src/models/gemma3.rs:98` | `"Missing critical metadata: gemma3.block_count"` | model path, available keys |
| `src/models/gemma3.rs:169` | `"Gemma3 generation config is missing 'eos_token_ids'..."` | config file path |
| `src/models/qwen3.rs:148` | `"Missing metadata key: qwen3.block_count"` | model path, available keys |
| `src/models/qwen3.rs:157` | `"Missing metadata key: qwen3.context_length"` | model path, available keys |
| `src/models/qwen3.rs:217` | `"Missing metadata key: qwen3.block_count"` | (duplicate site) |
| `src/models/qwen3.rs:226` | `"Missing metadata key: qwen3.context_length"` | (duplicate site) |
| `src/models/modernbert.rs:396` | `"Config missing 'entailment' in label2id"` | available labels |
| `src/models/modernbert.rs:470` | `"Config missing 'entailment' in label2id"` | available labels |

### ChatTemplate Errors (8 sites)

| Location | Current Code | Context Available |
|----------|--------------|-------------------|
| `src/models/gemma3.rs:142` | `"Missing 'chat_template' field in tokenizer config"` | config file path, model name |
| `src/models/gemma3.rs:154` | `e.to_string()` (minijinja error) | template name, model name |
| `src/models/gemma3.rs:310` | `e.to_string()` (get_template) | model name |
| `src/models/gemma3.rs:315` | `e.to_string()` (render) | model name, messages count |
| `src/models/qwen3.rs:104` | `"Missing 'chat_template' field in tokenizer config"` | config file path, model name |
| `src/models/qwen3.rs:137` | `e.to_string()` (add_template) | template name, model name |
| `src/models/qwen3.rs:468` | `e.to_string()` (get_template) | model name |
| `src/models/qwen3.rs:475` | `e.to_string()` (render) | model name, messages count, tools count |

### Tokenization Errors (18 sites)

| Location | Current Code | Context Available |
|----------|--------------|-------------------|
| `src/loaders.rs:125` | `"Failed to load tokenizer: {e}"` | `tokenizer_file_path` |
| `src/models/modernbert.rs:70` | `"Tokenization error: {e}"` | input text (first N chars) |
| `src/models/modernbert.rs:129` | `"Tokenization error: {e}"` | input text |
| `src/models/modernbert.rs:404` | `"Tokenization error: {e}"` | text, hypothesis |
| `src/models/modernbert.rs:712` | `"Tokenization error: {e}"` | input text |
| `src/models/modernbert.rs:755` | `"Tokenization error: {e}"` | input text |
| `src/models/modernbert.rs:913` | `"Failed to load tokenizer: {e}"` | tokenizer path |
| `src/pipelines/text_generation/pipeline.rs:93` | `e.to_string()` | input text |
| `src/pipelines/text_generation/pipeline.rs:176,203,266,278,439` | `e.to_string()` | templated prompt (first N chars) |
| `src/pipelines/text_generation/xml_pipeline.rs:74,88,145,157,339,440` | `e.to_string()` | templated prompt |
| `src/pipelines/text_generation/base_pipeline.rs:258,265,296,303` | `e.to_string()` | token being decoded |

### Generation Errors (16 sites)

| Location | Current Message | Context Available |
|----------|-----------------|-------------------|
| `src/pipelines/text_generation/base_pipeline.rs:127` | `"Model provided no EOS tokens; cannot run text generation"` | model name |
| `src/pipelines/text_generation/base_pipeline.rs:224` | `"Model provided no EOS tokens; cannot stream..."` | model name |
| `src/pipelines/fill_mask/pipeline.rs:23,34` | `"No predictions returned"` | input text |
| `src/models/modernbert.rs:77,136` | `"No [MASK] token found in input"` | input text (first N chars) |
| `src/models/modernbert.rs:234,272,517,559,821,862` | error msg or `"Unknown error"` | batch index |
| `src/models/modernbert.rs:725,772,885` | `"Predicted ID '{pred_id}' not in id2label"` | pred_id, available labels |

### ToolMessage Errors (8 sites)

| Location | Current Message | Context Available |
|----------|-----------------|-------------------|
| `tool_macro/src/lib.rs:218` | `e.to_string()` | tool name (from macro context) |
| `src/pipelines/text_generation/pipeline.rs:367` | `"Tool '{name}' not found"` | available tool names |
| `src/pipelines/text_generation/pipeline.rs:415,509` | `"No tools registered..."` | - |
| `src/pipelines/text_generation/xml_pipeline.rs:264` | `"Tool '{name}' not found"` | available tool names |
| `src/pipelines/text_generation/xml_pipeline.rs:315,409` | `"No tools registered..."` | - |

### ToolFormat Errors (5 sites)

| Location | Current Message | Context Available |
|----------|-----------------|-------------------|
| `tool_macro/src/lib.rs:211,226` | `e.to_string()` (serde) | tool name, expected schema, received JSON |
| `src/pipelines/text_generation/tools.rs:111` | `"schema serialization failed: {e}"` | tool name |
| `src/pipelines/text_generation/tools.rs:116` | `"invalid schema: {e}"` | tool name, schema |
| `src/pipelines/text_generation/tools.rs:128` | validation error messages | tool name, param name, expected type |

### Device Errors (1 site)

| Location | Current Code | Context Available |
|----------|--------------|-------------------|
| `src/pipelines/utils/mod.rs:36` | `e.to_string()` | `DeviceRequest` variant, device index |

---

## Design Decision: `#[non_exhaustive]` Enums

We use simple enums with `#[non_exhaustive]` rather than opaque structs with `.kind()` methods.

**Why not the `std::io::Error` pattern?**
- The `.kind()` + accessor methods pattern is verbose and awkward to use
- It exists for extreme semver stability in libraries with millions of users
- For a 0.x library, it's overkill

**Why `#[non_exhaustive]`?**
- Users must include `_ =>` in match arms
- We can add new variants in minor versions without breaking downstream code
- We keep nice destructuring syntax: `DownloadError::Timeout { repo, .. } =>`

---

## New Error Definitions

Replace the current `String` wrappers in `src/error.rs`:

```rust
use thiserror::Error;

// ============================================================================
// DOWNLOAD ERRORS
// ============================================================================

#[derive(Error, Debug)]
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

// ============================================================================
// MODEL METADATA ERRORS
// ============================================================================

#[derive(Error, Debug)]
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

/// Helper to format available keys nicely (max 5, then "...")
fn format_keys(keys: &[String]) -> String {
    if keys.len() <= 5 {
        keys.join(", ")
    } else {
        format!("{}, ... ({} more)", keys[..5].join(", "), keys.len() - 5)
    }
}

// ============================================================================
// CHAT TEMPLATE ERRORS
// ============================================================================

#[derive(Error, Debug)]
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

// ============================================================================
// TOKENIZATION ERRORS
// ============================================================================

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum TokenizationError {
    #[error("Failed to load tokenizer from '{path}': {reason}")]
    LoadFailed { path: String, reason: String },

    #[error("Tokenization failed on '{input_preview}': {reason}")]
    EncodeFailed {
        input_preview: String,  // first 50 chars
        reason: String,
    },

    #[error("Failed to decode token {token_id}: {reason}")]
    DecodeFailed { token_id: u32, reason: String },
}

impl TokenizationError {
    /// Create an encode error, truncating input to first 50 chars
    pub fn encode_failed(input: &str, reason: impl Into<String>) -> Self {
        let preview: String = input.chars().take(50).collect();
        Self::EncodeFailed {
            input_preview: preview,
            reason: reason.into(),
        }
    }
}

// ============================================================================
// GENERATION ERRORS
// ============================================================================

#[derive(Error, Debug)]
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

// ============================================================================
// TOOL ERRORS
// ============================================================================

#[derive(Error, Debug)]
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

// ============================================================================
// DEVICE ERRORS
// ============================================================================

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum DeviceError {
    #[error("Failed to init CUDA device {index}: {reason}. Try DeviceRequest::Cpu as fallback.")]
    CudaInitFailed { index: usize, reason: String },
}

// ============================================================================
// MAIN ERROR ENUM
// ============================================================================

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

    // Pass-through from dependencies
    #[error(transparent)]
    Candle(#[from] candle_core::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),

    // JSON mode (may expand later)
    #[error("JSON schema error: {0}")]
    JsonSchema(String),

    #[error("JSON parse error: {0}")]
    JsonParse(String),

    #[error("Invalid generation parameters: {0}")]
    InvalidParams(String),
}

pub type Result<T> = std::result::Result<T, TransformersError>;
```

---

## User-Facing Example

```rust
use transformers::{TransformersError, DownloadError, ToolError};

async fn run() -> transformers::Result<()> {
    let pipeline = TextGenerationPipeline::builder()
        .qwen3(Qwen3Size::Size0_6B)
        .build()
        .await?;

    // ... use pipeline
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        match e {
            TransformersError::Download(DownloadError::Timeout { repo, attempts, .. }) => {
                eprintln!("Download timed out from {repo} after {attempts} tries");
                std::process::exit(2);
            }
            TransformersError::Tool(ToolError::NotFound { name, available }) => {
                eprintln!("Unknown tool '{name}'. Did you mean: {}?", available.join(", "));
                std::process::exit(3);
            }
            _ => {
                // Required because of #[non_exhaustive] - handles future variants gracefully
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        }
    }
}
```

---

## Migration Examples

### Download Error

**Before** (`src/loaders.rs:96`):
```rust
return Err(TransformersError::Download(error_msg));
```

**After**:
```rust
return Err(DownloadError::Failed {
    repo: self.repo.clone(),
    file: self.filename.clone(),
    reason: error_msg,
}.into());
```

### Tool Not Found

**Before** (`src/pipelines/text_generation/pipeline.rs:367`):
```rust
TransformersError::ToolMessage(format!("Tool '{}' not found", call.name))
```

**After**:
```rust
ToolError::NotFound {
    name: call.name.clone(),
    available: tools.iter().map(|t| t.name.clone()).collect(),
}.into()
```

### Tokenization Error

**Before** (`src/models/modernbert.rs:70`):
```rust
.map_err(|e| TransformersError::Tokenization(format!("Tokenization error: {e}")))?
```

**After**:
```rust
.map_err(|e| TokenizationError::encode_failed(text, e.to_string()))?
```

### Model Metadata Error

**Before** (`src/models/qwen3.rs:148`):
```rust
TransformersError::ModelMetadata("Missing metadata key: qwen3.block_count".to_string())
```

**After**:
```rust
ModelMetadataError::MissingKey {
    key: "qwen3.block_count".into(),
    model_type: "Qwen3".into(),
    available: gguf.metadata.keys().cloned().collect(),
}.into()
```

---

## Test Cases

Add to `src/error.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // DOWNLOAD ERROR TESTS
    // ========================================================================

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

    // ========================================================================
    // MODEL METADATA ERROR TESTS
    // ========================================================================

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

    // ========================================================================
    // TOKENIZATION ERROR TESTS
    // ========================================================================

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

    // ========================================================================
    // GENERATION ERROR TESTS
    // ========================================================================

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

    // ========================================================================
    // TOOL ERROR TESTS
    // ========================================================================

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

    // ========================================================================
    // DEVICE ERROR TESTS
    // ========================================================================

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

    // ========================================================================
    // TRANSFORMERS ERROR CONVERSIONS
    // ========================================================================

    #[test]
    fn download_error_converts() {
        let err: TransformersError = DownloadError::Failed {
            repo: "org/model".into(),
            file: "weights.gguf".into(),
            reason: "404".into(),
        }.into();

        assert!(matches!(err, TransformersError::Download(_)));
        assert!(err.to_string().contains("org/model"));
    }

    #[test]
    fn tool_error_converts() {
        let err: TransformersError = ToolError::NoToolsRegistered.into();
        assert!(matches!(err, TransformersError::Tool(_)));
    }

    // ========================================================================
    // NON-EXHAUSTIVE BEHAVIOR
    // ========================================================================

    #[test]
    fn match_requires_wildcard() {
        let err = DownloadError::Failed {
            repo: "x".into(),
            file: "y".into(),
            reason: "z".into(),
        };

        // This compiles because we have _ arm (required by #[non_exhaustive])
        let msg = match err {
            DownloadError::Failed { reason, .. } => reason,
            DownloadError::Timeout { .. } => "timeout".into(),
            _ => "other".into(),  // Required!
        };

        assert_eq!(msg, "z");
    }
}
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/error.rs` | Replace current enum with new `#[non_exhaustive]` error enums |
| `src/loaders.rs` | 5 sites → `DownloadError`, `TokenizationError` |
| `src/models/qwen3.rs` | 8 sites → `ModelMetadataError`, `ChatTemplateError` |
| `src/models/gemma3.rs` | 6 sites → `ModelMetadataError`, `ChatTemplateError` |
| `src/models/modernbert.rs` | 15 sites → `TokenizationError`, `GenerationError`, `ModelMetadataError` |
| `src/pipelines/text_generation/pipeline.rs` | 10 sites → `TokenizationError`, `ToolError` |
| `src/pipelines/text_generation/xml_pipeline.rs` | 10 sites → `TokenizationError`, `ToolError` |
| `src/pipelines/text_generation/base_pipeline.rs` | 6 sites → `TokenizationError`, `GenerationError` |
| `src/pipelines/text_generation/tools.rs` | 3 sites → `ToolError` |
| `src/pipelines/utils/mod.rs` | 1 site → `DeviceError` |
| `tool_macro/src/lib.rs` | 3 sites → `ToolError` |

---

## Backward Compatibility

- `TransformersError` still implements `std::error::Error`, `Debug`, `Display`
- `Result<T>` type alias unchanged
- `?` propagation works via `#[from]` attributes
- Existing `impl From<X> for TransformersError` preserved for dependency errors

## Breaking Changes

- `TransformersError::Download(String)` → `TransformersError::Download(DownloadError)`
- `TransformersError::ToolMessage(String)` → `TransformersError::Tool(ToolError)`
- etc. for all error types
- Error message strings will change (this is the point!)
- Users matching on errors must now use `_ =>` arm (due to `#[non_exhaustive]`)
