# Improved Error Types

## Summary
Replace `anyhow::Error` with structured `TransformersError` enum across the entire codebase.

## Motivation
- Pattern matching on specific errors
- Better error messages
- Programmatic error handling (retry on certain errors, fail on others)
- Idiomatic Rust

## Current State
- `ToolError` exists in `pipelines::text_generation::tools`
- Everything else uses `anyhow::Result` (~150 usages across 27 files)
- Errors from candle wrapped as-is
- `GenerationStats` added (new stats module)

## Proposed Error Type

Single flat-ish enum, grouped by concept:

```rust
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
```

## Usage Examples

```rust
use transformers::{Result, TransformersError};

// User code
match pipeline.completion("test").await {
    Ok(result) => println!("{}", result),
    Err(TransformersError::MaxTokens) => {
        // Retry with shorter input
    }
    Err(TransformersError::Download(msg)) => {
        // Network issue, maybe retry
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Implementation — Big Bang

Replace all `anyhow::Result` with `transformers::Result` in one pass.

### Files to Modify

**Core:**
- `src/lib.rs` — export `TransformersError`, `Result`
- Create `src/error.rs` — define error types

**Loaders:**
- `src/loaders.rs` — return `TransformersError::Download`, `ModelNotFound`, etc.

**Models:**
- `src/models/qwen3.rs`
- `src/models/gemma3.rs`
- `src/models/modernbert.rs`

**Pipelines:**
- `src/pipelines/text_generation/*.rs` (includes `base_pipeline.rs`, `pipeline.rs`, `xml_pipeline.rs`, `streaming/*.rs`, `stats.rs`)
- `src/pipelines/fill_mask/*.rs`
- `src/pipelines/sentiment/*.rs`
- `src/pipelines/zero_shot/*.rs`
- `src/pipelines/cache.rs`
- `src/pipelines/utils/*.rs`

**Keep `ToolError` separate** — it's user-facing for tool authors, fold into `TransformersError::Tool`

### Conversion Patterns

```rust
// Before
anyhow::bail!("Model not found: {}", path)

// After
return Err(TransformersError::ModelNotFound(path.to_string()))
```

```rust
// Before
.map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?

// After
.map_err(|e| TransformersError::Tokenization(e.to_string()))?
```

```rust
// Before
.context("Failed to load model")?

// After (candle errors auto-convert via #[from])
?
// or for more context:
.map_err(|e| TransformersError::ModelFormat(format!("Failed to load: {}", e)))?
```

## Testing
- Ensure all existing tests still pass
- Add tests that check specific error variants are returned
- Test error Display output is helpful

## Notes
- Remove `anyhow` from dependencies after migration (or keep for internal use if needed)
- `thiserror` is already a dependency (used by `ToolError`)
