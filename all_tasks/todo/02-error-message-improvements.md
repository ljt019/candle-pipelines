# Error Message Improvements

## Summary
Enhance error messages with actionable context. Users should understand what went wrong and how to fix it without reading source code.

## Motivation
Current errors are bare-bones:
- `"Generation failed: {0}"` - what generation? what failed?
- `"Model not found: {0}"` - where did we look? is it a typo?
- `"Tokenization failed: {0}"` - what input caused it?

Users end up in source code trying to understand failures.

## Approach

### 1. Add Context Structs for Rich Errors
Instead of `String` wrappers, use structs with relevant context:

```rust
#[derive(Error, Debug)]
pub enum TransformersError {
    #[error("Model download failed: {0}")]
    Download(#[from] DownloadError),
    
    #[error("Generation failed: {0}")]
    Generation(#[from] GenerationError),
    // ...
}

#[derive(Error, Debug)]
#[error("Failed to download '{file}' from '{repo}': {reason}")]
pub struct DownloadError {
    pub repo: String,
    pub file: String,
    pub reason: String,
}

#[derive(Error, Debug)]  
#[error("Generation stopped: {reason} (after {tokens_generated} tokens)")]
pub struct GenerationError {
    pub reason: String,
    pub tokens_generated: usize,
}
```

### 2. Error Categories to Improve

#### Download Errors
Before: `"Download failed: connection reset"`
After: `"Failed to download 'Qwen3-0.6B-Q4_K_M.gguf' from 'unsloth/Qwen3-0.6B-GGUF': connection reset. Check your internet connection or try again."`

#### Model Loading Errors
Before: `"Model metadata missing: qwen3.block_count"`
After: `"Invalid GGUF file: missing required metadata 'qwen3.block_count'. The file may be corrupted or not a Qwen3 model."`

#### Tokenization Errors
Before: `"Tokenization failed: unknown token"`
After: `"Tokenization failed for input (first 100 chars): '...'. Error: unknown token. This may indicate incompatible input encoding."`

#### Generation Errors
Before: `"Max tokens exceeded"`
After: `"Generation stopped: reached max_len limit of 2048 tokens. Increase max_len or use a shorter prompt."`

#### Tool Errors
Before: `"Tool error: connection failed"`
After: `"Tool 'get_weather' failed after 3 retries: connection failed. Consider using ErrorStrategy::ReturnToModel to let the model handle failures."`

### 3. Suggestions in Error Messages
Add actionable hints where possible:
- "Did you mean X?" for typos
- "Try Y" for common fixes
- "See docs at Z" for complex issues

## Implementation

### Phase 1: Audit Current Errors
Go through `src/error.rs` and each `TransformersError::*` usage, catalog what context is available at error sites.

### Phase 2: Create Context Structs
For errors that benefit from structured context, create dedicated error structs.

### Phase 3: Update Error Sites
Modify code that creates errors to include available context.

### Phase 4: Keep Backward Compat
Ensure `TransformersError` still implements standard traits, existing error handling still works.

## Files to Modify
- `src/error.rs` - error type definitions
- `src/loaders.rs` - download error context
- `src/models/*.rs` - model loading errors
- `src/pipelines/text_generation/base_pipeline.rs` - generation errors
- `src/pipelines/text_generation/pipeline.rs` - tool errors
- `src/pipelines/text_generation/tools.rs` - tool validation errors

## Testing
- Unit tests for error formatting
- Verify error messages are helpful in practice
- Check that `?` propagation still works cleanly

