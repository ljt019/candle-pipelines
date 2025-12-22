# max_context_length() Public API

## Summary
Expose `max_context_length()` publicly so users can check context limits before sending prompts.

## Problem
Users can't easily check how much context a model supports:

```rust
// User wants to know: "Will my 10k token prompt fit?"
// Currently no way to check without hitting the limit
```

The method exists internally but isn't prominently exposed:

```rust
// pipeline.rs - exists but users may not discover it
pub async fn max_context_length(&self) -> usize {
    self.base.model.lock().await.get_max_seq_len()
}
```

## Solution
1. Ensure `max_context_length()` is documented and discoverable
2. Consider adding `context_position()` too (already exists)

```rust
impl<M: TextGenerationModel> TextGenerationPipeline<M> {
    /// Returns the maximum context length (in tokens) supported by the model.
    ///
    /// Use this to check if your prompt will fit before sending.
    ///
    /// # Example
    /// ```rust
    /// let max_tokens = pipeline.max_context_length().await;
    /// println!("Model supports up to {} tokens", max_tokens);
    /// ```
    pub async fn max_context_length(&self) -> usize { ... }
    
    /// Returns the current position in the context (tokens already cached).
    ///
    /// Useful for tracking how much of the context window is used.
    pub async fn context_position(&self) -> usize { ... }
}
```

## Bonus: Token Counting Helper
Could also add a helper to count tokens for a prompt:

```rust
/// Count tokens in a prompt without running generation.
pub fn count_tokens(&self, text: &str) -> Result<usize> {
    let tokens = self.base.model_tokenizer
        .encode(text, false)
        .map_err(|e| TransformersError::Tokenization(e.to_string()))?;
    Ok(tokens.get_ids().len())
}
```

Then users can do:
```rust
let prompt_tokens = pipeline.count_tokens(&my_prompt)?;
let max = pipeline.max_context_length().await;
if prompt_tokens > max {
    // truncate or error
}
```

## Files to Modify
- `src/pipelines/text_generation/pipeline.rs` - document existing methods, maybe add `count_tokens()`

## Notes
- Methods already exist, this is mostly documentation + discoverability
- `count_tokens()` would be new but simple to implement

