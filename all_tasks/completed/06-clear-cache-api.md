# clear_cache() / Context Reset API

## Summary
Expose a public method to reset the KV cache, allowing users to start fresh mid-conversation.

## Problem
Currently no public way to clear the internal context/KV cache:

```rust
// User wants to start a new conversation with same pipeline
let response1 = pipeline.completion(&messages1).await?;

// Now want to talk about something completely different
// But KV cache still has old context, affecting generation
let response2 = pipeline.completion(&messages2).await?;  // may be influenced by messages1
```

The pipeline internally manages cache reuse via prefix matching, but users can't force a reset.

## Solution
Add `clear_cache()` method to `TextGenerationPipeline`:

```rust
impl<M: TextGenerationModel> TextGenerationPipeline<M> {
    /// Clear the KV cache and reset context position.
    ///
    /// Call this when starting a completely new conversation to ensure
    /// no state from previous generations affects the new one.
    ///
    /// # Example
    /// ```rust
    /// // First conversation
    /// let response = pipeline.completion(&messages).await?;
    ///
    /// // Start fresh for unrelated conversation
    /// pipeline.clear_cache().await;
    /// let response = pipeline.completion(&new_messages).await?;
    /// ```
    pub async fn clear_cache(&self) {
        self.base.context.lock().await.reset();
        self.base.last_processed_tokens.lock().await.clear();
    }
}
```

## Implementation
The internals already exist:
- `context.reset()` clears KV cache
- `last_processed_tokens.clear()` resets prefix tracking

Just need to expose through public API.

## Files to Modify
- `src/pipelines/text_generation/pipeline.rs` - add `clear_cache()` method
- Same for `XmlGenerationPipeline` if it exists separately

## Notes
- Method is async because it needs to acquire locks
- Could also add to other pipelines (fill_mask, sentiment, zero_shot) if they have similar state
- Alternative name: `reset()`, `clear_context()`, `new_conversation()`

