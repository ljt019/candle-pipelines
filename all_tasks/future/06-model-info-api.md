# Model Info/Metadata API

## Summary
Provide a `model_info()` method that returns structured metadata about the loaded model.

## Motivation
Users want to know:
- What model is loaded?
- What size/variant?
- What's the context length?
- What quantization?
- What device is it running on?

Currently this info is scattered across internals or unavailable.

## API Design

```rust
/// Information about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model family name (e.g., "Qwen3", "Gemma3")
    pub family: String,
    
    /// Model size/variant (e.g., "0.6B", "4B")
    pub size: String,
    
    /// Full model identifier (e.g., "qwen3-0.6b")
    pub name: String,
    
    /// Maximum context length in tokens
    pub max_context_length: usize,
    
    /// Device the model is running on
    pub device: String,  // "cpu", "cuda:0", etc.
    
    /// Quantization format if applicable
    pub quantization: Option<String>,  // "Q4_K_M", etc.
}

impl<M: TextGenerationModel> TextGenerationPipeline<M> {
    /// Get information about the loaded model.
    ///
    /// # Example
    /// ```rust
    /// let info = pipeline.model_info().await;
    /// println!("Running {} on {}", info.name, info.device);
    /// println!("Context limit: {} tokens", info.max_context_length);
    /// ```
    pub async fn model_info(&self) -> ModelInfo { ... }
}
```

## Implementation

### Option A: Trait Method
Add to `TextGenerationModel` trait:

```rust
pub trait TextGenerationModel {
    // ... existing methods ...
    
    fn model_info(&self) -> ModelInfo;
}
```

Each model implements it with their specific info.

### Option B: Build from Existing Data
Pipeline constructs `ModelInfo` from available data:

```rust
pub async fn model_info(&self) -> ModelInfo {
    let model = self.base.model.lock().await;
    ModelInfo {
        family: M::family_name(),  // new trait method
        size: M::size_name(&self.model_options),  // new trait method
        name: format!("{}-{}", M::family_name(), M::size_name(...)),
        max_context_length: model.get_max_seq_len(),
        device: format!("{:?}", self.base.device.location()),
        quantization: Some("Q4_K_M".to_string()),  // hardcoded for now
    }
}
```

### Recommended: Option A
Cleaner, each model knows its own metadata.

## Files to Modify
- `src/pipelines/text_generation/model.rs` - add trait method or `ModelInfo` struct
- `src/pipelines/text_generation/pipeline.rs` - add `model_info()` method
- `src/models/qwen3.rs` - implement for Qwen3
- `src/models/gemma3.rs` - implement for Gemma3
- Could also add to other pipelines (fill_mask, sentiment, zero_shot)

## Notes
- Quantization info might need to come from GGUF metadata
- Device info available from `candle_core::Device`
- Could expose as `Display` impl for easy printing

