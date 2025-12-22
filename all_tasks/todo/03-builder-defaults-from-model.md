# Builder Defaults from Model

## Summary
Pipeline builders should use model-specific default generation params instead of hardcoded defaults.

## Problem
Currently `TextGenerationPipelineBuilder` uses `GenerationParams::default()`:

```rust
// builder.rs
impl<M: TextGenerationModel> TextGenerationPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self {
            model_options: options,
            gen_params: GenerationParams::default(),  // ← hardcoded!
            device_request: DeviceRequest::Default,
        }
    }
}
```

But models define their own optimal defaults:

```rust
// qwen3.rs
fn default_generation_params(&self) -> GenerationParams {
    GenerationParams {
        temperature: self.generation_config.temperature.unwrap_or(0.6),
        top_p: Some(self.generation_config.top_p.unwrap_or(0.95)),
        top_k: Some(self.generation_config.top_k.unwrap_or(20) as usize),
        // ... model-specific values from HF config
    }
}
```

Users get generic defaults instead of what the model was tuned for.

## Solution

### Option A: Defer to Model at Build Time
In `build()`, merge user-set params with model defaults:

```rust
pub async fn build(self) -> Result<TextGenerationPipeline<M>> {
    // ... load model ...
    
    // Start with model's recommended defaults
    let mut params = model.default_generation_params();
    
    // Override with user-specified values
    if let Some(temp) = self.user_temperature {
        params.temperature = temp;
    }
    if let Some(top_k) = self.user_top_k {
        params.top_k = Some(top_k);
    }
    // ...
}
```

Requires tracking which params user explicitly set vs left default.

### Option B: Builder Stores Option<T> for Each Param
```rust
pub struct TextGenerationPipelineBuilder<M> {
    model_options: M::Options,
    temperature: Option<f64>,      // None = use model default
    top_k: Option<usize>,          // None = use model default
    // ...
}
```

At build time, fill in `None` values from model defaults.

### Option C: Two-Phase Build (Recommended)
Keep current builder simple, but at `build()` time:
1. Load model
2. Get model's `default_generation_params()`
3. Apply user overrides from builder
4. Construct pipeline

```rust
pub async fn build(self) -> Result<TextGenerationPipeline<M>> {
    let model = /* load model */;
    
    // Start with model defaults, override with builder values
    let base = model.default_generation_params();
    let params = GenerationParams {
        temperature: if self.gen_params.temperature != GenerationParams::default().temperature {
            self.gen_params.temperature
        } else {
            base.temperature
        },
        // ... same pattern for each field
    };
    
    TextGenerationPipeline::new(model, params, device).await
}
```

Problem: Can't distinguish "user set to default value" from "user didn't set".

### Recommended: Option B
Track each param as `Option<T>` in builder. Most explicit, no ambiguity.

## Implementation

### 1. Update Builder Struct
```rust
pub struct TextGenerationPipelineBuilder<M: TextGenerationModel> {
    model_options: M::Options,
    device_request: DeviceRequest,
    // Individual params, None = use model default
    temperature: Option<f64>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,
    seed: Option<u64>,
    max_len: Option<usize>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    min_p: Option<f64>,
}
```

### 2. Update Builder Methods
```rust
pub fn temperature(mut self, temperature: f64) -> Self {
    self.temperature = Some(temperature);
    self
}
```

### 3. Update build()
```rust
pub async fn build(self) -> Result<TextGenerationPipeline<M>> {
    // ... load model ...
    
    let defaults = model.default_generation_params();
    let params = GenerationParams {
        temperature: self.temperature.unwrap_or(defaults.temperature),
        repeat_penalty: self.repeat_penalty.unwrap_or(defaults.repeat_penalty),
        repeat_last_n: self.repeat_last_n.unwrap_or(defaults.repeat_last_n),
        seed: self.seed.unwrap_or_else(|| rand::random()),
        max_len: self.max_len.unwrap_or(defaults.max_len),
        top_p: self.top_p.or(defaults.top_p),
        top_k: self.top_k.or(defaults.top_k),
        min_p: self.min_p.or(defaults.min_p),
    };
    
    TextGenerationPipeline::new(model, params, device).await
}
```

## Files to Modify
- `src/pipelines/text_generation/builder.rs`

## Testing
- Build pipeline without setting params → verify uses model defaults
- Build pipeline with some params set → verify overrides work
- Compare Qwen3 vs Gemma3 default behavior

