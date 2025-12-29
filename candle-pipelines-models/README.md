# candle-pipelines-models

Patched model implementations for candle-pipelines.

This crate provides fixed versions of models from `candle-transformers` where the upstream implementation has design issues.

## The Problem

Candle's quantized models embed KV cache internally with `&mut self` forward:

```rust
// Upstream candle-transformers (broken design)
pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor>
```

This requires:
1. **Manual position tracking** - you pass `offset`, model doesn't track it
2. **Clone entire model weights** for each conversation (to get independent KV cache)
3. **No sharing** - can't have multiple conversations use same weights

## The Fix

We use external cache like non-quantized llama does:

```rust
// Our patched version (correct design)
pub fn forward(&self, input: &Tensor, cache: &mut Cache) -> Result<Tensor>
```

Benefits:
1. **Automatic position tracking** - `cache.current_seq_len()`
2. **No weight cloning** - share `Arc<ModelWeights>` across conversations
3. **Independent caches** - each conversation gets its own `Cache`

## Patched Models

### `quantized_qwen3`

```rust
use candle_pipelines_models::quantized_qwen3::{ModelWeights, Cache};

let weights = Arc::new(ModelWeights::from_gguf(content, &mut reader, &device)?);

// Each conversation gets its own cache, shares weights
let mut cache1 = weights.new_cache();
let mut cache2 = weights.new_cache();

let logits = weights.forward(&input, &mut cache1)?;
cache1.reset(); // Clear for new conversation
```

### `quantized_gemma3`

Same API as qwen3.
