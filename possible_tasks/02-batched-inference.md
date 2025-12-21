# Batched Inference Support

## Summary
Enable processing multiple inputs in a single call across all pipelines for improved throughput and ergonomics.

## Motivation
Users often need to process many inputs. Batching improves GPU utilization and is a natural API expectation from HF transformers users.

## API Design

### Method naming
Separate methods for clarity:
```rust
// Single input
let result: String = pipeline.predict("text")?;

// Batch input
let results: Vec<Result<String, Error>> = pipeline.predict_batch(&["text1", "text2"])?;
```

### Return type
`Vec<Result<T, E>>` — standard Rust, no new types.
- Outer `Result` (on the method): batch-level failures (setup, tokenizer init, device error)
- Inner `Result` (per item): individual item failures

### Error handling
Partial results — if item #47 of 100 fails, you still get the other 99.

## Pipelines to Update

### 1. Fill-mask (`pipelines/fill_mask/`)
```rust
// Current
fn predict(&self, text: &str) -> Result<String>;
fn predict_top_k(&self, text: &str, k: usize) -> Result<Vec<FillMaskPrediction>>;

// Add
fn predict_batch(&self, texts: &[&str]) -> Result<Vec<Result<String, Error>>>;
fn predict_top_k_batch(&self, texts: &[&str], k: usize) -> Result<Vec<Result<Vec<FillMaskPrediction>, Error>>>;
```

### 2. Sentiment (`pipelines/sentiment/`)
```rust
// Current
fn predict(&self, text: &str) -> Result<String>;
fn predict_with_score(&self, text: &str) -> Result<SentimentResult>;

// Add
fn predict_batch(&self, texts: &[&str]) -> Result<Vec<Result<String, Error>>>;
fn predict_with_score_batch(&self, texts: &[&str]) -> Result<Vec<Result<SentimentResult, Error>>>;
```

### 3. Zero-shot (`pipelines/zero_shot/`)
```rust
// Current
fn predict(&self, text: &str, labels: &[&str]) -> Result<Vec<(String, f32)>>;

// Add
fn predict_batch(&self, texts: &[&str], labels: &[&str]) -> Result<Vec<Result<Vec<(String, f32)>, Error>>>;
```

### 4. Text generation (`pipelines/text_generation/`)
Most complex — variable output lengths, KV cache considerations.
```rust
// Current
async fn completion(&mut self, prompt: &str) -> Result<String>;
async fn chat(&mut self, messages: &[Message]) -> Result<String>;

// Add
async fn completion_batch(&mut self, prompts: &[&str]) -> Result<Vec<Result<String, Error>>>;
async fn chat_batch(&mut self, conversations: &[&[Message]]) -> Result<Vec<Result<String, Error>>>;
```

## Implementation Notes

### Simple pipelines (fill-mask, sentiment, zero-shot)
- Batch tokenization with padding
- Single forward pass with batch dimension
- Split results back to per-item

### Text generation (complex)
- Need attention mask for padded sequences
- KV cache per sequence in batch
- Handle different completion lengths (some finish early)
- Consider max batch size limits for memory

## Key Tasks

1. **Update model traits** in each pipeline's `model.rs`:
   - Add `_batch` method signatures

2. **Update pipeline structs** in each pipeline's `pipeline.rs`:
   - Implement batch methods
   - Handle padding/attention masks
   - Split results

3. **Update ModernBERT model** (`src/models/modernbert.rs`):
   - Already supports batch dimension, wire it up

4. **Update text-gen models** (`src/models/qwen3.rs`, `gemma3.rs`):
   - Batch KV cache management
   - Attention masking for variable lengths

5. **Add tests**:
   - Unit tests for padding logic
   - Integration tests for each pipeline's batch methods
   - Edge cases: empty batch, single item batch, mixed success/failure

## Testing Requirements
- Verify batch results match individual calls
- Test partial failure scenarios
- Test empty input handling
- Memory usage sanity checks for large batches

## Reference
- HF transformers batching behavior
- `src/models/modernbert.rs` already handles batch dimension in forward pass
