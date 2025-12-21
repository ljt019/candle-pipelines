# Generation Statistics

## Summary
Add performance statistics tracking for text generation — tokens per second, time to first token, total tokens generated.

## Motivation
Users want to know:
- "How fast is my setup?" → tokens/sec
- "Is model loading slow?" → time to first token
- "How much did it generate?" → token count
- Debugging performance issues

## API Design

### For streaming
```rust
let mut stream = pipeline.completion_stream("Hello").await?;

while let Some(chunk) = stream.next().await {
    let chunk = chunk?;
    println!("{}", chunk.text);
}

// After stream completes
let stats = stream.stats();
println!("Tokens: {}", stats.tokens_generated);
println!("TTFT: {:?}", stats.time_to_first_token);
println!("Speed: {:.1} tok/s", stats.tokens_per_second);
```

### For non-streaming
```rust
let (result, stats) = pipeline.completion_with_stats("Hello").await?;
// or
let result = pipeline.completion("Hello").await?;
let stats = pipeline.last_generation_stats();
```

## Stats Struct

```rust
#[derive(Debug, Clone)]
pub struct GenerationStats {
    /// Number of tokens generated
    pub tokens_generated: usize,
    
    /// Time from request start to first token
    pub time_to_first_token: Duration,
    
    /// Total generation time
    pub total_time: Duration,
    
    /// Tokens per second (tokens_generated / total_time)
    pub tokens_per_second: f64,
    
    /// Number of prompt tokens (input)
    pub prompt_tokens: usize,
}

impl GenerationStats {
    pub fn new() -> Self { ... }
    pub(crate) fn record_first_token(&mut self) { ... }
    pub(crate) fn record_token(&mut self) { ... }
    pub(crate) fn finalize(&mut self) { ... }
}
```

## Implementation

### 1. Create stats module
`src/pipelines/text_generation/stats.rs`
- `GenerationStats` struct
- Internal timing logic

### 2. Update streaming
`src/pipelines/text_generation/streaming/completion_stream.rs`
- Track stats during generation
- Expose `.stats()` method after completion

### 3. Update non-streaming
`src/pipelines/text_generation/pipeline.rs`
- Add `completion_with_stats()` method
- Or store last stats on pipeline for `last_generation_stats()`

### 4. Update base pipeline
`src/pipelines/text_generation/base_pipeline.rs`
- Integrate stats tracking into generation loop

## Files to Modify/Create
- `src/pipelines/text_generation/stats.rs` (new)
- `src/pipelines/text_generation/mod.rs` — export stats
- `src/pipelines/text_generation/streaming/completion_stream.rs`
- `src/pipelines/text_generation/pipeline.rs`
- `src/pipelines/text_generation/base_pipeline.rs`

## Testing
- Unit test stats calculations
- Integration test: verify stats are reasonable (ttft < total_time, tokens > 0, etc.)
- Test streaming and non-streaming both report stats

