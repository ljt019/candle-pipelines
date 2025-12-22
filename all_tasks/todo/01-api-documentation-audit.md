# API Documentation Audit

## Summary
Add comprehensive `#[doc]` comments to all public types, traits, and methods. Users shouldn't need to read source code to understand the API.

## Motivation
- Many public types lack documentation
- `TextGenerationModel` trait methods completely undocumented
- `Tool` struct fields/methods sparse docs
- Pipeline builder methods missing param explanations
- Users hitting API blind → frustration, GitHub issues

## Scope

### Priority 1: Core Public API
- `src/lib.rs` - module-level docs, re-exports
- `src/message.rs` - `Message`, `Role`, `MessageVecExt`
- `src/error.rs` - each error variant needs "when does this happen?"

### Priority 2: Text Generation Pipeline
- `src/pipelines/text_generation/mod.rs` - module docs
- `src/pipelines/text_generation/pipeline.rs` - `TextGenerationPipeline` methods
- `src/pipelines/text_generation/builder.rs` - each builder method
- `src/pipelines/text_generation/params.rs` - `GenerationParams` fields
- `src/pipelines/text_generation/stats.rs` - `GenerationStats` fields
- `src/pipelines/text_generation/tools.rs` - `Tool`, `ToolCalling`, `ErrorStrategy`

### Priority 3: Other Pipelines
- `src/pipelines/fill_mask/` - pipeline + builder
- `src/pipelines/sentiment/` - pipeline + builder
- `src/pipelines/zero_shot/` - pipeline + builder

### Priority 4: Models
- `src/models/qwen3.rs` - `Qwen3Size` variants
- `src/models/gemma3.rs` - `Gemma3Size` variants
- `src/models/modernbert.rs` - `ModernBertSize` variants

### Priority 5: Traits (for advanced users)
- `src/pipelines/text_generation/model.rs` - `TextGenerationModel`, `LanguageModelContext`

## Documentation Standards

### Types
```rust
/// A message in a chat conversation.
///
/// Messages have a role (system, user, or assistant) and content.
/// Use the constructor methods to create messages:
///
/// ```rust
/// use transformers::Message;
///
/// let system = Message::system("You are helpful");
/// let user = Message::user("Hello!");
/// let assistant = Message::assistant("Hi there!");
/// ```
pub struct Message { ... }
```

### Methods
```rust
/// Set the sampling temperature for generation.
///
/// Higher values (e.g., 1.0) produce more random output.
/// Lower values (e.g., 0.1) produce more deterministic output.
///
/// # Arguments
/// * `temperature` - Value between 0.0 and 2.0. Default: 0.7
///
/// # Example
/// ```rust
/// let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
///     .temperature(0.3)  // More focused responses
///     .build()
///     .await?;
/// ```
pub fn temperature(mut self, temperature: f64) -> Self
```

### Error Variants
```rust
/// Tokenization failed.
///
/// This occurs when:
/// - Input text contains characters the tokenizer can't handle
/// - The tokenizer file is corrupted or incompatible
///
/// The wrapped string contains details from the tokenizer.
#[error("Tokenization failed: {0}")]
Tokenization(String),
```

## Files to Modify
- `src/lib.rs`
- `src/message.rs`
- `src/error.rs`
- `src/pipelines/text_generation/mod.rs`
- `src/pipelines/text_generation/pipeline.rs`
- `src/pipelines/text_generation/builder.rs`
- `src/pipelines/text_generation/params.rs`
- `src/pipelines/text_generation/stats.rs`
- `src/pipelines/text_generation/tools.rs`
- `src/pipelines/text_generation/model.rs`
- `src/pipelines/fill_mask/*.rs`
- `src/pipelines/sentiment/*.rs`
- `src/pipelines/zero_shot/*.rs`
- `src/models/*.rs`

## Verification
- `cargo doc --no-deps` builds without warnings
- Browse generated docs, check all public items documented
- Examples in docs compile (`cargo test --doc`)

