# Project Philosophy

Candle provides Rust implementations of models and low-level ML machinery. This library sits on top, providing a simple high-level API like HuggingFace's `pipeline()` for Python Transformers.

## Core Principles 

1. Fill gaps, don't duplicate: Candle handles model implementations. We handle the "I just want to generate text" experience. Don't reimplement what Candle provides.
2. Primitives over frameworks: Give users building blocks (`Message`, `Tool`, pipelines) they compose themselves. Avoid over-abstraction.
3. Follow HuggingFace's lead: Their Python API for pipelines is our reference. If they don't have a feature, question whether we need it.
4. Rust-idiomatic but approachable: Proper Rust patterns, but learnable for devs who just want LLMs without becoming ML experts.

# Structure Overview

The project is broken up into 3 crates currently: `candle-pipelines`, `candle-pipelines-macros`, `candle-pipelines-models`.

## Candle Pipelines 

The main crate users will install. Implements all the pipelines for all the models, pipelines automatically
download weights from hugginface, configure chat template setup default hyperparameters, etc.

- `candle-pipelines/src/models/` - Model implementations (Qwen3, Gemma3, etc.)
- `candle-pipelines/src/pipelines/` - Pipeline logic (text_generation, fill_mask, etc.)

### Patterns
- Models implement `TextGenerationModel` trait
- Pipelines use builder pattern: `TextGenerationPipelineBuilder::qwen3(...).build()`
- Generation params come from HuggingFace's generation_config.json, user overrides via builder

## Candle Pipelines Macros 

Implements the tool proc macros (#[tool], tools!) for the candle pipeline crate, for now thats just the tool ones, may be more in the future but for now just those. The tool macros let you do fun stuff like:
```rust 
#[tool]
fn get_weather(city: String) -> String {
    format!("The weather in {city} is rainy")
}
```

## Candle Pipelines Models 

This crates is just for when i need to make patches to candle-transformers model implementations, which 
happens from time to time, it's convenient to not have to wait for them to merge in a fix, and helps
me find + verify any fixes I might want to merge back upstream eventually!

Current status:
  - Patched candle model weights with external KV cache

# Common Scripts

Run Tests:
`cargo test`

Run Tests, including slower inference ones: 
`cargo test --features cuda`

Run Benchmarks:
`cargo bench`

Create docs at './target/doc/candle_pipelines/:
`cargo doc --no-deps`