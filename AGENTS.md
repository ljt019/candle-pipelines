# Project Philosophy

Candle provides Rust implementations of models and low-level ML machinery. This library sits on top, providing a simple high-level API - like HuggingFace's `pipeline()` for Python Transformers.

## Core Principles 

1. Fill gaps, don't duplicate: Candle handles model implementations. We handle the "I just want to generate text" experience. Don't reimplement what Candle provides.

2. Primitives over frameworks: Give users building blocks (`Message`, `Tool`, pipelines) they compose themselves. Avoid over-abstraction.

3. Follow HuggingFace's lead: Their Python API is our reference. If they don't have a feature, question whether we need it.

4. Rust-idiomatic but approachable: Proper Rust patterns, but learnable for devs who just want LLMs without becoming ML experts.

# Running Tests

Tests that require CUDA (model loading, inference) are gated behind the `cuda` feature. You can't run these - no GPU available. If you need to test inference code, end your session and tell the user what to run. They'll run it locally and return results.

