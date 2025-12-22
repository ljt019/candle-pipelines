# Project Philosophy

Candle provides Rust implementations of models and low-level ML machinery. This library sits on top, providing a simple high-level API - like HuggingFace's `pipeline()` for Python Transformers.

## Core Principles 

1. Fill gaps, don't duplicate: Candle handles model implementations. We handle the "I just want to generate text" experience. Don't reimplement what Candle provides.

2. Primitives over frameworks: Give users building blocks (`Message`, `Tool`, pipelines) they compose themselves. Avoid over-abstraction.

3. Follow HuggingFace's lead: Their Python API is our reference. If they don't have a feature, question whether we need it.

4. Rust-idiomatic but approachable: Proper Rust patterns, but learnable for devs who just want LLMs without becoming ML experts.

# Task Management

Tasks live in `all_tasks/` with four subfolders:

- `todo/` - Ready to implement. 
- `future/` - Good ideas, not ready yet. May need design work or dependencies.
- `completed/` - Done and merged. Reference for how things were implemented.
- `not_planned/` - Considered but rejected. Kept for historical purposes.

# Candle Reference 

We don't depend on the local version but the entire candle repo is clone into './candle' so you can reference it's source code if you are ever confused whiel working with it or it's apis.


# Running Tests

Because the integration tests require actually loading the model into memory and running inference you can't run them. You can't run anything that requires actual inference because the machine you run on doesn't have access to a powerful enough cpu or a gpu at all. If you want to test any code that requires inference end your session and return a message to the user with what need to be ran/tested. The user will then run it on their machine and return the results to you so you can continue working.

