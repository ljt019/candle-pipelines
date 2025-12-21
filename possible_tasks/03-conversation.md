# Improved Conversation Management

## Summary
Create a first-class `Conversation` type for managing multi-turn dialogues with history, context windows, and persistence.

## Motivation
Currently, users manage `Vec<Message>` manually. A dedicated conversation type would simplify common patterns and handle edge cases automatically.

## Current State
- `Message` type exists in `src/message.rs` (re-exported from `lib.rs`)
- `MessageVecExt` trait provides convenience methods (`as_json()`)
- `Role` enum: `System`, `User`, `Assistant`
- Pipelines have `completion_batch` and `chat_batch` for batch processing
- KV cache reuse exists for multi-turn (prefix matching in `base_pipeline.rs`)
- No built-in conversation context management or truncation

## Proposed Features
- Automatic context window management (truncation strategies)
- Conversation history persistence (save/load)
- Message threading/branching support
- Token counting and budget management
- System prompt pinning (always include)

## Example API
```rust
let mut conversation = Conversation::new()
    .with_system("You are a helpful assistant")
    .with_context_limit(8192);

conversation.add_user("What is Rust?");
let response = pipeline.chat(&conversation).await?;
conversation.add_assistant(&response);

conversation.add_user("Tell me more about ownership");
let response = pipeline.chat(&conversation).await?;

// Automatic truncation if context exceeds limit
// Saves system prompt and recent messages

// Save/load
conversation.save("chat_history.json")?;
let loaded = Conversation::load("chat_history.json")?;
```

## Truncation Strategies
- Keep system + last N messages
- Keep system + first user + last N
- Summarize old messages
