# completion() Accepts String/&String

## Summary
Add `From<String>` and `From<&String>` implementations for `Input` so users don't need `.as_str()` everywhere.

## Problem
Current `Input` only accepts `&str` or `&[Message]`:

```rust
impl<'a> From<&'a str> for Input<'a> { ... }
impl<'a> From<&'a [Message]> for Input<'a> { ... }
```

Users with `String` values hit friction:

```rust
let prompt = format!("Tell me about {}", topic);
pipeline.completion(prompt.as_str()).await?;  // annoying .as_str()

let prompt: String = get_prompt_from_somewhere();
pipeline.completion(&prompt).await?;  // or & prefix
```

## Solution
Add implementations:

```rust
impl<'a> From<&'a String> for Input<'a> {
    fn from(s: &'a String) -> Self {
        Self::Prompt(s.as_str())
    }
}
```

Now users can write:

```rust
let prompt = format!("Tell me about {}", topic);
pipeline.completion(&prompt).await?;  // works!
```

## Files to Modify
- `src/pipelines/text_generation/pipeline.rs` - add `From` impl

## Notes
- Can't do `From<String>` (owned) because `Input` has lifetime tied to borrowed content
- `From<&String>` is the right solution - borrows the String
- Tiny change, big ergonomic win

