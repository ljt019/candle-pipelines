# JSON Mode / Structured Output

## Summary
Add grammar-based constrained decoding to guarantee model output matches a user-defined Rust type.

## Motivation
Many applications need structured data from LLMs. Currently models may produce invalid JSON or wrong shapes, requiring retry logic. Grammar-constrained decoding guarantees output always deserializes to the expected type.

## API Design

### Primary API — Type-safe JSON
```rust
#[derive(Deserialize, JsonSchema)]
struct FruitList {
    fruits: Vec<String>,
}

let result: FruitList = pipeline
    .completion("List 3 fruits")
    .json::<FruitList>()
    .await?;

// result.fruits == vec!["apple", "banana", "cherry"]
```

User gets their Rust type directly. Schema generation and constraint enforcement is invisible.

### Advanced API — Raw schema control
```rust
let schema = json!({
    "type": "object",
    "properties": {
        "name": { "type": "string" },
        "age": { "type": "integer" }
    },
    "required": ["name", "age"]
});

let result: serde_json::Value = pipeline
    .completion("Describe a person")
    .json_schema(&schema)
    .await?;
``` 

For users who need dynamic schemas or don't have a Rust type.

## Implementation Approach

### Grammar-based constrained decoding
1. Convert JSON Schema → grammar (defines valid token sequences)
2. At each sampling step, mask logits for tokens that would produce invalid JSON
3. Only allow tokens that continue a valid path through the grammar

This guarantees 100% valid output — no retries needed.

### Flow
```
User type (FruitList)
        ↓
schemars::schema_for!(FruitList)  →  JSON Schema
        ↓
schema_to_grammar(schema)  →  Grammar/State machine
        ↓
During generation:
  - Get logits from model
  - Get valid_tokens from grammar state
  - Mask invalid tokens (set logits to -inf)
  - Sample from remaining tokens
  - Update grammar state
        ↓
serde_json::from_str::<FruitList>(output)  →  User's type
```

## Key Tasks

### 1. JSON Schema → Grammar converter
- Parse JSON Schema
- Build state machine that tracks valid continuations
- Handle: objects, arrays, strings, numbers, booleans, null
- Handle: required fields, optional fields, nested structures

### 2. Grammar-constrained sampler
- New sampler wrapper in `src/pipelines/text_generation/params.rs`
- Takes grammar + base sampler
- Masks invalid tokens before sampling
- Updates grammar state after each token

### 3. Pipeline integration
- Add `.json::<T>()` method to pipeline builder
- Add `.json_schema(&schema)` method for raw schema
- Generate schema from T using schemars
- Wrap sampler with grammar constraint
- Deserialize output to T

### 4. Tokenizer integration
- Need to map grammar terminals to token IDs
- Handle tokenizer-specific details (BPE boundaries, etc.)

## Constraints

### No streaming with JSON mode
Streaming is disabled when JSON mode is active. Simplifies implementation — we validate/deserialize only at the end.

### Build our own (no external crates)
Grammar-based decoding needs tight integration with our sampling loop. External crates (llguidance, outlines-core) may not integrate cleanly with candle.

## Files to Modify/Create

- `src/pipelines/text_generation/json_mode.rs` (new) — grammar, schema conversion, constrained sampler
- `src/pipelines/text_generation/params.rs` — integrate grammar sampler
- `src/pipelines/text_generation/builder.rs` — add `.json::<T>()` and `.json_schema()` methods
- `src/pipelines/text_generation/pipeline.rs` — handle JSON mode in generation loop

## Testing

- Unit tests for schema → grammar conversion
- Unit tests for grammar state machine (valid/invalid token sequences)
- Integration tests with actual model:
  - Simple types (string, number, bool)
  - Objects with required/optional fields
  - Arrays
  - Nested structures
- Edge cases: empty objects, empty arrays, deeply nested

## Reference
- JSON Schema spec: https://json-schema.org/
- Outlines paper (for algorithm inspiration)
- llama.cpp GBNF grammars (for grammar format ideas)

## Complexity Note
This is significant work — grammar-based decoding is non-trivial. But it's the only approach that guarantees valid output. Worth doing right.
