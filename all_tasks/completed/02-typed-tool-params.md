# Typed Tool Parameters with JSON Schema

## Summary
Replace `HashMap<String, String>` tool parameters with proper typed parameters using `serde_json::Value` and auto-generated JSON Schema via `schemars`.

## Motivation
Current tool system loses type information — everything is strings. Models expect proper JSON Schema for tool definitions, and we should validate params before calling tools.

## Current State
```rust
// Tool stores params as string->string
pub(crate) parameters: HashMap<String, String>,
pub(crate) function: fn(HashMap<String, String>) -> Result<String, ToolError>,

// Macro does basic type mapping manually
fn type_to_json_type(ty: &Type) -> String {
    match name.as_str() {
        "String" => "string",
        "i32" | "u32" => "number",
        // ... limited, no arrays/options/nested
    }
}
```

## Proposed Changes

### 1. Add `schemars` dependency
```toml
[dependencies]
schemars = "0.8"
```

### 2. Change parameter storage to `serde_json::Value`
```rust
// Before
pub(crate) function: fn(HashMap<String, String>) -> Result<String, ToolError>,

// After  
pub(crate) function: fn(serde_json::Value) -> Result<String, ToolError>,
```

### 3. Update `#[tool]` macro to use schemars
```rust
#[tool]
/// Search for documents
fn search(
    query: String,           // required string
    limit: Option<u32>,      // optional integer  
    filters: Vec<String>,    // array of strings
) -> Result<String, ToolError> { ... }
```

Macro generates:
```rust
fn __search_schema() -> schemars::schema::RootSchema {
    // Auto-generated from function signature
}

fn __search_wrapper(params: serde_json::Value) -> Result<String, ToolError> {
    // Deserialize params into typed struct, call original fn
}
```

### 4. Add validation before tool execution
When model calls a tool:
1. Parse model output to get tool name + params (JSON)
2. Validate params against tool's schema
3. If invalid → return clear error (model can retry)
4. If valid → call tool with `serde_json::Value`

## Implementation Tasks

### 1. Update `Tool` struct (`src/pipelines/text_generation/tools.rs`)
```rust
pub struct Tool {
    name: String,
    description: String,
    schema: schemars::schema::RootSchema,  // Full JSON Schema
    function: fn(serde_json::Value) -> Result<String, ToolError>,
    // ... error_strategy, retries
}

impl Tool {
    pub fn validate(&self, params: &serde_json::Value) -> Result<(), ToolError> {
        // Validate params against self.schema
    }
}
```

### 2. Update `#[tool]` macro (`tool_macro/src/lib.rs`)
- Generate a params struct from function args
- Derive `JsonSchema` on that struct
- Generate wrapper that deserializes `Value` → struct → calls fn
- Remove manual `type_to_json_type` function

### 3. Update chat template rendering
- Pass full JSON Schema to template (not just `HashMap<String, String>`)

### 4. Update tool calling flow
- Add validation step before calling tool
- Return structured error on validation failure

## Example

User writes:
```rust
#[tool]
/// Search documents by query
fn search(query: String, limit: Option<u32>) -> String {
    format!("Searching for '{}' with limit {:?}", query, limit)
}
```

Macro generates schema:
```json
{
  "type": "object",
  "properties": {
    "query": { "type": "string" },
    "limit": { "type": "integer", "nullable": true }
  },
  "required": ["query"]
}
```

Model calls with wrong type:
```json
{"name": "search", "parameters": {"query": 123}}
```

Validation catches it:
```
ToolError::Format("query: expected string, got number")
```

## Files to Modify
- `tool_macro/src/lib.rs` — rewrite param handling, use schemars
- `src/pipelines/text_generation/tools.rs` — update Tool struct
- `src/models/qwen3.rs` — update tool calling to use new types

## Testing
- Unit tests for schema generation from various types
- Unit tests for validation (valid params, invalid params, missing required)
- Integration test with actual model tool calling
