# Async Tool Support

## Summary
Allow tools to be async functions, enabling network calls, database queries, and other I/O operations within tool implementations.

## Motivation
Many real-world tools need to perform I/O operations (API calls, database queries, file operations). Currently tools are synchronous, requiring workarounds like `block_on`.

## Current State
```rust
pub(crate) function: fn(parameters: serde_json::Value) -> Result<String, ToolError>,
```
Tools now use `serde_json::Value` for parameters (typed via schemars).

## Proposed Changes
- Support `async fn` in tool definitions
- Use `Box<dyn Future>` or similar for the function type
- Handle async execution in the pipeline
- Update `#[tool]` macro to detect and handle async functions

## Example API
```rust
#[tool]
async fn fetch_weather(city: String) -> Result<String, ToolError> {
    let response = reqwest::get(&format!("https://api.weather.com/{}", city))
        .await
        .map_err(|e| ToolError::Message(e.to_string()))?;
    // ...
}
```

## Considerations
- Both sync and async tools should be supported
- Execution timeout for async tools
- Cancellation support, (is this needed?)
