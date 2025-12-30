//! Llama 3.x tool call parser for streaming detection.
//!
//! Llama outputs raw JSON tool calls: `{"name": "func", "parameters": {...}}`
//! This parser uses a state machine to buffer potential tool calls during streaming
//! and detect them via brace-counting and JSON validation.

use serde::Deserialize;

/// Events emitted by the tool call parser during streaming.
#[derive(Debug, Clone)]
pub enum ParseEvent {
    /// Regular content to stream to user.
    Content(String),
    /// A complete tool call was detected.
    ToolCall(LlamaToolCall),
    /// Buffered content wasn't a tool call, flush as regular content.
    Flush(String),
}

/// A parsed Llama tool call.
#[derive(Debug, Clone, Deserialize)]
pub struct LlamaToolCall {
    pub name: String,
    pub parameters: serde_json::Value,
}

/// Parser state for streaming tool call detection.
#[derive(Debug, Clone, Default)]
enum ParseState {
    /// Normal streaming mode - pass tokens through.
    #[default]
    Streaming,
    /// Saw `{`, buffering tokens to determine if it's a tool call.
    Buffering,
}

/// Streaming parser for Llama tool calls.
///
/// Buffers tokens when a potential tool call is detected (`{`),
/// then either emits a `ToolCall` event or flushes the buffer as content.
#[derive(Debug, Clone, Default)]
pub struct LlamaToolParser {
    state: ParseState,
    buffer: String,
    brace_depth: i32,
}

impl LlamaToolParser {
    /// Create a new parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset parser state (call between generations).
    pub fn reset(&mut self) {
        self.state = ParseState::Streaming;
        self.buffer.clear();
        self.brace_depth = 0;
    }

    /// Process a token and return any resulting event.
    ///
    /// Returns `None` when buffering (no output yet).
    pub fn process_token(&mut self, token: &str) -> Option<ParseEvent> {
        match self.state {
            ParseState::Streaming => self.process_streaming(token),
            ParseState::Buffering => self.process_buffering(token),
        }
    }

    /// Finalize parsing when generation ends.
    ///
    /// If there's buffered content, tries to parse it as a tool call
    /// or flushes it as regular content.
    pub fn finalize(&mut self) -> Option<ParseEvent> {
        if self.buffer.is_empty() {
            return None;
        }

        let content = std::mem::take(&mut self.buffer);
        self.state = ParseState::Streaming;
        self.brace_depth = 0;

        // Try to parse as tool call
        if let Some(tool_call) = Self::try_parse_tool_call(&content) {
            Some(ParseEvent::ToolCall(tool_call))
        } else {
            Some(ParseEvent::Flush(content))
        }
    }

    fn process_streaming(&mut self, token: &str) -> Option<ParseEvent> {
        let trimmed = token.trim_start();

        // Check if this token starts a potential tool call
        if trimmed.starts_with('{') {
            self.state = ParseState::Buffering;
            self.buffer = token.to_string();
            self.brace_depth = Self::count_braces(token);

            // If braces already balanced, try parsing immediately
            if self.brace_depth == 0 {
                return self.try_complete_buffer();
            }

            None // Buffer, don't emit
        } else {
            Some(ParseEvent::Content(token.to_string()))
        }
    }

    fn process_buffering(&mut self, token: &str) -> Option<ParseEvent> {
        self.buffer.push_str(token);
        self.brace_depth += Self::count_braces(token);

        if self.brace_depth == 0 {
            // Braces balanced - JSON might be complete
            self.try_complete_buffer()
        } else if self.brace_depth < 0 {
            // Invalid nesting - flush as content
            Some(self.flush_buffer())
        } else if self.buffer.len() > 4096 {
            // Too long for a tool call - flush as content
            Some(self.flush_buffer())
        } else {
            None // Keep buffering
        }
    }

    fn try_complete_buffer(&mut self) -> Option<ParseEvent> {
        let content = std::mem::take(&mut self.buffer);
        self.state = ParseState::Streaming;
        self.brace_depth = 0;

        if let Some(tool_call) = Self::try_parse_tool_call(&content) {
            Some(ParseEvent::ToolCall(tool_call))
        } else {
            Some(ParseEvent::Flush(content))
        }
    }

    fn flush_buffer(&mut self) -> ParseEvent {
        let content = std::mem::take(&mut self.buffer);
        self.state = ParseState::Streaming;
        self.brace_depth = 0;
        ParseEvent::Flush(content)
    }

    fn count_braces(s: &str) -> i32 {
        let mut depth = 0i32;
        for c in s.chars() {
            match c {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
        }
        depth
    }

    fn try_parse_tool_call(content: &str) -> Option<LlamaToolCall> {
        let trimmed = content.trim();

        // Quick check - must start with { and be valid JSON
        if !trimmed.starts_with('{') {
            return None;
        }

        // Try parsing as LlamaToolCall
        let parsed: LlamaToolCall = serde_json::from_str(trimmed).ok()?;

        // Validate it has the expected structure
        if parsed.name.is_empty() {
            return None;
        }

        Some(parsed)
    }
}

/// Extract all tool calls from a complete response (non-streaming).
///
/// For Llama 3.x, tool calls are raw JSON objects with `name` and `parameters`.
pub fn extract_tool_calls(text: &str) -> Vec<LlamaToolCall> {
    let mut tool_calls = Vec::new();
    let trimmed = text.trim();

    // Try parsing the entire response as a single tool call
    if let Ok(call) = serde_json::from_str::<LlamaToolCall>(trimmed) {
        if !call.name.is_empty() {
            tool_calls.push(call);
            return tool_calls;
        }
    }

    // Fall back to scanning for JSON objects
    let mut decoder = JsonObjectScanner::new(trimmed);
    while let Some(json_str) = decoder.next_object() {
        if let Ok(call) = serde_json::from_str::<LlamaToolCall>(json_str) {
            if !call.name.is_empty() {
                tool_calls.push(call);
            }
        }
    }

    tool_calls
}

/// Simple scanner for extracting JSON objects from text.
struct JsonObjectScanner<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> JsonObjectScanner<'a> {
    fn new(text: &'a str) -> Self {
        Self { text, pos: 0 }
    }

    fn next_object(&mut self) -> Option<&'a str> {
        // Find next '{'
        let start = self.text[self.pos..].find('{')? + self.pos;

        // Count braces to find matching '}'
        let mut depth = 0;
        let mut end = start;

        for (i, c) in self.text[start..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = start + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }

        if depth != 0 {
            // Unbalanced braces
            self.pos = self.text.len();
            return None;
        }

        self.pos = end;
        Some(&self.text[start..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_tool_call() {
        let json = r#"{"name": "get_weather", "parameters": {"city": "Tokyo"}}"#;
        let calls = extract_tool_calls(json);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn test_streaming_detection() {
        let mut parser = LlamaToolParser::new();

        // Simulate streaming tokens
        let tokens = vec!["{", "\"name\"", ": ", "\"test\"", ", ", "\"parameters\"", ": ", "{}", "}"];

        let mut events = Vec::new();
        for token in tokens {
            if let Some(event) = parser.process_token(token) {
                events.push(event);
            }
        }

        // Should have one tool call event
        assert_eq!(events.len(), 1);
        match &events[0] {
            ParseEvent::ToolCall(call) => assert_eq!(call.name, "test"),
            _ => panic!("Expected ToolCall event"),
        }
    }

    #[test]
    fn test_non_tool_json_flushed() {
        let mut parser = LlamaToolParser::new();

        // JSON without 'name' field should be flushed
        let tokens = vec!["{", "\"foo\"", ": ", "\"bar\"", "}"];

        let mut events = Vec::new();
        for token in tokens {
            if let Some(event) = parser.process_token(token) {
                events.push(event);
            }
        }

        assert_eq!(events.len(), 1);
        match &events[0] {
            ParseEvent::Flush(content) => assert!(content.contains("foo")),
            _ => panic!("Expected Flush event"),
        }
    }
}
