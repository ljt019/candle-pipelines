//! Qwen3-specific tool call parser.
//!
//! Qwen3 uses XML-wrapped JSON for tool calls:
//! ```xml
//! <tool_call>
//! {"name": "function_name", "arguments": {"key": "value"}}
//! </tool_call>
//! ```

use std::collections::VecDeque;

use crate::models::capabilities::{ParseEvent, ToolCallInvocation, ToolCallParser};
use crate::pipelines::text_generation::xml_parser::{Event, TagParts, XmlParser, XmlTag};

/// Tags recognized by Qwen3's tool call format.
#[derive(Debug, Clone, PartialEq)]
pub enum Qwen3Tags {
    #[allow(dead_code)]
    ToolCall,
}

impl XmlTag for Qwen3Tags {
    fn from_tag_str(s: &str) -> Option<Self> {
        match s.trim() {
            "tool_call" => Some(Self::ToolCall),
            _ => None,
        }
    }

    fn as_tag_str(&self) -> &'static str {
        match self {
            Self::ToolCall => "tool_call",
        }
    }
}

/// Parser for Qwen3's XML-wrapped JSON tool call format.
///
/// Wraps [`XmlParser`] configured to detect `<tool_call>` tags and
/// parses the JSON content inside.
#[derive(Debug, Clone)]
pub struct Qwen3Parser {
    xml_parser: XmlParser<Qwen3Tags>,
    /// Buffer for events that need to be emitted (FIFO queue)
    pending_events: VecDeque<ParseEvent>,
}

impl Default for Qwen3Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl Qwen3Parser {
    /// Create a new Qwen3 tool call parser.
    pub fn new() -> Self {
        Self {
            xml_parser: XmlParser::new(),
            pending_events: VecDeque::new(),
        }
    }

    /// Process an XmlParser event and convert to ParseEvent.
    fn process_xml_event(&mut self, event: Event<Qwen3Tags>) -> Option<ParseEvent> {
        match event {
            Event::Tagged {
                tag: Qwen3Tags::ToolCall,
                part: TagParts::End,
                content,
                ..
            } => {
                // Tool call complete - try to parse
                Some(self.parse_tool_call_content(&content))
            }
            Event::Tagged {
                tag: Qwen3Tags::ToolCall,
                ..
            } => {
                // Start or Content - continue buffering
                None
            }
            Event::Output { content } => {
                if !content.is_empty() {
                    Some(ParseEvent::text(&content))
                } else {
                    None
                }
            }
        }
    }

    /// Parse the content of a tool_call tag.
    fn parse_tool_call_content(&self, full_xml: &str) -> ParseEvent {
        // Extract JSON from <tool_call>...</tool_call>
        let inner = match full_xml
            .strip_prefix("<tool_call>")
            .and_then(|s| s.strip_suffix("</tool_call>"))
        {
            Some(s) => s.trim(),
            None => {
                return ParseEvent::malformed(
                    full_xml.to_string(),
                    "Missing <tool_call> tags".to_string(),
                );
            }
        };

        // Parse JSON
        let parsed: serde_json::Value = match serde_json::from_str(inner) {
            Ok(v) => v,
            Err(e) => {
                return ParseEvent::malformed(full_xml.to_string(), format!("Invalid JSON: {e}"));
            }
        };

        // Extract name
        let name = match parsed.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => {
                return ParseEvent::malformed(
                    full_xml.to_string(),
                    "Missing 'name' field".to_string(),
                );
            }
        };

        // Extract arguments (default to empty object)
        let arguments = parsed
            .get("arguments")
            .cloned()
            .unwrap_or(serde_json::json!({}));

        ParseEvent::ToolCall(Ok(ToolCallInvocation::new(name, arguments)))
    }
}

impl ToolCallParser for Qwen3Parser {
    fn feed(&mut self, token: &str) -> ParseEvent {
        // Feed token to XML parser and process any resulting events
        if !token.is_empty() {
            let events = self.xml_parser.parse_token(token);

            // Process events and add to pending buffer
            for event in events {
                if let Some(parse_event) = self.process_xml_event(event) {
                    self.pending_events.push_back(parse_event);
                }
            }
        }

        // Return first pending event (FIFO), or Continue
        self.pending_events
            .pop_front()
            .unwrap_or(ParseEvent::Continue)
    }

    fn flush(&mut self) -> Option<ParseEvent> {
        // First return any pending events
        if let Some(event) = self.pending_events.pop_front() {
            return Some(event);
        }

        // Flush XML parser
        let events = self.xml_parser.flush();

        // Process flush events
        for event in events {
            if let Some(parse_event) = self.process_xml_event(event) {
                self.pending_events.push_back(parse_event);
            }
        }

        // Return first pending event
        self.pending_events.pop_front()
    }

    fn reset(&mut self) {
        self.xml_parser.reset();
        self.pending_events.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_tool_call() {
        let mut parser = Qwen3Parser::new();

        // Feed complete tool call
        let input = r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>"#;

        let mut result = None;
        for c in input.chars() {
            let event = parser.feed(&c.to_string());
            if !event.is_continue() {
                result = Some(event);
            }
        }

        // Flush to get final event
        if result.is_none() {
            result = parser.flush();
        }

        let event = result.expect("should have a result");
        assert!(event.is_tool_call(), "Expected tool call, got {:?}", event);

        let invocation = event.as_tool_call().unwrap();
        assert_eq!(invocation.name, "get_weather");
        assert_eq!(invocation.arguments, serde_json::json!({"city": "Paris"}));
    }

    #[test]
    fn test_malformed_json() {
        let mut parser = Qwen3Parser::new();

        let input = "<tool_call>{invalid json}</tool_call>";

        let mut result = None;
        for c in input.chars() {
            let event = parser.feed(&c.to_string());
            if !event.is_continue() {
                result = Some(event);
            }
        }

        if result.is_none() {
            result = parser.flush();
        }

        let event = result.expect("should have a result");
        assert!(event.is_malformed(), "Expected malformed, got {:?}", event);

        let error = event.as_malformed().unwrap();
        assert!(error.reason.contains("Invalid JSON"));
    }

    #[test]
    fn test_missing_name_field() {
        let mut parser = Qwen3Parser::new();

        let input = r#"<tool_call>{"arguments": {"x": 1}}</tool_call>"#;

        let mut result = None;
        for c in input.chars() {
            let event = parser.feed(&c.to_string());
            if !event.is_continue() {
                result = Some(event);
            }
        }

        if result.is_none() {
            result = parser.flush();
        }

        let event = result.expect("should have a result");
        assert!(event.is_malformed());

        let error = event.as_malformed().unwrap();
        assert!(error.reason.contains("name"));
    }

    #[test]
    fn test_text_before_tool_call() {
        let mut parser = Qwen3Parser::new();

        let input =
            r#"Here is the result: <tool_call>{"name": "test", "arguments": {}}</tool_call>"#;

        let mut events = Vec::new();
        for c in input.chars() {
            let event = parser.feed(&c.to_string());
            if !event.is_continue() {
                events.push(event);
            }
        }

        while let Some(event) = parser.flush() {
            events.push(event);
        }

        // Should have text and tool call events
        assert!(events.iter().any(|e| matches!(e, ParseEvent::Text(_))));
        assert!(events.iter().any(|e| e.is_tool_call()));
    }

    #[test]
    fn test_reset() {
        let mut parser = Qwen3Parser::new();

        // Feed partial input
        parser.feed("<tool_call>");

        // Reset
        parser.reset();

        // Feed new complete input
        let input = r#"<tool_call>{"name": "test", "arguments": {}}</tool_call>"#;

        let mut result = None;
        for c in input.chars() {
            let event = parser.feed(&c.to_string());
            if !event.is_continue() {
                result = Some(event);
            }
        }

        if result.is_none() {
            result = parser.flush();
        }

        let event = result.expect("should parse after reset");
        assert!(event.is_tool_call());
    }
}
