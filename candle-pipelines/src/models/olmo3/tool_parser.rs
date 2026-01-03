//! Parser for OLMo-3's function call format.
//!
//! OLMo-3 uses a Python-like syntax for tool calls:
//! ```text
//! <function_calls>
//! get_weather(location="Paris", unit="celsius")
//! calculate(x=42, y=13)
//! </function_calls>
//! ```

use std::collections::VecDeque;

use regex::Regex;
use serde_json::{json, Value};

use crate::models::capabilities::{ParseEvent, ToolCallInvocation, ToolCallParser};
use crate::pipelines::text_generation::xml_parser::{Event, TagParts, XmlParser, XmlTag};

/// Tags recognized by OLMo-3's tool call format.
#[derive(Debug, Clone, PartialEq)]
pub enum Olmo3Tags {
    FunctionCalls,
}

impl XmlTag for Olmo3Tags {
    fn from_tag_str(s: &str) -> Option<Self> {
        match s.trim() {
            "function_calls" => Some(Self::FunctionCalls),
            _ => None,
        }
    }

    fn as_tag_str(&self) -> &'static str {
        match self {
            Self::FunctionCalls => "function_calls",
        }
    }
}

/// Represents a parsed OLMo-3 tool call.
#[derive(Debug, Clone)]
pub struct Olmo3ToolCall {
    pub name: String,
    pub arguments: Value,
}

impl Olmo3ToolCall {
    /// Convert to the standard ToolCallInvocation format.
    pub fn into_invocation(self) -> ToolCallInvocation {
        ToolCallInvocation::new(self.name, self.arguments)
    }
}

/// Streaming parser for OLMo-3 tool calls.
///
/// Wraps [`XmlParser`] configured to detect `<function_calls>` tags and
/// parses the Python-like function call syntax inside.
///
/// Implements [`ToolCallParser`] for integration with the generic tool-aware iterator.
#[derive(Debug, Clone)]
pub struct Olmo3Parser {
    xml_parser: XmlParser<Olmo3Tags>,
    /// Queue of events to emit (FIFO)
    pending_events: VecDeque<ParseEvent>,
}

impl Default for Olmo3Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl Olmo3Parser {
    /// Create a new OLMo-3 tool call parser.
    pub fn new() -> Self {
        Self {
            xml_parser: XmlParser::new(),
            pending_events: VecDeque::new(),
        }
    }

    /// Process an XmlParser event and convert to ParseEvents.
    fn process_xml_event(&mut self, event: Event<Olmo3Tags>) {
        match event {
            Event::Tagged {
                tag: Olmo3Tags::FunctionCalls,
                part,
                content,
                ..
            } => {
                match part {
                    TagParts::Start | TagParts::Content => {
                        // Function calls block starting or content - continue buffering
                    }
                    TagParts::End => {
                        // Block complete - parse all function calls
                        self.parse_function_calls_block(&content);
                    }
                }
            }
            Event::Output { content } => {
                if !content.is_empty() {
                    self.pending_events.push_back(ParseEvent::text(&content));
                }
            }
        }
    }

    /// Parse a complete function_calls block and emit tool call events.
    fn parse_function_calls_block(&mut self, full_xml: &str) {
        // Extract content from tags
        let inner = full_xml
            .strip_prefix("<function_calls>")
            .and_then(|s| s.strip_suffix("</function_calls>"))
            .unwrap_or(full_xml)
            .trim();

        // Parse all function calls
        let mut remaining = inner;
        while !remaining.is_empty() {
            if let Some((call, rest)) = parse_function_call(remaining) {
                self.pending_events
                    .push_back(ParseEvent::ToolCall(Ok(call.into_invocation())));
                remaining = rest.trim();
            } else {
                // Couldn't parse - emit remaining as text
                if !remaining.trim().is_empty() {
                    self.pending_events.push_back(ParseEvent::malformed(
                        remaining,
                        "Failed to parse function call",
                    ));
                }
                break;
            }
        }
    }
}

impl ToolCallParser for Olmo3Parser {
    fn feed(&mut self, token: &str) -> ParseEvent {
        // Feed token to XML parser and process any resulting events
        if !token.is_empty() {
            let events = self.xml_parser.parse_token(token);

            // Process events
            for event in events {
                self.process_xml_event(event);
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
            self.process_xml_event(event);
        }

        // Return first pending event
        self.pending_events.pop_front()
    }

    fn reset(&mut self) {
        self.xml_parser.reset();
        self.pending_events.clear();
    }
}

/// Extract tool calls from OLMo-3's function call format.
pub fn extract_tool_calls(text: &str) -> Vec<Olmo3ToolCall> {
    let mut calls = Vec::new();

    // Match <function_calls>...</function_calls> blocks
    let block_re = Regex::new(r"(?s)<function_calls>(.*?)</function_calls>").unwrap();

    for block_cap in block_re.captures_iter(text) {
        let block_content = block_cap.get(1).unwrap().as_str();

        // Parse all function calls in block
        let mut remaining = block_content.trim();
        while !remaining.is_empty() {
            if let Some((call, rest)) = parse_function_call(remaining) {
                calls.push(call);
                remaining = rest.trim();
            } else {
                break;
            }
        }
    }

    calls
}

/// Parse a single function call, handling quoted strings that may contain parens.
/// Returns the parsed call and remaining text.
fn parse_function_call(text: &str) -> Option<(Olmo3ToolCall, &str)> {
    // Find function name (word chars before opening paren)
    let paren_pos = text.find('(')?;
    let name = text[..paren_pos].trim().to_string();

    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }

    // Find matching closing paren, respecting quotes
    let args_start = paren_pos + 1;
    let rest = &text[args_start..];
    let args_end = find_matching_paren(rest)?;
    let args_str = &rest[..args_end];

    let arguments = parse_arguments(args_str);
    let remaining = &text[args_start + args_end + 1..]; // +1 to skip closing paren
    Some((Olmo3ToolCall { name, arguments }, remaining))
}

/// Find the position of the closing paren, respecting quoted strings.
fn find_matching_paren(s: &str) -> Option<usize> {
    let mut depth = 1;
    let mut in_string = false;
    let mut string_char = '"';
    let mut escape_next = false;

    for (i, c) in s.char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }

        if c == '\\' && in_string {
            escape_next = true;
            continue;
        }

        if !in_string {
            match c {
                '"' | '\'' => {
                    in_string = true;
                    string_char = c;
                }
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        } else if c == string_char {
            in_string = false;
        }
    }

    None
}

/// Parse function arguments from "key=value, key2=value2" format.
/// Handles multi-line strings and escape sequences.
fn parse_arguments(args_str: &str) -> Value {
    let mut args = serde_json::Map::new();
    let mut chars = args_str.chars().peekable();

    loop {
        // Skip whitespace and commas
        while chars
            .peek()
            .map(|c| c.is_whitespace() || *c == ',')
            .unwrap_or(false)
        {
            chars.next();
        }

        if chars.peek().is_none() {
            break;
        }

        // Parse key (collect while alphanumeric or underscore)
        let mut key = String::new();
        while let Some(&c) = chars.peek() {
            if c.is_alphanumeric() || c == '_' {
                key.push(chars.next().unwrap());
            } else {
                break;
            }
        }

        if key.is_empty() {
            break;
        }

        // Skip whitespace
        while chars.peek().map(|c| c.is_whitespace()).unwrap_or(false) {
            chars.next();
        }

        // Expect =
        if chars.next() != Some('=') {
            continue;
        }

        // Skip whitespace
        while chars.peek().map(|c| c.is_whitespace()).unwrap_or(false) {
            chars.next();
        }

        // Parse value
        let value = match chars.peek().copied() {
            Some('"') | Some('\'') => {
                let quote = chars.next().unwrap();
                let mut val = String::new();
                let mut escape_next = false;

                loop {
                    match chars.next() {
                        None => break,
                        Some(c) if escape_next => {
                            match c {
                                'n' => val.push('\n'),
                                't' => val.push('\t'),
                                'r' => val.push('\r'),
                                '\\' => val.push('\\'),
                                '"' => val.push('"'),
                                '\'' => val.push('\''),
                                _ => {
                                    val.push('\\');
                                    val.push(c);
                                }
                            }
                            escape_next = false;
                        }
                        Some('\\') => escape_next = true,
                        Some(c) if c == quote => break,
                        Some(c) => val.push(c),
                    }
                }
                Value::String(val)
            }
            Some(c) if c.is_ascii_digit() || c == '-' => {
                let mut num = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_digit() || c == '.' || c == '-' {
                        num.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
                if let Ok(n) = num.parse::<i64>() {
                    Value::Number(n.into())
                } else if let Ok(n) = num.parse::<f64>() {
                    json!(n)
                } else {
                    Value::String(num)
                }
            }
            Some(_) => {
                let mut word = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_alphanumeric() || c == '_' {
                        word.push(chars.next().unwrap());
                    } else {
                        break;
                    }
                }
                match word.as_str() {
                    "true" | "True" => Value::Bool(true),
                    "false" | "False" => Value::Bool(false),
                    "null" | "None" => Value::Null,
                    _ => Value::String(word),
                }
            }
            None => continue,
        };

        args.insert(key, value);
    }

    Value::Object(args)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============ Non-streaming tests ============

    #[test]
    fn test_parse_simple_call() {
        let text = r#"<function_calls>get_weather(location="Paris")</function_calls>"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["location"], "Paris");
    }

    // ============ Streaming parser tests ============

    #[test]
    fn test_streaming_simple_call() {
        let mut parser = Olmo3Parser::new();

        let input = r#"<function_calls>get_weather(location="Paris")</function_calls>"#;

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

        // Should have a tool call
        assert!(
            events.iter().any(|e| e.is_tool_call()),
            "Expected tool call, got: {:?}",
            events
        );

        let tool_call = events.iter().find(|e| e.is_tool_call()).unwrap();
        let invocation = tool_call.as_tool_call().unwrap();
        assert_eq!(invocation.name, "get_weather");
        assert_eq!(invocation.arguments["location"], "Paris");
    }

    #[test]
    fn test_streaming_multiple_calls() {
        let mut parser = Olmo3Parser::new();

        let input = r#"<function_calls>
get_weather(location="Paris")
get_time(timezone="UTC")
</function_calls>"#;

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

        // Should have two tool calls
        let tool_calls: Vec<_> = events.iter().filter(|e| e.is_tool_call()).collect();
        assert_eq!(
            tool_calls.len(),
            2,
            "Expected 2 tool calls, got: {:?}",
            events
        );
    }

    #[test]
    fn test_streaming_text_before_call() {
        let mut parser = Olmo3Parser::new();

        let input = r#"Here is the result: <function_calls>test(x=1)</function_calls>"#;

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

        // Should have text and tool call
        assert!(events.iter().any(|e| matches!(e, ParseEvent::Text(_))));
        assert!(events.iter().any(|e| e.is_tool_call()));
    }

    #[test]
    fn test_streaming_reset() {
        let mut parser = Olmo3Parser::new();

        // Start parsing
        parser.feed("<function_calls>");

        // Reset
        parser.reset();

        // Parse new input
        let input = r#"<function_calls>test(x=1)</function_calls>"#;

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

        assert!(events.iter().any(|e| e.is_tool_call()));
    }

    #[test]
    fn test_parse_multiple_args() {
        let text = r#"<function_calls>calculate(x=42, y=13, op="add")</function_calls>"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "calculate");
        assert_eq!(calls[0].arguments["x"], 42);
        assert_eq!(calls[0].arguments["y"], 13);
        assert_eq!(calls[0].arguments["op"], "add");
    }

    #[test]
    fn test_parse_multiple_calls() {
        let text = r#"<function_calls>
get_weather(location="Paris")
get_time(timezone="UTC")
</function_calls>"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[1].name, "get_time");
    }

    #[test]
    fn test_parse_code_with_parens() {
        let text = r#"<function_calls>execute_python(code="print('hello world')\nfor i in range(10):\n    print(i)")</function_calls>"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "execute_python");
        assert!(calls[0].arguments["code"]
            .as_str()
            .unwrap()
            .contains("print"));
    }
}
