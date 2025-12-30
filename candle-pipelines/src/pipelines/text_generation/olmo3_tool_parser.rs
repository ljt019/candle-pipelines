//! Parser for OLMo-3's function call format.
//!
//! OLMo-3 uses a Python-like syntax for tool calls:
//! ```text
//! <function_calls>
//! get_weather(location="Paris", unit="celsius")
//! calculate(x=42, y=13)
//! </function_calls>
//! ```

use regex::Regex;
use serde_json::{json, Value};

/// Represents a parsed OLMo-3 tool call.
#[derive(Debug, Clone)]
pub struct Olmo3ToolCall {
    pub name: String,
    pub arguments: Value,
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

    #[test]
    fn test_parse_simple_call() {
        let text = r#"<function_calls>get_weather(location="Paris")</function_calls>"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["location"], "Paris");
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
