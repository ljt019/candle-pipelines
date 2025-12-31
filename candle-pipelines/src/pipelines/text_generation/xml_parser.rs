use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tag {
    name: String,
    id: usize,
}

impl Tag {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn id(&self) -> usize {
        self.id
    }
}

impl PartialEq<str> for Tag {
    fn eq(&self, other: &str) -> bool {
        self.name == other
    }
}

impl PartialEq<&str> for Tag {
    fn eq(&self, other: &&str) -> bool {
        self.name == *other
    }
}

impl PartialEq<String> for Tag {
    fn eq(&self, other: &String) -> bool {
        self.name == *other
    }
}

impl PartialEq<&Tag> for Tag {
    fn eq(&self, other: &&Tag) -> bool {
        self == *other
    }
}

/// Which part of a tag is being emitted during streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagParts {
    /// Opening tag (e.g., `<tool_call>`).
    Start,
    /// Content between the opening and closing tags.
    Content,
    /// Closing tag (e.g., `</tool_call>`).
    End,
}

/// An event emitted during XML stream parsing.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Content within a registered XML tag.
    Tagged {
        /// The tag being parsed.
        tag: Tag,
        /// Which part of the tag (start, content, end).
        part: TagParts,
        /// The content/text.
        content: String,
        /// Attributes from the opening tag.
        attributes: HashMap<String, String>,
    },
    /// Content outside any registered tags (plain output).
    Output {
        /// Which part of the output (start, content, end).
        part: TagParts,
        /// The content/text.
        content: String,
    },
    /// An error occurred in the underlying stream.
    Error {
        /// The error message.
        message: String,
    },
}

impl Event {
    fn tagged(
        tag: Tag,
        part: TagParts,
        content: impl Into<String>,
        attributes: HashMap<String, String>,
    ) -> Self {
        Self::Tagged {
            tag,
            part,
            content: content.into(),
            attributes,
        }
    }

    fn plain(part: TagParts, content: impl Into<String>) -> Self {
        Self::Output {
            part,
            content: content.into(),
        }
    }

    pub(crate) fn content(content: impl Into<String>) -> Self {
        Self::plain(TagParts::Content, content)
    }

    pub(crate) fn start(
        tag: Tag,
        opening_tag: impl Into<String>,
        attributes: HashMap<String, String>,
    ) -> Self {
        Self::tagged(tag, TagParts::Start, opening_tag, attributes)
    }

    pub(crate) fn end(
        tag: Tag,
        full_xml: impl Into<String>,
        attributes: HashMap<String, String>,
    ) -> Self {
        Self::tagged(tag, TagParts::End, full_xml, attributes)
    }

    fn tagged_internal(
        tag: Tag,
        content: impl Into<String>,
        attributes: HashMap<String, String>,
    ) -> Self {
        Self::tagged(tag, TagParts::Content, content, attributes)
    }

    /// Create an error event.
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error {
            message: message.into(),
        }
    }

    /// Returns true if this is an error event.
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error { .. })
    }

    /// Get the error message if this is an error event.
    pub fn error_message(&self) -> Option<&str> {
        match self {
            Self::Error { message } => Some(message),
            _ => None,
        }
    }

    /// Get the content/text of this event.
    pub fn get_content(&self) -> &str {
        match self {
            Self::Tagged { content, .. } | Self::Output { content, .. } => content,
            Self::Error { message } => message,
        }
    }

    /// Get the tag name if this is a tagged event.
    pub fn tag(&self) -> Option<&str> {
        match self {
            Self::Tagged { tag, .. } => Some(tag.name()),
            Self::Output { .. } | Self::Error { .. } => None,
        }
    }

    /// Get which part of the tag/output this event represents.
    pub fn part(&self) -> TagParts {
        match self {
            Self::Tagged { part, .. } | Self::Output { part, .. } => *part,
            Self::Error { .. } => TagParts::End, // Error terminates
        }
    }

    /// Get attributes if this is a tagged event with attributes.
    /// Returns None for Output/Error events, or tagged events with no attributes.
    pub fn attributes(&self) -> Option<&HashMap<String, String>> {
        match self {
            Self::Tagged { attributes, .. } if !attributes.is_empty() => Some(attributes),
            _ => None,
        }
    }

    /// Parse this event as a tool call if it's a complete `<tool_call>` end event.
    /// Returns `(name, arguments)` if successful.
    pub fn parse_tool_call(&self) -> Option<(String, serde_json::Value)> {
        let tag = self.tag()?;
        if tag != "tool_call" || self.part() != TagParts::End {
            return None;
        }

        let content = self.get_content();

        let inner = content
            .strip_prefix("<tool_call>")?
            .strip_suffix("</tool_call>")?
            .trim();

        let parsed: serde_json::Value = serde_json::from_str(inner).ok()?;
        let name = parsed.get("name")?.as_str()?.to_string();
        let arguments = parsed.get("arguments")?.clone();

        Some((name, arguments))
    }
}

/// Builder for creating an [`XmlParser`] with specific tags to track.
///
/// # Example
///
/// ```rust,ignore
/// let mut parser = XmlParserBuilder::new()
///     .register_tag("think")
///     .register_tag("answer")
///     .build();
/// ```
#[derive(Debug, Default)]
pub struct XmlParserBuilder {
    tags: Vec<String>,
    next_id: usize,
}

impl XmlParserBuilder {
    /// Create a new builder with no registered tags.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tag name for the parser to track. Returns self for chaining.
    pub fn register_tag(mut self, tag: impl Into<String>) -> Self {
        let name = tag.into();
        self.next_id += 1;
        self.tags.push(name);
        self
    }

    /// Build the parser with all registered tags.
    pub fn build(self) -> XmlParser {
        let mut tag_map = HashMap::new();
        let mut tags_set = HashSet::new();

        for (id, name) in self.tags.into_iter().enumerate() {
            tags_set.insert(name.clone());
            tag_map.insert(name.clone(), Tag { name, id });
        }

        XmlParser::new(tags_set, tag_map)
    }
}

#[derive(Debug, Clone)]
struct ParserState {
    /// (tag_name, content, attributes)
    open_tags: Vec<(String, String, HashMap<String, String>)>,
    content_buffer: String,
    tag_buffer: String,
    in_tag: bool,
    emitted_top_len: usize,
    emitted_tag_lens: std::collections::HashMap<String, usize>,
}

impl Default for ParserState {
    fn default() -> Self {
        Self {
            open_tags: Vec::with_capacity(4),
            content_buffer: String::with_capacity(1024),
            tag_buffer: String::with_capacity(64),
            in_tag: false,
            emitted_top_len: 0,
            emitted_tag_lens: HashMap::with_capacity(4),
        }
    }
}

/// Streaming XML parser for extracting structured content from LLM output.
///
/// Parses text containing XML-like tags and emits events for tag boundaries
/// and content. Useful for structured output like `<think>...</think>` blocks.
///
/// **Note:** This parser is `!Sync`. If you need to share it across threads,
/// wrap it in a `Mutex` or `RwLock`.
#[derive(Debug, Clone)]
pub struct XmlParser {
    registered_tags: HashSet<String>,
    tag_map: HashMap<String, Tag>,
    state: ParserState,
}

impl XmlParser {
    /// Create a new parser for the specified tags.
    pub fn new(tags: HashSet<String>, tag_map: HashMap<String, Tag>) -> Self {
        Self {
            registered_tags: tags,
            tag_map,
            state: ParserState::default(),
        }
    }

    /// Reset parser state for a new parsing session.
    pub fn reset(&mut self) {
        self.state = ParserState::default();
    }

    /// Parse a complete text string and return all events.
    pub fn parse(&mut self, text: &str) -> Vec<Event> {
        self.reset();
        let mut events = Vec::new();

        for c in text.chars() {
            events.extend(self.process_char_internal(c));
        }

        events.extend(self.flush_internal());
        events
    }

    /// Parse a single token in streaming mode. Call `flush()` when done.
    pub fn parse_token(&mut self, token: &str) -> Vec<Event> {
        let mut events = Vec::new();

        for c in token.chars() {
            events.extend(self.process_char_internal(c));
        }

        // Emit plain content as it comes in
        if self.state.open_tags.is_empty() {
            let current_len = self.state.content_buffer.len();
            if current_len > self.state.emitted_top_len {
                let new_slice = &self.state.content_buffer[self.state.emitted_top_len..];
                if !new_slice.is_empty() {
                    events.push(Event::content(new_slice));
                }
                self.state.emitted_top_len = current_len;
            }
        } else if let Some((tag_name_ref, content_ref, attrs_ref)) = self.state.open_tags.last() {
            // Emit tagged content as it comes in
            let tag_name = tag_name_ref.clone();
            let content = content_ref.clone();
            let attrs = attrs_ref.clone();
            let total_len = content.len();

            let already_emitted = *self.state.emitted_tag_lens.get(&tag_name).unwrap_or(&0);

            if total_len > already_emitted {
                let new_slice = &content[already_emitted..];
                if !new_slice.is_empty() {
                    if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                        events.push(Event::tagged_internal(
                            tag_handle.clone(),
                            new_slice,
                            attrs.clone(),
                        ));
                    }
                }
                self.state
                    .emitted_tag_lens
                    .insert(tag_name.clone(), total_len);
            }
        }

        events
    }

    fn process_char_internal(&mut self, c: char) -> Vec<Event> {
        let mut events = Vec::new();

        match c {
            '<' => {
                self.state.in_tag = true;
                self.state.tag_buffer.clear();
                self.state.tag_buffer.push(c);
            }
            '>' if self.state.in_tag => {
                self.state.tag_buffer.push(c);
                self.state.in_tag = false;

                let tag_content = self.state.tag_buffer.clone();
                self.state.tag_buffer.clear();

                events.extend(self.handle_tag(&tag_content));
            }
            _ if self.state.in_tag => {
                self.state.tag_buffer.push(c);
            }
            _ => {
                if let Some((_, ref mut content, _)) = self.state.open_tags.last_mut() {
                    content.push(c);
                } else {
                    self.state.content_buffer.push(c);
                }
            }
        }

        events
    }

    fn handle_tag(&mut self, tag_content: &str) -> Vec<Event> {
        let mut events = Vec::new();

        if let Some(tag_name) = self.parse_tag_name(tag_content) {
            if self.registered_tags.contains(&tag_name) {
                if tag_content.starts_with("</") {
                    // Closing tag - only close if it matches the outermost open tag
                    if let Some((outermost_name, _, _)) = self.state.open_tags.first() {
                        if outermost_name == &tag_name {
                            // Matches outermost - close it
                            let (_, content, attrs) = self.state.open_tags.remove(0);
                            let already_emitted =
                                self.state.emitted_tag_lens.remove(&tag_name).unwrap_or(0);

                            if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                                // Emit any remaining content
                                if content.len() > already_emitted {
                                    let remaining = &content[already_emitted..];
                                    if !remaining.is_empty() {
                                        events.push(Event::tagged_internal(
                                            tag_handle.clone(),
                                            remaining,
                                            attrs.clone(),
                                        ));
                                    }
                                }
                                let full_xml = format!("<{}>{}</{}>", tag_name, content, tag_name);
                                events.push(Event::end(tag_handle.clone(), full_xml, attrs));
                            }
                        } else {
                            // Doesn't match outermost - treat as content (greedy)
                            if let Some((_, ref mut content, _)) = self.state.open_tags.last_mut() {
                                content.push_str(tag_content);
                            }
                        }
                    } else {
                        // No open tags - closing tag becomes plain content
                        self.state.content_buffer.push_str(tag_content);
                    }
                } else if tag_content.ends_with("/>") {
                    // Self-closing tag
                    if self.state.open_tags.is_empty() {
                        // Emit any buffered plain content first
                        if !self.state.content_buffer.is_empty() {
                            let content = &self.state.content_buffer[self.state.emitted_top_len..];
                            if !content.is_empty() {
                                events.push(Event::content(content));
                            }
                            self.state.emitted_top_len = self.state.content_buffer.len();
                        }

                        let attrs = self.parse_attributes(tag_content);
                        if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                            // Emit start + end for self-closing (no content event)
                            events.push(Event::start(
                                tag_handle.clone(),
                                format!("<{}>", tag_name),
                                attrs.clone(),
                            ));
                            events.push(Event::end(tag_handle.clone(), tag_content, attrs));
                        }
                    } else {
                        // Inside another tag - treat as content (greedy)
                        if let Some((_, ref mut content, _)) = self.state.open_tags.last_mut() {
                            content.push_str(tag_content);
                        }
                    }
                } else {
                    // Opening tag
                    if self.state.open_tags.is_empty() {
                        // Only open new tags at top level (greedy parsing)
                        // Emit any buffered plain content first
                        if !self.state.content_buffer.is_empty() {
                            let content = &self.state.content_buffer[self.state.emitted_top_len..];
                            if !content.is_empty() {
                                events.push(Event::content(content));
                            }
                            self.state.emitted_top_len = self.state.content_buffer.len();
                        }

                        let attrs = self.parse_attributes(tag_content);
                        self.state
                            .open_tags
                            .push((tag_name.clone(), String::new(), attrs.clone()));

                        if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                            events.push(Event::start(tag_handle.clone(), tag_content, attrs));
                        }
                    } else {
                        // Inside another tag - treat as content (greedy)
                        if let Some((_, ref mut content, _)) = self.state.open_tags.last_mut() {
                            content.push_str(tag_content);
                        }
                    }
                }
            } else if self.state.open_tags.is_empty() {
                self.state.content_buffer.push_str(tag_content);
            } else if let Some((_, ref mut content, _)) = self.state.open_tags.last_mut() {
                content.push_str(tag_content);
            }
        } else if self.state.open_tags.is_empty() {
            self.state.content_buffer.push_str(tag_content);
        } else if let Some((_, ref mut content, _)) = self.state.open_tags.last_mut() {
            content.push_str(tag_content);
        }

        events
    }

    fn parse_tag_name(&self, tag_content: &str) -> Option<String> {
        if tag_content.len() < 3 || !tag_content.starts_with('<') || !tag_content.ends_with('>') {
            return None;
        }

        let inner = &tag_content[1..tag_content.len() - 1];

        if let Some(name) = inner.strip_prefix('/') {
            Some(name.split_whitespace().next()?.to_string())
        } else {
            let name = inner.split_whitespace().next()?;
            if let Some(stripped) = name.strip_suffix('/') {
                Some(stripped.to_string())
            } else {
                Some(name.to_string())
            }
        }
    }

    /// Parse attributes from a tag like `<tag name=value foo="bar">`.
    fn parse_attributes(&self, tag_content: &str) -> HashMap<String, String> {
        let mut attrs = HashMap::new();

        // Strip < and >
        if tag_content.len() < 3 {
            return attrs;
        }
        let inner = &tag_content[1..tag_content.len() - 1];

        // Skip the tag name (first token)
        let after_name = match inner.split_whitespace().next() {
            Some(name) => {
                let name_end = inner.find(name).unwrap_or(0) + name.len();
                &inner[name_end..]
            }
            None => return attrs,
        };

        let mut chars = after_name.chars().peekable();

        while let Some(c) = chars.next() {
            // Skip whitespace
            if c.is_whitespace() {
                continue;
            }

            // Read attribute name
            let mut attr_name = String::new();
            attr_name.push(c);
            while let Some(&next) = chars.peek() {
                if next == '=' || next.is_whitespace() {
                    break;
                }
                attr_name.push(chars.next().unwrap());
            }

            // Skip whitespace and find =
            while let Some(&next) = chars.peek() {
                if next == '=' {
                    chars.next();
                    break;
                } else if next.is_whitespace() {
                    chars.next();
                } else {
                    break;
                }
            }

            // Read attribute value
            let mut attr_value = String::new();

            // Skip whitespace before value
            while let Some(&next) = chars.peek() {
                if !next.is_whitespace() {
                    break;
                }
                chars.next();
            }

            if let Some(&quote) = chars.peek() {
                if quote == '"' || quote == '\'' {
                    chars.next(); // consume opening quote
                    while let Some(c) = chars.next() {
                        if c == quote {
                            break;
                        }
                        attr_value.push(c);
                    }
                } else {
                    // Unquoted value - read until whitespace or /
                    while let Some(&next) = chars.peek() {
                        if next.is_whitespace() || next == '/' {
                            break;
                        }
                        attr_value.push(chars.next().unwrap());
                    }
                }
            }

            if !attr_name.is_empty() {
                attrs.insert(attr_name, attr_value);
            }
        }

        attrs
    }

    /// Flush any remaining buffered content as events.
    pub fn flush(&mut self) -> Vec<Event> {
        self.flush_internal()
    }

    fn flush_internal(&mut self) -> Vec<Event> {
        let mut events = Vec::new();

        // Emit any remaining plain content
        if self.state.content_buffer.len() > self.state.emitted_top_len {
            let remaining = &self.state.content_buffer[self.state.emitted_top_len..];
            if !remaining.is_empty() {
                events.push(Event::content(remaining));
            }
        }
        self.state.content_buffer.clear();
        self.state.emitted_top_len = 0;

        // Emit remaining content for any unclosed tags (but no End event since they weren't closed)
        let drained: Vec<_> = self.state.open_tags.drain(..).collect();
        for (tag_name, content, attrs) in drained {
            let already_emitted = self.state.emitted_tag_lens.remove(&tag_name).unwrap_or(0);

            if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                if content.len() > already_emitted {
                    let remaining = &content[already_emitted..];
                    if !remaining.is_empty() {
                        events.push(Event::tagged_internal(
                            tag_handle.clone(),
                            remaining,
                            attrs.clone(),
                        ));
                    }
                }
                // Don't emit End event for unclosed tags
            }
        }

        events
    }

    /// Returns the set of tag names this parser recognizes.
    pub fn registered_tags(&self) -> &HashSet<String> {
        &self.registered_tags
    }

    /// Wrap a token iterator to produce XML parsing events.
    ///
    /// Use this to compose XML parsing with any text generation iterator:
    ///
    /// ```rust,ignore
    /// let mut parser = XmlParserBuilder::new().register_tag("think").build();
    /// let tokens = pipeline.run_iter("...")?;
    /// let events = parser.parse_iter(tokens);
    ///
    /// for event in events {
    ///     match (event.tag(), event.part()) {
    ///         (Some("think"), TagParts::Content) => println!("[thinking] {}", event.get_content()),
    ///         (None, TagParts::Content) => print!("{}", event.get_content()),
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub fn parse_iter<I>(&self, iter: I) -> EventIterator<I>
    where
        I: Iterator<Item = crate::error::Result<String>>,
    {
        EventIterator::new(self.clone(), iter)
    }
}

/// Sync iterator of XML parsing events.
///
/// Wraps a token iterator and parses XML tags as they arrive.
pub struct EventIterator<I> {
    parser: XmlParser,
    inner: I,
    buffer: Vec<Event>,
    flushed: bool,
}

impl<I> EventIterator<I> {
    fn new(mut parser: XmlParser, iter: I) -> Self {
        parser.reset();
        Self {
            parser,
            inner: iter,
            buffer: Vec::new(),
            flushed: false,
        }
    }
}

impl<I> Iterator for EventIterator<I>
where
    I: Iterator<Item = crate::error::Result<String>>,
{
    type Item = Event;

    fn next(&mut self) -> Option<Self::Item> {
        // Return buffered events first
        if !self.buffer.is_empty() {
            return Some(self.buffer.remove(0));
        }

        // Get more tokens and parse
        for result in self.inner.by_ref() {
            match result {
                Ok(token) => {
                    let events = self.parser.parse_token(&token);
                    if !events.is_empty() {
                        self.buffer.extend(events);
                        return Some(self.buffer.remove(0));
                    }
                }
                Err(e) => {
                    // Emit error as event instead of silently breaking
                    return Some(Event::error(e.to_string()));
                }
            }
        }

        // Flush remaining events
        if !self.flushed {
            self.flushed = true;
            let events = self.parser.flush();
            if !events.is_empty() {
                self.buffer.extend(events);
                return Some(self.buffer.remove(0));
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plain_text_only_parsing() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "Regular content";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].tag(), None);
        assert_eq!(events[0].get_content(), text);
    }

    #[test]
    fn test_empty_parsing() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 0);
    }

    #[test]
    fn test_whitespace_parsing() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "  ";
        let events = parser.parse(&text);

        assert!(events.len() == 1);
        assert!(events[0].tag() == None);
        assert_eq!(events[0].get_content(), text);
    }

    #[test]
    fn test_single_tag_parsing() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "<think>Hello world</think>";
        let events = parser.parse(&text);

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::Content, "Hello world"),
            (Some("think"), TagParts::End, text),
        ];

        assert_eq!(events.len(), 3);
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }
    }

    #[test]
    fn test_plain_text_and_single_tag_parsing() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "<think>Hello world</think>Regular content";
        let events = parser.parse(&text);

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::Content, "Hello world"),
            (Some("think"), TagParts::End, "<think>Hello world</think>"),
            (None, TagParts::Content, "Regular content"),
        ];

        assert_eq!(events.len(), expected.len());
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }
    }

    #[test]
    fn test_plain_text_before_and_after_single_tag_parsing() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "How are <think>Hello world</think>you doing today?";
        let events = parser.parse(&text);

        let expected = &[
            (None, TagParts::Content, "How are "),
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::Content, "Hello world"),
            (Some("think"), TagParts::End, "<think>Hello world</think>"),
            (None, TagParts::Content, "you doing today?"),
        ];

        let expected_content = "How are you doing today?";

        let mut final_content = String::new();
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            match event.tag() {
                None => final_content.push_str(event.get_content()),
                Some(_) => {}
            }

            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }

        assert_eq!(final_content, expected_content);
    }

    #[test]
    fn test_multiple_tags_parsing() {
        let mut parser = XmlParserBuilder::new()
            .register_tag("think")
            .register_tag("answer")
            .build();

        let text = "<think>Hm the answer to 1 + 1 is 2</think><answer>2</answer>";
        let events = parser.parse(&text);

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (
                Some("think"),
                TagParts::Content,
                "Hm the answer to 1 + 1 is 2",
            ),
            (
                Some("think"),
                TagParts::End,
                "<think>Hm the answer to 1 + 1 is 2</think>",
            ),
            (Some("answer"), TagParts::Start, "<answer>"),
            (Some("answer"), TagParts::Content, "2"),
            (Some("answer"), TagParts::End, "<answer>2</answer>"),
        ];

        assert_eq!(events.len(), expected.len());
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }
    }

    #[test]
    fn test_empty_tags_parsing() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "<think></think>Regular content";
        let events = parser.parse(&text);

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::End, "<think></think>"),
            (None, TagParts::Content, "Regular content"),
        ];

        assert_eq!(events.len(), expected.len());
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }
    }

    #[test]
    fn test_unregistered_tag_parsing() {
        let mut parser = XmlParserBuilder::new().register_tag("answer").build();

        let text = "<think>Hm the answer to 1 + 1 is 2</think><answer>2</answer>";
        let events = parser.parse(&text);

        let expected = &[
            (
                None,
                TagParts::Content,
                "<think>Hm the answer to 1 + 1 is 2</think>",
            ),
            (Some("answer"), TagParts::Start, "<answer>"),
            (Some("answer"), TagParts::Content, "2"),
            (Some("answer"), TagParts::End, "<answer>2</answer>"),
        ];

        assert_eq!(events.len(), expected.len());
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }
    }

    #[test]
    fn test_is_greedy_parsing() {
        let mut parser = XmlParserBuilder::new()
            .register_tag("think")
            .register_tag("answer")
            .build();

        let text = "<think>Hm the answer to 1 + 1 is 2 so I should output <answer> tags containing 2 like <answer>2</answer>, I'll do that now</think><answer>2</answer>";
        let events = parser.parse(&text);

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::Content, "Hm the answer to 1 + 1 is 2 so I should output <answer> tags containing 2 like <answer>2</answer>, I'll do that now"),
            (Some("think"), TagParts::End, "<think>Hm the answer to 1 + 1 is 2 so I should output <answer> tags containing 2 like <answer>2</answer>, I'll do that now</think>"),
            (Some("answer"), TagParts::Start, "<answer>"),
            (Some("answer"), TagParts::Content, "2"),
            (Some("answer"), TagParts::End, "<answer>2</answer>"),
        ];

        assert_eq!(events.len(), expected.len());
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }
    }

    #[test]
    fn test_greedy_multiple_same_tag_parsing() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "<think>Hello world<think>Regular content</think></think>";
        let events = parser.parse(&text);

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (
                Some("think"),
                TagParts::Content,
                "Hello world<think>Regular content",
            ),
            (
                Some("think"),
                TagParts::End,
                "<think>Hello world<think>Regular content</think>",
            ),
            (None, TagParts::Content, "</think>"),
        ];

        assert_eq!(events.len(), expected.len());
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }
    }

    #[test]
    fn test_greedy_multiple_same_open_tag_parsing() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "<think>Hello world<think> Regular content</think>";
        let events = parser.parse(&text);

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (
                Some("think"),
                TagParts::Content,
                "Hello world<think> Regular content",
            ),
            (
                Some("think"),
                TagParts::End,
                "<think>Hello world<think> Regular content</think>",
            ),
        ];

        assert_eq!(events.len(), expected.len());
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }
    }

    #[test]
    fn test_mismatched_close_parsing() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "<think>Hm I think the answer to 1 + 1 is 2</answer>";
        let events = parser.parse(&text);

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (
                Some("think"),
                TagParts::Content,
                "Hm I think the answer to 1 + 1 is 2</answer>",
            ),
        ];

        assert_eq!(events.len(), expected.len());
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }
    }

    #[test]
    fn test_unclosed_tag_parsing() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "<think>Hm I think the answer to 1 + 1 is 2";
        let events = parser.parse(&text);

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (
                Some("think"),
                TagParts::Content,
                "Hm I think the answer to 1 + 1 is 2",
            ),
        ];

        assert_eq!(events.len(), expected.len());
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }
    }

    #[test]
    fn test_self_closing_tag_parsing() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "<think/>Regular Content<think />";
        let events = parser.parse(&text);

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::End, "<think/>"),
            (None, TagParts::Content, "Regular Content"),
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::End, "<think />"),
        ];

        assert_eq!(events.len(), expected.len());
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }
    }

    #[test]
    fn test_multiple_same_tag_parsing() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "<think>Hello world</think><think>Regular content</think>";

        let events = parser.parse(&text);

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::Content, "Hello world"),
            (Some("think"), TagParts::End, "<think>Hello world</think>"),
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::Content, "Regular content"),
            (
                Some("think"),
                TagParts::End,
                "<think>Regular content</think>",
            ),
        ];

        assert_eq!(events.len(), expected.len());
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }
    }

    #[test]
    fn test_reset_isolation() {
        let mut parser = XmlParserBuilder::new()
            .register_tag("think")
            .register_tag("answer")
            .build();

        let text = "Before <think>thinking here</think> middle <answer>42</answer> after";

        let first_parse = parser.parse(&text);
        let second_parse = parser.parse(&text);

        assert_eq!(
            first_parse.len(),
            second_parse.len(),
            "parse() should return same number of events on repeated calls"
        );

        for (first, second) in first_parse.iter().zip(second_parse.iter()) {
            assert_eq!(first.tag(), second.tag());
            assert_eq!(first.part(), second.part());
            assert_eq!(first.get_content(), second.get_content());
            assert_eq!(first.attributes(), second.attributes());
        }
    }

    #[test]
    fn test_no_attributes_returns_none() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "<think>Hello world</think>Regular content";

        let events = parser.parse(&text);

        assert_eq!(events[0].tag(), Some("think"));
        assert_eq!(events[0].attributes(), None);
    }

    #[test]
    fn test_attribute_no_quotes() {
        let mut parser = XmlParserBuilder::new()
            .register_tag("function_call")
            .build();

        let text = "<function_call name=get_weather>Tokyo</function_call>";

        let events = parser.parse(&text);

        assert_eq!(events[0].tag(), Some("function_call"));
        assert_eq!(
            events[0].attributes().unwrap().get("name"),
            Some(&"get_weather".to_string())
        );
    }

    #[test]
    fn test_attributes_self_closing_tag_parsing() {
        let mut parser = XmlParserBuilder::new()
            .register_tag("function_call")
            .build();

        let text = "<function_call name=get_weather/>";
        let events = parser.parse(&text);

        assert_eq!(events[0].tag(), Some("function_call"));
        assert_eq!(
            events[0].attributes().unwrap().get("name"),
            Some(&"get_weather".to_string())
        );
    }

    #[test]
    fn test_attribute_double_quotes() {
        let mut parser = XmlParserBuilder::new()
            .register_tag("function_call")
            .build();

        let text = "<function_call name=\"get_weather\">Tokyo</function_call>";

        let events = parser.parse(&text);

        assert_eq!(events[0].tag(), Some("function_call"));
        assert_eq!(
            events[0].attributes().unwrap().get("name"),
            Some(&"get_weather".to_string())
        );
    }

    #[test]
    fn test_attribute_single_quotes() {
        let mut parser = XmlParserBuilder::new()
            .register_tag("function_call")
            .build();

        let text = "<function_call name='get_weather'>Tokyo</function_call>";

        let events = parser.parse(&text);

        assert_eq!(events[0].tag(), Some("function_call"));
        assert_eq!(
            events[0].attributes().unwrap().get("name"),
            Some(&"get_weather".to_string())
        );
    }

    #[test]
    fn test_attribute_quotes_with_spaces() {
        let mut parser = XmlParserBuilder::new()
            .register_tag("function_call")
            .build();

        let text = "<function_call name=\"get weather\">Tokyo</function_call>";

        let events = parser.parse(&text);

        assert_eq!(events[0].tag(), Some("function_call"));
        assert_eq!(
            events[0].attributes().unwrap().get("name"),
            Some(&"get weather".to_string())
        );
    }

    #[test]
    fn test_multiple_attributes() {
        let mut parser = XmlParserBuilder::new()
            .register_tag("function_call")
            .build();

        let text = "<function_call name='get_weather' id=0>Tokyo</function_call>";

        let events = parser.parse(&text);

        assert_eq!(events[0].tag(), Some("function_call"));
        assert_eq!(
            events[0].attributes().unwrap().get("name"),
            Some(&"get_weather".to_string())
        );
        assert_eq!(
            events[0].attributes().unwrap().get("id"),
            Some(&"0".to_string())
        );
    }

    #[test]
    fn test_attributes_persist_across_tag_sequence() {
        let mut parser = XmlParserBuilder::new()
            .register_tag("function_call")
            .build();

        let text = "<function_call name=test>content</function_call>";
        let events = parser.parse(&text);

        // All three events should have the same attributes
        for event in &events {
            if event.tag() == Some("function_call") {
                assert_eq!(
                    event.attributes().unwrap().get("name"),
                    Some(&"test".to_string())
                );
            }
        }
    }

    #[test]
    fn test_attribute_returns_none_for_plain_content() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "Regular content";

        let events = parser.parse(&text);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].tag(), None);
        assert_eq!(events[0].attributes(), None);
    }

    #[test]
    fn test_iter_attributes_preserved() {
        let parser = XmlParserBuilder::new()
            .register_tag("function_call")
            .build();

        let tokens = vec![
            Ok("<function_call na".to_string()),
            Ok("me=get_weather>Tokyo</function_call>".to_string()),
        ];

        let events: Vec<Event> = parser.parse_iter(tokens.into_iter()).collect();

        assert_eq!(events[0].tag(), Some("function_call"));
        assert_eq!(
            events[0].attributes().unwrap().get("name"),
            Some(&"get_weather".to_string())
        );
    }

    #[test]
    fn test_iter_greedy_multiple_same_tag_parsing() {
        let parser = XmlParserBuilder::new().register_tag("think").build();

        let tokens = vec![
            Ok("<think>Hello ".to_string()),
            Ok("world<think> Regular content</think>".to_string()),
            Ok("</think>".to_string()),
        ];

        let events = parser
            .parse_iter(tokens.into_iter())
            .collect::<Vec<Event>>();

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::Content, "Hello "),
            (
                Some("think"),
                TagParts::Content,
                "world<think> Regular content",
            ),
            (
                Some("think"),
                TagParts::End,
                "<think>Hello world<think> Regular content</think>",
            ),
            (None, TagParts::Content, "</think>"),
        ];

        assert_eq!(events.len(), expected.len());
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }
    }

    #[test]
    fn test_iter_greedy_multiple_same_open_tag_parsing() {
        let parser = XmlParserBuilder::new().register_tag("think").build();

        let tokens = vec![
            Ok("<think>Hello ".to_string()),
            Ok("world<think> Regular content".to_string()),
            Ok("</think>".to_string()),
        ];

        let events = parser
            .parse_iter(tokens.into_iter())
            .collect::<Vec<Event>>();

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::Content, "Hello "),
            (
                Some("think"),
                TagParts::Content,
                "world<think> Regular content",
            ),
            (
                Some("think"),
                TagParts::End,
                "<think>Hello world<think> Regular content</think>",
            ),
        ];

        assert_eq!(events.len(), expected.len());
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }
    }

    #[test]
    fn test_iter_split_open_tag_parsing() {
        let parser = XmlParserBuilder::new().register_tag("think").build();

        let tokens = vec![
            Ok("<".to_string()),
            Ok("think>Hello world</think>".to_string()),
        ];

        let events = parser
            .parse_iter(tokens.into_iter())
            .collect::<Vec<Event>>();

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::Content, "Hello world"),
            (Some("think"), TagParts::End, "<think>Hello world</think>"),
        ];

        assert_eq!(events.len(), expected.len());
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }
    }

    #[test]
    fn test_iter_split_close_tag_parsing() {
        let parser = XmlParserBuilder::new().register_tag("think").build();

        let tokens = vec![
            Ok("<".to_string()),
            Ok("think>Hello world</".to_string()),
            Ok("think>".to_string()),
        ];

        let events = parser
            .parse_iter(tokens.into_iter())
            .collect::<Vec<Event>>();

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::Content, "Hello world"),
            (Some("think"), TagParts::End, "<think>Hello world</think>"),
        ];

        assert_eq!(events.len(), expected.len());
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }
    }

    #[test]
    fn test_iter_multiple_splits_tag_parsing() {
        let parser = XmlParserBuilder::new().register_tag("think").build();

        let tokens = vec![
            Ok("<".to_string()),
            Ok("think".to_string()),
            Ok(">Hello world</think>".to_string()),
        ];

        let events = parser
            .parse_iter(tokens.into_iter())
            .collect::<Vec<Event>>();

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::Content, "Hello world"),
            (Some("think"), TagParts::End, "<think>Hello world</think>"),
        ];

        assert_eq!(events.len(), expected.len());
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }
    }

    #[test]
    fn test_iter_split_content_in_tag_parsing() {
        let parser = XmlParserBuilder::new().register_tag("think").build();

        let tokens = vec![
            Ok("<".to_string()),
            Ok("think>Hello ".to_string()),
            Ok("world</think>".to_string()),
        ];
        let events = parser
            .parse_iter(tokens.into_iter())
            .collect::<Vec<Event>>();

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::Content, "Hello "),
            (Some("think"), TagParts::Content, "world"),
            (Some("think"), TagParts::End, "<think>Hello world</think>"),
        ];

        assert_eq!(events.len(), expected.len());
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }
    }

    #[test]
    fn test_iter_char_by_char() {
        let parser = XmlParserBuilder::new().register_tag("think").build();

        let input = "<think>Hello world</think>";
        let tokens: Vec<Result<String, _>> = input.chars().map(|c| Ok(c.to_string())).collect();
        let events: Vec<Event> = parser.parse_iter(tokens.into_iter()).collect();

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::Content, "H"),
            (Some("think"), TagParts::Content, "e"),
            (Some("think"), TagParts::Content, "l"),
            (Some("think"), TagParts::Content, "l"),
            (Some("think"), TagParts::Content, "o"),
            (Some("think"), TagParts::Content, " "),
            (Some("think"), TagParts::Content, "w"),
            (Some("think"), TagParts::Content, "o"),
            (Some("think"), TagParts::Content, "r"),
            (Some("think"), TagParts::Content, "l"),
            (Some("think"), TagParts::Content, "d"),
            (Some("think"), TagParts::End, "<think>Hello world</think>"),
        ];

        assert_eq!(events.len(), expected.len());
        for (event, (tag, part, content)) in events.iter().zip(expected) {
            assert_eq!(event.tag(), *tag);
            assert_eq!(event.part(), *part);
            assert_eq!(event.get_content(), *content);
        }
    }
}
