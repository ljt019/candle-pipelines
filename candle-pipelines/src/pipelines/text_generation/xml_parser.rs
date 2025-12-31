use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tag {
    name: String,
}

impl Tag {
    pub fn name(&self) -> &str {
        &self.name
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
        /// The content/text.
        content: String,
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

    fn plain(content: impl Into<String>) -> Self {
        Self::Output {
            content: content.into(),
        }
    }

    pub(crate) fn content(content: impl Into<String>) -> Self {
        Self::plain(content)
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

    /// Get the content/text of this event.
    pub fn get_content(&self) -> &str {
        match self {
            Self::Tagged { content, .. } | Self::Output { content, .. } => content,
        }
    }

    /// Get the tag name if this is a tagged event.
    pub fn tag(&self) -> Option<&str> {
        match self {
            Self::Tagged { tag, .. } => Some(tag.name()),
            Self::Output { .. } => None,
        }
    }

    /// Get which part of the tag this event represents.
    /// Returns `Content` for plain output events.
    pub fn part(&self) -> TagParts {
        match self {
            Self::Tagged { part, .. } => *part,
            Self::Output { .. } => TagParts::Content,
        }
    }

    /// Get attributes if this is a tagged event with attributes.
    /// Returns None for Output events, or tagged events with no attributes.
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
}

impl XmlParserBuilder {
    /// Create a new builder with no registered tags.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tag name for the parser to track. Returns self for chaining.
    /// Registering the same tag twice is idempotent.
    pub fn register_tag(mut self, tag: impl Into<String>) -> Self {
        let name = tag.into();
        if !self.tags.contains(&name) {
            self.tags.push(name);
        }
        self
    }

    /// Build the parser with all registered tags.
    pub fn build(self) -> XmlParser {
        let mut tag_map = HashMap::new();
        let mut tags_set = HashSet::new();

        for name in self.tags {
            tags_set.insert(name.clone());
            tag_map.insert(name.clone(), Tag { name });
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
    pub(crate) fn parse_token(&mut self, token: &str) -> Vec<Event> {
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
            // Use trimmed name for registration lookup (whitespace-insensitive matching)
            let trimmed_name = tag_name.trim().to_string();
            if self.registered_tags.contains(&trimmed_name) {
                // Create tag handle with original name (preserving whitespace)
                let tag_handle = self.tag_map.get(&trimmed_name).map(|_| Tag {
                    name: tag_name.clone(),
                });

                if tag_content.starts_with("</") {
                    // Closing tag - only close if it matches the outermost open tag
                    if let Some((outermost_name, _, _)) = self.state.open_tags.first() {
                        // Compare trimmed names for matching
                        if outermost_name.trim() == trimmed_name {
                            // Matches outermost - close it
                            let (stored_name, content, attrs) = self.state.open_tags.remove(0);
                            let already_emitted = self
                                .state
                                .emitted_tag_lens
                                .remove(&stored_name)
                                .unwrap_or(0);

                            if let Some(tag_handle) = tag_handle {
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
                                // End event contains full XML with original tag name (preserving whitespace)
                                let full_xml =
                                    format!("<{}>{}</{}>", stored_name, content, stored_name);
                                events.push(Event::end(tag_handle, full_xml, attrs));
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
                        if let Some(tag_handle) = tag_handle {
                            // Emit start + end for self-closing (no content event)
                            events.push(Event::start(
                                tag_handle.clone(),
                                format!("<{}>", tag_name),
                                attrs.clone(),
                            ));
                            events.push(Event::end(tag_handle, tag_content, attrs));
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

                        if let Some(tag_handle) = tag_handle {
                            events.push(Event::start(tag_handle, tag_content, attrs));
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
            // Preserve leading whitespace in tag name
            let trimmed_end = name.trim_end();
            if trimmed_end.is_empty() {
                None
            } else {
                Some(trimmed_end.to_string())
            }
        } else {
            // Preserve leading whitespace, trim only trailing (for attrs/self-closing)
            let trimmed_end = inner.trim_end_matches('/').split_whitespace().next()?;
            // But we want to preserve leading space, so find where it starts
            let leading_ws = inner.len() - inner.trim_start().len();
            let name_with_leading = &inner[..leading_ws + trimmed_end.len()];
            if let Some(stripped) = name_with_leading.strip_suffix('/') {
                Some(stripped.to_string())
            } else {
                Some(name_with_leading.to_string())
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
    pub(crate) fn flush(&mut self) -> Vec<Event> {
        self.flush_internal()
    }

    fn flush_internal(&mut self) -> Vec<Event> {
        let mut events = Vec::new();

        // Handle partial tag at EOF (e.g., "hello <think" - the "<think" is stuck in tag_buffer)
        if self.state.in_tag && !self.state.tag_buffer.is_empty() {
            let partial = std::mem::take(&mut self.state.tag_buffer);
            if let Some((_, ref mut content, _)) = self.state.open_tags.last_mut() {
                // Inside a registered tag - append to its content
                content.push_str(&partial);
            } else {
                // Top level - append to content buffer
                self.state.content_buffer.push_str(&partial);
            }
            self.state.in_tag = false;
        }

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
    pending_error: Option<crate::error::PipelineError>,
}

impl<I> EventIterator<I> {
    fn new(mut parser: XmlParser, iter: I) -> Self {
        parser.reset();
        Self {
            parser,
            inner: iter,
            buffer: Vec::new(),
            flushed: false,
            pending_error: None,
        }
    }
}

impl<I> Iterator for EventIterator<I>
where
    I: Iterator<Item = crate::error::Result<String>>,
{
    type Item = crate::error::Result<Event>;

    fn next(&mut self) -> Option<Self::Item> {
        // Return buffered events first
        if !self.buffer.is_empty() {
            return Some(Ok(self.buffer.remove(0)));
        }

        // If we have a pending error (after flushing), return it and stop
        if let Some(e) = self.pending_error.take() {
            return Some(Err(e));
        }

        // If already flushed (error occurred), stop
        if self.flushed {
            return None;
        }

        // Get more tokens and parse
        for result in self.inner.by_ref() {
            match result {
                Ok(token) => {
                    let events = self.parser.parse_token(&token);
                    if !events.is_empty() {
                        self.buffer.extend(events);
                        return Some(Ok(self.buffer.remove(0)));
                    }
                }
                Err(e) => {
                    // Flush partial state before returning error
                    let flush_events = self.parser.flush();
                    self.buffer.extend(flush_events);
                    self.flushed = true;
                    self.pending_error = Some(e);

                    // Return buffered content first, error comes after
                    if !self.buffer.is_empty() {
                        return Some(Ok(self.buffer.remove(0)));
                    }
                    // No buffered content, return error immediately
                    return Some(Err(self.pending_error.take().unwrap()));
                }
            }
        }

        // Flush remaining events (normal completion)
        if !self.flushed {
            self.flushed = true;
            let events = self.parser.flush();
            if !events.is_empty() {
                self.buffer.extend(events);
                return Some(Ok(self.buffer.remove(0)));
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Expected<'a> = (Option<&'a str>, TagParts, &'a str);

    fn assert_events(events: &[Event], expected: &[Expected<'_>]) {
        assert_eq!(events.len(), expected.len(), "events: {events:#?}");
        for (i, (event, (tag, part, content))) in events.iter().zip(expected).enumerate() {
            assert_eq!(event.tag(), *tag, "idx={i} event={event:#?}");
            assert_eq!(event.part(), *part, "idx={i} event={event:#?}");
            assert_eq!(event.get_content(), *content, "idx={i} event={event:#?}");
        }
    }

    fn assert_attr(event: &Event, key: &str, expected: &str) {
        let attrs = event.attributes().expect("expected attributes");
        assert_eq!(attrs.get(key).map(String::as_str), Some(expected));
    }

    #[test]
    fn test_plain_text_only_parsing() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "Regular content";
        let events = parser.parse(&text);

        assert_events(&events, &[(None, TagParts::Content, "Regular content")]);
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

        assert_events(&events, &[(None, TagParts::Content, "  ")]);
    }

    #[test]
    fn test_single_tag_parsing() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "<think>Hello world</think>";
        let events = parser.parse(&text);

        assert_events(
            &events,
            &[
                (Some("think"), TagParts::Start, "<think>"),
                (Some("think"), TagParts::Content, "Hello world"),
                (Some("think"), TagParts::End, text),
            ],
        );
    }

    #[test]
    fn test_plain_text_and_single_tag_parsing() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "<think>Hello world</think>Regular content";
        let events = parser.parse(&text);

        assert_events(
            &events,
            &[
                (Some("think"), TagParts::Start, "<think>"),
                (Some("think"), TagParts::Content, "Hello world"),
                (Some("think"), TagParts::End, "<think>Hello world</think>"),
                (None, TagParts::Content, "Regular content"),
            ],
        );
    }

    #[test]
    fn test_unicode_tag_parsing() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();
        let text = "<think>你好世界</think>普通内容";
        let events = parser.parse(&text);
        assert_events(
            &events,
            &[
                (Some("think"), TagParts::Start, "<think>"),
                (Some("think"), TagParts::Content, "你好世界"),
                (Some("think"), TagParts::End, "<think>你好世界</think>"),
                (None, TagParts::Content, "普通内容"),
            ],
        );
    }

    #[test]
    fn test_plain_text_before_and_after_single_tag_parsing() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "How are <think>Hello world</think>you doing today?";
        let events = parser.parse(&text);

        assert_events(
            &events,
            &[
                (None, TagParts::Content, "How are "),
                (Some("think"), TagParts::Start, "<think>"),
                (Some("think"), TagParts::Content, "Hello world"),
                (Some("think"), TagParts::End, "<think>Hello world</think>"),
                (None, TagParts::Content, "you doing today?"),
            ],
        );

        // Verify plain content reconstruction
        let plain: String = events
            .iter()
            .filter(|e| e.tag().is_none())
            .map(|e| e.get_content())
            .collect();
        assert_eq!(plain, "How are you doing today?");
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

        assert_events(&events, expected);
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

        assert_events(&events, expected);
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

        assert_events(&events, expected);
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

        assert_events(&events, expected);
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

        assert_events(&events, expected);
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

        assert_events(&events, expected);
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

        assert_events(&events, expected);
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

        assert_events(&events, expected);
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

        assert_events(&events, expected);
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

        assert_events(&events, expected);
    }

    #[test]
    fn test_tag_with_leading_whitespace() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "hello < think>content</ think>";
        let events = parser.parse(&text);

        let expected = &[
            (None, TagParts::Content, "hello "),
            (Some(" think"), TagParts::Start, "< think>"),
            (Some(" think"), TagParts::Content, "content"),
            (Some(" think"), TagParts::End, "< think>content</ think>"),
        ];

        assert_events(&events, expected);
    }

    #[test]
    fn test_empty_tag_name() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "hello <> world </>";
        let events = parser.parse(&text);

        assert_events(&events, &[(None, TagParts::Content, "hello <> world </>")]);
    }

    #[test]
    fn test_reset_clears_streaming_state() {
        let mut parser = XmlParserBuilder::new()
            .register_tag("think")
            .register_tag("answer")
            .build();

        parser.parse_token("<thi");
        parser.parse_token("nk>partial content");
        parser.parse_token(" more");
        parser.reset();

        let mut events = parser.parse_token("<answer>42</answer>");
        events.extend(parser.flush());

        let expected = &[
            (Some("answer"), TagParts::Start, "<answer>"),
            (Some("answer"), TagParts::Content, "42"),
            (Some("answer"), TagParts::End, "<answer>42</answer>"),
        ];

        assert_events(&events, expected);
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
    fn test_attribute_parsing_variants() {
        let cases = [
            // (input, attr_key, expected_value)
            ("<fn name=get_weather>x</fn>", "name", "get_weather"), // no quotes
            ("<fn name=\"get_weather\">x</fn>", "name", "get_weather"), // double quotes
            ("<fn name='get_weather'>x</fn>", "name", "get_weather"), // single quotes
            ("<fn name = get_weather>x</fn>", "name", "get_weather"), // spaces around =
            ("<fn name=\"get weather\">x</fn>", "name", "get weather"), // space in value
            ("<fn name=\"unterminated>x</fn>", "name", "unterminated"), // unterminated quote
        ];

        for (input, key, expected) in cases {
            let mut parser = XmlParserBuilder::new().register_tag("fn").build();
            let events = parser.parse(input);
            assert_attr(&events[0], key, expected);
        }
    }

    #[test]
    fn test_attribute_boolean() {
        let mut parser = XmlParserBuilder::new().register_tag("fn").build();
        let events = parser.parse("<fn disabled>x</fn>");
        assert!(events[0].attributes().unwrap().contains_key("disabled"));
    }

    #[test]
    fn test_attributes_self_closing() {
        let mut parser = XmlParserBuilder::new().register_tag("fn").build();
        let events = parser.parse("<fn name=test/>");
        assert_attr(&events[0], "name", "test");
    }

    #[test]
    fn test_multiple_attributes() {
        let mut parser = XmlParserBuilder::new().register_tag("fn").build();
        let events = parser.parse("<fn name='get_weather' id=0>x</fn>");

        assert_attr(&events[0], "name", "get_weather");
        assert_attr(&events[0], "id", "0");
    }

    #[test]
    fn test_attributes_persist_across_tag_sequence() {
        let mut parser = XmlParserBuilder::new()
            .register_tag("function_call")
            .build();

        let text = "<function_call name=test>content</function_call>";
        let events = parser.parse(&text);

        for event in &events {
            if event.tag() == Some("function_call") {
                assert_attr(event, "name", "test");
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

        let events: Vec<Event> = parser
            .parse_iter(tokens.into_iter())
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(events[0].tag(), Some("function_call"));
        assert_attr(&events[0], "name", "get_weather");
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
            .map(|r| r.unwrap())
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

        assert_events(&events, expected);
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
            .map(|r| r.unwrap())
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

        assert_events(&events, expected);
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
            .map(|r| r.unwrap())
            .collect::<Vec<Event>>();

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::Content, "Hello world"),
            (Some("think"), TagParts::End, "<think>Hello world</think>"),
        ];

        assert_events(&events, expected);
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
            .map(|r| r.unwrap())
            .collect::<Vec<Event>>();

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::Content, "Hello world"),
            (Some("think"), TagParts::End, "<think>Hello world</think>"),
        ];

        assert_events(&events, expected);
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
            .map(|r| r.unwrap())
            .collect::<Vec<Event>>();

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::Content, "Hello "),
            (Some("think"), TagParts::Content, "world"),
            (Some("think"), TagParts::End, "<think>Hello world</think>"),
        ];

        assert_events(&events, expected);
    }

    #[test]
    fn test_iter_char_by_char() {
        let parser = XmlParserBuilder::new().register_tag("think").build();

        let input = "<think>Hello world</think>";
        let tokens: Vec<Result<String, _>> = input.chars().map(|c| Ok(c.to_string())).collect();
        let events: Vec<Event> = parser
            .parse_iter(tokens.into_iter())
            .map(|r| r.unwrap())
            .collect();

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

        assert_events(&events, expected);
    }

    #[test]
    fn test_partial_tag_at_end_emitted() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "hello <think";
        let events = parser.parse(&text);

        assert_events(&events, &[(None, TagParts::Content, "hello <think")]);
    }

    #[test]
    fn test_partial_tag_inside_registered_tag() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();
        let events = parser.parse("<think>hi <ans");

        let expected = &[
            (Some("think"), TagParts::Start, "<think>"),
            (Some("think"), TagParts::Content, "hi <ans"),
        ];

        assert_events(&events, expected);
    }

    #[test]
    fn test_parse_token_no_double_emit() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let events1 = parser.parse_token("hello ");
        let events2 = parser.parse_token("world");
        let events3 = parser.flush();

        let all_content: String = events1
            .iter()
            .chain(events2.iter())
            .chain(events3.iter())
            .map(|e| e.get_content())
            .collect();

        assert_eq!(all_content, "hello world");
    }

    #[test]
    fn test_parse_token_flush_trailing_plain() {
        let mut parser = XmlParserBuilder::new().register_tag("think").build();

        let _ = parser.parse_token("<think>content</think>");
        let events1 = parser.parse_token("trailing");
        let events2 = parser.flush();

        let trailing_count = events1
            .iter()
            .chain(events2.iter())
            .filter(|e| e.tag().is_none() && e.get_content().contains("trailing"))
            .count();

        assert_eq!(trailing_count, 1);
    }

    #[test]
    fn test_parse_tool_call_valid() {
        let mut parser = XmlParserBuilder::new().register_tag("tool_call").build();

        let text =
            r#"<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>"#;
        let events = parser.parse(&text);

        let end_event = events.iter().find(|e| e.part() == TagParts::End).unwrap();
        let result = end_event.parse_tool_call();

        assert!(result.is_some());
        let (name, args) = result.unwrap();
        assert_eq!(name, "get_weather");
        assert_eq!(args["city"], "Tokyo");
    }

    #[test]
    fn test_parse_tool_call_invalid() {
        let mut parser = XmlParserBuilder::new().register_tag("tool_call").build();

        let text = "<tool_call>not valid json</tool_call>";
        let events = parser.parse(&text);

        let end_event = events.iter().find(|e| e.part() == TagParts::End).unwrap();
        let result = end_event.parse_tool_call();

        assert!(result.is_none());
    }
}
