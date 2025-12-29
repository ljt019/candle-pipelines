use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

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
    fn tagged(tag: Tag, part: TagParts, content: impl Into<String>) -> Self {
        Self::Tagged {
            tag,
            part,
            content: content.into(),
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

    pub(crate) fn plain_start() -> Self {
        Self::plain(TagParts::Start, "")
    }

    pub(crate) fn plain_end() -> Self {
        Self::plain(TagParts::End, "")
    }

    pub(crate) fn start(tag: Tag) -> Self {
        Self::tagged(tag, TagParts::Start, "")
    }

    pub(crate) fn end(tag: Tag, full_xml: impl Into<String>) -> Self {
        Self::tagged(tag, TagParts::End, full_xml)
    }

    fn tagged_internal(tag: Tag, content: impl Into<String>) -> Self {
        Self::tagged(tag, TagParts::Content, content)
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
/// let parser = XmlParserBuilder::new()
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

#[derive(Debug, Clone, Default)]
struct ParserState {
    open_tags: Vec<(String, String)>,
    content_buffer: String,
    tag_buffer: String,
    in_tag: bool,
    emitted_top_len: usize,
    emitted_tag_lens: std::collections::HashMap<String, usize>,
    top_level_open: bool,
    last_content_had_newline: bool,
}

/// Streaming XML parser for extracting structured content from LLM output.
///
/// Parses text containing XML-like tags and emits events for tag boundaries
/// and content. Useful for structured output like `<think>...</think>` blocks.
#[derive(Debug, Clone)]
pub struct XmlParser {
    registered_tags: HashSet<String>,
    tag_map: HashMap<String, Tag>,
    state: Arc<Mutex<ParserState>>,
}

impl XmlParser {
    /// Create a new parser for the specified tags.
    pub fn new(tags: HashSet<String>, tag_map: HashMap<String, Tag>) -> Self {
        Self {
            registered_tags: tags,
            tag_map,
            state: Arc::new(Mutex::new(ParserState::default())),
        }
    }

    /// Reset parser state for a new parsing session.
    pub fn reset(&self) {
        *self.state.lock().expect("parser lock poisoned") = ParserState::default();
    }

    /// Parse a complete text string and return all events.
    pub fn parse_complete(&self, text: &str) -> Vec<Event> {
        self.reset();
        let mut events = Vec::new();

        for char in text.chars() {
            let mut evs = self.process_char(char);
            events.append(&mut evs);
        }

        events.extend(self.flush());
        events
    }

    /// Parse a single token in streaming mode. Call `flush()` when done.
    pub fn parse_token(&self, token: &str) -> Vec<Event> {
        let mut events = Vec::new();

        for char in token.chars() {
            let mut evs = self.process_char(char);
            events.append(&mut evs);
        }

        {
            let mut state = self.state.lock().expect("parser lock poisoned");

            if state.open_tags.is_empty() {
                let current_len = state.content_buffer.len();
                if current_len > state.emitted_top_len {
                    let mut new_slice = &state.content_buffer[state.emitted_top_len..];

                    if state.emitted_top_len == 0 {
                        new_slice = new_slice.trim_start_matches('\n');
                    }

                    let content_to_emit = new_slice.to_string();

                    if !content_to_emit.is_empty() {
                        if !state.top_level_open {
                            events.push(Event::plain_start());
                            state.top_level_open = true;
                        }
                        events.push(Event::content(content_to_emit.clone()));
                        state.last_content_had_newline = content_to_emit.ends_with('\n');
                    }
                    state.emitted_top_len = current_len;
                }
            } else if let Some((tag_name_ref, content_ref)) = state.open_tags.last() {
                let tag_name = tag_name_ref.clone();
                let content = content_ref.clone();
                let total_len = content.len();

                let already_emitted = *state.emitted_tag_lens.get(&tag_name).unwrap_or(&0);

                if total_len > already_emitted {
                    let new_slice = &content[already_emitted..];

                    if already_emitted == 0 {
                        let trimmed = new_slice.trim_start_matches('\n');
                        if !trimmed.is_empty() {
                            if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                                events.push(Event::tagged_internal(tag_handle.clone(), trimmed));
                            }
                        }
                        state.emitted_tag_lens.insert(tag_name.clone(), total_len);
                    } else if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                        events.push(Event::tagged_internal(tag_handle.clone(), new_slice));
                        state.emitted_tag_lens.insert(tag_name.clone(), total_len);
                    } else {
                        state.emitted_tag_lens.insert(tag_name.clone(), total_len);
                    }
                }
            }
        }

        events
    }

    fn process_char(&self, c: char) -> Vec<Event> {
        let mut events = Vec::new();
        let mut state = self.state.lock().expect("parser lock poisoned");

        match c {
            '<' => {
                state.in_tag = true;
                state.tag_buffer.clear();
                state.tag_buffer.push(c);
            }
            '>' if state.in_tag => {
                state.tag_buffer.push(c);
                state.in_tag = false;

                let tag_content = state.tag_buffer.clone();
                state.tag_buffer.clear();

                events.extend(self.handle_tag(&mut state, &tag_content));
            }
            _ if state.in_tag => {
                state.tag_buffer.push(c);
            }
            _ => {
                if let Some((_, ref mut content)) = state.open_tags.last_mut() {
                    content.push(c);
                } else {
                    state.content_buffer.push(c);
                }
            }
        }

        events
    }

    fn handle_tag(&self, state: &mut ParserState, tag_content: &str) -> Vec<Event> {
        let mut events = Vec::new();

        if let Some(tag_name) = self.parse_tag_name(tag_content) {
            if self.registered_tags.contains(&tag_name) {
                if tag_content.starts_with("</") {
                    if let Some(pos) = state
                        .open_tags
                        .iter()
                        .rposition(|(name, _)| name == &tag_name)
                    {
                        let (_, content) = state.open_tags.remove(pos);

                        let already_emitted = state.emitted_tag_lens.remove(&tag_name).unwrap_or(0);

                        if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                            if content.len() > already_emitted {
                                let remaining_content = &content[already_emitted..];
                                let content_to_emit = if already_emitted == 0 {
                                    remaining_content.trim_start_matches('\n')
                                } else {
                                    remaining_content
                                };

                                let trimmed = content_to_emit.trim_end_matches('\n');
                                if !trimmed.is_empty() {
                                    let mut final_str = trimmed.to_string();
                                    final_str.push('\n');
                                    events.push(Event::tagged_internal(
                                        tag_handle.clone(),
                                        final_str,
                                    ));
                                }
                            }
                            let full_xml = format!("<{}>{}</{}>", tag_name, content, tag_name);
                            events.push(Event::end(tag_handle.clone(), full_xml));
                        }
                    }
                } else if !tag_content.ends_with("/>") {
                    if state.open_tags.is_empty() && !state.content_buffer.is_empty() {
                        let content = &state.content_buffer[state.emitted_top_len..];

                        let mut slice = content;
                        if state.emitted_top_len == 0 {
                            slice = slice.trim_start_matches('\n');
                        }
                        let content_to_emit = if slice.is_empty() {
                            String::new()
                        } else {
                            let mut content_str = slice.to_string();
                            if !content_str.ends_with('\n') {
                                content_str.push('\n');
                            }
                            content_str
                        };

                        state.emitted_top_len = state.content_buffer.len();
                        if !content_to_emit.is_empty() {
                            if !state.top_level_open {
                                events.push(Event::plain_start());
                                state.top_level_open = true;
                            }
                            events.push(Event::content(content_to_emit.clone()));
                            state.last_content_had_newline = content_to_emit.ends_with('\n');
                        }
                    }

                    state.open_tags.push((tag_name.clone(), String::new()));

                    if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                        events.push(Event::start(tag_handle.clone()));
                    }
                }
            } else if state.open_tags.is_empty() {
                state.content_buffer.push_str(tag_content);
            } else if let Some((_, ref mut content)) = state.open_tags.last_mut() {
                content.push_str(tag_content);
            }
        } else if state.open_tags.is_empty() {
            state.content_buffer.push_str(tag_content);
        } else if let Some((_, ref mut content)) = state.open_tags.last_mut() {
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

    /// Flush any remaining buffered content as events.
    pub fn flush(&self) -> Vec<Event> {
        let mut state = self.state.lock().expect("parser lock poisoned");
        let mut events = Vec::new();

        if state.content_buffer.len() > state.emitted_top_len {
            let remaining = &state.content_buffer[state.emitted_top_len..];

            let slice = remaining.trim_start_matches('\n').trim_end_matches('\n');

            let content_to_emit = if slice.is_empty() {
                String::new()
            } else {
                let mut content_str = slice.to_string();
                if !content_str.ends_with('\n') {
                    content_str.push('\n');
                }
                content_str
            };

            state.emitted_top_len = state.content_buffer.len();
            if !content_to_emit.is_empty() {
                if !state.top_level_open {
                    events.push(Event::plain_start());
                    state.top_level_open = true;
                }
                events.push(Event::content(content_to_emit.clone()));
                state.last_content_had_newline = content_to_emit.ends_with('\n');
            }
        }
        if state.top_level_open {
            if !state.last_content_had_newline {
                events.push(Event::content("\n"));
            }
            events.push(Event::plain_end());
        }
        state.top_level_open = false;
        state.content_buffer.clear();
        state.emitted_top_len = 0;

        let drained: Vec<_> = state.open_tags.drain(..).collect();
        for (tag_name, content) in drained {
            let already_emitted = state.emitted_tag_lens.remove(&tag_name).unwrap_or(0);

            if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                if content.len() > already_emitted {
                    let remaining_content = &content[already_emitted..];
                    let content_to_emit = if already_emitted == 0 {
                        remaining_content.trim_start_matches('\n')
                    } else {
                        remaining_content
                    };

                    let trimmed = content_to_emit.trim_end_matches('\n');
                    if !trimmed.is_empty() {
                        let mut final_str = trimmed.to_string();
                        final_str.push('\n');
                        events.push(Event::tagged_internal(tag_handle.clone(), final_str));
                    }
                }
                let full_xml = format!("<{}>{}</{}>", tag_name, content, tag_name);
                events.push(Event::end(tag_handle.clone(), full_xml));
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
    /// let parser = XmlParserBuilder::new().register_tag("think").build();
    /// let token_iter = pipeline.completion_stream("...")?;
    /// let mut event_iter = parser.wrap_iterator(token_iter);
    ///
    /// for event in event_iter {
    ///     match (event.tag(), event.part()) {
    ///         (Some("think"), TagParts::Content) => println!("[thinking] {}", event.get_content()),
    ///         (None, TagParts::Content) => print!("{}", event.get_content()),
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub fn wrap_iterator<I>(&self, iter: I) -> EventIterator<I>
    where
        I: Iterator<Item = crate::error::Result<String>>,
    {
        EventIterator::new(self.clone(), iter)
    }

    /// Wrap a token stream to produce XML parsing events (async version).
    ///
    /// Use this for async streams like those from the reader pipeline.
    pub fn wrap_stream<S>(&self, stream: S) -> EventStream<S>
    where
        S: futures::Stream<Item = crate::error::Result<String>> + Send,
    {
        EventStream::new(self.clone(), stream)
    }
}

/// Stream of XML parsing events.
///
/// Wraps a token stream and parses XML tags as they arrive.
/// Has inherent methods like `.next()` - no need to import `StreamExt`.
pub struct EventStream<S> {
    parser: XmlParser,
    inner: std::pin::Pin<Box<S>>,
    buffer: Vec<Event>,
    flushed: bool,
}

impl<S> EventStream<S> {
    fn new(parser: XmlParser, stream: S) -> Self {
        parser.reset();
        Self {
            parser,
            inner: Box::pin(stream),
            buffer: Vec::new(),
            flushed: false,
        }
    }

    /// Get the next event from the stream.
    pub async fn next(&mut self) -> Option<Event>
    where
        S: futures::Stream<Item = crate::error::Result<String>>,
    {
        use futures::StreamExt;

        // Return buffered events first
        if !self.buffer.is_empty() {
            return Some(self.buffer.remove(0));
        }

        // Get more tokens and parse
        while let Some(result) = self.inner.as_mut().next().await {
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

    /// Collect all events into a vector.
    pub async fn collect(mut self) -> Vec<Event>
    where
        S: futures::Stream<Item = crate::error::Result<String>>,
    {
        let mut events = Vec::new();
        while let Some(event) = self.next().await {
            events.push(event);
        }
        events
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
    fn new(parser: XmlParser, iter: I) -> Self {
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
    fn test_basic_parsing() {
        let parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "<think>Hello world</think>Regular content";
        let events = parser.parse_complete(text);

        assert_eq!(events.len(), 6);
        assert_eq!(events[0].part(), TagParts::Start);
        assert_eq!(events[0].tag(), Some("think"));
        assert_eq!(events[1].part(), TagParts::Content);
        assert_eq!(events[1].get_content(), "Hello world\n");
        assert_eq!(events[2].part(), TagParts::End);
        assert_eq!(events[2].tag(), Some("think"));
        assert_eq!(events[3].part(), TagParts::Start);
        assert_eq!(events[3].tag(), None);
        assert_eq!(events[4].part(), TagParts::Content);
        assert_eq!(events[4].get_content(), "Regular content\n");
        assert_eq!(events[5].part(), TagParts::End);
        assert_eq!(events[5].tag(), None);
    }

    #[test]
    fn test_unregistered_tags() {
        let parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "<think>Registered</think><other>Not registered</other>";
        let events = parser.parse_complete(text);

        assert_eq!(events.len(), 6);
        assert_eq!(events[0].part(), TagParts::Start);
        assert_eq!(events[0].tag(), Some("think"));
        assert_eq!(events[1].part(), TagParts::Content);
        assert_eq!(events[1].get_content(), "Registered\n");
        assert_eq!(events[2].part(), TagParts::End);
        assert_eq!(events[2].tag(), Some("think"));
        assert_eq!(events[3].part(), TagParts::Start);
        assert_eq!(events[3].tag(), None);
        assert_eq!(events[4].part(), TagParts::Content);
        assert_eq!(events[4].get_content(), "<other>Not registered</other>\n");
        assert_eq!(events[5].part(), TagParts::End);
        assert_eq!(events[5].tag(), None);
    }

    #[test]
    fn test_malformed_xml() {
        let parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "<think>Unclosed tag content";
        let events = parser.parse_complete(text);

        assert_eq!(events.len(), 3);
        assert_eq!(events[0].part(), TagParts::Start);
        assert_eq!(events[0].tag(), Some("think"));
        assert_eq!(events[1].part(), TagParts::Content);
        assert_eq!(events[1].get_content(), "Unclosed tag content\n");
        assert_eq!(events[2].part(), TagParts::End);
        assert_eq!(events[2].tag(), Some("think"));
    }

    #[test]
    fn test_self_closing_tag_ignored() {
        let parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "<think/>Hello";
        let events = parser.parse_complete(text);

        assert_eq!(events.len(), 3);
        assert_eq!(events[0].part(), TagParts::Start);
        assert_eq!(events[0].tag(), None);
        assert_eq!(events[1].part(), TagParts::Content);
        assert_eq!(events[1].get_content(), "Hello\n");
        assert_eq!(events[2].part(), TagParts::End);
        assert_eq!(events[2].tag(), None);
    }

    #[test]
    fn test_tag_with_attributes() {
        let parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "<think answer=\"yes\">Content</think>";
        let events = parser.parse_complete(text);

        assert_eq!(events.len(), 3);
        assert_eq!(events[0].part(), TagParts::Start);
        assert_eq!(events[0].tag(), Some("think"));
        assert_eq!(events[1].part(), TagParts::Content);
        assert_eq!(events[1].get_content(), "Content\n");
        assert_eq!(events[2].part(), TagParts::End);
        assert_eq!(events[2].tag(), Some("think"));
    }

    #[test]
    fn test_nested_registered_tags() {
        let parser = XmlParserBuilder::new()
            .register_tag("think")
            .register_tag("inner")
            .build();

        let text = "<think>hi<inner>there</inner>end</think>";
        let events = parser.parse_complete(text);

        assert_eq!(events.len(), 6);
        assert_eq!(events[0].part(), TagParts::Start);
        assert_eq!(events[0].tag(), Some("think"));
        assert_eq!(events[1].part(), TagParts::Start);
        assert_eq!(events[1].tag(), Some("inner"));
        assert_eq!(events[2].part(), TagParts::Content);
        assert_eq!(events[2].tag(), Some("inner"));
        assert_eq!(events[2].get_content(), "there\n");
        assert_eq!(events[3].part(), TagParts::End);
        assert_eq!(events[3].tag(), Some("inner"));
        assert_eq!(events[4].part(), TagParts::Content);
        assert_eq!(events[4].tag(), Some("think"));
        assert_eq!(events[4].get_content(), "hiend\n");
        assert_eq!(events[5].part(), TagParts::End);
        assert_eq!(events[5].tag(), Some("think"));
    }

    #[test]
    fn test_parse_token_equivalent_to_complete() {
        let parser = XmlParserBuilder::new().register_tag("think").build();

        let text = "<think>Hello</think>";
        let expected = parser.parse_complete(text);

        parser.reset();
        let mut actual = parser.parse_token(text);
        actual.extend(parser.flush());

        assert_eq!(expected, actual);
    }
}
