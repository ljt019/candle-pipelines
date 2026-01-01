use std::collections::{HashMap, VecDeque};

/// A tag name recognized by the parser.
///
/// Tag names may include leading whitespace from the source text (for example,
/// `< think>` yields a tag name of `" think"`). The parser matches tags using
/// `trim()` internally, but the stored name preserves the original formatting.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tag {
    name: String,
}

impl Tag {
    /// Returns the tag name as stored by the parser.
    ///
    /// The returned name may include leading whitespace from the source text.
    /// Do not assume the name is normalized.
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

/// Identifies which portion of a tag an [`Event`] represents.
///
/// For [`Event::Output`], [`Event::part`] returns [`TagParts::Content`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagParts {
    /// An opening tag (for example, `<tool_call>`).
    Start,
    /// Text content inside a tracked tag, or plain output outside any tracked tag.
    Content,
    /// A closing tag (for example, `</tool_call>`).
    End,
}

/// An event emitted by [`XmlParser`].
///
/// This parser recognizes a small XML-like syntax for a configured set of tag names.
/// It emits:
/// - [`Event::Tagged`] for tracked tags, with [`TagParts::Start`], [`TagParts::Content`],
///   and [`TagParts::End`].
/// - [`Event::Output`] for text outside any tracked tag.
///
/// For [`Event::Tagged`]:
/// - `Start` events store the opening tag text in `content`.
/// - `Content` events store text inside the tag in `content`.
/// - `End` events store a reconstructed full element string in `content`
///   (see [`XmlParser`] docs for details and limitations).
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Data associated with a tracked tag.
    Tagged {
        /// The parsed tag name as produced by the parser.
        ///
        /// Note: the parser may preserve leading whitespace from the source text
        /// (for example, `< think>` yields a tag name of `" think"`).
        tag: Tag,
        /// Which portion of the tag this event represents.
        part: TagParts,
        /// The payload for this event.
        ///
        /// - For `Start`, this is the opening tag text.
        /// - For `Content`, this is tag inner text (possibly chunked in streaming mode).
        /// - For `End`, this is the reconstructed full element string.
        content: String,
        /// Parsed attributes from the opening tag.
        ///
        /// The parser attaches the same attribute map to all events for the element.
        attributes: HashMap<String, String>,
    },
    /// Text outside any tracked tag.
    Output {
        /// The emitted text (possibly chunked in streaming mode).
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

    /// Returns the payload string for this event.
    ///
    /// For [`Event::Tagged`], the returned string depends on [`Event::part`]:
    /// - `Start`: opening tag text
    /// - `Content`: inner text
    /// - `End`: reconstructed full element text
    pub fn get_content(&self) -> &str {
        match self {
            Self::Tagged { content, .. } | Self::Output { content, .. } => content,
        }
    }

    /// Returns the tag name for [`Event::Tagged`], or `None` for [`Event::Output`].
    ///
    /// The returned name is the parser's stored name for this element and may include
    /// leading whitespace from the source text (for example, `< think>` yields `" think"`).
    /// Do not assume the name is normalized.
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

    /// Returns parsed attributes for this event.
    ///
    /// Returns `Some(...)` only when this is [`Event::Tagged`] and the attribute map is non-empty.
    /// Returns `None` for [`Event::Output`] and for tagged events whose opening tag had no attributes.
    ///
    /// Boolean attributes (for example, `<fn disabled>`) appear with an empty string value.
    pub fn attributes(&self) -> Option<&HashMap<String, String>> {
        match self {
            Self::Tagged { attributes, .. } if !attributes.is_empty() => Some(attributes),
            _ => None,
        }
    }

    /// Parses this event as a `<tool_call>` payload.
    ///
    /// This returns `Some((name, arguments))` only when:
    /// - `self` is a tagged event whose `tag()` is exactly `"tool_call"`,
    /// - `part()` is [`TagParts::End`], and
    /// - `get_content()` starts with `<tool_call>` and ends with `</tool_call>`.
    ///
    /// The parser trims whitespace inside the element, parses the inner text as JSON,
    /// and expects an object with:
    /// - `"name"`: a string
    /// - `"arguments"`: any JSON value
    ///
    /// Returns `None` on any mismatch or JSON parsing/shape error.
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

/// Builder for creating an [`XmlParser`] with a set of tag names to track.
///
/// The parser matches input tag names using `trim()` on the parsed name.
/// Register normalized tag names (for example, `"think"`, not `" think"`).
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

    /// Registers a tag name for the parser to track and returns `self` for chaining.
    ///
    /// Registering the same name multiple times is idempotent.
    ///
    /// This method does not normalize the provided name. Register normalized names
    /// (for example, `"think"`).
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

        for name in self.tags {
            tag_map.insert(name.clone(), Tag { name });
        }

        XmlParser::new(tag_map)
    }
}

#[derive(Debug, Clone)]
struct OpenTag {
    name: String,
    content: String,
    attrs: HashMap<String, String>,
    emitted_len: usize,
}

#[derive(Debug, Clone, Default)]
struct ParserState {
    open_tag: Option<OpenTag>,
    content_buffer: String,
    tag_buffer: String,
    in_tag: bool,
    emitted_top_len: usize,
}

/// Streaming parser for extracting XML-like tagged regions from text.
///
/// The parser recognizes a small tag syntax and emits [`Event`] values for:
/// - text outside tracked tags ([`Event::Output`])
/// - the opening tag, inner text, and closing tag of tracked tags ([`Event::Tagged`]).
///
/// # Tag tracking and limitations
///
/// - The parser tracks at most one open tracked tag at a time. It does not support
///   nested tracked tags.
/// - If a tracked tag appears inside the content of another tracked tag, the parser
///   treats it as literal text until the outer tag closes.
/// - The parser does not validate XML and does not handle namespaces, entities,
///   comments, CDATA, or processing instructions. Malformed or mismatched tags are
///   emitted as plain text.
///
/// # Event payloads
///
/// For a tracked element `<name ...attrs...>inner</name>`:
/// - A `Start` event is emitted with `get_content()` set to the opening tag text
///   as it appeared in the input.
/// - Zero or more `Content` events are emitted with slices of `inner`
///   (streaming mode may chunk this arbitrarily).
/// - An `End` event is emitted with `get_content()` set to a reconstructed string
///   of the form `<name>{inner}</name>`. This reconstruction does not preserve
///   original attribute text or whitespace in the opening/closing tags.
///
/// For a self-closing tracked tag (for example, `<name a=b/>`), the parser emits:
/// - `Start` with `get_content()` set to `"<name>"` (normalized; attributes not included),
/// - `End` with `get_content()` set to the original self-closing tag text.
///
/// # Threading
///
/// This type is `Sync`, but parsing mutates internal state and requires `&mut self`.
/// If you need shared mutable access across threads, wrap it in a `Mutex` or `RwLock`.
#[derive(Debug, Clone)]
pub struct XmlParser {
    tag_map: HashMap<String, Tag>,
    state: ParserState,
}

impl XmlParser {
    /// Creates a parser that tracks the provided tag names.
    ///
    /// The parser matches an input tag by comparing `trim()` of the parsed tag name
    /// against the keys of `tag_map`.
    pub fn new(tag_map: HashMap<String, Tag>) -> Self {
        Self {
            tag_map,
            state: ParserState::default(),
        }
    }

    /// Reset parser state for a new parsing session.
    pub fn reset(&mut self) {
        self.state = ParserState::default();
    }

    /// Parses `text` as a complete input and returns all produced events.
    ///
    /// This method calls [`XmlParser::reset`] before parsing and flushes any buffered
    /// data at the end. In non-streaming mode, inner content for a tracked tag is
    /// typically emitted as a single `Content` event when the closing tag arrives.
    pub fn parse(&mut self, text: &str) -> Vec<Event> {
        self.reset();
        let mut events = Vec::new();

        for c in text.chars() {
            self.process_char_internal(c, &mut events);
        }

        self.flush_internal(&mut events);
        events
    }

    /// Parses a chunk of input in streaming mode and returns newly available events.
    ///
    /// This may emit `Content` events incrementally as additional characters arrive.
    /// Call [`XmlParser::flush`] after the final chunk to emit any remaining buffered text.
    pub(crate) fn parse_token(&mut self, token: &str) -> Vec<Event> {
        let mut events = Vec::new();

        for c in token.chars() {
            self.process_char_internal(c, &mut events);
        }

        if let Some(ref mut open) = self.state.open_tag {
            let total_len = open.content.len();
            if total_len > open.emitted_len {
                let new_slice = &open.content[open.emitted_len..];
                if !new_slice.is_empty() {
                    if let Some(tag_handle) = self.tag_map.get(&open.name) {
                        events.push(Event::tagged_internal(
                            tag_handle.clone(),
                            new_slice,
                            open.attrs.clone(),
                        ));
                    }
                }
                open.emitted_len = total_len;
            }
        } else {
            let current_len = self.state.content_buffer.len();
            if current_len > self.state.emitted_top_len {
                let new_slice = &self.state.content_buffer[self.state.emitted_top_len..];
                if !new_slice.is_empty() {
                    events.push(Event::content(new_slice));
                }
                self.state.content_buffer.clear();
                self.state.emitted_top_len = 0;
            }
        }

        events
    }

    fn process_char_internal(&mut self, c: char, out: &mut Vec<Event>) {
        match c {
            '<' => {
                self.state.in_tag = true;
                self.state.tag_buffer.clear();
                self.state.tag_buffer.push(c);
            }
            '>' if self.state.in_tag => {
                self.state.tag_buffer.push(c);
                self.state.in_tag = false;

                let tag_content = std::mem::take(&mut self.state.tag_buffer);
                self.handle_tag(&tag_content, out);
            }
            _ if self.state.in_tag => {
                self.state.tag_buffer.push(c);
            }
            _ => {
                if let Some(ref mut open) = self.state.open_tag {
                    open.content.push(c);
                } else {
                    self.state.content_buffer.push(c);
                }
            }
        }
    }

    fn handle_tag(&mut self, tag_content: &str, out: &mut Vec<Event>) {
        if let Some(tag_name) = self.parse_tag_name(tag_content) {
            let trimmed_name = tag_name.trim().to_string();
            if self.tag_map.contains_key(&trimmed_name) {
                let tag_handle = Tag {
                    name: tag_name.clone(),
                };

                if tag_content.starts_with("</") {
                    if let Some(ref open) = self.state.open_tag {
                        if open.name.trim() == trimmed_name {
                            let open = self.state.open_tag.take().unwrap();

                            if open.content.len() > open.emitted_len {
                                let remaining = &open.content[open.emitted_len..];
                                if !remaining.is_empty() {
                                    out.push(Event::tagged_internal(
                                        tag_handle.clone(),
                                        remaining,
                                        open.attrs.clone(),
                                    ));
                                }
                            }

                            let full_xml =
                                format!("<{}>{}</{}>", open.name, open.content, open.name);
                            out.push(Event::end(tag_handle, full_xml, open.attrs));
                        } else {
                            if let Some(ref mut open) = self.state.open_tag {
                                open.content.push_str(tag_content);
                            }
                        }
                    } else {
                        self.state.content_buffer.push_str(tag_content);
                    }
                } else if tag_content.ends_with("/>") {
                    if self.state.open_tag.is_none() {
                        if !self.state.content_buffer.is_empty() {
                            let content = &self.state.content_buffer[self.state.emitted_top_len..];
                            if !content.is_empty() {
                                out.push(Event::content(content));
                            }
                            self.state.emitted_top_len = self.state.content_buffer.len();
                        }

                        let attrs = self.parse_attributes(tag_content);

                        out.push(Event::start(
                            tag_handle.clone(),
                            format!("<{}>", tag_name),
                            attrs.clone(),
                        ));
                        out.push(Event::end(tag_handle, tag_content, attrs));
                    } else {
                        if let Some(ref mut open) = self.state.open_tag {
                            open.content.push_str(tag_content);
                        }
                    }
                } else {
                    if self.state.open_tag.is_none() {
                        if !self.state.content_buffer.is_empty() {
                            let content = &self.state.content_buffer[self.state.emitted_top_len..];
                            if !content.is_empty() {
                                out.push(Event::content(content));
                            }
                            self.state.emitted_top_len = self.state.content_buffer.len();
                        }

                        let attrs = self.parse_attributes(tag_content);
                        self.state.open_tag = Some(OpenTag {
                            name: tag_name.clone(),
                            content: String::new(),
                            attrs: attrs.clone(),
                            emitted_len: 0,
                        });

                        out.push(Event::start(tag_handle, tag_content, attrs));
                    } else {
                        if let Some(ref mut open) = self.state.open_tag {
                            open.content.push_str(tag_content);
                        }
                    }
                }
            } else if let Some(ref mut open) = self.state.open_tag {
                open.content.push_str(tag_content);
            } else {
                self.state.content_buffer.push_str(tag_content);
            }
        } else if let Some(ref mut open) = self.state.open_tag {
            open.content.push_str(tag_content);
        } else {
            self.state.content_buffer.push_str(tag_content);
        }
    }

    fn parse_tag_name(&self, tag_content: &str) -> Option<String> {
        if tag_content.len() < 3 || !tag_content.starts_with('<') || !tag_content.ends_with('>') {
            return None;
        }

        let inner = &tag_content[1..tag_content.len() - 1];

        if let Some(name) = inner.strip_prefix('/') {
            let trimmed_end = name.trim_end();
            if trimmed_end.is_empty() {
                None
            } else {
                Some(trimmed_end.to_string())
            }
        } else {
            let trimmed_end = inner.trim_end_matches('/').split_whitespace().next()?;
            let leading_ws = inner.len() - inner.trim_start().len();
            let name_with_leading = &inner[..leading_ws + trimmed_end.len()];
            if let Some(stripped) = name_with_leading.strip_suffix('/') {
                Some(stripped.to_string())
            } else {
                Some(name_with_leading.to_string())
            }
        }
    }

    /// Parses attributes from an opening or self-closing tag.
    ///
    /// This is a permissive, non-XML-compliant parser intended for LLM-produced text.
    /// It supports:
    /// - unquoted values: `<tag name=value>`
    /// - single/double quoted values: `<tag name="value">`, `<tag name='value'>`
    /// - boolean attributes: `<tag disabled>` (stored with an empty string value)
    ///
    /// It does not handle escapes or entities. Unterminated quoted values are accepted
    /// and read until the end of the tag.
    fn parse_attributes(&self, tag_content: &str) -> HashMap<String, String> {
        let mut attrs = HashMap::new();

        if tag_content.len() < 3 {
            return attrs;
        }
        let inner = &tag_content[1..tag_content.len() - 1];

        let after_name = match inner.split_whitespace().next() {
            Some(name) => {
                let name_end = inner.find(name).unwrap_or(0) + name.len();
                &inner[name_end..]
            }
            None => return attrs,
        };

        let mut chars = after_name.chars().peekable();

        while let Some(c) = chars.next() {
            if c.is_whitespace() {
                continue;
            }

            let mut attr_name = String::new();
            attr_name.push(c);
            while let Some(&next) = chars.peek() {
                if next == '=' || next.is_whitespace() {
                    break;
                }
                attr_name.push(chars.next().unwrap());
            }

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

            let mut attr_value = String::new();

            while let Some(&next) = chars.peek() {
                if !next.is_whitespace() {
                    break;
                }
                chars.next();
            }

            if let Some(&quote) = chars.peek() {
                if quote == '"' || quote == '\'' {
                    chars.next();
                    while let Some(c) = chars.next() {
                        if c == quote {
                            break;
                        }
                        attr_value.push(c);
                    }
                } else {
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

    /// Flushes any buffered input and returns the resulting events.
    ///
    /// This emits:
    /// - any buffered plain text as [`Event::Output`], and
    /// - any buffered inner text for an unclosed tracked tag as `Tagged` `Content`.
    ///
    /// If the parser has seen a `<` without a matching `>`, `flush` treats the buffered
    /// partial tag text as literal text.
    pub(crate) fn flush(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        self.flush_internal(&mut events);
        events
    }

    fn flush_internal(&mut self, out: &mut Vec<Event>) {
        if self.state.in_tag && !self.state.tag_buffer.is_empty() {
            let partial = std::mem::take(&mut self.state.tag_buffer);
            if let Some(ref mut open) = self.state.open_tag {
                open.content.push_str(&partial);
            } else {
                self.state.content_buffer.push_str(&partial);
            }
            self.state.in_tag = false;
        }

        if self.state.content_buffer.len() > self.state.emitted_top_len {
            let remaining = &self.state.content_buffer[self.state.emitted_top_len..];
            if !remaining.is_empty() {
                out.push(Event::content(remaining));
            }
        }
        self.state.content_buffer.clear();
        self.state.emitted_top_len = 0;

        if let Some(open) = self.state.open_tag.take() {
            if let Some(tag_handle) = self.tag_map.get(&open.name) {
                if open.content.len() > open.emitted_len {
                    let remaining = &open.content[open.emitted_len..];
                    if !remaining.is_empty() {
                        out.push(Event::tagged_internal(
                            tag_handle.clone(),
                            remaining,
                            open.attrs.clone(),
                        ));
                    }
                }
            }
        }
    }

    /// Returns an iterator over the tag names this parser recognizes.
    pub fn registered_tags(&self) -> impl Iterator<Item = &str> {
        self.tag_map.keys().map(|s| s.as_str())
    }

    /// Wraps a token iterator and yields parsing events as they become available.
    ///
    /// This method clones `self`, resets the clone, and parses the provided token stream.
    /// Event boundaries depend on token boundaries; callers should be prepared to
    /// concatenate adjacent `Content`/`Output` payloads.
    ///
    /// # Errors
    ///
    /// The inner iterator yields `Result<String, PipelineError>`. When it yields `Err(e)`,
    /// the returned iterator first yields any events produced by flushing buffered input,
    /// then yields `Err(e)`, and then terminates.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let parser = XmlParserBuilder::new().register_tag("think").build();
    /// let tokens = pipeline.run_iter("...")?;
    ///
    /// for event in parser.parse_iter(tokens) {
    ///     match event {
    ///         Ok(event) => match (event.tag(), event.part()) {
    ///             (Some("think"), TagParts::Content) => println!("[thinking] {}", event.get_content()),
    ///             (None, TagParts::Content) => print!("{}", event.get_content()),
    ///             _ => {}
    ///         },
    ///         Err(e) => eprintln!("Error: {}", e),
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

/// Iterator that converts a token stream into [`Event`] values.
///
/// This iterator wraps a token iterator and parses XML-like tags incrementally.
/// See [`XmlParser::parse_iter`] for event chunking and error/flush behavior.
pub struct EventIterator<I> {
    parser: XmlParser,
    inner: I,
    buffer: VecDeque<Event>,
    flushed: bool,
    pending_error: Option<crate::error::PipelineError>,
}

impl<I> EventIterator<I> {
    fn new(mut parser: XmlParser, iter: I) -> Self {
        parser.reset();
        Self {
            parser,
            inner: iter,
            buffer: VecDeque::new(),
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
        if let Some(event) = self.buffer.pop_front() {
            return Some(Ok(event));
        }

        if let Some(e) = self.pending_error.take() {
            return Some(Err(e));
        }

        if self.flushed {
            return None;
        }

        for result in self.inner.by_ref() {
            match result {
                Ok(token) => {
                    let events = self.parser.parse_token(&token);
                    if !events.is_empty() {
                        self.buffer.extend(events);
                        return Some(Ok(self.buffer.pop_front().unwrap()));
                    }
                }
                Err(e) => {
                    let flush_events = self.parser.flush();
                    self.buffer.extend(flush_events);
                    self.flushed = true;
                    self.pending_error = Some(e);

                    if let Some(event) = self.buffer.pop_front() {
                        return Some(Ok(event));
                    }
                    return Some(Err(self.pending_error.take().unwrap()));
                }
            }
        }

        if !self.flushed {
            self.flushed = true;
            let events = self.parser.flush();
            if !events.is_empty() {
                self.buffer.extend(events);
                return Some(Ok(self.buffer.pop_front().unwrap()));
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
