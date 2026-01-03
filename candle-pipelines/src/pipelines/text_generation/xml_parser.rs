//! Streaming XML-like tag parser for LLM outputs.
//!
//! Parse structured output like `<think>...</think>` or `<tool_call>...</tool_call>` tags.
//!
//! # Quick Start
//!
//! Define your tags as an enum and derive `XmlTag`:
//!
//! ```rust,ignore
//! use candle_pipelines::text_generation::{Event, TagParts, XmlTag};
//!
//! #[derive(XmlTag, Clone, PartialEq, Debug)]
//! enum MyTags {
//!     #[tag("think")]
//!     Think,
//!     #[tag("answer")]
//!     Answer,
//! }
//!
//! let mut parser = MyTags::parser();
//! let events = parser.parse("<think>reasoning</think><answer>42</answer>");
//!
//! for event in events {
//!     match event {
//!         Event::Tagged { tag: MyTags::Think, part: TagParts::Content, content, .. } => {
//!             println!("[thinking] {}", content);
//!         }
//!         Event::Tagged { tag: MyTags::Answer, part: TagParts::End, content, .. } => {
//!             println!("[answer] {}", content);
//!         }
//!         Event::Output { content } => print!("{}", content),
//!         _ => {}
//!     }
//! }
//! ```
//!
//! # Core Types
//!
//! - `#[derive(XmlTag)]` - Derive macro for defining tag enums
//! - [`XmlParser<T>`] - The parser, generic over your tag enum
//! - [`Event<T>`] - Emitted events: `Tagged` or `Output`
//! - [`TagParts`] - Which part of a tag: `Start`, `Content`, or `End`

use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;

// ##############################################
//                 Public Api
// ##############################################

/// Identifies which portion of a tag an [`Event`] represents.
///
/// For [`Event::Output`], [`Event::part()`] returns [`TagParts::Content`].
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
/// This parser recognizes a small XML-like syntax for tags defined in your `T` enum.
/// It emits:
/// - [`Event::Tagged`] for recognized tags, with [`TagParts::Start`], [`TagParts::Content`],
///   and [`TagParts::End`].
/// - [`Event::Output`] for text outside any recognized tag.
///
/// For [`Event::Tagged`]:
/// - `Start` events store the opening tag text in `content`.
/// - `Content` events store text inside the tag in `content`.
/// - `End` events store a reconstructed full element string in `content`
///   (see [`XmlParser`] docs for details and limitations).
#[derive(Debug, Clone, PartialEq)]
pub enum Event<T: XmlTag> {
    /// Data associated with a recognized tag.
    Tagged {
        /// The tag variant that was matched.
        tag: T,
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
    /// Text outside any recognized tag (including unrecognized tags).
    Output {
        /// The emitted text (possibly chunked in streaming mode).
        content: String,
    },
}

impl<T: XmlTag> Event<T> {
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

    /// Returns a reference to the tag for [`Event::Tagged`], or `None` for [`Event::Output`].
    pub fn tag(&self) -> Option<&T> {
        match self {
            Self::Tagged { tag, .. } => Some(tag),
            Self::Output { .. } => None,
        }
    }

    /// Returns the tag name as a string for [`Event::Tagged`], or `None` for [`Event::Output`].
    pub fn tag_str(&self) -> Option<&'static str> {
        self.tag().map(|t| t.as_tag_str())
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
    /// - `self` is a tagged event whose tag maps to `"tool_call"`,
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
        if tag.as_tag_str() != "tool_call" || self.part() != TagParts::End {
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

/// Streaming parser for extracting XML-like tagged regions from text.
///
/// Define your tags as an enum with `#[derive(XmlTag)]`, then call `.parser()`
/// on the enum to create a parser.
///
/// # Example
///
/// ```rust,ignore
/// use candle_pipelines::text_generation::{XmlTag, Event, TagParts};
///
/// #[derive(XmlTag, Clone, PartialEq, Debug)]
/// enum MyTags {
///     #[tag("think")]
///     Think,
///     #[tag("answer")]
///     Answer,
/// }
///
/// let mut parser = MyTags::parser();
/// let events = parser.parse("<think>reasoning</think>");
/// ```
///
/// # Tag tracking and limitations
///
/// - The parser tracks at most one open tag at a time. It does not support nested tags.
/// - If a tag appears inside another tag's content, the parser treats it as literal text.
/// - The parser does not validate XML and does not handle namespaces, entities,
///   comments, CDATA, or processing instructions. Malformed or mismatched tags are
///   emitted as plain text.
///
/// # Event payloads
///
/// For a recognized element `<name ...attrs...>inner</name>`:
/// - A `Start` event is emitted with `get_content()` set to the opening tag text.
/// - Zero or more `Content` events are emitted with slices of `inner`.
/// - An `End` event is emitted with `get_content()` set to a reconstructed string
///   of the form `<name>{inner}</name>`.
///
/// For a self-closing tag (for example, `<name a=b/>`), the parser emits:
/// - `Start` with `get_content()` set to `"<name>"`,
/// - `End` with `get_content()` set to the original self-closing tag text.
#[derive(Debug, Clone)]
pub struct XmlParser<T: XmlTag> {
    _marker: PhantomData<T>,
    state: ParserState<T>,
}

impl<T: XmlTag> Default for XmlParser<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: XmlTag> XmlParser<T> {
    /// Creates a new parser.
    ///
    /// Typically you'll use `MyTags::parser()` instead of calling this directly.
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
            state: ParserState::default(),
        }
    }

    /// Clears any partial state from previous parsing.
    ///
    /// Call this before reusing a parser for a new input.
    /// Note: [`parse`](Self::parse) calls this automatically.
    pub fn reset(&mut self) {
        self.state = ParserState::default();
    }

    /// Parses `text` and returns all events.
    ///
    /// Use this for complete strings. For streaming token-by-token parsing,
    /// use [`parse_iter`](Self::parse_iter) instead.
    pub fn parse(&mut self, text: &str) -> Vec<Event<T>> {
        self.reset();
        let mut events = Vec::new();

        for c in text.chars() {
            self.process_char_internal(c, &mut events);
        }

        self.flush_internal(&mut events);
        events
    }

    /// Wraps a token iterator and yields parsing events as they become available.
    ///
    /// Use this for streaming output from an LLM. Events are emitted as tokens arrive,
    /// so `Content` events may be split across multiple chunks. Concatenate adjacent
    /// payloads if you need the full content.
    ///
    /// # Errors
    ///
    /// If the inner iterator yields an error, it is propagated after any pending events.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use candle_pipelines::text_generation::{XmlTag, Event, TagParts};
    ///
    /// #[derive(XmlTag, Clone, PartialEq, Debug)]
    /// enum Tags { #[tag("think")] Think }
    ///
    /// let parser = Tags::parser();
    /// let tokens = pipeline.run_iter("...")?;
    ///
    /// for event in parser.parse_iter(tokens) {
    ///     match event? {
    ///         Event::Tagged { tag: Tags::Think, part: TagParts::Content, content, .. } => {
    ///             println!("[thinking] {}", content);
    ///         }
    ///         Event::Output { content } => print!("{}", content),
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub fn parse_iter<I>(&self, iter: I) -> impl Iterator<Item = crate::error::Result<Event<T>>>
    where
        I: Iterator<Item = crate::error::Result<String>>,
    {
        EventIterator::new(self.clone(), iter)
    }
}

// ##############################################
//                 Internal Api
// ##############################################

#[doc(hidden)]
pub trait XmlTag: Sized + Clone + PartialEq + std::fmt::Debug {
    fn from_tag_str(s: &str) -> Option<Self>;
    fn as_tag_str(&self) -> &'static str;
}

/// Iterator that converts a token stream into [`Event`] values.
pub(crate) struct EventIterator<T: XmlTag, I> {
    parser: XmlParser<T>,
    inner: I,
    buffer: VecDeque<Event<T>>,
    flushed: bool,
    pending_error: Option<crate::error::PipelineError>,
}

impl<T: XmlTag, I> Iterator for EventIterator<T, I>
where
    I: Iterator<Item = crate::error::Result<String>>,
{
    type Item = crate::error::Result<Event<T>>;

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

impl<T: XmlTag, I> EventIterator<T, I> {
    fn new(mut parser: XmlParser<T>, iter: I) -> Self {
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

impl<T: XmlTag> Event<T> {
    fn tagged(
        tag: T,
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
        tag: T,
        opening_tag: impl Into<String>,
        attributes: HashMap<String, String>,
    ) -> Self {
        Self::tagged(tag, TagParts::Start, opening_tag, attributes)
    }

    pub(crate) fn end(
        tag: T,
        full_xml: impl Into<String>,
        attributes: HashMap<String, String>,
    ) -> Self {
        Self::tagged(tag, TagParts::End, full_xml, attributes)
    }

    fn tagged_internal(
        tag: T,
        content: impl Into<String>,
        attributes: HashMap<String, String>,
    ) -> Self {
        Self::tagged(tag, TagParts::Content, content, attributes)
    }
}

impl<T: XmlTag> XmlParser<T> {
    /// Parses a chunk of input in streaming mode and returns newly available events.
    pub(crate) fn parse_token(&mut self, token: &str) -> Vec<Event<T>> {
        let mut events = Vec::new();

        for c in token.chars() {
            self.process_char_internal(c, &mut events);
        }

        if let Some(ref mut open) = self.state.open_tag {
            let total_len = open.content.len();
            if total_len > open.emitted_len {
                let new_slice = &open.content[open.emitted_len..];
                if !new_slice.is_empty() {
                    events.push(Event::tagged_internal(
                        open.tag.clone(),
                        new_slice,
                        open.attrs.clone(),
                    ));
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

    fn process_char_internal(&mut self, c: char, out: &mut Vec<Event<T>>) {
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

    fn handle_tag(&mut self, tag_content: &str, out: &mut Vec<Event<T>>) {
        if let Some(tag_name) = Self::parse_tag_name(tag_content) {
            let trimmed_name = tag_name.trim();

            // Try to parse as a known tag
            if let Some(tag) = T::from_tag_str(trimmed_name) {
                let tag_str = tag.as_tag_str();

                if tag_content.starts_with("</") {
                    // Closing tag
                    if let Some(ref open) = self.state.open_tag {
                        if open.tag.as_tag_str() == tag_str {
                            let open = self.state.open_tag.take().unwrap();

                            if open.content.len() > open.emitted_len {
                                let remaining = &open.content[open.emitted_len..];
                                if !remaining.is_empty() {
                                    out.push(Event::tagged_internal(
                                        open.tag.clone(),
                                        remaining,
                                        open.attrs.clone(),
                                    ));
                                }
                            }

                            let full_xml =
                                format!("<{}>{}</{}>", tag_str, open.content, tag_str);
                            out.push(Event::end(open.tag, full_xml, open.attrs));
                        } else {
                            // Mismatched close - treat as content
                            if let Some(ref mut open) = self.state.open_tag {
                                open.content.push_str(tag_content);
                            }
                        }
                    } else {
                        self.state.content_buffer.push_str(tag_content);
                    }
                } else if tag_content.ends_with("/>") {
                    // Self-closing tag
                    if self.state.open_tag.is_none() {
                        if !self.state.content_buffer.is_empty() {
                            let content = &self.state.content_buffer[self.state.emitted_top_len..];
                            if !content.is_empty() {
                                out.push(Event::content(content));
                            }
                            self.state.emitted_top_len = self.state.content_buffer.len();
                        }

                        let attrs = Self::parse_attributes(tag_content);

                        out.push(Event::start(
                            tag.clone(),
                            format!("<{}>", tag_str),
                            attrs.clone(),
                        ));
                        out.push(Event::end(tag, tag_content.to_string(), attrs));
                    } else {
                        if let Some(ref mut open) = self.state.open_tag {
                            open.content.push_str(tag_content);
                        }
                    }
                } else {
                    // Opening tag
                    if self.state.open_tag.is_none() {
                        if !self.state.content_buffer.is_empty() {
                            let content = &self.state.content_buffer[self.state.emitted_top_len..];
                            if !content.is_empty() {
                                out.push(Event::content(content));
                            }
                            self.state.emitted_top_len = self.state.content_buffer.len();
                        }

                        let attrs = Self::parse_attributes(tag_content);
                        self.state.open_tag = Some(OpenTag {
                            tag: tag.clone(),
                            content: String::new(),
                            attrs: attrs.clone(),
                            emitted_len: 0,
                        });

                        out.push(Event::start(tag, tag_content.to_string(), attrs));
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

    fn parse_tag_name(tag_content: &str) -> Option<String> {
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
    fn parse_attributes(tag_content: &str) -> HashMap<String, String> {
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
    pub(crate) fn flush(&mut self) -> Vec<Event<T>> {
        let mut events = Vec::new();
        self.flush_internal(&mut events);
        events
    }

    fn flush_internal(&mut self, out: &mut Vec<Event<T>>) {
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
            if open.content.len() > open.emitted_len {
                let remaining = &open.content[open.emitted_len..];
                if !remaining.is_empty() {
                    out.push(Event::tagged_internal(
                        open.tag,
                        remaining,
                        open.attrs,
                    ));
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct OpenTag<T: XmlTag> {
    tag: T,
    content: String,
    attrs: HashMap<String, String>,
    emitted_len: usize,
}

#[derive(Debug, Clone)]
struct ParserState<T: XmlTag> {
    open_tag: Option<OpenTag<T>>,
    content_buffer: String,
    tag_buffer: String,
    in_tag: bool,
    emitted_top_len: usize,
}

impl<T: XmlTag> Default for ParserState<T> {
    fn default() -> Self {
        Self {
            open_tag: None,
            content_buffer: String::new(),
            tag_buffer: String::new(),
            in_tag: false,
            emitted_top_len: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============ Test Tag Enums ============

    /// Single tag for basic tests
    #[derive(Debug, Clone, PartialEq)]
    enum Think {
        Think,
    }

    impl XmlTag for Think {
        fn from_tag_str(s: &str) -> Option<Self> {
            match s.trim() {
                "think" => Some(Self::Think),
                _ => None,
            }
        }
        fn as_tag_str(&self) -> &'static str {
            "think"
        }
    }

    /// Two tags for multi-tag tests
    #[derive(Debug, Clone, PartialEq)]
    enum ThinkAnswer {
        Think,
        Answer,
    }

    impl XmlTag for ThinkAnswer {
        fn from_tag_str(s: &str) -> Option<Self> {
            match s.trim() {
                "think" => Some(Self::Think),
                "answer" => Some(Self::Answer),
                _ => None,
            }
        }
        fn as_tag_str(&self) -> &'static str {
            match self {
                Self::Think => "think",
                Self::Answer => "answer",
            }
        }
    }

    /// Just answer tag (for unregistered tag tests)
    #[derive(Debug, Clone, PartialEq)]
    enum Answer {
        Answer,
    }

    impl XmlTag for Answer {
        fn from_tag_str(s: &str) -> Option<Self> {
            match s.trim() {
                "answer" => Some(Self::Answer),
                _ => None,
            }
        }
        fn as_tag_str(&self) -> &'static str {
            "answer"
        }
    }

    /// Fn tag for attribute tests
    #[derive(Debug, Clone, PartialEq)]
    enum Fn {
        Fn,
    }

    impl XmlTag for Fn {
        fn from_tag_str(s: &str) -> Option<Self> {
            match s.trim() {
                "fn" => Some(Self::Fn),
                _ => None,
            }
        }
        fn as_tag_str(&self) -> &'static str {
            "fn"
        }
    }

    /// FunctionCall tag
    #[derive(Debug, Clone, PartialEq)]
    enum FunctionCall {
        FunctionCall,
    }

    impl XmlTag for FunctionCall {
        fn from_tag_str(s: &str) -> Option<Self> {
            match s.trim() {
                "function_call" => Some(Self::FunctionCall),
                _ => None,
            }
        }
        fn as_tag_str(&self) -> &'static str {
            "function_call"
        }
    }

    /// ToolCall tag
    #[derive(Debug, Clone, PartialEq)]
    enum ToolCall {
        ToolCall,
    }

    impl XmlTag for ToolCall {
        fn from_tag_str(s: &str) -> Option<Self> {
            match s.trim() {
                "tool_call" => Some(Self::ToolCall),
                _ => None,
            }
        }
        fn as_tag_str(&self) -> &'static str {
            "tool_call"
        }
    }

    // ============ Helper Functions ============

    fn assert_attr<T: XmlTag>(event: &Event<T>, key: &str, expected: &str) {
        let attrs = event.attributes().expect("expected attributes");
        assert_eq!(attrs.get(key).map(String::as_str), Some(expected));
    }

    // ============ Tests ============

    #[test]
    fn test_plain_text_only_parsing() {
        let mut parser = XmlParser::<Think>::new();

        let text = "Regular content";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], Event::Output { content } if content == "Regular content"));
    }

    #[test]
    fn test_empty_parsing() {
        let mut parser = XmlParser::<Think>::new();

        let text = "";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 0);
    }

    #[test]
    fn test_whitespace_parsing() {
        let mut parser = XmlParser::<Think>::new();

        let text = "  ";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], Event::Output { content } if content == "  "));
    }

    #[test]
    fn test_single_tag_parsing() {
        let mut parser = XmlParser::<Think>::new();

        let text = "<think>Hello world</think>";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], Event::Tagged { tag: Think::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[1], Event::Tagged { tag: Think::Think, part: TagParts::Content, content, .. } if content == "Hello world"));
        assert!(matches!(&events[2], Event::Tagged { tag: Think::Think, part: TagParts::End, content, .. } if content == text));
    }

    #[test]
    fn test_plain_text_and_single_tag_parsing() {
        let mut parser = XmlParser::<Think>::new();

        let text = "<think>Hello world</think>Regular content";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], Event::Tagged { tag: Think::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[1], Event::Tagged { tag: Think::Think, part: TagParts::Content, content, .. } if content == "Hello world"));
        assert!(matches!(&events[2], Event::Tagged { tag: Think::Think, part: TagParts::End, .. }));
        assert!(matches!(&events[3], Event::Output { content } if content == "Regular content"));
    }

    #[test]
    fn test_unicode_tag_parsing() {
        let mut parser = XmlParser::<Think>::new();
        let text = "<think>你好世界</think>普通内容";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 4);
        assert!(matches!(&events[1], Event::Tagged { tag: Think::Think, part: TagParts::Content, content, .. } if content == "你好世界"));
        assert!(matches!(&events[3], Event::Output { content } if content == "普通内容"));
    }

    #[test]
    fn test_plain_text_before_and_after_single_tag_parsing() {
        let mut parser = XmlParser::<Think>::new();

        let text = "How are <think>Hello world</think>you doing today?";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 5);
        assert!(matches!(&events[0], Event::Output { content } if content == "How are "));
        assert!(matches!(&events[1], Event::Tagged { tag: Think::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[2], Event::Tagged { tag: Think::Think, part: TagParts::Content, content, .. } if content == "Hello world"));
        assert!(matches!(&events[3], Event::Tagged { tag: Think::Think, part: TagParts::End, .. }));
        assert!(matches!(&events[4], Event::Output { content } if content == "you doing today?"));

        let plain: String = events
            .iter()
            .filter(|e| matches!(e, Event::Output { .. }))
            .map(|e| e.get_content())
            .collect();
        assert_eq!(plain, "How are you doing today?");
    }

    #[test]
    fn test_multiple_tags_parsing() {
        let mut parser = XmlParser::<ThinkAnswer>::new();

        let text = "<think>Hm the answer to 1 + 1 is 2</think><answer>2</answer>";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 6);
        assert!(matches!(&events[0], Event::Tagged { tag: ThinkAnswer::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[1], Event::Tagged { tag: ThinkAnswer::Think, part: TagParts::Content, .. }));
        assert!(matches!(&events[2], Event::Tagged { tag: ThinkAnswer::Think, part: TagParts::End, .. }));
        assert!(matches!(&events[3], Event::Tagged { tag: ThinkAnswer::Answer, part: TagParts::Start, .. }));
        assert!(matches!(&events[4], Event::Tagged { tag: ThinkAnswer::Answer, part: TagParts::Content, content, .. } if content == "2"));
        assert!(matches!(&events[5], Event::Tagged { tag: ThinkAnswer::Answer, part: TagParts::End, .. }));
    }

    #[test]
    fn test_multiple_same_tag_parsing() {
        let mut parser = XmlParser::<Think>::new();

        let text = "<think>Hello world</think><think>Regular content</think>";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 6);
        assert!(matches!(&events[0], Event::Tagged { tag: Think::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[1], Event::Tagged { tag: Think::Think, part: TagParts::Content, content, .. } if content == "Hello world"));
        assert!(matches!(&events[2], Event::Tagged { tag: Think::Think, part: TagParts::End, .. }));
        assert!(matches!(&events[3], Event::Tagged { tag: Think::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[4], Event::Tagged { tag: Think::Think, part: TagParts::Content, content, .. } if content == "Regular content"));
        assert!(matches!(&events[5], Event::Tagged { tag: Think::Think, part: TagParts::End, .. }));
    }

    #[test]
    fn test_empty_tags_parsing() {
        let mut parser = XmlParser::<Think>::new();

        let text = "<think></think>Regular content";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], Event::Tagged { tag: Think::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[1], Event::Tagged { tag: Think::Think, part: TagParts::End, .. }));
        assert!(matches!(&events[2], Event::Output { content } if content == "Regular content"));
    }

    #[test]
    fn test_unregistered_tag_parsing() {
        // Using Answer enum - <think> is not registered, only <answer>
        let mut parser = XmlParser::<Answer>::new();

        let text = "<think>Hm the answer to 1 + 1 is 2</think><answer>2</answer>";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 4);
        // <think> is not recognized, so it becomes output
        assert!(matches!(&events[0], Event::Output { content } if content == "<think>Hm the answer to 1 + 1 is 2</think>"));
        assert!(matches!(&events[1], Event::Tagged { tag: Answer::Answer, part: TagParts::Start, .. }));
        assert!(matches!(&events[2], Event::Tagged { tag: Answer::Answer, part: TagParts::Content, content, .. } if content == "2"));
        assert!(matches!(&events[3], Event::Tagged { tag: Answer::Answer, part: TagParts::End, .. }));
    }

    #[test]
    fn test_self_closing_tag_parsing() {
        let mut parser = XmlParser::<Think>::new();

        let text = "<think/>Regular Content<think />";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 5);
        assert!(matches!(&events[0], Event::Tagged { tag: Think::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[1], Event::Tagged { tag: Think::Think, part: TagParts::End, .. }));
        assert!(matches!(&events[2], Event::Output { content } if content == "Regular Content"));
        assert!(matches!(&events[3], Event::Tagged { tag: Think::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[4], Event::Tagged { tag: Think::Think, part: TagParts::End, .. }));
    }

    #[test]
    fn test_is_greedy_parsing() {
        let mut parser = XmlParser::<ThinkAnswer>::new();

        let text = "<think>Hm the answer to 1 + 1 is 2 so I should output <answer> tags containing 2 like <answer>2</answer>, I'll do that now</think><answer>2</answer>";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 6);
        assert!(matches!(&events[0], Event::Tagged { tag: ThinkAnswer::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[1], Event::Tagged { tag: ThinkAnswer::Think, part: TagParts::Content, .. }));
        assert!(matches!(&events[2], Event::Tagged { tag: ThinkAnswer::Think, part: TagParts::End, .. }));
        assert!(matches!(&events[3], Event::Tagged { tag: ThinkAnswer::Answer, part: TagParts::Start, .. }));
        assert!(matches!(&events[4], Event::Tagged { tag: ThinkAnswer::Answer, part: TagParts::Content, content, .. } if content == "2"));
        assert!(matches!(&events[5], Event::Tagged { tag: ThinkAnswer::Answer, part: TagParts::End, .. }));
    }

    #[test]
    fn test_greedy_multiple_same_tag_parsing() {
        let mut parser = XmlParser::<Think>::new();

        let text = "<think>Hello world<think>Regular content</think></think>";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], Event::Tagged { tag: Think::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[1], Event::Tagged { tag: Think::Think, part: TagParts::Content, content, .. } if content == "Hello world<think>Regular content"));
        assert!(matches!(&events[2], Event::Tagged { tag: Think::Think, part: TagParts::End, .. }));
        assert!(matches!(&events[3], Event::Output { content } if content == "</think>"));
    }

    #[test]
    fn test_greedy_multiple_same_open_tag_parsing() {
        let mut parser = XmlParser::<Think>::new();

        let text = "<think>Hello world<think> Regular content</think>";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], Event::Tagged { tag: Think::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[1], Event::Tagged { tag: Think::Think, part: TagParts::Content, content, .. } if content == "Hello world<think> Regular content"));
        assert!(matches!(&events[2], Event::Tagged { tag: Think::Think, part: TagParts::End, .. }));
    }

    #[test]
    fn test_mismatched_close_parsing() {
        let mut parser = XmlParser::<Think>::new();

        let text = "<think>Hm I think the answer to 1 + 1 is 2</answer>";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], Event::Tagged { tag: Think::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[1], Event::Tagged { tag: Think::Think, part: TagParts::Content, content, .. } if content == "Hm I think the answer to 1 + 1 is 2</answer>"));
    }

    #[test]
    fn test_unclosed_tag_parsing() {
        let mut parser = XmlParser::<Think>::new();

        let text = "<think>Hm I think the answer to 1 + 1 is 2";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], Event::Tagged { tag: Think::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[1], Event::Tagged { tag: Think::Think, part: TagParts::Content, content, .. } if content == "Hm I think the answer to 1 + 1 is 2"));
    }

    #[test]
    fn test_empty_tag_name() {
        let mut parser = XmlParser::<Think>::new();

        let text = "hello <> world </>";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], Event::Output { content } if content == "hello <> world </>"));
    }

    #[test]
    fn test_reset_clears_streaming_state() {
        let mut parser = XmlParser::<ThinkAnswer>::new();

        parser.parse_token("<thi");
        parser.parse_token("nk>partial content");
        parser.parse_token(" more");
        parser.reset();

        let mut events = parser.parse_token("<answer>42</answer>");
        events.extend(parser.flush());

        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], Event::Tagged { tag: ThinkAnswer::Answer, part: TagParts::Start, .. }));
        assert!(matches!(&events[1], Event::Tagged { tag: ThinkAnswer::Answer, part: TagParts::Content, content, .. } if content == "42"));
        assert!(matches!(&events[2], Event::Tagged { tag: ThinkAnswer::Answer, part: TagParts::End, .. }));
    }

    #[test]
    fn test_no_attributes_returns_none() {
        let mut parser = XmlParser::<Think>::new();

        let text = "<think>Hello world</think>Regular content";
        let events = parser.parse(&text);

        assert!(matches!(&events[0], Event::Tagged { tag: Think::Think, .. }));
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
            let mut parser = XmlParser::<Fn>::new();
            let events = parser.parse(input);
            assert_attr(&events[0], key, expected);
        }
    }

    #[test]
    fn test_attribute_boolean() {
        let mut parser = XmlParser::<Fn>::new();
        let events = parser.parse("<fn disabled>x</fn>");
        assert!(events[0].attributes().unwrap().contains_key("disabled"));
    }

    #[test]
    fn test_attributes_self_closing() {
        let mut parser = XmlParser::<Fn>::new();
        let events = parser.parse("<fn name=test/>");
        assert_attr(&events[0], "name", "test");
    }

    #[test]
    fn test_multiple_attributes() {
        let mut parser = XmlParser::<Fn>::new();
        let events = parser.parse("<fn name='get_weather' id=0>x</fn>");

        assert_attr(&events[0], "name", "get_weather");
        assert_attr(&events[0], "id", "0");
    }

    #[test]
    fn test_attributes_persist_across_tag_sequence() {
        let mut parser = XmlParser::<FunctionCall>::new();

        let text = "<function_call name=test>content</function_call>";
        let events = parser.parse(&text);

        for event in &events {
            if matches!(event, Event::Tagged { tag: FunctionCall::FunctionCall, .. }) {
                assert_attr(event, "name", "test");
            }
        }
    }

    #[test]
    fn test_attribute_returns_none_for_plain_content() {
        let mut parser = XmlParser::<Think>::new();

        let text = "Regular content";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], Event::Output { .. }));
        assert_eq!(events[0].attributes(), None);
    }

    #[test]
    fn test_iter_attributes_preserved() {
        let parser = XmlParser::<FunctionCall>::new();

        let tokens = vec![
            Ok("<function_call na".to_string()),
            Ok("me=get_weather>Tokyo</function_call>".to_string()),
        ];

        let events: Vec<Event<FunctionCall>> = parser
            .parse_iter(tokens.into_iter())
            .map(|r| r.unwrap())
            .collect();

        assert!(matches!(&events[0], Event::Tagged { tag: FunctionCall::FunctionCall, .. }));
        assert_attr(&events[0], "name", "get_weather");
    }

    #[test]
    fn test_iter_greedy_multiple_same_tag_parsing() {
        let parser = XmlParser::<Think>::new();

        let tokens = vec![
            Ok("<think>Hello ".to_string()),
            Ok("world<think> Regular content</think>".to_string()),
            Ok("</think>".to_string()),
        ];

        let events: Vec<Event<Think>> = parser
            .parse_iter(tokens.into_iter())
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(events.len(), 5);
        assert!(matches!(&events[0], Event::Tagged { tag: Think::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[1], Event::Tagged { tag: Think::Think, part: TagParts::Content, content, .. } if content == "Hello "));
        assert!(matches!(&events[2], Event::Tagged { tag: Think::Think, part: TagParts::Content, content, .. } if content == "world<think> Regular content"));
        assert!(matches!(&events[3], Event::Tagged { tag: Think::Think, part: TagParts::End, .. }));
        assert!(matches!(&events[4], Event::Output { content } if content == "</think>"));
    }

    #[test]
    fn test_iter_greedy_multiple_same_open_tag_parsing() {
        let parser = XmlParser::<Think>::new();

        let tokens = vec![
            Ok("<think>Hello ".to_string()),
            Ok("world<think> Regular content".to_string()),
            Ok("</think>".to_string()),
        ];

        let events: Vec<Event<Think>> = parser
            .parse_iter(tokens.into_iter())
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], Event::Tagged { tag: Think::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[3], Event::Tagged { tag: Think::Think, part: TagParts::End, .. }));
    }

    #[test]
    fn test_iter_split_open_tag_parsing() {
        let parser = XmlParser::<Think>::new();

        let tokens = vec![
            Ok("<".to_string()),
            Ok("think>Hello world</think>".to_string()),
        ];

        let events: Vec<Event<Think>> = parser
            .parse_iter(tokens.into_iter())
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], Event::Tagged { tag: Think::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[1], Event::Tagged { tag: Think::Think, part: TagParts::Content, content, .. } if content == "Hello world"));
        assert!(matches!(&events[2], Event::Tagged { tag: Think::Think, part: TagParts::End, .. }));
    }

    #[test]
    fn test_iter_split_close_tag_parsing() {
        let parser = XmlParser::<Think>::new();

        let tokens = vec![
            Ok("<".to_string()),
            Ok("think>Hello world</".to_string()),
            Ok("think>".to_string()),
        ];

        let events: Vec<Event<Think>> = parser
            .parse_iter(tokens.into_iter())
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], Event::Tagged { tag: Think::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[1], Event::Tagged { tag: Think::Think, part: TagParts::Content, content, .. } if content == "Hello world"));
        assert!(matches!(&events[2], Event::Tagged { tag: Think::Think, part: TagParts::End, .. }));
    }

    #[test]
    fn test_iter_split_content_in_tag_parsing() {
        let parser = XmlParser::<Think>::new();

        let tokens = vec![
            Ok("<".to_string()),
            Ok("think>Hello ".to_string()),
            Ok("world</think>".to_string()),
        ];
        let events: Vec<Event<Think>> = parser
            .parse_iter(tokens.into_iter())
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], Event::Tagged { tag: Think::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[1], Event::Tagged { tag: Think::Think, part: TagParts::Content, content, .. } if content == "Hello "));
        assert!(matches!(&events[2], Event::Tagged { tag: Think::Think, part: TagParts::Content, content, .. } if content == "world"));
        assert!(matches!(&events[3], Event::Tagged { tag: Think::Think, part: TagParts::End, .. }));
    }

    #[test]
    fn test_iter_char_by_char() {
        let parser = XmlParser::<Think>::new();

        let input = "<think>Hello world</think>";
        let tokens: Vec<Result<String, crate::error::PipelineError>> = input.chars().map(|c| Ok(c.to_string())).collect();
        let events: Vec<Event<Think>> = parser
            .parse_iter(tokens.into_iter())
            .map(|r| r.unwrap())
            .collect();

        // Should have Start, many Content events (one per char), and End
        assert!(events.len() >= 3);
        assert!(matches!(&events[0], Event::Tagged { tag: Think::Think, part: TagParts::Start, .. }));
        assert!(matches!(events.last().unwrap(), Event::Tagged { tag: Think::Think, part: TagParts::End, .. }));
    }

    #[test]
    fn test_partial_tag_at_end_emitted() {
        let mut parser = XmlParser::<Think>::new();

        let text = "hello <think";
        let events = parser.parse(&text);

        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], Event::Output { content } if content == "hello <think"));
    }

    #[test]
    fn test_partial_tag_inside_registered_tag() {
        let mut parser = XmlParser::<Think>::new();
        let events = parser.parse("<think>hi <ans");

        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], Event::Tagged { tag: Think::Think, part: TagParts::Start, .. }));
        assert!(matches!(&events[1], Event::Tagged { tag: Think::Think, part: TagParts::Content, content, .. } if content == "hi <ans"));
    }

    #[test]
    fn test_parse_token_no_double_emit() {
        let mut parser = XmlParser::<Think>::new();

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
        let mut parser = XmlParser::<Think>::new();

        let _ = parser.parse_token("<think>content</think>");
        let events1 = parser.parse_token("trailing");
        let events2 = parser.flush();

        let trailing_count = events1
            .iter()
            .chain(events2.iter())
            .filter(|e| matches!(e, Event::Output { .. }) && e.get_content().contains("trailing"))
            .count();

        assert_eq!(trailing_count, 1);
    }

    #[test]
    fn test_parse_tool_call_valid() {
        let mut parser = XmlParser::<ToolCall>::new();

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
        let mut parser = XmlParser::<ToolCall>::new();

        let text = "<tool_call>not valid json</tool_call>";
        let events = parser.parse(&text);

        let end_event = events.iter().find(|e| e.part() == TagParts::End).unwrap();
        let result = end_event.parse_tool_call();

        assert!(result.is_none());
    }
}
