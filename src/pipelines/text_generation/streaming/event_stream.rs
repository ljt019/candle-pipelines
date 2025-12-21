use crate::pipelines::text_generation::parser::{Event, TagParts};
use futures::Stream;
use futures::StreamExt;
use pin_project_lite::pin_project;
use std::pin::Pin;
use std::task::{Context, Poll};

pin_project! {
    /// Convenience wrapper around the streaming output of XML pipeline.
    pub struct EventStream<S> {
        #[pin]
        inner: Pin<Box<S>>,
    }
}

impl<S> EventStream<S> {
    pub(crate) fn new(inner: S) -> Self {
        Self {
            inner: Box::pin(inner),
        }
    }

    /// Get the next event from the stream.
    ///
    /// Returns `None` when the stream is exhausted.
    pub async fn next(&mut self) -> Option<Event>
    where
        S: Stream<Item = Event>,
    {
        self.inner.as_mut().next().await
    }

    /// Collect all events into a vector.
    pub async fn collect(mut self) -> Vec<Event>
    where
        S: Stream<Item = Event>,
    {
        let mut events = Vec::new();
        while let Some(event) = self.inner.as_mut().next().await {
            events.push(event);
        }
        events
    }

    /// Collect only the content from events, ignoring start/end markers.
    pub async fn collect_content(mut self) -> String
    where
        S: Stream<Item = Event>,
    {
        let mut out = String::new();
        while let Some(event) = self.inner.as_mut().next().await {
            if event.part() == TagParts::Content {
                out.push_str(event.get_content());
            }
        }
        out
    }

    /// Take up to `n` events from the stream.
    pub async fn take(mut self, n: usize) -> Vec<Event>
    where
        S: Stream<Item = Event>,
    {
        let mut events = Vec::new();
        for _ in 0..n {
            match self.inner.as_mut().next().await {
                Some(event) => events.push(event),
                None => break,
            }
        }
        events
    }

    /// Filter events based on a predicate.
    pub fn filter<F>(self, mut f: F) -> EventStream<impl Stream<Item = Event>>
    where
        S: Stream<Item = Event>,
        F: FnMut(&Event) -> bool,
    {
        EventStream::new(self.inner.filter(move |item| std::future::ready(f(item))))
    }

    /// Map each event through a function.
    pub fn map<F, T>(self, f: F) -> EventStream<impl Stream<Item = T>>
    where
        S: Stream<Item = Event>,
        F: FnMut(Event) -> T,
    {
        EventStream::new(self.inner.map(f))
    }

    /// Filter events by tag name.
    pub fn filter_tag(self, tag_name: &str) -> EventStream<impl Stream<Item = Event>>
    where
        S: Stream<Item = Event>,
    {
        let tag_name = tag_name.to_string();
        self.filter(move |event| event.tag() == Some(&tag_name))
    }

    /// Get only content events (excluding start/end markers).
    pub fn content_only(self) -> EventStream<impl Stream<Item = Event>>
    where
        S: Stream<Item = Event>,
    {
        self.filter(|event| event.part() == TagParts::Content)
    }
}

impl<S> Stream for EventStream<S>
where
    S: Stream<Item = Event>,
{
    type Item = Event;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        this.inner.as_mut().as_mut().poll_next(cx)
    }
}
