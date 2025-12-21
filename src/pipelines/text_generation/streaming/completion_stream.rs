use futures::Stream;
use futures::StreamExt;
use pin_project_lite::pin_project;
use std::pin::Pin;
use std::task::{Context, Poll};

pin_project! {
    pub struct CompletionStream<S> {
        #[pin]
        inner: Pin<Box<S>>,
    }
}

impl<S> CompletionStream<S> {
    pub(crate) fn new(inner: S) -> Self {
        Self {
            inner: Box::pin(inner),
        }
    }

    /// Get the next chunk from the stream.
    ///
    /// Returns `None` when the stream is exhausted.
    pub async fn next(&mut self) -> Option<anyhow::Result<String>>
    where
        S: Stream<Item = anyhow::Result<String>>,
    {
        self.inner.as_mut().next().await
    }

    /// Collect the entire stream into a single `String`.
    pub async fn collect(mut self) -> anyhow::Result<String>
    where
        S: Stream<Item = anyhow::Result<String>>,
    {
        let mut out = String::new();
        while let Some(chunk) = self.inner.as_mut().next().await {
            out.push_str(&chunk?);
        }
        Ok(out)
    }

    /// Take up to `n` chunks from the stream.
    ///
    /// If the underlying stream ends before `n` chunks are yielded,
    /// the returned vector will contain fewer elements.
    pub async fn take(mut self, n: usize) -> anyhow::Result<Vec<String>>
    where
        S: Stream<Item = anyhow::Result<String>>,
    {
        let mut out = Vec::new();
        for _ in 0..n {
            match self.inner.as_mut().next().await {
                Some(chunk) => out.push(chunk?),
                None => break,
            }
        }
        Ok(out)
    }

    /// Map each chunk in the stream through a function.
    pub fn map<F, T>(self, f: F) -> CompletionStream<impl Stream<Item = T>>
    where
        S: Stream<Item = anyhow::Result<String>>,
        F: FnMut(anyhow::Result<String>) -> T,
    {
        CompletionStream::new(self.inner.map(f))
    }

    /// Filter chunks in the stream based on a predicate.
    pub fn filter<F>(self, mut f: F) -> CompletionStream<impl Stream<Item = anyhow::Result<String>>>
    where
        S: Stream<Item = anyhow::Result<String>>,
        F: FnMut(&anyhow::Result<String>) -> bool,
    {
        CompletionStream::new(self.inner.filter(move |item| std::future::ready(f(item))))
    }

    /// Fold over the stream, producing a single value.
    pub async fn fold<T, F>(self, init: T, mut f: F) -> T
    where
        S: Stream<Item = anyhow::Result<String>>,
        F: FnMut(T, anyhow::Result<String>) -> T,
    {
        self.inner
            .fold(init, |acc, item| std::future::ready(f(acc, item)))
            .await
    }
}

impl<S> Stream for CompletionStream<S>
where
    S: Stream<Item = anyhow::Result<String>>,
{
    type Item = anyhow::Result<String>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        this.inner.as_mut().as_mut().poll_next(cx)
    }
}
