use crate::Result;
use futures::Stream;
use futures::StreamExt;
use pin_project_lite::pin_project;
use std::pin::Pin;
use std::task::{Context, Poll};

pin_project! {
    pub struct CompletionStream<S> {
        #[pin]
        inner: Pin<Box<S>>,
        stats: std::sync::Arc<std::sync::Mutex<crate::pipelines::text_generation::stats::GenerationStats>>,
    }
}

impl<S> CompletionStream<S> {
    pub(crate) fn new(
        inner: S,
        stats: std::sync::Arc<
            std::sync::Mutex<crate::pipelines::text_generation::stats::GenerationStats>,
        >,
    ) -> Self {
        Self {
            inner: Box::pin(inner),
            stats,
        }
    }

    /// Get the next chunk from the stream.
    ///
    /// Returns `None` when the stream is exhausted.
    pub async fn next(&mut self) -> Option<Result<String>>
    where
        S: Stream<Item = Result<String>>,
    {
        self.inner.as_mut().next().await
    }

    /// Collect the entire stream into a single `String`.
    pub async fn collect(mut self) -> Result<String>
    where
        S: Stream<Item = Result<String>>,
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
    pub async fn take(mut self, n: usize) -> Result<Vec<String>>
    where
        S: Stream<Item = Result<String>>,
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
        S: Stream<Item = Result<String>>,
        F: FnMut(Result<String>) -> T,
    {
        CompletionStream::new(self.inner.map(f), self.stats)
    }

    /// Filter chunks in the stream based on a predicate.
    pub fn filter<F>(self, mut f: F) -> CompletionStream<impl Stream<Item = Result<String>>>
    where
        S: Stream<Item = Result<String>>,
        F: FnMut(&Result<String>) -> bool,
    {
        CompletionStream::new(
            self.inner.filter(move |item| std::future::ready(f(item))),
            self.stats,
        )
    }

    /// Fold over the stream, producing a single value.
    pub async fn fold<T, F>(self, init: T, mut f: F) -> T
    where
        S: Stream<Item = Result<String>>,
        F: FnMut(T, Result<String>) -> T,
    {
        self.inner
            .fold(init, |acc, item| std::future::ready(f(acc, item)))
            .await
    }

    /// Return the generation statistics collected during streaming.
    ///
    /// Statistics are typically finalized once the stream has completed.
    pub fn stats(&self) -> crate::pipelines::text_generation::stats::GenerationStats {
        self.stats.lock().unwrap().clone()
    }
}

impl<S> Stream for CompletionStream<S>
where
    S: Stream<Item = Result<String>>,
{
    type Item = Result<String>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut this = self.project();
        this.inner.as_mut().as_mut().poll_next(cx)
    }
}
