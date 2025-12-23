use crate::error::Result;
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

    pub async fn next(&mut self) -> Option<Result<String>>
    where
        S: Stream<Item = Result<String>>,
    {
        self.inner.as_mut().next().await
    }

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

    pub fn map<F, T>(self, f: F) -> CompletionStream<impl Stream<Item = T>>
    where
        S: Stream<Item = Result<String>>,
        F: FnMut(Result<String>) -> T,
    {
        CompletionStream::new(self.inner.map(f), self.stats)
    }

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

    pub async fn fold<T, F>(self, init: T, mut f: F) -> T
    where
        S: Stream<Item = Result<String>>,
        F: FnMut(T, Result<String>) -> T,
    {
        self.inner
            .fold(init, |acc, item| std::future::ready(f(acc, item)))
            .await
    }

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
