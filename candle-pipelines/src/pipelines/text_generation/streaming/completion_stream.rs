use crate::error::Result;
use std::sync::{Arc, Mutex};

/// A streaming iterator over generated tokens.
///
/// This is a sync iterator - each call to `next()` blocks while generating the next token.
/// For web server integration, wrap with `tokio::task::spawn_blocking` or use
/// `tokio_stream::iter()`.
pub struct CompletionStream<I> {
    inner: I,
    stats: Arc<Mutex<crate::pipelines::text_generation::stats::GenerationStats>>,
}

impl<I> CompletionStream<I> {
    pub(crate) fn new(
        inner: I,
        stats: Arc<Mutex<crate::pipelines::text_generation::stats::GenerationStats>>,
    ) -> Self {
        Self { inner, stats }
    }

    /// Get current generation statistics.
    pub fn stats(&self) -> crate::pipelines::text_generation::stats::GenerationStats {
        self.stats.lock().unwrap().clone()
    }
}

impl<I> Iterator for CompletionStream<I>
where
    I: Iterator<Item = Result<String>>,
{
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<I> CompletionStream<I>
where
    I: Iterator<Item = Result<String>>,
{
    /// Collect all tokens into a single string.
    pub fn collect_string(self) -> Result<String> {
        let mut out = String::new();
        for chunk in self {
            out.push_str(&chunk?);
        }
        Ok(out)
    }

    /// Take up to n tokens.
    pub fn take_tokens(self, n: usize) -> impl Iterator<Item = Result<String>> {
        self.take(n)
    }

    /// Map over the token results.
    pub fn map_tokens<F, T>(self, f: F) -> impl Iterator<Item = T>
    where
        F: FnMut(Result<String>) -> T,
    {
        self.map(f)
    }

    /// Filter tokens based on a predicate.
    pub fn filter_tokens<F>(self, f: F) -> impl Iterator<Item = Result<String>>
    where
        F: FnMut(&Result<String>) -> bool,
    {
        self.filter(f)
    }
}
