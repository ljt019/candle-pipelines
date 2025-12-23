// ============ Internal API ============

pub(crate) mod loaders;
pub(crate) mod models;
pub(crate) mod pipelines;

// ============ Public API ============

pub mod error;

pub use pipelines::{fill_mask, sentiment, text_generation, zero_shot};
