//! Patched model implementations for candle-pipelines.
//!
//! This crate provides fixed versions of models from candle-transformers
//! where the upstream implementation has design issues.
//!
//! ## The Problem
//!
//! Upstream quantized models embed KV cache internally:
//! ```ignore
//! pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor>
//! ```
//!
//! This requires cloning entire model weights for each conversation.
//!
//! ## The Fix
//!
//! We use external cache (like non-quantized llama):
//! ```ignore
//! pub fn forward(&self, input: &Tensor, cache: &mut Cache) -> Result<Tensor>
//! ```
//!
//! Share `Arc<ModelWeights>` across conversations, each gets its own `Cache`.

pub mod quantized_gemma3;
pub mod quantized_qwen3;
