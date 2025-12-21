//! Shared builder patterns for pipeline construction.
//!
//! This module provides common traits and implementations that eliminate code duplication
//! across different pipeline builders. Most pipeline builders follow a similar pattern:
//! 1. Store model options and device request
//! 2. Implement device selection methods via `DeviceSelectable`
//! 3. Build the pipeline by resolving device, creating model from cache, and getting tokenizer
//!
//! The `BasePipelineBuilder` trait captures this common pattern.

use super::{build_cache_key, DeviceRequest, DeviceSelectable};
use crate::pipelines::cache::{global_cache, ModelOptions};
use anyhow::Result;

/// A trait that captures the common pipeline building pattern.
///
/// Most pipeline builders follow these steps:
/// 1. Resolve the device request to an actual device
/// 2. Generate a cache key from model options and device
/// 3. Get or create the model from the global cache
/// 4. Get the tokenizer for the model
/// 5. Construct the final pipeline with model and tokenizer
///
/// This trait provides a default implementation of this pattern while allowing
/// customization through the associated methods.
pub trait BasePipelineBuilder<M>: DeviceSelectable + Sized
where
    M: Clone + Send + Sync + 'static,
{
    /// The model trait that this builder creates (e.g., `EmbeddingModel`, `FillMaskModel`)
    type Model: Clone + Send + Sync + 'static;

    /// The final pipeline type (e.g., `EmbeddingPipeline<M>`, `FillMaskPipeline<M>`)
    type Pipeline;

    /// The model options type
    type Options: ModelOptions + Clone;

    /// Get the model options from the builder
    fn options(&self) -> &Self::Options;

    /// Get the device request from the builder
    fn device_request(&self) -> &DeviceRequest;

    /// Create a new model instance with the given options and device.
    /// This is typically just `M::new(options, device)`.
    fn create_model(options: Self::Options, device: candle_core::Device) -> Result<M>;

    /// Get a tokenizer for the given options.
    /// This is typically just `M::get_tokenizer(options)`.
    fn get_tokenizer(options: Self::Options) -> Result<tokenizers::Tokenizer>;

    /// Construct the final pipeline from the model and tokenizer.
    /// This is where each pipeline type provides its specific construction logic.
    fn construct_pipeline(model: M, tokenizer: tokenizers::Tokenizer) -> Result<Self::Pipeline>;

    /// Build the pipeline using the common pattern.
    ///
    /// This method implements the standard pipeline building flow:
    /// 1. Resolve device from device request
    /// 2. Generate cache key from options and device
    /// 3. Get or create model from global cache
    /// 4. Get tokenizer
    /// 5. Construct final pipeline
    fn build(self) -> Result<Self::Pipeline> {
        // Resolve the device request
        let device = self.device_request().clone().resolve()?;

        // Generate cache key
        let key = build_cache_key(self.options(), &device);

        // Get or create model from cache
        let model = global_cache().get_or_create(&key, || {
            Self::create_model(self.options().clone(), device.clone())
        })?;

        // Get tokenizer
        let tokenizer = Self::get_tokenizer(self.options().clone())?;

        // Construct final pipeline
        Self::construct_pipeline(model, tokenizer)
    }
}

/// A standard pipeline builder struct that can be used by most pipeline types.
///
/// This struct implements the most common builder pattern with just model options
/// and device request. Pipeline types can use this directly or create their own
/// struct with additional fields and implement `BasePipelineBuilder` manually.
pub struct StandardPipelineBuilder<Opts> {
    pub(crate) options: Opts,
    pub(crate) device_request: DeviceRequest,
}

impl<Opts> StandardPipelineBuilder<Opts> {
    /// Create a new standard pipeline builder with the given options
    pub fn new(options: Opts) -> Self {
        Self {
            options,
            device_request: DeviceRequest::Default,
        }
    }
}

impl<Opts> DeviceSelectable for StandardPipelineBuilder<Opts> {
    fn device_request_mut(&mut self) -> &mut DeviceRequest {
        &mut self.device_request
    }
}
