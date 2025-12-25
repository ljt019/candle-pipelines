use super::model::FillMaskModel;
use super::pipeline::FillMaskPipeline;
use crate::error::Result;
use crate::pipelines::cache::ModelOptions;
use crate::pipelines::utils::{BasePipelineBuilder, DeviceRequest, StandardPipelineBuilder};

crate::pipelines::utils::impl_device_methods!(delegated: FillMaskPipelineBuilder<M: FillMaskModel>);

/// Builder for creating [`FillMaskPipeline`] instances.
///
/// Use [`Self::modernbert`] as the entry point.
///
/// # Examples
///
/// ```rust,no_run
/// # use candle_pipelines::fill_mask::{FillMaskPipelineBuilder, ModernBertSize};
/// # fn main() -> candle_pipelines::error::Result<()> {
/// let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base)
///     .cuda(0)
///     .build()?;
/// # Ok(())
/// # }
/// ```
pub struct FillMaskPipelineBuilder<M: FillMaskModel>(StandardPipelineBuilder<M::Options>);

impl<M: FillMaskModel> FillMaskPipelineBuilder<M> {
    pub(crate) fn new(options: M::Options) -> Self {
        Self(StandardPipelineBuilder::new(options))
    }

    /// Builds the pipeline with configured settings.
    ///
    /// # Errors
    ///
    /// Returns an error if model loading or device initialization fails.
    pub fn build(self) -> Result<FillMaskPipeline<M>>
    where
        M: Clone + Send + Sync + 'static,
        M::Options: ModelOptions + Clone,
    {
        BasePipelineBuilder::build(self)
    }
}

impl<M: FillMaskModel> BasePipelineBuilder<M> for FillMaskPipelineBuilder<M>
where
    M: Clone + Send + Sync + 'static,
    M::Options: ModelOptions + Clone,
{
    type Model = M;
    type Pipeline = FillMaskPipeline<M>;
    type Options = M::Options;

    fn options(&self) -> &Self::Options {
        &self.0.options
    }

    fn device_request(&self) -> &DeviceRequest {
        &self.0.device_request
    }

    fn create_model(options: Self::Options, device: candle_core::Device) -> Result<M> {
        M::new(options, device)
    }

    fn get_tokenizer(options: Self::Options) -> Result<tokenizers::Tokenizer> {
        M::get_tokenizer(options)
    }

    fn construct_pipeline(model: M, tokenizer: tokenizers::Tokenizer) -> Result<Self::Pipeline> {
        Ok(FillMaskPipeline { model, tokenizer })
    }
}

impl FillMaskPipelineBuilder<super::FillMaskModernBert> {
    /// Creates a builder for a ModernBERT fill-mask model.
    pub fn modernbert(size: crate::models::ModernBertSize) -> Self {
        Self::new(size)
    }
}
