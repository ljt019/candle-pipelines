use super::model::ZeroShotClassificationModel;
use super::pipeline::ZeroShotClassificationPipeline;
use crate::error::Result;
use crate::pipelines::cache::ModelOptions;
use crate::pipelines::utils::{BasePipelineBuilder, DeviceRequest, StandardPipelineBuilder};

crate::pipelines::utils::impl_device_methods!(delegated: ZeroShotClassificationPipelineBuilder<M: ZeroShotClassificationModel>);

pub struct ZeroShotClassificationPipelineBuilder<M: ZeroShotClassificationModel>(
    StandardPipelineBuilder<M::Options>,
);

impl<M: ZeroShotClassificationModel> ZeroShotClassificationPipelineBuilder<M> {
    pub(crate) fn new(options: M::Options) -> Self {
        Self(StandardPipelineBuilder::new(options))
    }

    pub fn build(self) -> Result<ZeroShotClassificationPipeline<M>>
    where
        M: Clone + Send + Sync + 'static,
        M::Options: ModelOptions + Clone,
    {
        BasePipelineBuilder::build(self)
    }
}

impl<M: ZeroShotClassificationModel> BasePipelineBuilder<M>
    for ZeroShotClassificationPipelineBuilder<M>
where
    M: Clone + Send + Sync + 'static,
    M::Options: ModelOptions + Clone,
{
    type Model = M;
    type Pipeline = ZeroShotClassificationPipeline<M>;
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
        Ok(ZeroShotClassificationPipeline { model, tokenizer })
    }
}

impl ZeroShotClassificationPipelineBuilder<crate::models::modernbert::ZeroShotModernBertModel> {
    pub fn modernbert(size: crate::models::ModernBertSize) -> Self {
        Self::new(size)
    }
}
