use super::model::ZeroShotClassificationModel;
use super::pipeline::ZeroShotClassificationPipeline;
use crate::pipelines::cache::ModelOptions;
use crate::pipelines::utils::{
    BasePipelineBuilder, DeviceRequest, DeviceSelectable, StandardPipelineBuilder,
};

pub struct ZeroShotClassificationPipelineBuilder<M: ZeroShotClassificationModel>(
    StandardPipelineBuilder<M::Options>,
);

impl<M: ZeroShotClassificationModel> ZeroShotClassificationPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self(StandardPipelineBuilder::new(options))
    }
}

impl<M: ZeroShotClassificationModel> DeviceSelectable for ZeroShotClassificationPipelineBuilder<M> {
    fn device_request_mut(&mut self) -> &mut DeviceRequest {
        self.0.device_request_mut()
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

    fn create_model(options: Self::Options, device: candle_core::Device) -> anyhow::Result<M> {
        M::new(options, device)
    }

    fn get_tokenizer(options: Self::Options) -> anyhow::Result<tokenizers::Tokenizer> {
        M::get_tokenizer(options)
    }

    fn construct_pipeline(
        model: M,
        tokenizer: tokenizers::Tokenizer,
    ) -> anyhow::Result<Self::Pipeline> {
        Ok(ZeroShotClassificationPipeline { model, tokenizer })
    }
}

impl ZeroShotClassificationPipelineBuilder<crate::models::modernbert::ZeroShotModernBertModel> {
    pub fn modernbert(size: crate::models::ModernBertSize) -> Self {
        Self::new(size)
    }
}
