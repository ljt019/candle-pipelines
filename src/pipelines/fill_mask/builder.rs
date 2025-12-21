use super::model::FillMaskModel;
use super::pipeline::FillMaskPipeline;
use crate::pipelines::cache::ModelOptions;
use crate::pipelines::utils::{
    BasePipelineBuilder, DeviceRequest, DeviceSelectable, StandardPipelineBuilder,
};
use crate::Result;

pub struct FillMaskPipelineBuilder<M: FillMaskModel>(StandardPipelineBuilder<M::Options>);

impl<M: FillMaskModel> FillMaskPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self(StandardPipelineBuilder::new(options))
    }
}

impl<M: FillMaskModel> DeviceSelectable for FillMaskPipelineBuilder<M> {
    fn device_request_mut(&mut self) -> &mut DeviceRequest {
        self.0.device_request_mut()
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

impl FillMaskPipelineBuilder<crate::models::modernbert::FillMaskModernBertModel> {
    pub fn modernbert(size: crate::models::ModernBertSize) -> Self {
        Self::new(size)
    }
}
