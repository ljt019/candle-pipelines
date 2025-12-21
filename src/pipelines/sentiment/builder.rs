use super::model::SentimentAnalysisModel;
use super::pipeline::SentimentAnalysisPipeline;
use crate::pipelines::cache::ModelOptions;
use crate::pipelines::utils::{
    BasePipelineBuilder, DeviceRequest, DeviceSelectable, StandardPipelineBuilder,
};
use crate::Result;

pub struct SentimentAnalysisPipelineBuilder<M: SentimentAnalysisModel>(
    StandardPipelineBuilder<M::Options>,
);

impl<M: SentimentAnalysisModel> SentimentAnalysisPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self(StandardPipelineBuilder::new(options))
    }
}

impl<M: SentimentAnalysisModel> DeviceSelectable for SentimentAnalysisPipelineBuilder<M> {
    fn device_request_mut(&mut self) -> &mut DeviceRequest {
        self.0.device_request_mut()
    }
}

impl<M: SentimentAnalysisModel> BasePipelineBuilder<M> for SentimentAnalysisPipelineBuilder<M>
where
    M: Clone + Send + Sync + 'static,
    M::Options: ModelOptions + Clone,
{
    type Model = M;
    type Pipeline = SentimentAnalysisPipeline<M>;
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
        Ok(SentimentAnalysisPipeline { model, tokenizer })
    }
}

impl SentimentAnalysisPipelineBuilder<crate::models::modernbert::SentimentModernBertModel> {
    pub fn modernbert(size: crate::models::ModernBertSize) -> Self {
        Self::new(size)
    }
}
