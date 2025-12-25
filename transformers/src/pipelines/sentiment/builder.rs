use super::model::SentimentAnalysisModel;
use super::pipeline::SentimentAnalysisPipeline;
use crate::error::Result;
use crate::pipelines::cache::ModelOptions;
use crate::pipelines::utils::{BasePipelineBuilder, DeviceRequest, StandardPipelineBuilder};

crate::pipelines::utils::impl_device_methods!(delegated: SentimentAnalysisPipelineBuilder<M: SentimentAnalysisModel>);

/// Builder for creating [`SentimentAnalysisPipeline`] instances.
///
/// Use [`Self::modernbert`] as the entry point.
///
/// # Examples
///
/// ```rust,no_run
/// # use transformers::sentiment::{SentimentAnalysisPipelineBuilder, ModernBertSize};
/// # fn main() -> transformers::error::Result<()> {
/// let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base)
///     .cuda(0)
///     .build()?;
/// # Ok(())
/// # }
/// ```
pub struct SentimentAnalysisPipelineBuilder<M: SentimentAnalysisModel>(
    StandardPipelineBuilder<M::Options>,
);

impl<M: SentimentAnalysisModel> SentimentAnalysisPipelineBuilder<M> {
    pub(crate) fn new(options: M::Options) -> Self {
        Self(StandardPipelineBuilder::new(options))
    }

    /// Builds the pipeline with configured settings.
    ///
    /// # Errors
    ///
    /// Returns an error if model loading or device initialization fails.
    pub fn build(self) -> Result<SentimentAnalysisPipeline<M>>
    where
        M: Clone + Send + Sync + 'static,
        M::Options: ModelOptions + Clone,
    {
        BasePipelineBuilder::build(self)
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

impl SentimentAnalysisPipelineBuilder<super::SentimentModernBert> {
    /// Creates a builder for a ModernBERT sentiment analysis model.
    pub fn modernbert(size: crate::models::ModernBertSize) -> Self {
        Self::new(size)
    }
}
