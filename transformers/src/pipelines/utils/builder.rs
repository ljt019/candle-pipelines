use super::{build_cache_key, DeviceRequest, DeviceSelectable};
use crate::pipelines::cache::{global_cache, ModelOptions};
use crate::Result;

pub trait BasePipelineBuilder<M>: DeviceSelectable + Sized
where
    M: Clone + Send + Sync + 'static,
{
    type Model: Clone + Send + Sync + 'static;
    type Pipeline;

    type Options: ModelOptions + Clone;

    fn options(&self) -> &Self::Options;

    fn device_request(&self) -> &DeviceRequest;

    fn create_model(options: Self::Options, device: candle_core::Device) -> Result<M>;

    fn get_tokenizer(options: Self::Options) -> Result<tokenizers::Tokenizer>;

    fn construct_pipeline(model: M, tokenizer: tokenizers::Tokenizer) -> Result<Self::Pipeline>;

    fn build(self) -> Result<Self::Pipeline> {
        let device = self.device_request().clone().resolve()?;

        let key = build_cache_key(self.options(), &device);

        let model = global_cache().get_or_create(&key, || {
            Self::create_model(self.options().clone(), device.clone())
        })?;

        let tokenizer = Self::get_tokenizer(self.options().clone())?;

        Self::construct_pipeline(model, tokenizer)
    }
}

pub struct StandardPipelineBuilder<Opts> {
    pub(crate) options: Opts,
    pub(crate) device_request: DeviceRequest,
}

impl<Opts> StandardPipelineBuilder<Opts> {
    pub fn new(options: Opts) -> Self {
        Self {
            options,
            device_request: DeviceRequest::Cpu,
        }
    }
}

impl<Opts> DeviceSelectable for StandardPipelineBuilder<Opts> {
    fn device_request_mut(&mut self) -> &mut DeviceRequest {
        &mut self.device_request
    }
}
