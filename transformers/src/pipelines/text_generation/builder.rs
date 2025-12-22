use super::params::GenerationParams;
use crate::models::{Gemma3Model, Gemma3Size, Qwen3Model, Qwen3Size};
use crate::pipelines::cache::{global_cache, ModelOptions};
use crate::pipelines::utils::{build_cache_key, DeviceRequest, DeviceSelectable};
use crate::Result;

use super::model::TextGenerationModel;
use super::parser::XmlParserBuilder;
use super::pipeline::TextGenerationPipeline;
use super::xml_pipeline::XmlGenerationPipeline;

pub struct TextGenerationPipelineBuilder<M: TextGenerationModel> {
    model_options: M::Options,
    gen_params: GenerationParams,
    device_request: DeviceRequest,
}

impl<M: TextGenerationModel> TextGenerationPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self {
            model_options: options,
            gen_params: GenerationParams::default(),
            device_request: DeviceRequest::Cpu,
        }
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.gen_params.temperature = temperature;
        self
    }

    pub fn repeat_penalty(mut self, repeat_penalty: f32) -> Self {
        self.gen_params.repeat_penalty = repeat_penalty;
        self
    }

    pub fn repetition_penalty(self, repetition_penalty: f32) -> Self {
        self.repeat_penalty(repetition_penalty)
    }

    pub fn repeat_last_n(mut self, repeat_last_n: usize) -> Self {
        self.gen_params.repeat_last_n = repeat_last_n;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.gen_params.seed = seed;
        self
    }

    pub fn max_len(mut self, max_len: usize) -> Self {
        self.gen_params.max_len = max_len;
        self
    }

    pub fn max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.gen_params.max_len = max_new_tokens;
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.gen_params.top_p = Some(top_p.clamp(0.0, 1.0));
        self
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
        self.gen_params.top_k = Some(top_k);
        self
    }

    pub fn min_p(mut self, min_p: f64) -> Self {
        self.gen_params.min_p = Some(min_p.clamp(0.0, 1.0));
        self
    }

    pub async fn build(self) -> Result<TextGenerationPipeline<M>>
    where
        M: Clone + Send + Sync + 'static,
        M::Options: ModelOptions + Clone,
    {
        let device = self.device_request.resolve()?;
        let cache_key = build_cache_key(&self.model_options, &device);

        let options = self.model_options.clone();
        let device_for_model = device.clone();
        let model = global_cache()
            .get_or_create_async(&cache_key, || async move {
                M::new(options, device_for_model).await
            })
            .await?;

        TextGenerationPipeline::new(model, self.gen_params, device).await
    }

    pub async fn build_xml(self, tags: &[&str]) -> Result<XmlGenerationPipeline<M>>
    where
        M: Clone + Send + Sync + 'static,
        M::Options: ModelOptions + Clone,
    {
        let device = self.device_request.resolve()?;
        let cache_key = build_cache_key(&self.model_options, &device);

        let options = self.model_options.clone();
        let device_for_model = device.clone();
        let model = global_cache()
            .get_or_create_async(&cache_key, || async move {
                M::new(options, device_for_model).await
            })
            .await?;

        let mut builder = XmlParserBuilder::new();
        for tag in tags {
            builder.register_tag(*tag);
        }
        let xml_parser = builder.build();
        XmlGenerationPipeline::new(model, self.gen_params, xml_parser, device).await
    }
}

impl<M: TextGenerationModel> DeviceSelectable for TextGenerationPipelineBuilder<M> {
    fn device_request_mut(&mut self) -> &mut DeviceRequest {
        &mut self.device_request
    }
}

impl TextGenerationPipelineBuilder<Qwen3Model> {
    pub fn qwen3(size: Qwen3Size) -> Self {
        Self::new(size)
    }
}

impl TextGenerationPipelineBuilder<Gemma3Model> {
    pub fn gemma3(size: Gemma3Size) -> Self {
        Self::new(size)
    }
}
