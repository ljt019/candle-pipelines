use super::params::GenerationParams;
use crate::models::{Gemma3Model, Gemma3Size, Qwen3Model, Qwen3Size};
use crate::pipelines::cache::{global_cache, ModelOptions};
use crate::pipelines::utils::{build_cache_key, DeviceRequest, DeviceSelectable};
use crate::Result;

use super::model::TextGenerationModel;
use super::parser::XmlParserBuilder;
use super::pipeline::TextGenerationPipeline;
use super::xml_pipeline::XmlGenerationPipeline;

/// Builder for text generation pipelines.
///
/// Generation parameters default to `None`, meaning the model's recommended
/// defaults (from `generation_config.json`) will be used. Any explicitly set
/// parameter overrides the model default.
pub struct TextGenerationPipelineBuilder<M: TextGenerationModel> {
    model_options: M::Options,
    device_request: DeviceRequest,
    // Individual params - None = use model default at build time
    temperature: Option<f64>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,
    seed: Option<u64>,
    max_len: Option<usize>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    min_p: Option<f64>,
}

impl<M: TextGenerationModel> TextGenerationPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self {
            model_options: options,
            device_request: DeviceRequest::Cpu,
            temperature: None,
            repeat_penalty: None,
            repeat_last_n: None,
            seed: None,
            max_len: None,
            top_p: None,
            top_k: None,
            min_p: None,
        }
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn repeat_penalty(mut self, repeat_penalty: f32) -> Self {
        self.repeat_penalty = Some(repeat_penalty);
        self
    }

    pub fn repetition_penalty(self, repetition_penalty: f32) -> Self {
        self.repeat_penalty(repetition_penalty)
    }

    pub fn repeat_last_n(mut self, repeat_last_n: usize) -> Self {
        self.repeat_last_n = Some(repeat_last_n);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn max_len(mut self, max_len: usize) -> Self {
        self.max_len = Some(max_len);
        self
    }

    pub fn max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_len = Some(max_new_tokens);
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p.clamp(0.0, 1.0));
        self
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn min_p(mut self, min_p: f64) -> Self {
        self.min_p = Some(min_p.clamp(0.0, 1.0));
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

        // Start with model's recommended defaults, override with user-specified values
        let defaults = model.default_generation_params();
        let params = GenerationParams {
            temperature: self.temperature.unwrap_or(defaults.temperature),
            repeat_penalty: self.repeat_penalty.unwrap_or(defaults.repeat_penalty),
            repeat_last_n: self.repeat_last_n.unwrap_or(defaults.repeat_last_n),
            seed: self.seed.unwrap_or_else(rand::random),
            max_len: self.max_len.unwrap_or(defaults.max_len),
            top_p: self.top_p.or(defaults.top_p),
            top_k: self.top_k.or(defaults.top_k),
            min_p: self.min_p.or(defaults.min_p),
        };

        TextGenerationPipeline::new(model, params, device).await
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

        // Start with model's recommended defaults, override with user-specified values
        let defaults = model.default_generation_params();
        let params = GenerationParams {
            temperature: self.temperature.unwrap_or(defaults.temperature),
            repeat_penalty: self.repeat_penalty.unwrap_or(defaults.repeat_penalty),
            repeat_last_n: self.repeat_last_n.unwrap_or(defaults.repeat_last_n),
            seed: self.seed.unwrap_or_else(rand::random),
            max_len: self.max_len.unwrap_or(defaults.max_len),
            top_p: self.top_p.or(defaults.top_p),
            top_k: self.top_k.or(defaults.top_k),
            min_p: self.min_p.or(defaults.min_p),
        };

        let mut builder = XmlParserBuilder::new();
        for tag in tags {
            builder.register_tag(*tag);
        }
        let xml_parser = builder.build();
        XmlGenerationPipeline::new(model, params, xml_parser, device).await
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
