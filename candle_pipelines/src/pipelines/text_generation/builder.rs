use super::params::GenerationParams;
use super::tools::ErrorStrategy;
use crate::error::Result;
use crate::models::{Gemma3, Gemma3Size, Qwen3, Qwen3Size};
use crate::pipelines::cache::{global_cache, ModelOptions};
use crate::pipelines::utils::{build_cache_key, DeviceRequest};

use super::model::TextGenerationModel;
use super::parser::XmlParserBuilder;
use super::pipeline::TextGenerationPipeline;
use super::xml_pipeline::XmlTextGenerationPipeline;

crate::pipelines::utils::impl_device_methods!(direct: TextGenerationPipelineBuilder<M: TextGenerationModel>);

/// Builder for constructing [`TextGenerationPipeline`] instances.
///
/// # Example
///
/// ```rust,no_run
/// use candle_pipelines::text_generation::{TextGenerationPipelineBuilder, Qwen3Size};
///
/// # async fn example() -> candle_pipelines::error::Result<()> {
/// let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
///     .temperature(0.7)
///     .top_p(0.9)
///     .max_len(512)
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct TextGenerationPipelineBuilder<M: TextGenerationModel> {
    model_options: M::Options,
    gen_params: GenerationParams,
    device_request: DeviceRequest,
    tool_error_strategy: ErrorStrategy,
}

impl<M: TextGenerationModel> TextGenerationPipelineBuilder<M> {
    /// Create a builder with the given model options.
    pub fn new(options: M::Options) -> Self {
        Self {
            model_options: options,
            gen_params: GenerationParams::default(),
            device_request: DeviceRequest::Cpu,
            tool_error_strategy: ErrorStrategy::default(),
        }
    }

    /// Set how tool execution errors are handled.
    pub fn tool_error_strategy(mut self, strategy: ErrorStrategy) -> Self {
        self.tool_error_strategy = strategy;
        self
    }

    /// Set sampling temperature. 0.0 = deterministic, higher = more random.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.gen_params.temperature = temperature;
        self
    }

    /// Set penalty for repeating tokens. 1.0 = no penalty.
    pub fn repeat_penalty(mut self, repeat_penalty: f32) -> Self {
        self.gen_params.repeat_penalty = repeat_penalty;
        self
    }

    /// Alias for [`repeat_penalty`](Self::repeat_penalty).
    pub fn repetition_penalty(self, repetition_penalty: f32) -> Self {
        self.repeat_penalty(repetition_penalty)
    }

    /// Set how many recent tokens to consider for repeat penalty.
    pub fn repeat_last_n(mut self, repeat_last_n: usize) -> Self {
        self.gen_params.repeat_last_n = repeat_last_n;
        self
    }

    /// Set random seed for reproducible generation.
    pub fn seed(mut self, seed: u64) -> Self {
        self.gen_params.seed = seed;
        self
    }

    /// Set maximum tokens to generate per turn.
    pub fn max_len(mut self, max_len: usize) -> Self {
        self.gen_params.max_len = max_len;
        self
    }

    /// Alias for [`max_len`](Self::max_len).
    pub fn max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.gen_params.max_len = max_new_tokens;
        self
    }

    /// Set nucleus sampling threshold (0.0-1.0).
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.gen_params.top_p = Some(top_p.clamp(0.0, 1.0));
        self
    }

    /// Only sample from the top k most likely tokens.
    pub fn top_k(mut self, top_k: usize) -> Self {
        self.gen_params.top_k = Some(top_k);
        self
    }

    /// Filter tokens below min_p * max_probability (0.0-1.0).
    pub fn min_p(mut self, min_p: f64) -> Self {
        self.gen_params.min_p = Some(min_p.clamp(0.0, 1.0));
        self
    }

    /// Build the pipeline, downloading and loading the model if needed.
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

        TextGenerationPipeline::new(model, self.gen_params, device, self.tool_error_strategy).await
    }

    /// Build an XML-parsing pipeline that extracts specified tags from output.
    pub async fn build_xml(self, tags: &[&str]) -> Result<XmlTextGenerationPipeline<M>>
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
        XmlTextGenerationPipeline::new(
            model,
            self.gen_params,
            xml_parser,
            device,
            self.tool_error_strategy,
        )
        .await
    }
}

impl TextGenerationPipelineBuilder<Qwen3> {
    /// Create a builder for a Qwen 3 model.
    pub fn qwen3(size: Qwen3Size) -> Self {
        Self::new(size)
    }
}

impl TextGenerationPipelineBuilder<Gemma3> {
    /// Create a builder for a Gemma 3 model.
    pub fn gemma3(size: Gemma3Size) -> Self {
        Self::new(size)
    }
}
