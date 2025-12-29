use std::sync::Arc;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_pipelines_models::quantized_gemma3 as candle_gemma3;
use minijinja::UndefinedBehavior;
use minijinja::{context, Environment};
use minijinja_contrib::{add_to_environment, pycompat};
use tokenizers::Tokenizer;

use crate::error::{PipelineError, Result};
use crate::loaders::GenerationConfig;
use crate::loaders::{GenerationConfigLoader, GgufModelLoader, HfLoader, TokenizerLoader};
use crate::pipelines::text_generation::model::{ModelCache, TextGenerationModel};

/// Available Gemma 3 model sizes.
#[derive(Debug, Clone, Copy)]
pub enum Gemma3Size {
    /// 1 billion parameters.
    Size1B,
    /// 4 billion parameters.
    Size4B,
    /// 12 billion parameters.
    Size12B,
    /// 27 billion parameters.
    Size27B,
}

impl Gemma3Size {
    pub(crate) fn weight_repo_id(&self) -> &str {
        match self {
            Gemma3Size::Size1B => "unsloth/gemma-3-1b-it-GGUF",
            Gemma3Size::Size4B => "unsloth/gemma-3-4b-it-GGUF",
            Gemma3Size::Size12B => "unsloth/gemma-3-12b-it-GGUF",
            Gemma3Size::Size27B => "unsloth/gemma-3-27b-it-GGUF",
        }
    }

    pub(crate) fn weight_filename(&self) -> &str {
        match self {
            Gemma3Size::Size1B => "gemma-3-1b-it-Q4_K_M.gguf",
            Gemma3Size::Size4B => "gemma-3-4b-it-Q4_K_M.gguf",
            Gemma3Size::Size12B => "gemma-3-12b-it-Q4_K_M.gguf",
            Gemma3Size::Size27B => "gemma-3-27b-it-Q4_K_M.gguf",
        }
    }

    pub(crate) fn config_repo_id(&self) -> &str {
        match self {
            Gemma3Size::Size1B => "google/gemma-3-1b-it",
            Gemma3Size::Size4B => "google/gemma-3-4b-it",
            Gemma3Size::Size12B => "google/gemma-3-12b-it",
            Gemma3Size::Size27B => "google/gemma-3-27b-it",
        }
    }
}

impl std::fmt::Display for Gemma3Size {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Gemma3Size::Size1B => "gemma3-1b",
            Gemma3Size::Size4B => "gemma3-4b",
            Gemma3Size::Size12B => "gemma3-12b",
            Gemma3Size::Size27B => "gemma3-27b",
        };
        write!(f, "{name}")
    }
}

impl crate::pipelines::cache::ModelOptions for Gemma3Size {
    fn cache_key(&self) -> String {
        self.to_string()
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub dtype: DType,
    pub device: Device,
    pub target_device: Option<String>,
}

/// Only for generic annotations. Use [`TextGenerationPipelineBuilder::gemma3`](crate::text_generation::TextGenerationPipelineBuilder::gemma3).
#[derive(Clone)]
pub struct Gemma3 {
    weights: Arc<candle_gemma3::ModelWeights>,
    info: ModelInfo,
    tokenizer_repo_id: String,
    generation_config: GenerationConfig,
    chat_template_env: Arc<Environment<'static>>,
}

impl Gemma3 {
    fn parse_metadata(content: &gguf_file::Content, device: &Device) -> Result<ModelInfo> {
        let num_layers = content
            .metadata
            .get("gemma3.block_count")
            .and_then(|v| v.to_u32().ok())
            .ok_or_else(|| {
                PipelineError::Unexpected(
                    "Missing 'gemma3.block_count' in Gemma3 model metadata".to_string(),
                )
            })? as usize;

        let max_seq_len = content
            .metadata
            .get("gemma3.context_length")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(131_072) as usize;

        let dtype = content
            .metadata
            .get("general.dtype")
            .and_then(|v| v.to_u32().ok())
            .and_then(|v| match v {
                0 => Some(DType::F32),
                1 => Some(DType::F16),
                _ => None,
            })
            .unwrap_or(DType::F16);

        let target_device = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned();

        Ok(ModelInfo {
            num_layers,
            max_seq_len,
            dtype,
            device: device.clone(),
            target_device,
        })
    }

    fn parse_chat_template(
        tokenizer_config_path: std::path::PathBuf,
    ) -> Result<Arc<Environment<'static>>> {
        let tokenizer_config_content = std::fs::read_to_string(tokenizer_config_path)?;
        let config_json: serde_json::Value = serde_json::from_str(&tokenizer_config_content)?;

        let chat_template_str = config_json["chat_template"].as_str().ok_or_else(|| {
            PipelineError::Unexpected(
                "Missing 'chat_template' in tokenizer config for Gemma3".to_string(),
            )
        })?;

        let mut env = Environment::new();
        env.set_undefined_behavior(UndefinedBehavior::Lenient);
        add_to_environment(&mut env);
        env.set_unknown_method_callback(pycompat::unknown_method_callback);
        env.add_filter("tojson", minijinja::filters::tojson);

        env.add_template_owned("chat", chat_template_str.to_string())
            .map_err(|e| {
                PipelineError::Unexpected(format!("Failed to parse chat template for Gemma3: {e}"))
            })?;

        Ok(Arc::new(env))
    }

    fn load_chat_template_env(repo_id: &str) -> Result<Arc<Environment<'static>>> {
        let tokenizer_config_loader = HfLoader::new(repo_id, "tokenizer_config.json");
        let tokenizer_config_path = tokenizer_config_loader.load()?;
        Self::parse_chat_template(tokenizer_config_path)
    }

    async fn load_chat_template_env_async(repo_id: &str) -> Result<Arc<Environment<'static>>> {
        let tokenizer_config_loader = HfLoader::new(repo_id, "tokenizer_config.json");
        let tokenizer_config_path = tokenizer_config_loader.load_async().await?;
        Self::parse_chat_template(tokenizer_config_path)
    }

    fn eos_tokens(&self) -> Vec<u32> {
        self.generation_config
            .eos_token_ids
            .iter()
            .map(|&id| id as u32)
            .collect()
    }

    fn ensure_eos_tokens(config: &GenerationConfig) -> Result<()> {
        if config.eos_token_ids.is_empty() {
            return Err(PipelineError::Unexpected(
                "Missing 'eos_token_ids' in generation config for Gemma3".to_string(),
            ));
        }

        Ok(())
    }

    fn build_model(
        device: &Device,
        mut file: std::fs::File,
        content: gguf_file::Content,
        tokenizer_repo_id: String,
        generation_config: GenerationConfig,
        chat_template_env: Arc<Environment<'static>>,
    ) -> Result<Self> {
        let info = Self::parse_metadata(&content, device)?;
        let weights = Arc::new(candle_gemma3::ModelWeights::from_gguf(
            content, &mut file, device,
        )?);

        Ok(Self {
            weights,
            info,
            tokenizer_repo_id,
            generation_config,
            chat_template_env,
        })
    }

    pub(crate) fn from_hf(device: &Device, size: Gemma3Size) -> Result<Self> {
        let loader = GgufModelLoader::new(size.weight_repo_id(), size.weight_filename());
        let (file, content) = loader.load()?;

        let tokenizer_repo_id = size.config_repo_id().to_string();
        let generation_config =
            GenerationConfigLoader::new(&tokenizer_repo_id, "generation_config.json").load()?;
        Self::ensure_eos_tokens(&generation_config)?;
        let chat_template_env = Self::load_chat_template_env(&tokenizer_repo_id)?;

        Self::build_model(
            device,
            file,
            content,
            tokenizer_repo_id,
            generation_config,
            chat_template_env,
        )
    }

    pub(crate) async fn from_hf_async(device: &Device, size: Gemma3Size) -> Result<Self> {
        let loader = GgufModelLoader::new(size.weight_repo_id(), size.weight_filename());
        let (file, content) = loader.load_async().await?;

        let tokenizer_repo_id = size.config_repo_id().to_string();
        let generation_config =
            GenerationConfigLoader::new(&tokenizer_repo_id, "generation_config.json")
                .load_async()
                .await?;
        Self::ensure_eos_tokens(&generation_config)?;
        let chat_template_env = Self::load_chat_template_env_async(&tokenizer_repo_id).await?;

        Self::build_model(
            device,
            file,
            content,
            tokenizer_repo_id,
            generation_config,
            chat_template_env,
        )
    }

    pub(crate) fn get_tokenizer(&self) -> Result<Tokenizer> {
        let tokenizer_loader = TokenizerLoader::new(&self.tokenizer_repo_id, "tokenizer.json");
        tokenizer_loader.load()
    }
}

// Implement ModelCache for the external cache type
impl ModelCache for candle_gemma3::Cache {
    fn reset(&mut self) {
        candle_gemma3::Cache::reset(self);
    }

    fn current_seq_len(&self) -> usize {
        candle_gemma3::Cache::current_seq_len(self)
    }
}

impl TextGenerationModel for Gemma3 {
    type Options = Gemma3Size;
    type Cache = candle_gemma3::Cache;

    fn new(options: Self::Options, device: Device) -> Result<Self> {
        Gemma3::from_hf(&device, options)
    }

    async fn new_async(options: Self::Options, device: Device) -> Result<Self> {
        Gemma3::from_hf_async(&device, options).await
    }

    fn get_tokenizer(&self) -> Result<Tokenizer> {
        Gemma3::get_tokenizer(self)
    }

    fn apply_chat_template(
        &self,
        messages: &[crate::pipelines::text_generation::message::Message],
        _tools: &[crate::pipelines::text_generation::tools::Tool],
    ) -> Result<String> {
        // Gemma3 doesn't support tool calling, so tools parameter is ignored
        let message_count = messages.len();

        let rendered = self
            .chat_template_env
            .get_template("chat")
            .map_err(|e| {
                PipelineError::Unexpected(format!("Failed to get chat template for Gemma3: {e}"))
            })?
            .render(context! {
                messages => messages,
                add_generation_prompt => true,
            })
            .map_err(|e| {
                PipelineError::Unexpected(format!(
                    "Failed to render template for Gemma3 ({message_count} messages): {e}"
                ))
            })?;
        Ok(rendered)
    }

    fn get_eos_token(&self) -> Option<u32> {
        self.eos_tokens().into_iter().next()
    }

    fn get_eos_tokens(&self) -> Vec<u32> {
        self.eos_tokens()
    }

    fn get_max_seq_len(&self) -> usize {
        self.info.max_seq_len
    }

    fn new_cache(&self) -> Self::Cache {
        self.weights.new_cache()
    }

    fn forward(&self, input: &Tensor, cache: &mut Self::Cache) -> candle_core::Result<Tensor> {
        let input = if input.dtype() != DType::I64 {
            input.to_dtype(DType::I64)?
        } else {
            input.clone()
        };
        self.weights.forward(&input, cache)
    }

    fn default_generation_params(
        &self,
    ) -> crate::pipelines::text_generation::params::GenerationParams {
        crate::pipelines::text_generation::params::GenerationParams {
            temperature: self.generation_config.temperature.unwrap_or(1.0),
            repeat_penalty: self.generation_config.repeat_penalty.unwrap_or(1.15),
            repeat_last_n: self.generation_config.repeat_last_n.unwrap_or(64),
            seed: rand::random(),
            max_len: 8192,
            top_p: Some(self.generation_config.top_p.unwrap_or(0.95)),
            top_k: Some(self.generation_config.top_k.unwrap_or(64) as usize),
            min_p: Some(self.generation_config.min_p.unwrap_or(0.0)).filter(|v| *v > 0.0),
        }
    }
}
