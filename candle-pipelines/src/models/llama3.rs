use std::sync::Arc;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_pipelines_models::quantized_llama as candle_llama;
use minijinja::UndefinedBehavior;
use minijinja::{context, Environment};
use minijinja_contrib::{add_to_environment, pycompat};
use tokenizers::Tokenizer;

use crate::error::{PipelineError, Result};
use crate::loaders::GenerationConfig;
use crate::loaders::{GenerationConfigLoader, GgufModelLoader, HfLoader, TokenizerLoader};
use crate::pipelines::text_generation::model::{ModelCache, TextGenerationModel};

/// Available Llama 3.2 model sizes.
#[derive(Debug, Clone, Copy)]
pub enum Llama3Size {
    /// 1 billion parameters.
    Size1B,
    /// 3 billion parameters.
    Size3B,
}

impl Llama3Size {
    pub(crate) fn weight_repo_id(&self) -> &str {
        match self {
            Llama3Size::Size1B => "unsloth/Llama-3.2-1B-Instruct-GGUF",
            Llama3Size::Size3B => "unsloth/Llama-3.2-3B-Instruct-GGUF",
        }
    }

    pub(crate) fn weight_filename(&self) -> &str {
        match self {
            Llama3Size::Size1B => "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            Llama3Size::Size3B => "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        }
    }

    pub(crate) fn config_repo_id(&self) -> &str {
        match self {
            Llama3Size::Size1B => "meta-llama/Llama-3.2-1B-Instruct",
            Llama3Size::Size3B => "meta-llama/Llama-3.2-3B-Instruct",
        }
    }
}

impl std::fmt::Display for Llama3Size {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Llama3Size::Size1B => "llama3.2-1b",
            Llama3Size::Size3B => "llama3.2-3b",
        };
        write!(f, "{name}")
    }
}

impl crate::pipelines::cache::ModelOptions for Llama3Size {
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
}

/// Only for generic annotations. Use [`TextGenerationPipelineBuilder::llama3`](crate::text_generation::TextGenerationPipelineBuilder::llama3).
#[derive(Clone)]
pub struct Llama3 {
    weights: Arc<candle_llama::ModelWeights>,
    info: ModelInfo,
    tokenizer_repo_id: String,
    generation_config: GenerationConfig,
    chat_template_env: Arc<Environment<'static>>,
}

impl Llama3 {
    fn parse_metadata(content: &gguf_file::Content, device: &Device) -> Result<ModelInfo> {
        let num_layers = content
            .metadata
            .get("llama.block_count")
            .and_then(|v| v.to_u32().ok())
            .ok_or_else(|| {
                PipelineError::Unexpected(
                    "Missing 'llama.block_count' in Llama model metadata".to_string(),
                )
            })? as usize;

        let max_seq_len = content
            .metadata
            .get("llama.context_length")
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

        Ok(ModelInfo {
            num_layers,
            max_seq_len,
            dtype,
            device: device.clone(),
        })
    }

    fn parse_chat_template(
        tokenizer_config_path: std::path::PathBuf,
    ) -> Result<Arc<Environment<'static>>> {
        let tokenizer_config_content = std::fs::read_to_string(tokenizer_config_path)?;
        let config_json: serde_json::Value = serde_json::from_str(&tokenizer_config_content)?;

        let chat_template_str = config_json["chat_template"].as_str().ok_or_else(|| {
            PipelineError::Unexpected(
                "Missing 'chat_template' in tokenizer config for Llama3".to_string(),
            )
        })?;

        let mut env = Environment::new();
        env.set_undefined_behavior(UndefinedBehavior::Lenient);
        add_to_environment(&mut env);
        env.set_unknown_method_callback(pycompat::unknown_method_callback);
        env.add_filter("tojson", minijinja::filters::tojson);

        env.add_template_owned("chat", chat_template_str.to_string())
            .map_err(|e| {
                PipelineError::Unexpected(format!("Failed to parse chat template for Llama3: {e}"))
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
                "Missing 'eos_token_ids' in generation config for Llama3".to_string(),
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
        let weights = Arc::new(candle_llama::ModelWeights::from_gguf(
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

    pub(crate) fn from_hf(device: &Device, size: Llama3Size) -> Result<Self> {
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

    pub(crate) async fn from_hf_async(device: &Device, size: Llama3Size) -> Result<Self> {
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
impl ModelCache for candle_llama::Cache {
    fn reset(&mut self) {
        candle_llama::Cache::reset(self);
    }

    fn current_seq_len(&self) -> usize {
        candle_llama::Cache::current_seq_len(self)
    }
}

impl TextGenerationModel for Llama3 {
    type Options = Llama3Size;
    type Cache = candle_llama::Cache;

    fn new(options: Self::Options, device: Device) -> Result<Self> {
        Llama3::from_hf(&device, options)
    }

    async fn new_async(options: Self::Options, device: Device) -> Result<Self> {
        Llama3::from_hf_async(&device, options).await
    }

    fn get_tokenizer(&self) -> Result<Tokenizer> {
        Llama3::get_tokenizer(self)
    }

    fn apply_chat_template(
        &self,
        messages: &[crate::pipelines::text_generation::message::Message],
        tools: &[crate::pipelines::text_generation::tools::Tool],
    ) -> Result<String> {
        use crate::pipelines::text_generation::message::Role;

        let message_count = messages.len();

        // Llama uses "ipython" role for tool results, so we remap internally
        #[derive(serde::Serialize)]
        struct LlamaMessage<'a> {
            role: &'a str,
            content: &'a str,
        }

        let llama_messages: Vec<LlamaMessage> = messages
            .iter()
            .map(|m| LlamaMessage {
                role: match m.role() {
                    Role::Tool => "ipython", // Llama's tool result role
                    other => other.as_str(),
                },
                content: m.content(),
            })
            .collect();

        // Convert tools to JSON format expected by Llama template
        let tools_json: Option<Vec<serde_json::Value>> = if tools.is_empty() {
            None
        } else {
            Some(
                tools
                    .iter()
                    .map(|t| {
                        serde_json::json!({
                            "type": "function",
                            "function": {
                                "name": t.name(),
                                "description": t.description(),
                                "parameters": t.schema()
                            }
                        })
                    })
                    .collect(),
            )
        };

        let rendered = self
            .chat_template_env
            .get_template("chat")
            .map_err(|e| {
                PipelineError::Unexpected(format!("Failed to get chat template for Llama3: {e}"))
            })?
            .render(context! {
                messages => llama_messages,
                tools => tools_json,
                add_generation_prompt => true,
            })
            .map_err(|e| {
                PipelineError::Unexpected(format!(
                    "Failed to render template for Llama3 ({message_count} messages): {e}"
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
        self.weights.forward(input, cache)
    }

    fn default_generation_params(
        &self,
    ) -> crate::pipelines::text_generation::params::GenerationParams {
        crate::pipelines::text_generation::params::GenerationParams {
            temperature: self.generation_config.temperature.unwrap_or(0.6),
            repeat_penalty: self.generation_config.repeat_penalty.unwrap_or(1.1),
            repeat_last_n: self.generation_config.repeat_last_n.unwrap_or(64),
            seed: rand::random(),
            max_len: 2048,
            top_p: Some(self.generation_config.top_p.unwrap_or(0.9)),
            top_k: Some(self.generation_config.top_k.unwrap_or(50) as usize),
            min_p: Some(self.generation_config.min_p.unwrap_or(0.0)).filter(|v| *v > 0.0),
        }
    }
}

use crate::pipelines::text_generation::tools::ToolCalling;

impl ToolCalling for Llama3 {}
