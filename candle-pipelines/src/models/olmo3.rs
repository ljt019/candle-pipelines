use candle_core::{Device, Tensor};
use candle_pipelines_models::quantized_olmo3 as candle_olmo3;
use minijinja::UndefinedBehavior;
use minijinja::{context, Environment};
use minijinja_contrib::{add_to_environment, pycompat};
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::error::{PipelineError, Result};

/// Available OLMo-3 model sizes.
#[derive(Debug, Clone, Copy)]
pub enum Olmo3Size {
    /// 7 billion parameters.
    Size7B,
    /// 32 billion parameters.
    Size32B,
}

impl Olmo3Size {
    pub(crate) fn to_id(self) -> (String, String) {
        match self {
            Olmo3Size::Size7B => (
                "unsloth/Olmo-3-7B-Instruct-GGUF".into(),
                "Olmo-3-7B-Instruct-Q4_K_M.gguf".into(),
            ),
            Olmo3Size::Size32B => (
                "unsloth/Olmo-3-32B-Instruct-GGUF".into(),
                "Olmo-3-32B-Instruct-Q4_K_M.gguf".into(),
            ),
        }
    }
}

impl std::fmt::Display for Olmo3Size {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Olmo3Size::Size7B => "olmo3-7b",
            Olmo3Size::Size32B => "olmo3-32b",
        };
        write!(f, "{name}")
    }
}

impl crate::pipelines::cache::ModelOptions for Olmo3Size {
    fn cache_key(&self) -> String {
        self.to_string()
    }
}

use crate::loaders::{GgufModelLoader, TokenizerLoader};

/// Only for generic annotations. Use [`TextGenerationPipelineBuilder::olmo3`](crate::text_generation::TextGenerationPipelineBuilder::olmo3).
pub struct Olmo3 {
    weights: Arc<candle_olmo3::ModelWeights>,
    info: ModelInfo,
    generation_config: crate::loaders::GenerationConfig,
    chat_template_env: Arc<Environment<'static>>,
}

impl Clone for Olmo3 {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            info: self.info.clone(),
            generation_config: self.generation_config.clone(),
            chat_template_env: self.chat_template_env.clone(),
        }
    }
}

impl Olmo3 {
    fn parse_chat_template(template_path: std::path::PathBuf) -> Result<Arc<Environment<'static>>> {
        let chat_template_str = std::fs::read_to_string(&template_path).map_err(|e| {
            PipelineError::Unexpected(format!(
                "Failed to read chat template file {}: {e}",
                template_path.display()
            ))
        })?;

        // Apply compatibility fixes for minijinja
        let mut chat_template_owned = chat_template_str;

        // Fix: selectattr with equalto test is not supported in minijinja
        // We'll compute has_system in Rust and pass it to the template
        chat_template_owned = chat_template_owned.replace(
            "{%- set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 -%}",
            "",  // Remove this line, we'll pass has_system from Rust
        );

        // Fix: .get('key', none) -> .key with default filter for safe access
        // Also fix "is not none" checks to work with undefined
        chat_template_owned = chat_template_owned
            .replace("message.get('content', none) is not none", "message.content")
            .replace("message.get('function_calls', none) is not none", "message.function_calls")
            .replace("message.get('tool_calls', none) is not none", "message.tool_calls")
            .replace("message.get('functions', none) is not none", "message.functions")
            .replace("tool_call.get('function', none) is not none", "tool_call.function")
            // Now replace remaining .get() calls with attribute access
            .replace(".get('content', none)", ".content")
            .replace(".get('function_calls', none)", ".function_calls")
            .replace(".get('tool_calls', none)", ".tool_calls")
            .replace(".get('functions', none)", ".functions")
            .replace(".get('function', none)", ".function");

        let mut env = Environment::new();
        env.set_undefined_behavior(UndefinedBehavior::Lenient);

        add_to_environment(&mut env);
        env.set_unknown_method_callback(pycompat::unknown_method_callback);

        env.add_filter("tojson", minijinja::filters::tojson);

        let chat_template_static = Box::leak(chat_template_owned.into_boxed_str());
        env.add_template("chat", chat_template_static)
            .map_err(|e| {
                PipelineError::Unexpected(format!(
                    "Failed to parse chat template for OLMo-3: {e}"
                ))
            })?;

        Ok(Arc::new(env))
    }

    fn load_chat_template_env() -> Result<Arc<Environment<'static>>> {
        let template_loader =
            crate::loaders::HfLoader::new("allenai/Olmo-3-7B-Instruct", "chat_template.jinja");
        let template_path = template_loader.load()?;
        Self::parse_chat_template(template_path)
    }

    async fn load_chat_template_env_async() -> Result<Arc<Environment<'static>>> {
        let template_loader =
            crate::loaders::HfLoader::new("allenai/Olmo-3-7B-Instruct", "chat_template.jinja");
        let template_path = template_loader.load_async().await?;
        Self::parse_chat_template(template_path)
    }

    fn build_model(
        device: &Device,
        mut file: std::fs::File,
        content: candle_core::quantized::gguf_file::Content,
        generation_config: crate::loaders::GenerationConfig,
        chat_template_env: Arc<Environment<'static>>,
    ) -> Result<Self> {
        let max_seq_len = content
            .metadata
            .get("olmo2.context_length")
            .ok_or_else(|| {
                PipelineError::Unexpected(
                    "Missing 'olmo2.context_length' in OLMo-3 model metadata".to_string(),
                )
            })?
            .to_u32()? as usize;
        let info = ModelInfo {
            max_seq_len,
            _device: device.clone(),
        };

        let weights = Arc::new(candle_olmo3::ModelWeights::from_gguf(
            content, &mut file, device,
        )?);
        Ok(Self {
            weights,
            info,
            generation_config,
            chat_template_env,
        })
    }

    pub(crate) fn from_hf(device: &Device, size: Olmo3Size) -> Result<Self> {
        let (repo_id, file_name) = size.to_id();

        let model_loader = GgufModelLoader::new(&repo_id, &file_name);
        let (file, content) = model_loader.load()?;

        let generation_config = crate::loaders::GenerationConfigLoader::new(
            "allenai/Olmo-3-7B-Instruct",
            "generation_config.json",
        )
        .load()?;

        let chat_template_env = Self::load_chat_template_env()?;
        Self::build_model(device, file, content, generation_config, chat_template_env)
    }

    pub(crate) async fn from_hf_async(device: &Device, size: Olmo3Size) -> Result<Self> {
        let (repo_id, file_name) = size.to_id();

        let model_loader = GgufModelLoader::new(&repo_id, &file_name);
        let (file, content) = model_loader.load_async().await?;

        let generation_config = crate::loaders::GenerationConfigLoader::new(
            "allenai/Olmo-3-7B-Instruct",
            "generation_config.json",
        )
        .load_async()
        .await?;

        let chat_template_env = Self::load_chat_template_env_async().await?;
        Self::build_model(device, file, content, generation_config, chat_template_env)
    }

    pub(crate) fn get_tokenizer(&self) -> Result<Tokenizer> {
        let tokenizer_loader =
            TokenizerLoader::new("allenai/Olmo-3-7B-Instruct", "tokenizer.json");
        tokenizer_loader.load()
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub max_seq_len: usize,
    pub _device: Device,
}

use crate::pipelines::text_generation::model::{ModelCache, TextGenerationModel};
use crate::pipelines::text_generation::tools::ToolCalling;

// Implement ModelCache for the external cache type
impl ModelCache for candle_olmo3::Cache {
    fn reset(&mut self) {
        candle_olmo3::Cache::reset(self);
    }

    fn current_seq_len(&self) -> usize {
        candle_olmo3::Cache::current_seq_len(self)
    }
}

impl TextGenerationModel for Olmo3 {
    type Cache = candle_olmo3::Cache;
    type Options = Olmo3Size;

    fn get_eos_token(&self) -> Option<u32> {
        self.generation_config
            .eos_token_ids
            .first()
            .copied()
            .map(|id| id as u32)
    }

    fn get_eos_tokens(&self) -> Vec<u32> {
        self.generation_config
            .eos_token_ids
            .iter()
            .map(|&id| id as u32)
            .collect()
    }

    fn get_max_seq_len(&self) -> usize {
        self.info.max_seq_len
    }

    fn new(options: Self::Options, device: candle_core::Device) -> Result<Self> {
        Olmo3::from_hf(&device, options)
    }

    async fn new_async(options: Self::Options, device: candle_core::Device) -> Result<Self> {
        Olmo3::from_hf_async(&device, options).await
    }

    fn get_tokenizer(&self) -> Result<tokenizers::Tokenizer> {
        Olmo3::get_tokenizer(self)
    }

    fn apply_chat_template(
        &self,
        messages: &[crate::pipelines::text_generation::message::Message],
        tools: &[crate::pipelines::text_generation::tools::Tool],
    ) -> Result<String> {
        // Tool instructions that OLMo-3 was trained with - must be present for reliable tool calling
        const TOOL_INSTRUCTIONS: &str = "You are provided with function signatures within <functions></functions> XML tags. You may call one or more functions to assist with the user query. Output any function calls within <function_calls></function_calls> XML tags. Do not make assumptions about what values to plug into functions.";

        let messages_dicts: Vec<serde_json::Value> = messages
            .iter()
            .map(|msg| {
                // If this is a system message AND tools are provided, append tool instructions
                // This ensures user's persona comes first, then technical tool-calling instructions
                let content = if msg.role().as_str() == "system" && !tools.is_empty() {
                    format!("{} {}", msg.content(), TOOL_INSTRUCTIONS)
                } else {
                    msg.content().to_string()
                };
                serde_json::json!({
                    "role": msg.role().as_str(),
                    "content": content,
                })
            })
            .collect();

        let message_count = messages_dicts.len();

        // Compute has_system in Rust (minijinja doesn't support selectattr with 3 args)
        let has_system = messages.iter().any(|m| m.role().as_str() == "system");

        // OLMo-3 supports function calling - pass tools with name, description, parameters
        let tools_json: Option<Vec<serde_json::Value>> = if tools.is_empty() {
            None
        } else {
            Some(
                tools
                    .iter()
                    .map(|t| serde_json::to_value(t).unwrap_or_default())
                    .collect(),
            )
        };

        let rendered = self
            .chat_template_env
            .get_template("chat")
            .map_err(|e| {
                PipelineError::Unexpected(format!("Failed to get chat template for OLMo-3: {e}"))
            })?
            .render(context! {
                messages => messages_dicts,
                add_generation_prompt => true,
                tools => tools_json,
                eos_token => "<|im_end|>",
                has_system => has_system,
            })
            .map_err(|e| {
                PipelineError::Unexpected(format!(
                    "Failed to render template for OLMo-3 ({message_count} messages): {e}"
                ))
            })?;

        Ok(rendered)
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
            top_p: Some(self.generation_config.top_p.unwrap_or(0.95)),
            top_k: Some(self.generation_config.top_k.unwrap_or(20) as usize),
            min_p: Some(self.generation_config.min_p.unwrap_or(0.0)).filter(|v| *v > 0.0),
        }
    }
}

impl ToolCalling for Olmo3 {}
