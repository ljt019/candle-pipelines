use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_transformers::models::quantized_qwen3 as candle_qwen3;
use minijinja::UndefinedBehavior;
use minijinja::{context, Environment};
use minijinja_contrib::{add_to_environment, pycompat};
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::error::{PipelineError, Result};

/// Available Qwen 3 model sizes.
#[derive(Debug, Clone, Copy)]
pub enum Qwen3Size {
    /// 0.6 billion parameters.
    Size0_6B,
    /// 1.7 billion parameters.
    Size1_7B,
    /// 4 billion parameters.
    Size4B,
    /// 8 billion parameters.
    Size8B,
    /// 14 billion parameters.
    Size14B,
    /// 32 billion parameters.
    Size32B,
}

impl Qwen3Size {
    pub(crate) fn to_id(self) -> (String, String) {
        match self {
            Qwen3Size::Size0_6B => (
                "unsloth/Qwen3-0.6B-GGUF".into(),
                "Qwen3-0.6B-Q4_K_M.gguf".into(),
            ),
            Qwen3Size::Size1_7B => (
                "unsloth/Qwen3-1.7B-GGUF".into(),
                "Qwen3-1.7B-Q4_K_M.gguf".into(),
            ),
            Qwen3Size::Size4B => (
                "unsloth/Qwen3-4B-GGUF".into(),
                "Qwen3-4B-Q4_K_M.gguf".into(),
            ),
            Qwen3Size::Size8B => (
                "unsloth/Qwen3-8B-GGUF".into(),
                "Qwen3-8B-Q4_K_M.gguf".into(),
            ),
            Qwen3Size::Size14B => (
                "unsloth/Qwen3-14B-GGUF".into(),
                "Qwen3-14B-Q4_K_M.gguf".into(),
            ),
            Qwen3Size::Size32B => (
                "unsloth/Qwen3-32B-GGUF".into(),
                "Qwen3-32B-Q4_K_M.gguf".into(),
            ),
        }
    }
}

impl std::fmt::Display for Qwen3Size {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Qwen3Size::Size0_6B => "qwen3-0.6b",
            Qwen3Size::Size1_7B => "qwen3-1.7b",
            Qwen3Size::Size4B => "qwen3-4b",
            Qwen3Size::Size8B => "qwen3-8b",
            Qwen3Size::Size14B => "qwen3-14b",
            Qwen3Size::Size32B => "qwen3-32b",
        };
        write!(f, "{name}")
    }
}

impl crate::pipelines::cache::ModelOptions for Qwen3Size {
    fn cache_key(&self) -> String {
        self.to_string()
    }
}

use crate::loaders::{GgufModelLoader, TokenizerLoader};

/// Only for generic annotations. Use [`TextGenerationPipelineBuilder::qwen3`](crate::text_generation::TextGenerationPipelineBuilder::qwen3).
pub struct Qwen3 {
    weights: Arc<candle_qwen3::ModelWeights>,
    info: ModelInfo,
    reasoning: std::sync::atomic::AtomicBool,
    generation_config: crate::loaders::GenerationConfig,
    chat_template_env: Arc<Environment<'static>>,
}

impl Clone for Qwen3 {
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            info: self.info.clone(),
            reasoning: std::sync::atomic::AtomicBool::new(
                self.reasoning.load(std::sync::atomic::Ordering::SeqCst),
            ),
            generation_config: self.generation_config.clone(),
            chat_template_env: self.chat_template_env.clone(),
        }
    }
}

impl Qwen3 {
    async fn load_chat_template_env() -> Result<Arc<Environment<'static>>> {
        let tokenizer_config_loader =
            crate::loaders::HfLoader::new("Qwen/Qwen3-0.6B", "tokenizer_config.json");

        let tokenizer_config_path = tokenizer_config_loader.load().await?;
        let tokenizer_config_content = std::fs::read_to_string(tokenizer_config_path)?;
        let config_json: serde_json::Value = serde_json::from_str(&tokenizer_config_content)?;

        let chat_template_str = config_json["chat_template"].as_str().ok_or_else(|| {
            PipelineError::Unexpected(
                "Missing 'chat_template' in tokenizer config for Qwen3".to_string(),
            )
        })?;

        let mut chat_template_owned = chat_template_str.to_string();

        chat_template_owned = chat_template_owned.replace("messages[::-1]", "messages|reverse");

        chat_template_owned = chat_template_owned.replace(
            "(messages|length - 1) - loop.index0",
            "((messages|length - 1)|int - loop.index0|int)",
        );

        chat_template_owned =
            chat_template_owned.replace("messages[-1]", "messages[(messages|length - 1)]");

        let mut env = Environment::new();
        env.set_undefined_behavior(UndefinedBehavior::Lenient);

        add_to_environment(&mut env);
        env.set_unknown_method_callback(pycompat::unknown_method_callback);

        env.add_filter("tojson", minijinja::filters::tojson);

        let chat_template_static = Box::leak(chat_template_owned.into_boxed_str());
        env.add_template("chat", chat_template_static)
            .map_err(|e| {
                PipelineError::Unexpected(format!("Failed to parse chat template for Qwen3: {e}"))
            })?;

        Ok(Arc::new(env))
    }

    pub(crate) async fn from_hf(device: &Device, size: Qwen3Size) -> Result<Self> {
        let (repo_id, file_name) = size.to_id();

        let model_loader = GgufModelLoader::new(&repo_id, &file_name);
        let (mut file, content) = model_loader.load().await?;

        let generation_config = crate::loaders::GenerationConfigLoader::new(
            "Qwen/Qwen3-0.6B",
            "generation_config.json",
        )
        .load()
        .await?;

        let num_layers = content
            .metadata
            .get("qwen3.block_count")
            .ok_or_else(|| {
                PipelineError::Unexpected(
                    "Missing 'qwen3.block_count' in Qwen3 model metadata".to_string(),
                )
            })?
            .to_u32()? as usize;
        let max_seq_len = content
            .metadata
            .get("qwen3.context_length")
            .ok_or_else(|| {
                PipelineError::Unexpected(
                    "Missing 'qwen3.context_length' in Qwen3 model metadata".to_string(),
                )
            })?
            .to_u32()? as usize;
        let dtype = match content.metadata.get("general.dtype") {
            Some(v) => match v.to_u32().unwrap_or(1) {
                0 => DType::F32,
                1 => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };
        let info = ModelInfo {
            num_layers,
            max_seq_len,
            dtype,
            device: device.clone(),
        };

        let weights = Arc::new(candle_qwen3::ModelWeights::from_gguf(
            content, &mut file, device,
        )?);
        let chat_template_env = Self::load_chat_template_env().await?;
        Ok(Self {
            weights,
            info,
            reasoning: std::sync::atomic::AtomicBool::new(true),
            generation_config,
            chat_template_env,
        })
    }

    pub(crate) async fn get_tokenizer(&self) -> Result<Tokenizer> {
        let tokenizer_loader = TokenizerLoader::new("Qwen/Qwen3-0.6B", "tokenizer.json");
        let tokenizer = tokenizer_loader.load().await?;
        Ok(tokenizer)
    }
}

pub struct Context {
    weights: candle_qwen3::ModelWeights,
    info: ModelInfo,
    position: usize,
}

impl Context {
    pub fn new(weights: Arc<candle_qwen3::ModelWeights>, info: ModelInfo) -> Self {
        let mut weights = (*weights).clone();
        weights.clear_kv_cache();
        Self {
            weights,
            info,
            position: 0,
        }
    }

    pub fn with_cache_size(
        weights: Arc<candle_qwen3::ModelWeights>,
        info: ModelInfo,
        _cache_size: usize,
    ) -> Self {
        Self::new(weights, info)
    }

    pub fn generate(&mut self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let seq_len = input_ids.dim(1)?;

        if self.position == 0 {
            self.weights.clear_kv_cache();
        }

        let logits = self.weights.forward(input_ids, self.position)?;
        self.position += seq_len;
        Ok(logits)
    }

    pub fn reset(&mut self) {
        self.weights.clear_kv_cache();
        self.position = 0;
    }

    pub fn current_position(&self) -> usize {
        self.position
    }

    pub fn set_position(&mut self, position: usize) {
        self.position = position;
    }

    pub fn info(&self) -> ModelInfo {
        self.info.clone()
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub dtype: DType,
    pub device: Device,
}

use crate::pipelines::text_generation::model::{
    LanguageModelContext, Reasoning, TextGenerationModel, ToggleableReasoning,
};
use crate::pipelines::text_generation::tools::ToolCalling;

impl LanguageModelContext for Context {
    fn generate(&mut self, input: &Tensor) -> candle_core::Result<Tensor> {
        Context::generate(self, input)
    }

    fn reset(&mut self) {
        Context::reset(self);
    }

    fn position(&self) -> usize {
        self.position
    }
}

impl TextGenerationModel for Qwen3 {
    type Context = Context;
    type Options = Qwen3Size;

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

    async fn new(options: Self::Options, device: candle_core::Device) -> Result<Self> {
        Qwen3::from_hf(&device, options).await
    }

    async fn get_tokenizer(&self) -> Result<tokenizers::Tokenizer> {
        Qwen3::get_tokenizer(self).await
    }

    fn apply_chat_template(
        &self,
        messages: &[crate::pipelines::text_generation::message::Message],
        tools: &[crate::pipelines::text_generation::tools::Tool],
    ) -> Result<String> {
        let mut enable_thinking = self.reasoning.load(std::sync::atomic::Ordering::SeqCst);
        if let Some(last_user_msg) = messages
            .iter()
            .rev()
            .find(|msg| msg.role() == &crate::text_generation::message::Role::User)
        {
            let content = last_user_msg.content();
            if content.contains("/think") {
                enable_thinking = true;
            } else if content.contains("/no_think") {
                enable_thinking = false;
            }
        }

        let messages_dicts: Vec<serde_json::Value> = messages
            .iter()
            .map(|msg| {
                let mut content = msg.content().to_string();
                if msg.role() == &crate::pipelines::text_generation::message::Role::User {
                    content = content
                        .replace("/think", "")
                        .replace("/no_think", "")
                        .trim()
                        .to_string();
                }
                serde_json::json!({
                    "role": msg.role().as_str(),
                    "content": content,
                })
            })
            .collect();

        let message_count = messages_dicts.len();

        let rendered = self
            .chat_template_env
            .get_template("chat")
            .map_err(|e| {
                PipelineError::Unexpected(format!("Failed to get chat template for Qwen3: {e}"))
            })?
            .render(context! {
                messages => messages_dicts,
                add_generation_prompt => true,
                enable_thinking => enable_thinking,
                tools => tools,
            })
            .map_err(|e| {
                PipelineError::Unexpected(format!(
                    "Failed to render template for Qwen3 ({message_count} messages): {e}"
                ))
            })?;

        Ok(rendered)
    }

    fn new_context(&self) -> Context {
        Context::new(self.weights.clone(), self.info.clone())
    }

    fn clear_context(&self, context: &mut Context) {
        context.reset();
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

impl Reasoning for Qwen3 {}

impl ToggleableReasoning for Qwen3 {
    fn enable_reasoning(&self, enable: bool) {
        self.reasoning
            .store(enable, std::sync::atomic::Ordering::SeqCst);
    }
}

impl ToolCalling for Qwen3 {}
