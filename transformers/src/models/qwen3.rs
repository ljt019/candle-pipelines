use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_transformers::models::quantized_qwen3 as candle_qwen3;
use minijinja::UndefinedBehavior;
use minijinja::{context, Environment};
use minijinja_contrib::{add_to_environment, pycompat};
use std::io::{Read, Seek};
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::error::Result;
use crate::error::{ChatTemplateError, ModelMetadataError};

#[derive(Debug, Clone, Copy)]
pub enum Qwen3Size {
    Size0_6B,
    Size1_7B,
    Size4B,
    Size8B,
    Size14B,
    Size32B,
}

impl Qwen3Size {
    pub(crate) fn to_id(&self) -> (String, String) {
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

#[derive(Clone)]
pub struct Qwen3Model {
    weights: Arc<candle_qwen3::ModelWeights>,
    info: ModelInfo,
    reasoning: bool,
    generation_config: crate::loaders::GenerationConfig,
    tools: Vec<crate::pipelines::text_generation::tools::Tool>,
    chat_template_env: Arc<Environment<'static>>,
}

impl Qwen3Model {
    async fn load_chat_template_env() -> Result<Arc<Environment<'static>>> {
        let tokenizer_config_loader =
            crate::loaders::HfLoader::new("Qwen/Qwen3-0.6B", "tokenizer_config.json");

        let tokenizer_config_path = tokenizer_config_loader.load().await?;
        let tokenizer_config_content = std::fs::read_to_string(tokenizer_config_path)?;
        let config_json: serde_json::Value = serde_json::from_str(&tokenizer_config_content)?;

        let chat_template_str = config_json["chat_template"].as_str().ok_or_else(|| {
            ChatTemplateError::MissingTemplate {
                model: "Qwen3".into(),
            }
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
            .map_err(|e| ChatTemplateError::ParseFailed {
                model: "Qwen3".into(),
                reason: e.to_string(),
            })?;

        Ok(Arc::new(env))
    }
    pub async fn from_gguf<R: Read + Seek>(reader: &mut R, device: &Device) -> Result<Self> {
        let content = gguf_file::Content::read(reader)?;
        let available_keys: Vec<String> = content.metadata.keys().cloned().collect();

        let num_layers = content
            .metadata
            .get("qwen3.block_count")
            .ok_or_else(|| ModelMetadataError::MissingKey {
                key: "qwen3.block_count".into(),
                model_type: "Qwen3".into(),
                available: available_keys.clone(),
            })?
            .to_u32()? as usize;
        let max_seq_len = content
            .metadata
            .get("qwen3.context_length")
            .ok_or_else(|| ModelMetadataError::MissingKey {
                key: "qwen3.context_length".into(),
                model_type: "Qwen3".into(),
                available: available_keys.clone(),
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
            content, reader, device,
        )?);
        let generation_config = crate::loaders::GenerationConfigLoader::new(
            "Qwen/Qwen3-0.6B",
            "generation_config.json",
        )
        .load()
        .await?;
        let chat_template_env = Self::load_chat_template_env().await?;
        Ok(Self {
            weights,
            info,
            reasoning: true,
            generation_config,
            tools: Vec::new(),
            chat_template_env,
        })
    }

    pub async fn from_hf(device: &Device, size: Qwen3Size) -> Result<Self> {
        let (repo_id, file_name) = size.to_id();

        let model_loader = GgufModelLoader::new(&repo_id, &file_name);
        let (mut file, content) = model_loader.load().await?;

        let generation_config = crate::loaders::GenerationConfigLoader::new(
            "Qwen/Qwen3-0.6B",
            "generation_config.json",
        )
        .load()
        .await?;

        let available_keys: Vec<String> = content.metadata.keys().cloned().collect();

        let num_layers = content
            .metadata
            .get("qwen3.block_count")
            .ok_or_else(|| ModelMetadataError::MissingKey {
                key: "qwen3.block_count".into(),
                model_type: "Qwen3".into(),
                available: available_keys.clone(),
            })?
            .to_u32()? as usize;
        let max_seq_len = content
            .metadata
            .get("qwen3.context_length")
            .ok_or_else(|| ModelMetadataError::MissingKey {
                key: "qwen3.context_length".into(),
                model_type: "Qwen3".into(),
                available: available_keys.clone(),
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
            reasoning: true,
            generation_config,
            tools: Vec::new(),
            chat_template_env,
        })
    }

    pub async fn get_tokenizer(&self) -> Result<Tokenizer> {
        let tokenizer_loader = TokenizerLoader::new("Qwen/Qwen3-0.6B", "tokenizer.json");
        let tokenizer = tokenizer_loader.load().await?;
        Ok(tokenizer)
    }

    pub fn new_context(&self) -> Context {
        Context::new(self.weights.clone(), self.info.clone())
    }

    pub fn new_context_with_cache_size(&self, cache_size: usize) -> Context {
        Context::with_cache_size(self.weights.clone(), self.info.clone(), cache_size)
    }

    pub fn info(&self) -> ModelInfo {
        self.info.clone()
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
    LanguageModelContext, TextGenerationModel, ToggleableReasoning,
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

impl TextGenerationModel for Qwen3Model {
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
        Qwen3Model::from_hf(&device, options).await
    }

    async fn get_tokenizer(&self) -> Result<tokenizers::Tokenizer> {
        Qwen3Model::get_tokenizer(self).await
    }

    fn apply_chat_template(
        &self,
        messages: &[crate::pipelines::text_generation::message::Message],
    ) -> Result<String> {
        let mut enable_thinking = self.reasoning;
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
            .map_err(|e| ChatTemplateError::ParseFailed {
                model: "Qwen3".into(),
                reason: e.to_string(),
            })?
            .render(context! {
                messages => messages_dicts,
                add_generation_prompt => true,
                enable_thinking => enable_thinking,
                tools => self.registered_tools(),
            })
            .map_err(|e| ChatTemplateError::RenderFailed {
                model: "Qwen3".into(),
                message_count,
                reason: e.to_string(),
            })?;

        Ok(rendered)
    }

    fn new_context(&self) -> Context {
        Context::new(self.weights.clone(), self.info.clone())
    }

    fn clear_context(&self, context: &mut Context) -> Result<()> {
        context.reset();
        Ok(())
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

impl ToggleableReasoning for Qwen3Model {
    fn set_reasoning(&mut self, enable: bool) -> Result<()> {
        self.reasoning = enable;
        Ok(())
    }
}

use crate::pipelines::text_generation::tools::Tool;

impl ToolCalling for Qwen3Model {
    fn register_tool(&mut self, tool: Tool) -> Result<()> {
        if let Some(pos) = self.tools.iter().position(|t| t.name() == tool.name()) {
            self.tools[pos] = tool;
        } else {
            self.tools.push(tool);
        }
        Ok(())
    }

    fn unregister_tool(&mut self, name: &str) -> Result<()> {
        if let Some(pos) = self.tools.iter().position(|t| t.name() == name) {
            self.tools.remove(pos);
        }
        Ok(())
    }

    fn clear_tools(&mut self) -> Result<()> {
        self.tools.clear();
        Ok(())
    }

    fn registered_tools(&self) -> Vec<Tool> {
        self.tools.clone()
    }
}
