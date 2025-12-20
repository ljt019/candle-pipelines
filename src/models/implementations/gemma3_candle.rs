//! Gemma3 wrapper built on Candle's quantized implementation.
//!
//! This module delegates model execution to `candle_transformers::models::quantized_gemma3::ModelWeights`
//! and only keeps the repository-specific glue code for loading from the hub, device selection,
//! and pipeline integration.

use std::sync::Arc;

use candle_core::{quantized::gguf_file, DType, Device, Tensor};
use candle_transformers::models::quantized_gemma3 as candle_gemma3;
use minijinja::{context, Environment};
use tokenizers::Tokenizer;

use crate::loaders::{GenerationConfigLoader, GgufModelLoader, HfLoader, TokenizerLoader};
use crate::pipelines::text_generation_pipeline::model::{
    LanguageModelContext, TextGenerationModel,
};

#[derive(Debug, Clone, Copy)]
pub enum Gemma3Size {
    Size1B,
    Size4B,
    Size12B,
    Size27B,
}

impl Gemma3Size {
    pub fn to_id(&self) -> (String, String) {
        match self {
            Gemma3Size::Size1B => (
                "unsloth/gemma-3-1b-it-GGUF".into(),
                "gemma-3-1b-it-Q4_K_M.gguf".into(),
            ),
            Gemma3Size::Size4B => (
                "unsloth/gemma-3-4b-it-GGUF".into(),
                "gemma-3-4b-it-Q4_K_M.gguf".into(),
            ),
            Gemma3Size::Size12B => (
                "unsloth/gemma-3-12b-it-GGUF".into(),
                "gemma-3-12b-it-Q4_K_M.gguf".into(),
            ),
            Gemma3Size::Size27B => (
                "unsloth/gemma-3-27b-it-GGUF".into(),
                "gemma-3-27b-it-Q4_K_M.gguf".into(),
            ),
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

impl crate::core::ModelOptions for Gemma3Size {
    fn cache_key(&self) -> String {
        self.to_string()
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub dtype: DType,
    pub device: Device,
}

fn parse_metadata(content: &gguf_file::Content, device: &Device) -> anyhow::Result<ModelInfo> {
    let get_metadata = |key: &str| -> anyhow::Result<&gguf_file::Value> {
        content
            .metadata
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("Missing metadata key: {key}"))
    };

    let num_layers = get_metadata("gemma3.block_count")?.to_u32()? as usize;
    let dtype = match content.metadata.get("general.dtype") {
        Some(v) => match v.to_u32().unwrap_or(1) {
            0 => DType::F32,
            1 => DType::F16,
            _ => DType::F16,
        },
        None => DType::F16,
    };

    Ok(ModelInfo {
        num_layers,
        max_seq_len: candle_gemma3::MAX_SEQ_LEN,
        dtype,
        device: device.clone(),
    })
}

/// High-level Gemma3 model interface for text generation built on Candle.
#[derive(Clone)]
pub struct Gemma3Model {
    weights: Arc<candle_gemma3::ModelWeights>,
    generation_config: crate::core::GenerationConfig,
    chat_template_env: Arc<Environment<'static>>,
    info: ModelInfo,
    repo_id: String,
}

impl Gemma3Model {
    async fn load_chat_template_env(repo_id: &str) -> anyhow::Result<Arc<Environment<'static>>> {
        let tokenizer_config_loader = HfLoader::new(repo_id, "tokenizer_config.json");

        let tokenizer_config_path = tokenizer_config_loader.load().await?;
        let tokenizer_config_content = std::fs::read_to_string(tokenizer_config_path)?;
        let config_json: serde_json::Value = serde_json::from_str(&tokenizer_config_content)?;

        let chat_template_str = config_json["chat_template"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'chat_template' field in tokenizer config"))?;

        let mut env = Environment::new();
        let chat_template_static = Box::leak(chat_template_str.to_string().into_boxed_str());
        env.add_template("chat", chat_template_static)?;

        Ok(Arc::new(env))
    }

    pub async fn from_gguf<R: std::io::Read + std::io::Seek>(
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        const DEFAULT_REPO_ID: &str = "google/gemma-3-1b-it";

        let content = gguf_file::Content::read(reader)?;
        let info = parse_metadata(&content, device)?;
        let weights = Arc::new(candle_gemma3::ModelWeights::from_gguf(
            content, reader, device,
        )?);

        let generation_config =
            GenerationConfigLoader::new(DEFAULT_REPO_ID, "generation_config.json")
                .load()
                .await?;
        let chat_template_env = Self::load_chat_template_env(DEFAULT_REPO_ID).await?;

        Ok(Self {
            weights,
            generation_config,
            chat_template_env,
            info,
            repo_id: DEFAULT_REPO_ID.to_string(),
        })
    }

    pub async fn from_hf(device: &Device, size: Gemma3Size) -> anyhow::Result<Self> {
        let (repo_id, file_name) = size.to_id();
        let model_loader = GgufModelLoader::new(&repo_id, &file_name);
        let (mut file, content) = model_loader.load().await?;
        let info = parse_metadata(&content, device)?;
        let weights = Arc::new(candle_gemma3::ModelWeights::from_gguf(
            content, &mut file, device,
        )?);

        let generation_config = GenerationConfigLoader::new(&repo_id, "generation_config.json")
            .load()
            .await?;
        let chat_template_env = Self::load_chat_template_env(&repo_id).await?;

        Ok(Self {
            weights,
            generation_config,
            chat_template_env,
            info,
            repo_id,
        })
    }

    pub async fn get_tokenizer(&self) -> anyhow::Result<Tokenizer> {
        let tokenizer_loader = TokenizerLoader::new(&self.repo_id, "tokenizer.json");
        tokenizer_loader.load().await
    }

    pub fn new_context(&self) -> Context {
        Context::new(self.weights.clone(), self.info.clone())
    }

    pub fn new_context_with_cache_size(&self, _cache_size: usize) -> Context {
        // Candle manages KV cache internally; cloning the weights resets the cache state.
        Context::new(self.weights.clone(), self.info.clone())
    }

    pub fn info(&self) -> ModelInfo {
        self.info.clone()
    }
}

/// Per-inference context wrapping Candle's internal KV cache state.
pub struct Context {
    template: Arc<candle_gemma3::ModelWeights>,
    weights: candle_gemma3::ModelWeights,
    info: ModelInfo,
    position: usize,
}

impl Context {
    pub fn new(template: Arc<candle_gemma3::ModelWeights>, info: ModelInfo) -> Self {
        let weights = template.as_ref().clone();
        Self {
            template,
            weights,
            info,
            position: 0,
        }
    }

    pub fn generate(&mut self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        let logits = self.weights.forward(input_ids, self.position)?;
        let (_, seq_len) = input_ids.dims2()?;
        self.position += seq_len;
        Ok(logits)
    }

    pub fn reset(&mut self) {
        self.weights = self.template.as_ref().clone();
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

    fn can_continue_from(&self, position: usize) -> bool {
        self.position == position
    }
}

impl TextGenerationModel for Gemma3Model {
    type Options = Gemma3Size;
    type Context = Context;

    async fn new(options: Self::Options, device: candle_core::Device) -> anyhow::Result<Self> {
        Gemma3Model::from_hf(&device, options).await
    }

    async fn get_tokenizer(&self) -> anyhow::Result<Tokenizer> {
        Gemma3Model::get_tokenizer(self).await
    }

    fn apply_chat_template(&self, messages: &[crate::Message]) -> anyhow::Result<String> {
        let rendered = self
            .chat_template_env
            .get_template("chat")?
            .render(context! { messages => messages, add_generation_prompt => true, })?;
        Ok(rendered)
    }

    fn get_eos_token(&self) -> u32 {
        self.generation_config.eos_token_ids[0] as u32
    }

    fn get_eos_tokens(&self) -> Vec<u32> {
        self.generation_config
            .eos_token_ids
            .iter()
            .map(|&id| id as u32)
            .collect()
    }

    fn get_max_seq_len(&self) -> usize {
        candle_gemma3::MAX_SEQ_LEN
    }

    fn new_context(&self) -> Context {
        Gemma3Model::new_context(self)
    }

    fn clear_context(&self, context: &mut Context) -> anyhow::Result<()> {
        context.reset();
        Ok(())
    }

    fn default_generation_params(&self) -> crate::models::generation::GenerationParams {
        crate::models::generation::GenerationParams {
            temperature: self.generation_config.temperature.unwrap_or(1.0),
            repeat_penalty: self.generation_config.repeat_penalty.unwrap_or(1.15),
            repeat_last_n: self.generation_config.repeat_last_n.unwrap_or(64),
            seed: 42,
            max_len: 8192,
            top_p: self.generation_config.top_p.unwrap_or(0.95),
            top_k: self.generation_config.top_k.unwrap_or(64) as usize,
            min_p: self.generation_config.min_p.unwrap_or(0.0),
        }
    }
}
