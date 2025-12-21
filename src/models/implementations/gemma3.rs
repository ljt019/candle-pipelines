use std::io::{Read, Seek};
use std::sync::Arc;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_transformers::models::quantized_gemma3 as candle_gemma3;
use minijinja::UndefinedBehavior;
use minijinja::{context, Environment};
use minijinja_contrib::{add_to_environment, pycompat};
use tokenizers::Tokenizer;
use tokio::fs;

use crate::core::GenerationConfig;
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
    pub fn weight_repo_id(&self) -> &str {
        match self {
            Gemma3Size::Size1B => "unsloth/gemma-3-1b-it-GGUF",
            Gemma3Size::Size4B => "unsloth/gemma-3-4b-it-GGUF",
            Gemma3Size::Size12B => "unsloth/gemma-3-12b-it-GGUF",
            Gemma3Size::Size27B => "unsloth/gemma-3-27b-it-GGUF",
        }
    }

    pub fn weight_filename(&self) -> &str {
        match self {
            Gemma3Size::Size1B => "gemma-3-1b-it-Q4_K_M.gguf",
            Gemma3Size::Size4B => "gemma-3-4b-it-Q4_K_M.gguf",
            Gemma3Size::Size12B => "gemma-3-12b-it-Q4_K_M.gguf",
            Gemma3Size::Size27B => "gemma-3-27b-it-Q4_K_M.gguf",
        }
    }

    pub fn config_repo_id(&self) -> &str {
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
    pub target_device: Option<String>,
}

#[derive(Clone)]
pub struct Gemma3Model {
    base_weights: Arc<candle_gemma3::ModelWeights>,
    info: ModelInfo,
    tokenizer_repo_id: String,
    generation_config: GenerationConfig,
    chat_template_env: Arc<Environment<'static>>,
}

impl Gemma3Model {
    fn parse_metadata(content: &gguf_file::Content, device: &Device) -> anyhow::Result<ModelInfo> {
        let num_layers = content
            .metadata
            .get("gemma3.block_count")
            .and_then(|v| v.to_u32().ok())
            .ok_or_else(|| anyhow::anyhow!("Missing critical metadata: gemma3.block_count"))?
            as usize;

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

    async fn load_chat_template_env(repo_id: &str) -> anyhow::Result<Arc<Environment<'static>>> {
        let tokenizer_config_loader = HfLoader::new(repo_id, "tokenizer_config.json");
        let tokenizer_config_path = tokenizer_config_loader.load().await?;
        let tokenizer_config_content = fs::read_to_string(tokenizer_config_path).await?;
        let config_json: serde_json::Value = serde_json::from_str(&tokenizer_config_content)?;

        let chat_template_str = config_json["chat_template"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'chat_template' field in tokenizer config"))?;

        let mut env = Environment::new();
        env.set_undefined_behavior(UndefinedBehavior::Lenient);
        add_to_environment(&mut env);
        env.set_unknown_method_callback(pycompat::unknown_method_callback);
        env.add_filter("tojson", minijinja::filters::tojson);

        env.add_template_owned("chat", chat_template_str.to_string())?;

        Ok(Arc::new(env))
    }

    fn eos_tokens(&self) -> Vec<u32> {
        self.generation_config
            .eos_token_ids
            .iter()
            .map(|&id| id as u32)
            .collect()
    }

    fn ensure_eos_tokens(config: &GenerationConfig) -> anyhow::Result<()> {
        if config.eos_token_ids.is_empty() {
            anyhow::bail!(
                "Gemma3 generation config is missing 'eos_token_ids'; cannot determine end-of-sequence tokens"
            );
        }

        Ok(())
    }

    pub async fn from_gguf<R: Read + Seek>(
        reader: &mut R,
        device: &Device,
        size: Gemma3Size,
    ) -> anyhow::Result<Self> {
        let content = gguf_file::Content::read(reader)?;
        let info = Self::parse_metadata(&content, device)?;
        let weights = Arc::new(candle_gemma3::ModelWeights::from_gguf(
            content, reader, device,
        )?);

        let tokenizer_repo_id = size.config_repo_id().to_string();
        let generation_config =
            GenerationConfigLoader::new(&tokenizer_repo_id, "generation_config.json")
                .load()
                .await?;
        Self::ensure_eos_tokens(&generation_config)?;
        let chat_template_env = Self::load_chat_template_env(&tokenizer_repo_id).await?;

        Ok(Self {
            base_weights: weights,
            info,
            tokenizer_repo_id,
            generation_config,
            chat_template_env,
        })
    }

    pub async fn from_hf(device: &Device, size: Gemma3Size) -> anyhow::Result<Self> {
        let loader = GgufModelLoader::new(size.weight_repo_id(), size.weight_filename());
        let (mut file, content) = loader.load().await?;
        let info = Self::parse_metadata(&content, device)?;
        let weights = Arc::new(candle_gemma3::ModelWeights::from_gguf(
            content, &mut file, device,
        )?);

        let tokenizer_repo_id = size.config_repo_id().to_string();
        let generation_config =
            GenerationConfigLoader::new(&tokenizer_repo_id, "generation_config.json")
                .load()
                .await?;
        Self::ensure_eos_tokens(&generation_config)?;
        let chat_template_env = Self::load_chat_template_env(&tokenizer_repo_id).await?;

        Ok(Self {
            base_weights: weights,
            info,
            tokenizer_repo_id,
            generation_config,
            chat_template_env,
        })
    }

    pub async fn get_tokenizer(&self) -> anyhow::Result<Tokenizer> {
        let tokenizer_loader = TokenizerLoader::new(&self.tokenizer_repo_id, "tokenizer.json");
        tokenizer_loader.load().await
    }

    pub fn create_context(&self) -> Context {
        Context::new((*self.base_weights).clone())
    }
}

pub struct Context {
    base_weights: candle_gemma3::ModelWeights,
    weights: candle_gemma3::ModelWeights,
    position: usize,
}

impl Context {
    pub fn new(weights: candle_gemma3::ModelWeights) -> Self {
        Self {
            base_weights: weights.clone(),
            weights,
            position: 0,
        }
    }

    pub fn reset_with(&mut self, weights: &candle_gemma3::ModelWeights) {
        self.base_weights = weights.clone();
        self.weights = weights.clone();
        self.position = 0;
    }

    pub fn generate(&mut self, input: &Tensor) -> CandleResult<Tensor> {
        let input = if input.dtype() != DType::I64 {
            input.to_dtype(DType::I64)?
        } else {
            input.clone()
        };
        let seq_len = input.dim(1)? as usize;
        let logits = self.weights.forward(&input, self.position)?;
        self.position += seq_len;
        Ok(logits)
    }
}

impl LanguageModelContext for Context {
    fn generate(&mut self, input: &Tensor) -> candle_core::Result<Tensor> {
        Context::generate(self, input)
    }

    fn reset(&mut self) {
        self.weights = self.base_weights.clone();
        self.position = 0;
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

    async fn new(options: Self::Options, device: Device) -> anyhow::Result<Self> {
        Gemma3Model::from_hf(&device, options).await
    }

    async fn get_tokenizer(&self) -> anyhow::Result<Tokenizer> {
        Gemma3Model::get_tokenizer(self).await
    }

    fn apply_chat_template(&self, messages: &[crate::Message]) -> anyhow::Result<String> {
        let rendered = self
            .chat_template_env
            .get_template("chat")?
            .render(context! {
                messages => messages,
                add_generation_prompt => true,
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

    fn new_context(&self) -> Context {
        self.create_context()
    }

    fn clear_context(&self, context: &mut Context) -> anyhow::Result<()> {
        context.reset_with(&self.base_weights);
        Ok(())
    }

    fn default_generation_params(&self) -> crate::models::generation::GenerationParams {
        crate::models::generation::GenerationParams {
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
