//! High-performance Gemma3 implementation with quantization support.
//!
//! This implementation provides efficient inference for Gemma3 models with:
//! - Quantized weights for reduced memory usage
//! - KV caching for autoregressive generation
//! - Multi-context support for concurrent conversations
//! - GPU acceleration via Candle framework
//! - Sliding window attention patterns
//!

use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{kv_cache::KvCache, Embedding, Module};
use std::collections::HashMap;
use std::io::{Read, Seek};
use std::sync::Arc;

use crate::models::RmsNorm;

// Constants
const MAX_SEQ_LEN: usize = 131072;
const DEFAULT_CACHE_SIZE: usize = 64;
const KV_CACHE_DIMS: usize = 2;
const DEFAULT_SLIDING_WINDOW_TYPE: usize = 6;
const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
const DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;

/// Mask cache key for efficient caching
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct MaskCacheKey {
    seq_len: usize,
    index_pos: usize,
    sliding_window_size: Option<usize>,
}

/// Cached mask entry
#[derive(Debug, Clone)]
struct CachedMask {
    mask: Tensor,
}

impl CachedMask {
    fn get_mask(&self, target_dtype: DType, b_sz: usize) -> Result<Tensor> {
        let mask = if self.mask.dtype() != target_dtype {
            self.mask.to_dtype(target_dtype)?
        } else {
            self.mask.clone()
        };

        // Expand to batch size if needed
        if mask.dims()[0] != b_sz {
            let mask_dims = mask.dims();
            match mask_dims.len() {
                4 => mask.expand((b_sz, mask_dims[1], mask_dims[2], mask_dims[3])),
                3 => {
                    let expanded = mask.expand((b_sz, mask_dims[1], mask_dims[2]))?;
                    expanded.unsqueeze(1)
                }
                2 => {
                    let expanded = mask.expand((b_sz, mask_dims[0], mask_dims[1]))?;
                    expanded.unsqueeze(1)
                }
                _ => candle_core::bail!("Unsupported mask tensor shape: {:?}", mask_dims),
            }
        } else {
            Ok(mask)
        }
    }
}

/// Repeats key/value tensors for Grouped Query Attention.
fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }

    let (batch, num_kv_heads, seq_len, head_dim) = xs.dims4()?;
    Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((batch, num_kv_heads * n_rep, seq_len, head_dim))
}

/// Rotary Position Embedding (RoPE) implementation.
#[derive(Debug, Clone)]
struct RoPE {
    cos: Tensor,
    sin: Tensor,
}

impl RoPE {
    fn new(head_dim: usize, rope_frequency: f32, device: &Device) -> Result<Self> {
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_frequency.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0f32, MAX_SEQ_LEN as f32, device)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok(Self { sin, cos })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, position_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, position_offset, seq_len)?;
        let sin = self.sin.narrow(0, position_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

/// Feed-forward network with SwiGLU activation.
#[derive(Debug, Clone)]
struct FeedForward {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl FeedForward {
    fn load<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        layer_prefix: &str,
        device: &Device,
    ) -> Result<Self> {
        let gate_proj = QMatMul::from_qtensor(content.tensor(
            reader,
            &format!("{layer_prefix}.ffn_gate.weight"),
            device,
        )?)?;
        let up_proj = QMatMul::from_qtensor(content.tensor(
            reader,
            &format!("{layer_prefix}.ffn_up.weight"),
            device,
        )?)?;
        let down_proj = QMatMul::from_qtensor(content.tensor(
            reader,
            &format!("{layer_prefix}.ffn_down.weight"),
            device,
        )?)?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let silu = candle_nn::ops::silu(&gate)?;
        let gated = (silu * up)?;
        self.down_proj.forward(&gated)
    }
}

/// Multi-head attention with Grouped Query Attention and sliding window support.
#[derive(Debug, Clone)]
struct Attention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window_size: Option<usize>,
    rope: Arc<RoPE>,
}

impl Attention {
    fn load<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        layer_prefix: &str,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window_size: Option<usize>,
        rope: Arc<RoPE>,
        rms_eps: f64,
        device: &Device,
    ) -> Result<Self> {
        let q_proj = QMatMul::from_qtensor(content.tensor(
            reader,
            &format!("{layer_prefix}.attn_q.weight"),
            device,
        )?)?;
        let k_proj = QMatMul::from_qtensor(content.tensor(
            reader,
            &format!("{layer_prefix}.attn_k.weight"),
            device,
        )?)?;
        let v_proj = QMatMul::from_qtensor(content.tensor(
            reader,
            &format!("{layer_prefix}.attn_v.weight"),
            device,
        )?)?;
        let o_proj = QMatMul::from_qtensor(content.tensor(
            reader,
            &format!("{layer_prefix}.attn_output.weight"),
            device,
        )?)?;

        let q_norm = RmsNorm::from_qtensor(
            content.tensor(
                reader,
                &format!("{layer_prefix}.attn_q_norm.weight"),
                device,
            )?,
            rms_eps,
        )?;
        let k_norm = RmsNorm::from_qtensor(
            content.tensor(
                reader,
                &format!("{layer_prefix}.attn_k_norm.weight"),
                device,
            )?,
            rms_eps,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window_size,
            rope,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_offset: usize,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden_states.dims3()?;

        // Project to Q, K, V
        let queries = self
            .q_proj
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let keys = self
            .k_proj
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let values = self
            .v_proj
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply Q/K normalization
        let q_flat = queries.flatten(0, 2)?;
        let k_flat = keys.flatten(0, 2)?;
        let q_normed = self.q_norm.forward(&q_flat)?.reshape(queries.shape())?;
        let k_normed = self.k_norm.forward(&k_flat)?.reshape(keys.shape())?;

        // Apply RoPE
        let (queries, keys) = self.rope.apply(&q_normed, &k_normed, position_offset)?;

        // Update KV cache
        if position_offset == 0 {
            kv_cache.reset();
        }
        let (keys, values) = kv_cache.append(&keys.contiguous()?, &values.contiguous()?)?;

        // Expand KV for Grouped Query Attention
        let num_groups = self.num_heads / self.num_kv_heads;
        let keys = repeat_kv(keys, num_groups)?.contiguous()?;
        let values = repeat_kv(values, num_groups)?.contiguous()?;

        // Compute attention
        let scale = (self.head_dim as f64).sqrt().recip();
        let mut attention_scores = (queries.matmul(&keys.transpose(2, 3)?)? * scale)?;

        if let Some(mask) = attention_mask {
            attention_scores = attention_scores.broadcast_add(mask)?;
        }

        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;
        let context = attention_probs.matmul(&values)?;

        // Reshape and project output
        let output =
            context
                .transpose(1, 2)?
                .reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&output)
    }
}

/// Single transformer layer with pre-normalization.
#[derive(Debug, Clone)]
struct TransformerLayer {
    attention: Attention,
    feed_forward: FeedForward,
    attention_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,
}

impl TransformerLayer {
    fn load<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        layer_idx: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        sliding_window_size: Option<usize>,
        rope: Arc<RoPE>,
        rms_eps: f64,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        let attention = Attention::load(
            content,
            reader,
            &prefix,
            num_heads,
            num_kv_heads,
            head_dim,
            sliding_window_size,
            rope,
            rms_eps,
            device,
        )?;
        let feed_forward = FeedForward::load(content, reader, &prefix, device)?;

        let attention_norm = RmsNorm::from_qtensor(
            content.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
            rms_eps,
        )?;
        let post_attention_norm = RmsNorm::from_qtensor(
            content.tensor(
                reader,
                &format!("{prefix}.post_attention_norm.weight"),
                device,
            )?,
            rms_eps,
        )?;
        let ffn_norm = RmsNorm::from_qtensor(
            content.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
            rms_eps,
        )?;
        let post_ffn_norm = RmsNorm::from_qtensor(
            content.tensor(reader, &format!("{prefix}.post_ffw_norm.weight"), device)?,
            rms_eps,
        )?;

        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            post_attention_norm,
            ffn_norm,
            post_ffn_norm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_offset: usize,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        // Attention block
        let residual = hidden_states;
        let hidden_states = self.attention_norm.forward(hidden_states)?;
        let attention_out =
            self.attention
                .forward(&hidden_states, attention_mask, position_offset, kv_cache)?;
        let hidden_states = self.post_attention_norm.forward(&attention_out)?;
        let hidden_states = (hidden_states + residual)?;

        // Feed-forward block
        let residual = &hidden_states;
        let hidden_states = self.ffn_norm.forward(&hidden_states)?;
        let ffn_out = self.feed_forward.forward(&hidden_states)?;
        let hidden_states = self.post_ffn_norm.forward(&ffn_out)?;
        hidden_states + residual
    }
}

/// Shared model weights that can be used across multiple inference contexts.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    embeddings: Embedding,
    embedding_length: usize,
    layers: Vec<TransformerLayer>,
    final_norm: RmsNorm,
    output_projection: QMatMul,
    device: Device,
    dtype: DType,
    max_seq_len: usize,
}

impl ModelWeights {
    /// Load model weights from a GGUF file.
    pub fn from_gguf<R: Read + Seek>(
        content: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let get_metadata = |key: &str| -> Result<&gguf_file::Value> {
            content
                .metadata
                .get(key)
                .ok_or_else(|| candle_core::Error::Msg(format!("Missing metadata key: {key}")))
        };

        // Parse model configuration
        let num_heads = get_metadata("gemma3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = get_metadata("gemma3.attention.head_count_kv")?.to_u32()? as usize;
        let num_layers = get_metadata("gemma3.block_count")?.to_u32()? as usize;
        let embedding_length = get_metadata("gemma3.embedding_length")?.to_u32()? as usize;
        let head_dim = get_metadata("gemma3.attention.key_length")?.to_u32()? as usize;
        let rms_eps = get_metadata("gemma3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let sliding_window_size =
            get_metadata("gemma3.attention.sliding_window")?.to_u32()? as usize;

        let sliding_window_type = get_metadata("gemma3.attention.sliding_window_type")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(DEFAULT_SLIDING_WINDOW_TYPE);

        let rope_freq_base = get_metadata("gemma3.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);

        let rope_freq_base_sliding = get_metadata("gemma3.rope.local_freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SLIDING);

        let dtype = match content.metadata.get("general.dtype") {
            Some(v) => match v.to_u32().unwrap_or(1) {
                0 => DType::F32,
                1 => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        // Load embeddings
        let embed_tensor = content.tensor(reader, "token_embd.weight", device)?;
        let embeddings = Embedding::new(embed_tensor.dequantize(device)?, embedding_length);

        // Load final norm and output projection
        let final_norm = RmsNorm::from_qtensor(
            content.tensor(reader, "output_norm.weight", device)?,
            rms_eps,
        )?;

        let output_tensor = content
            .tensor(reader, "output.weight", device)
            .or_else(|_| content.tensor(reader, "token_embd.weight", device))?;
        let output_projection = QMatMul::from_qtensor(output_tensor)?;

        // Load transformer layers
        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            // Determine sliding window configuration for this layer
            let is_sliding = (layer_idx + 1) % sliding_window_type > 0;
            let layer_sliding_window = is_sliding.then_some(sliding_window_size);
            let layer_rope_frequency = if is_sliding {
                rope_freq_base_sliding
            } else {
                rope_freq_base
            };

            let rope = Arc::new(RoPE::new(head_dim, layer_rope_frequency, device)?);

            layers.push(TransformerLayer::load(
                &content,
                reader,
                layer_idx,
                num_heads,
                num_kv_heads,
                head_dim,
                layer_sliding_window,
                rope,
                rms_eps,
                device,
            )?);
        }

        Ok(Self {
            embeddings,
            embedding_length,
            layers,
            final_norm,
            output_projection,
            device: device.clone(),
            dtype,
            max_seq_len: MAX_SEQ_LEN,
        })
    }

    /// Create a causal attention mask with optional sliding window.
    fn create_causal_mask(
        &self,
        batch_size: usize,
        seq_len: usize,
        position_offset: usize,
        sliding_window_size: Option<usize>,
    ) -> Result<Tensor> {
        let total_len = seq_len + position_offset;

        // Create position indices
        let row_ids = Tensor::arange(0f32, seq_len as f32, &self.device)?.reshape((seq_len, 1))?;
        let col_ids =
            Tensor::arange(0f32, total_len as f32, &self.device)?.reshape((1, total_len))?;

        // Add position offset to rows
        let offset_tensor = Tensor::new(&[[position_offset as f32]], &self.device)?;
        let current_pos = row_ids.broadcast_add(&offset_tensor)?;

        // Create causal mask: j > (i + position_offset)
        let causal_condition = col_ids.broadcast_gt(&current_pos)?;
        let neg_inf_tensor = Tensor::full(f32::NEG_INFINITY, (seq_len, total_len), &self.device)?;
        let zeros_tensor = Tensor::zeros((seq_len, total_len), DType::F32, &self.device)?;
        let mut mask = causal_condition.where_cond(&neg_inf_tensor, &zeros_tensor)?;

        // Apply sliding window if specified
        if let Some(window_size) = sliding_window_size {
            let window_size_tensor = Tensor::new(&[[window_size as f32]], &self.device)?;
            // Sliding window condition: (i + position_offset) > (j + window_size)
            let sliding_condition =
                current_pos.broadcast_gt(&col_ids.broadcast_add(&window_size_tensor)?)?;
            let sliding_neg_inf =
                Tensor::full(f32::NEG_INFINITY, (seq_len, total_len), &self.device)?;
            let sliding_zeros = Tensor::zeros((seq_len, total_len), DType::F32, &self.device)?;
            let sliding_mask = sliding_condition.where_cond(&sliding_neg_inf, &sliding_zeros)?;
            mask = mask.maximum(&sliding_mask)?;
        }

        // Add batch and head dimensions
        mask.unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as(&[batch_size, 1, seq_len, total_len])
    }
}

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

use crate::loaders::{GgufModelLoader, TokenizerLoader};
use tokenizers::Tokenizer;

/// High-level Gemma3 model interface for text generation.
/// This struct manages the shared weights and creates individual contexts.
#[derive(Clone)]
pub struct Gemma3Model {
    weights: Arc<ModelWeights>,
    generation_config: crate::core::GenerationConfig,
    chat_template_env: Arc<Environment<'static>>,
}

impl Gemma3Model {
    /// Load and prepare the chat template environment
    async fn load_chat_template_env() -> anyhow::Result<Arc<Environment<'static>>> {
        // Load the tokenizer config and extract the chat template
        let tokenizer_config_loader =
            crate::loaders::HfLoader::new("google/gemma-3-1b-it", "tokenizer_config.json");

        let tokenizer_config_path = tokenizer_config_loader.load().await?;
        let tokenizer_config_content = std::fs::read_to_string(tokenizer_config_path)?;
        let config_json: serde_json::Value = serde_json::from_str(&tokenizer_config_content)?;

        let chat_template_str = config_json["chat_template"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'chat_template' field in tokenizer config"))?;

        // Build the MiniJinja environment
        let mut env = Environment::new();

        // Leak the string to get 'static lifetime - this is fine since we're storing it in the model
        let chat_template_static = Box::leak(chat_template_str.to_string().into_boxed_str());
        env.add_template("chat", chat_template_static)?;

        Ok(Arc::new(env))
    }
    /// Load a Gemma3 model from a GGUF file.
    pub async fn from_gguf<R: Read + Seek>(
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let content = gguf_file::Content::read(reader)?;
        let weights = Arc::new(ModelWeights::from_gguf(content, reader, device)?);
        let generation_config = crate::loaders::GenerationConfigLoader::new(
            "google/gemma-3-1b-it",
            "generation_config.json",
        )
        .load()
        .await?;
        let chat_template_env = Self::load_chat_template_env().await?;
        Ok(Self {
            weights,
            generation_config,
            chat_template_env,
        })
    }

    /// Load the model from hf
    pub async fn from_hf(device: &Device, size: Gemma3Size) -> anyhow::Result<Self> {
        let (repo_id, file_name) = size.to_id();

        // Download the model from hf
        let model_loader = GgufModelLoader::new(&repo_id, &file_name);
        let (mut file, content) = model_loader.load().await?;

        let weights = Arc::new(ModelWeights::from_gguf(content, &mut file, device)?);
        let generation_config = crate::loaders::GenerationConfigLoader::new(
            "google/gemma-3-1b-it",
            "generation_config.json",
        )
        .load()
        .await?;
        let chat_template_env = Self::load_chat_template_env().await?;
        Ok(Self {
            weights,
            generation_config,
            chat_template_env,
        })
    }

    /// Get the models suggested tokenizer
    pub async fn get_tokenizer(&self) -> anyhow::Result<Tokenizer> {
        let tokenizer_loader = TokenizerLoader::new("google/gemma-3-1b-it", "tokenizer.json");
        let tokenizer = tokenizer_loader.load().await?;
        Ok(tokenizer)
    }

    /// Create a new inference context with this model.
    /// Each context maintains its own KV cache and position tracking.
    pub fn new_context(&self) -> Context {
        Context::new(self.weights.clone())
    }

    /// Create a new context with custom KV cache size.
    pub fn new_context_with_cache_size(&self, cache_size: usize) -> Context {
        Context::with_cache_size(self.weights.clone(), cache_size)
    }

    /// Get model information.
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            num_layers: self.weights.layers.len(),
            max_seq_len: self.weights.max_seq_len,
            dtype: self.weights.dtype,
            device: self.weights.device.clone(),
        }
    }
}

/// A single inference context with independent state.
/// Multiple contexts can share the same model weights.
pub struct Context {
    weights: Arc<ModelWeights>,
    kv_caches: Vec<KvCache>,
    mask_cache: HashMap<MaskCacheKey, CachedMask>,
    position: usize,
}

impl Context {
    /// Create a new context with shared weights.
    pub fn new(weights: Arc<ModelWeights>) -> Self {
        let num_layers = weights.layers.len();
        let kv_caches = (0..num_layers)
            .map(|_| KvCache::new(KV_CACHE_DIMS, DEFAULT_CACHE_SIZE))
            .collect();

        Self {
            weights,
            kv_caches,
            mask_cache: HashMap::new(),
            position: 0,
        }
    }

    /// Create a new context with custom KV cache size.
    pub fn with_cache_size(weights: Arc<ModelWeights>, cache_size: usize) -> Self {
        let num_layers = weights.layers.len();
        let kv_caches = (0..num_layers)
            .map(|_| KvCache::new(KV_CACHE_DIMS, cache_size))
            .collect();

        Self {
            weights,
            kv_caches,
            mask_cache: HashMap::new(),
            position: 0,
        }
    }

    /// Get or create mask with caching
    fn get_cached_mask(
        &mut self,
        seq_len: usize,
        position_offset: usize,
        sliding_window_size: Option<usize>,
        target_dtype: DType,
        batch_size: usize,
    ) -> Result<Tensor> {
        // Clear mask cache when starting a new sequence
        if position_offset == 0 {
            self.mask_cache.clear();
        }

        let cache_key = MaskCacheKey {
            seq_len,
            index_pos: position_offset,
            sliding_window_size,
        };

        // Check if mask is in cache
        if let Some(cached) = self.mask_cache.get(&cache_key) {
            return cached.get_mask(target_dtype, batch_size);
        }

        // Create new mask
        let mask = self.weights.create_causal_mask(
            batch_size,
            seq_len,
            position_offset,
            sliding_window_size,
        )?;

        // Cache the mask
        let cached_mask = CachedMask { mask: mask.clone() };
        self.mask_cache.insert(cache_key, cached_mask);

        Ok(mask)
    }

    /// Generate next token logits given input token IDs.
    /// Position is tracked automatically within this context.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs with shape [batch_size, sequence_length]
    ///
    /// # Returns
    /// Logits for next token prediction with shape [batch_size, vocab_size]
    pub fn generate(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        // Use current position as offset
        let position_offset = self.position;

        // Reset caches if starting new sequence
        if position_offset == 0 {
            for kv_cache in &mut self.kv_caches {
                kv_cache.reset();
            }
            self.mask_cache.clear();
        }

        // Embed input tokens and scale
        let mut hidden_states = self.weights.embeddings.forward(input_ids)?;
        hidden_states = (hidden_states * (self.weights.embedding_length as f64).sqrt())?;

        // Forward pass through transformer layers
        for layer_idx in 0..self.weights.layers.len() {
            // First, fetch data we need from the layer **without** keeping a reference alive
            let sliding_window_size = self.weights.layers[layer_idx].attention.sliding_window_size;

            // Compute attention mask (needs &mut self)
            let attention_mask = if seq_len > 1 {
                Some(self.get_cached_mask(
                    seq_len,
                    position_offset,
                    sliding_window_size,
                    DType::F32,
                    batch_size,
                )?)
            } else {
                None
            };

            // Now safely borrow the layer immutably for the forward call.
            let layer = &self.weights.layers[layer_idx];

            hidden_states = layer.forward(
                &hidden_states,
                attention_mask.as_ref(),
                position_offset,
                &mut self.kv_caches[layer_idx],
            )?;
        }

        // Final normalization
        hidden_states = self.weights.final_norm.forward(&hidden_states)?;

        // Extract last token and compute logits
        let last_hidden = hidden_states.i((.., seq_len - 1, ..))?;
        let logits = self.weights.output_projection.forward(&last_hidden)?;

        // Update position after successful generation
        self.position += seq_len;

        Ok(logits)
    }

    /// Reset context state and position counter.
    pub fn reset(&mut self) {
        for cache in &mut self.kv_caches {
            cache.reset();
        }
        self.mask_cache.clear();
        self.position = 0;
    }

    /// Get current position in this context.
    pub fn current_position(&self) -> usize {
        self.position
    }

    /// Manually set position (for advanced use cases).
    pub fn set_position(&mut self, position: usize) {
        self.position = position;
    }

    /// Get model information.
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            num_layers: self.weights.layers.len(),
            max_seq_len: self.weights.max_seq_len,
            dtype: self.weights.dtype,
            device: self.weights.device.clone(),
        }
    }
}

/// Model information structure.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub dtype: DType,
    pub device: Device,
}

/*

Pipeline stuff

*/

use crate::pipelines::text_generation_pipeline::model::{
    LanguageModelContext, TextGenerationModel,
};

use minijinja::{context, Environment};

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
        // Check if we can continue from the given position
        // The cache is valid if the requested position matches our current position
        self.position == position
    }
}

impl TextGenerationModel for Gemma3Model {
    type Options = Gemma3Size;
    type Context = Context;

    async fn new(options: Self::Options) -> anyhow::Result<Self> {
        Gemma3Model::from_hf(&candle_core::Device::Cpu, options).await
    }

    async fn get_tokenizer(&self) -> anyhow::Result<Tokenizer> {
        Gemma3Model::get_tokenizer(self).await
    }

    fn apply_chat_template(&self, messages: &[crate::Message]) -> anyhow::Result<String> {
        // Render the template using the pre-loaded environment
        let rendered = self
            .chat_template_env
            .get_template("chat")?
            .render(context! {
                messages => messages,
                add_generation_prompt => true,
            })?;

        Ok(rendered)
    }

    fn get_eos_token(&self) -> u32 {
        // Return the first EOS token ID from the generation config
        self.generation_config.eos_token_ids[0] as u32
    }

    fn get_eos_tokens(&self) -> Vec<u32> {
        // Return all EOS token IDs for robust termination detection
        self.generation_config
            .eos_token_ids
            .iter()
            .map(|&id| id as u32)
            .collect()
    }

    fn get_max_seq_len(&self) -> usize {
        self.weights.max_seq_len
    }

    fn new_context(&self) -> Context {
        Context::new(self.weights.clone())
    }

    fn clear_context(&self, context: &mut Context) -> anyhow::Result<()> {
        context.reset();
        Ok(())
    }

    fn default_generation_params(&self) -> crate::models::generation::GenerationParams {
        // Recommended Gemma3 inference settings (confirmed with HF team)
        crate::models::generation::GenerationParams {
            temperature: self.generation_config.temperature.unwrap_or(1.0),
            repeat_penalty: self.generation_config.repeat_penalty.unwrap_or(1.15),
            repeat_last_n: self.generation_config.repeat_last_n.unwrap_or(64),
            seed: 42,
            // Gemma3 supports very long context, but keep a sane default
            max_len: 8192,
            sampling: crate::models::generation::params::SamplingParams {
                top_p: Some(self.generation_config.top_p.unwrap_or(0.95)),
                top_k: Some(self.generation_config.top_k.unwrap_or(64) as usize),
                min_p: Some(self.generation_config.min_p.unwrap_or(0.0)).filter(|v| *v > 0.0),
            },
        }
    }
}
