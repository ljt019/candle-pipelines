//! Patched OLMo-3 quantized implementation with external KV cache.
//!
//! Fixes from upstream candle-transformers:
//! - Uses external `Cache` struct instead of internal cache
//! - Model weights are `&self` (immutable, shareable across conversations)
//! - Each conversation gets its own `Cache` instance
//!
//! OLMo-3 specific features:
//! - Sliding window attention (mixed with full attention layers)
//! - QK-Norm (query/key normalization)
//! - 65K context window support
//! - Post-layer normalization
//!
//! References:
//! - [OLMo-3 Model Card](https://huggingface.co/allenai/Olmo-3-7B-Instruct)
//! - Architecture: olmo2 (OLMo-3 uses olmo2 architecture tag in GGUF)

use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Activation, Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;
use std::io::{Read, Seek};
use std::sync::Arc;

use candle_transformers::models::with_tracing::QMatMul;

pub const MAX_SEQ_LEN: usize = 131072; // 128K for safety margin
pub const DEFAULT_ROPE_THETA: f64 = 500000.0;
pub const DEFAULT_SLIDING_WINDOW: usize = 4096;
pub const DEFAULT_SLIDING_WINDOW_PATTERN: usize = 6; // Every 6th layer uses full attention

struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        QMatMul::from_weights(ws.into())
    }

    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }
}

#[derive(Debug, Clone)]
struct MlpWeights {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: Activation,
    span: tracing::Span,
}

impl MlpWeights {
    fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        let gate_proj = gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
        let up_proj = gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
        let down_proj = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        let act_fn = Activation::Silu;
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
            span,
        })
    }
}

impl Module for MlpWeights {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let gate = self.gate_proj.forward(x)?.apply(&self.act_fn)?;
        let up = self.up_proj.forward(x)?;
        let gated = (gate * up)?;
        self.down_proj.forward(&gated)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let dim = head_dim;
        let max_seq_len = max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct AttentionWeights {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    sliding_window_size: Option<usize>,
    rotary_emb: Arc<RotaryEmbedding>,
    span_attn: tracing::Span,
}

impl AttentionWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        sliding_window_size: Option<usize>,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let k_proj = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let v_proj = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;
        let o_proj = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        // OLMo-3 uses QK-Norm
        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;

        let span_attn = tracing::span!(tracing::Level::TRACE, "attn");

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            sliding_window_size,
            rotary_emb,
            span_attn,
        })
    }

    fn prepare_attention_mask(
        &self,
        b_sz: usize,
        seq_len: usize,
        offset: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let mask: Vec<_> = if let Some(window_size) = self.sliding_window_size {
            // Sliding window attention
            (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len + offset).map(move |j| {
                        let i_pos = i + offset;
                        // Allow attention to past tokens within window
                        if j <= i_pos && i_pos.saturating_sub(j) < window_size {
                            0f32
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                })
                .collect()
        } else {
            // Full causal attention
            (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len + offset).map(move |j| {
                        if j <= i + offset {
                            0f32
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                })
                .collect()
        };

        Tensor::from_slice(&mask, (seq_len, seq_len + offset), device)?
            .expand((b_sz, 1, seq_len, seq_len + offset))?
            .to_dtype(dtype)
    }

    fn forward(
        &self,
        x: &Tensor,
        offset: usize,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Apply QK-Norm before reshaping (weights are [hidden_size])
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // Use external cache
        let (k, v) = match kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if offset == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[&*k_cache, &k], 2)?;
                    let v = Tensor::cat(&[&*v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        *kv_cache = Some((k.clone(), v.clone()));

        // Repeat KV for GQA
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        // Generate per-layer attention mask (with sliding window if configured)
        if l > 1 {
            let attn_mask =
                self.prepare_attention_mask(b, l, offset, q.dtype(), &q.device())?;
            scores = scores.broadcast_add(&attn_mask)?;
        }

        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;
        let reshaped_ctx = ctx
            .transpose(1, 2)?
            .reshape((b, l, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&reshaped_ctx)
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    self_attn: AttentionWeights,
    mlp: MlpWeights,
    post_attention_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
}

impl LayerWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        sliding_window_size: Option<usize>,
        rotary_emb: Arc<RotaryEmbedding>,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        let post_attention_layernorm =
            gg.rms_norm(&format!("{prefix}.post_attention_norm.weight"), rms_norm_eps)?;
        let post_feedforward_layernorm =
            gg.rms_norm(&format!("{prefix}.post_ffw_norm.weight"), rms_norm_eps)?;

        let self_attn = AttentionWeights::new(
            gg,
            num_heads,
            num_kv_heads,
            head_dim,
            rms_norm_eps,
            sliding_window_size,
            rotary_emb,
            &prefix,
        )?;
        let mlp = MlpWeights::new(gg, &prefix)?;

        Ok(Self {
            self_attn,
            mlp,
            post_attention_layernorm,
            post_feedforward_layernorm,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        offset: usize,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        // OLMo-3 uses post-layer normalization
        let residual = x;
        let attn_out = self.self_attn.forward(x, offset, kv_cache)?;
        let attn_out = self.post_attention_layernorm.forward(&attn_out)?;
        let h = (residual + attn_out)?;

        let residual = &h;
        let mlp_out = self.mlp.forward(&h)?;
        let mlp_out = self.post_feedforward_layernorm.forward(&mlp_out)?;
        residual + mlp_out
    }
}

/// External KV cache for OLMo-3 - one per conversation.
#[derive(Debug, Clone)]
pub struct Cache {
    /// KV cache per layer: Vec<Option<(K, V)>>
    kvs: Vec<Option<(Tensor, Tensor)>>,
}

impl Cache {
    /// Create a new empty cache for the given number of layers.
    pub fn new(num_layers: usize) -> Self {
        Self {
            kvs: vec![None; num_layers],
        }
    }

    /// Reset the cache (clear all KV state).
    pub fn reset(&mut self) {
        for kv in &mut self.kvs {
            *kv = None;
        }
    }

    /// Get current sequence length from cache.
    pub fn current_seq_len(&self) -> usize {
        self.kvs
            .first()
            .and_then(|kv| kv.as_ref())
            .map(|(k, _)| k.dim(2).unwrap_or(0))
            .unwrap_or(0)
    }
}

/// Patched ModelWeights with external cache.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    num_layers: usize,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());

        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // OLMo-3 uses "olmo2" architecture tag in GGUF
        let num_heads = md_get("olmo2.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("olmo2.attention.head_count_kv")?.to_u32()? as usize;
        let num_layers = md_get("olmo2.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("olmo2.embedding_length")?.to_u32()? as usize;
        // head_dim is typically hidden_size / num_heads
        let head_dim = hidden_size / num_heads;
        let max_position_embeddings = md_get("olmo2.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("olmo2.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_theta = md_get("olmo2.rope.freq_base")
            .and_then(|v| Ok(v.to_f32()? as f64))
            .unwrap_or(DEFAULT_ROPE_THETA);

        // Check for sliding window configuration
        // OLMo-3 paper: SWA window = 4096, every 6th layer uses full attention
        let sliding_window_size = md_get("olmo2.attention.sliding_window")
            .and_then(|v| Ok(v.to_u32()? as usize))
            .unwrap_or(DEFAULT_SLIDING_WINDOW);

        // Sliding window pattern: some layers use SWA, some use full attention
        // Default: every 6th layer uses full attention (from OLMo-3 paper)
        let sliding_window_pattern = md_get("olmo2.attention.sliding_window_pattern")
            .and_then(|v| Ok(v.to_u32()? as usize))
            .unwrap_or(DEFAULT_SLIDING_WINDOW_PATTERN);

        let dtype = match gg.metadata().get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            max_position_embeddings,
            rope_theta,
            device,
        )?);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            // Apply sliding window to most layers, full attention periodically
            // Every Nth layer (where N=sliding_window_pattern) uses full attention
            let layer_sliding_window = if (i + 1) % sliding_window_pattern == 0 {
                None // Full attention layer
            } else {
                Some(sliding_window_size) // Sliding window layer
            };

            layers.push(LayerWeights::new(
                &mut gg,
                num_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                layer_sliding_window,
                rotary.clone(),
                i,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg.tensor("token_embd.weight")?, // Tied embeddings
        };
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            num_layers,
            span,
            span_output,
        })
    }

    /// Get number of layers (needed to create Cache).
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Create a new cache for this model.
    pub fn new_cache(&self) -> Cache {
        Cache::new(self.num_layers)
    }

    /// Forward pass with external cache.
    ///
    /// Model weights are `&self` (immutable). Cache is passed externally.
    pub fn forward(&self, input: &Tensor, cache: &mut Cache) -> Result<Tensor> {
        let _enter = self.span.enter();
        let offset = cache.current_seq_len();
        let (_b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, offset, &mut cache.kvs[layer_idx])?;
        }
        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();
        let last_hidden = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }
}
