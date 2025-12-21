# Add Llama 3.2 Model Support

## Summary
Add support for Llama 3.2 models (1B, 3B) to the text generation pipeline.

## Motivation
Llama 3.2 is one of the most popular open-source LLM families. Adding support expands model coverage and user choice.

## Implementation Pattern
Follow the exact same pattern as `src/models/gemma3.rs` and `src/models/qwen3.rs`:
- Thin wrapper around `candle_transformers::models::quantized_llama`
- MiniJinja for chat template (loaded from `tokenizer_config.json`)
- Generation params from `generation_config.json`
- Implement `TextGenerationModel` trait
- Implement `ToolCalling` trait (Llama 3.2 supports tool calling)

## Model Sources

### GGUF Weights (quantized)
- **1B**: `bartowski/Llama-3.2-1B-Instruct-GGUF` 
- **3B**: `bartowski/Llama-3.2-3B-Instruct-GGUF`

### Config/Tokenizer (from Meta)
- **1B**: `meta-llama/Llama-3.2-1B-Instruct`
- **3B**: `meta-llama/Llama-3.2-3B-Instruct`

## Key Tasks

1. **Create `src/models/llama.rs`** with:
   - `LlamaSize` enum (`Size1B`, `Size3B`)
   - `LlamaModel` struct wrapping `candle_transformers::models::quantized_llama::ModelWeights`
   - `Context` struct for KV cache management
   - `impl TextGenerationModel for LlamaModel`
   - `impl ToolCalling for LlamaModel`

2. **Chat template handling**:
   - Load from `meta-llama/Llama-3.2-{size}-Instruct` repo's `tokenizer_config.json`
   - Use MiniJinja (same as Gemma3/Qwen3)
   - Try clean first — only patch if MiniJinja has issues with Llama's template

3. **Generation params**:
   - Load from `generation_config.json` in Meta's repo
   - Use HF values with sensible fallbacks

4. **Update `src/models/mod.rs`**:
   - Add `pub mod llama;`
   - Export `LlamaModel`, `LlamaSize`

5. **Update `src/lib.rs`**:
   - Re-export `LlamaModel`, `LlamaSize`

6. **Add integration test**: `tests/text_generation.rs` — add Llama test case

## Reference Files
- `src/models/gemma3.rs` — primary reference for structure
- `src/models/qwen3.rs` — reference for tool calling implementation
- `candle/candle-transformers/src/models/quantized_llama.rs` — upstream implementation

## Notes
- Llama 3.2 only has 1B and 3B sizes (no larger instruct models in 3.2 series)
- Use Q4_K_M quantization like other models
- Llama 3.2 has native tool calling support — implement `ToolCalling` trait
