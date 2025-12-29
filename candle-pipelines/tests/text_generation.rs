#![cfg(feature = "cuda")]

use candle_pipelines::error::Result;
use candle_pipelines::text_generation::{
    GenerationParams, Qwen3Size, TextGenerationPipelineBuilder,
};

#[tokio::test]
async fn text_generation_basic() -> Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .cuda(0)
        .seed(42)
        .temperature(0.7)
        .max_len(8)
        .build_async()
        .await?;

    let out = pipeline.run("Rust is a")?;
    assert!(!out.text.trim().is_empty());
    Ok(())
}

#[tokio::test]
async fn text_generation_streaming() -> Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .cuda(0)
        .seed(42)
        .max_len(8)
        .build_async()
        .await?;

    let stream = pipeline.run_iter("Hello")?;
    let mut acc = String::new();
    for tok in stream {
        acc.push_str(&tok?);
    }
    assert!(!acc.trim().is_empty());
    Ok(())
}

#[tokio::test]
async fn text_generation_params_update() -> Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .cuda(0)
        .seed(42)
        .max_len(1)
        .build_async()
        .await?;

    let short = pipeline.run("Rust is a")?;

    let mut new_params = GenerationParams::default();
    new_params.max_len = 8;
    pipeline.set_generation_params(new_params);

    let longer = pipeline.run("Rust is a")?;
    assert!(longer.text.len() >= short.text.len());
    Ok(())
}
