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
        .build()
        .await?;

    let out = pipeline.completion("Rust is a").await?;
    assert!(!out.trim().is_empty());
    Ok(())
}

#[tokio::test]
async fn text_generation_streaming() -> Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .cuda(0)
        .seed(42)
        .max_len(8)
        .build()
        .await?;

    let mut stream = pipeline.completion_stream("Hello").await?;
    let mut acc = String::new();
    while let Some(tok) = stream.next().await {
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
        .build()
        .await?;

    let short = pipeline.completion("Rust is a").await?;

    let mut new_params = GenerationParams::default();
    new_params.max_len = 8;
    pipeline.set_generation_params(new_params).await;

    let longer = pipeline.completion("Rust is a").await?;
    assert!(longer.len() >= short.len());
    Ok(())
}
