//! Integration test for model switching.
//! Run with: cargo test --features cuda model_switching -- --nocapture

#[cfg(feature = "cuda")]
use candle_pipelines::error::Result;
#[cfg(feature = "cuda")]
use candle_pipelines::text_generation::{Gemma3Size, Qwen3Size, TextGenerationPipelineBuilder};

#[tokio::test]
#[cfg(feature = "cuda")]
async fn gemma_qwen_gemma_switch() -> Result<()> {
    let gemma1 = TextGenerationPipelineBuilder::gemma3(Gemma3Size::Size1B)
        .cuda(0)
        .max_len(50)
        .build_async()
        .await?;

    let response = gemma1.completion("Say hello in one word.").await?;
    println!("Gemma3 (1st): {}", response);

    drop(gemma1);

    let qwen = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .cuda(0)
        .max_len(50)
        .build_async()
        .await?;

    let response = qwen.completion("Say hello in one word.").await?;
    println!("Qwen3: {}", response);

    drop(qwen);

    let gemma2 = TextGenerationPipelineBuilder::gemma3(Gemma3Size::Size1B)
        .cuda(0)
        .max_len(50)
        .build_async()
        .await?;

    let response = gemma2.completion("Say hello in one word.").await?;
    println!("Gemma3 (2nd): {}", response);

    Ok(())
}

#[tokio::test]
#[cfg(feature = "cuda")]
async fn gemma_gemma_switch() -> Result<()> {
    let gemma1 = TextGenerationPipelineBuilder::gemma3(Gemma3Size::Size1B)
        .cuda(0)
        .max_len(50)
        .build_async()
        .await?;

    let response = gemma1.completion("Say hi.").await?;
    println!("Gemma3 (1st): {}", response);

    // Drop should now sync the CUDA stream before freeing resources
    drop(gemma1);

    let gemma2 = TextGenerationPipelineBuilder::gemma3(Gemma3Size::Size1B)
        .cuda(0)
        .max_len(50)
        .build_async()
        .await?;

    let response = gemma2.completion("Say hi.").await?;
    println!("Gemma3 (2nd): {}", response);

    Ok(())
}

#[tokio::test]
#[cfg(feature = "cuda")]
async fn qwen_qwen_switch() -> Result<()> {
    let qwen1 = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .cuda(0)
        .max_len(50)
        .build_async()
        .await?;

    let response = qwen1.completion("Say hi.").await?;
    println!("Qwen3 0.6B: {}", response);

    drop(qwen1);

    let qwen2 = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size4B)
        .cuda(0)
        .max_len(50)
        .build_async()
        .await?;

    let response = qwen2.completion("Say hi.").await?;
    println!("Qwen3 4B: {}", response);

    Ok(())
}

/// Tests reloading the EXACT same model (same cache key) after drop.
/// With weak refs, this forces a fresh reload into potentially reused GPU memory.
#[tokio::test]
#[cfg(feature = "cuda")]
async fn qwen_same_model_reload() -> Result<()> {
    let qwen1 = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .cuda(0)
        .max_len(50)
        .build_async()
        .await?;

    let response = qwen1.completion("Say hi.").await?;
    println!("Qwen3 0.6B (1st): {}", response);

    drop(qwen1);

    // Same exact model - with weak refs this triggers fresh reload
    let qwen2 = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .cuda(0)
        .max_len(50)
        .build_async()
        .await?;

    let response = qwen2.completion("Say hi.").await?;
    println!("Qwen3 0.6B (2nd): {}", response);

    Ok(())
}
