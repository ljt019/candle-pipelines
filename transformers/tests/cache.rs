#![cfg(feature = "integration")]

use transformers::error::Result;
use transformers::text_generation::{Qwen3Size, TextGenerationPipelineBuilder};

#[tokio::test]
async fn multiple_pipelines_work_independently() -> Result<()> {
    let mut pipelines = Vec::new();
    for _ in 0..3 {
        let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
            .cuda_device(0)
            .temperature(0.7)
            .max_len(10)
            .build()
            .await?;
        pipelines.push(pipeline);
    }

    // Use first pipeline
    let _ = pipelines[0].completion("Hello").await?;
    assert!(pipelines[0].context_position().await > 0);

    // Other pipelines should have independent context
    for p in pipelines.iter().skip(1) {
        assert_eq!(p.context_position().await, 0);
    }

    Ok(())
}
