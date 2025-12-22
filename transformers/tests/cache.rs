#![cfg(feature = "integration")]

use transformers::pipelines::cache::global_cache;
use transformers::pipelines::text_generation::*;
use transformers::pipelines::utils::DeviceSelectable;

#[tokio::test]
async fn pipelines_share_weights() -> transformers::Result<()> {
    global_cache().clear();

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

    assert_eq!(global_cache().len(), 1);

    let _ = pipelines[0].completion("Hello").await?;

    assert!(pipelines[0].context_position().await > 0);

    for p in pipelines.iter().skip(1) {
        assert_eq!(p.context_position().await, 0);
    }

    Ok(())
}
