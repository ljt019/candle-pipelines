//! Integration tests for zero shot classification pipeline
//! Run with: cargo test --features integration

#![cfg(feature = "integration")]

use transformers::pipelines::utils::{BasePipelineBuilder, DeviceSelectable};
use transformers::pipelines::zero_shot::*;

#[tokio::test]
async fn zero_shot_basic() -> anyhow::Result<()> {
    let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda_device(0)
        .build()
        .await?;

    let labels = ["politics", "sports"];
    let res = pipeline.classify("The election results were surprising", &labels)?;
    assert_eq!(res.len(), 2);
    Ok(())
}
