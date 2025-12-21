//! Integration tests for fill mask pipeline
//! Run with: cargo test --features integration

#![cfg(feature = "integration")]

use transformers::pipelines::fill_mask::*;
use transformers::pipelines::utils::{BasePipelineBuilder, DeviceSelectable};

#[tokio::test]
async fn fill_mask_basic() -> anyhow::Result<()> {
    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda_device(0)
        .build()
        .await?;

    let res = pipeline.predict("The capital of France is [MASK].")?;
    assert!(!res.word.trim().is_empty());
    assert!(res.score >= 0.0 && res.score <= 1.0);
    Ok(())
}

#[tokio::test]
async fn fill_mask_empty_input_errors() -> anyhow::Result<()> {
    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda_device(0)
        .build()
        .await?;

    assert!(pipeline.predict("").is_err());
    Ok(())
}
