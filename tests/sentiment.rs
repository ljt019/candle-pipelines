//! Integration tests for sentiment analysis pipeline
//! Run with: cargo test --features integration

#![cfg(feature = "integration")]

use transformers::pipelines::sentiment::*;
use transformers::pipelines::utils::{BasePipelineBuilder, DeviceSelectable};

#[tokio::test]
async fn sentiment_basic() -> anyhow::Result<()> {
    let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda_device(0)
        .build()
        .await?;

    let res = pipeline.predict("I love Rust!")?;
    assert!(!res.label.trim().is_empty());
    assert!(res.score >= 0.0 && res.score <= 1.0);
    Ok(())
}
