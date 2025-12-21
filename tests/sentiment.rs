//! Integration tests for sentiment analysis pipeline
//! Run with: cargo test --features integration

#![cfg(feature = "integration")]

use std::time::Instant;
use transformers::pipelines::sentiment::*;
use transformers::pipelines::utils::{BasePipelineBuilder, DeviceSelectable};

#[test]
fn sentiment_basic() -> anyhow::Result<()> {
    let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda_device(0)
        .build()?;

    let res = pipeline.predict("I love Rust!")?;
    assert!(!res.label.trim().is_empty());
    assert!(res.score >= 0.0 && res.score <= 1.0);
    Ok(())
}

#[test]
fn sentiment_batch_faster_than_sequential() -> anyhow::Result<()> {
    let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda_device(0)
        .build()?;

    let texts: Vec<&str> = vec![
        "I absolutely love this product!",
        "This is terrible, worst experience ever.",
        "The weather is nice today.",
        "Great service, highly recommend!",
        "Complete waste of money.",
        "Just an ordinary day.",
        "Fantastic movie!",
        "The staff was rude and unhelpful.",
    ];

    // Warmup
    let _ = pipeline.predict(texts[0]);

    // Sequential
    let start = Instant::now();
    let sequential_results: Vec<_> = texts.iter().map(|t| pipeline.predict(t)).collect();
    let sequential_time = start.elapsed();

    // Batched
    let start = Instant::now();
    let batched_results = pipeline.predict_batch(&texts)?;
    let batched_time = start.elapsed();

    // Verify correctness
    for (seq, batch) in sequential_results.iter().zip(batched_results.iter()) {
        let seq = seq.as_ref().unwrap();
        let batch = batch.as_ref().unwrap();
        assert_eq!(seq.label, batch.label, "Labels should match");
    }

    // Verify batching is faster
    assert!(
        batched_time < sequential_time,
        "Batching should be faster: batched={:?}, sequential={:?}",
        batched_time,
        sequential_time
    );

    Ok(())
}
