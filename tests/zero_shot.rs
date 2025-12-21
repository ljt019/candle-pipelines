//! Integration tests for zero shot classification pipeline
//! Run with: cargo test --features integration

#![cfg(feature = "integration")]

use std::time::Instant;
use transformers::pipelines::utils::{BasePipelineBuilder, DeviceSelectable};
use transformers::pipelines::zero_shot::*;

#[test]
fn zero_shot_basic() -> anyhow::Result<()> {
    let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda_device(0)
        .build()?;

    let labels = ["politics", "sports"];
    let res = pipeline.classify("The election results were surprising", &labels)?;
    assert_eq!(res.len(), 2);
    Ok(())
}

#[test]
fn zero_shot_batch_faster_than_sequential() -> anyhow::Result<()> {
    let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda_device(0)
        .build()?;

    let texts: Vec<&str> = vec![
        "The election results were surprising.",
        "The team won the championship game.",
        "New research shows promising results.",
        "The stock market reached new highs.",
        "The concert was absolutely amazing.",
        "The recipe calls for fresh ingredients.",
        "Climate change affects global weather.",
        "The movie broke box office records.",
    ];
    let labels = ["politics", "sports", "science", "business", "entertainment"];

    // Warmup
    let _ = pipeline.classify(texts[0], &labels);

    // Sequential
    let start = Instant::now();
    let sequential_results: Vec<_> = texts
        .iter()
        .map(|t| pipeline.classify(t, &labels))
        .collect();
    let sequential_time = start.elapsed();

    // Batched
    let start = Instant::now();
    let batched_results = pipeline.classify_batch(&texts, &labels)?;
    let batched_time = start.elapsed();

    // Verify correctness - top label should match
    for (seq, batch) in sequential_results.iter().zip(batched_results.iter()) {
        let seq = seq.as_ref().unwrap();
        let batch = batch.as_ref().unwrap();
        // Results are Vec<ClassificationResult>, compare top label (first element)
        assert_eq!(
            seq[0].label, batch[0].label,
            "Top predicted label should match"
        );
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
