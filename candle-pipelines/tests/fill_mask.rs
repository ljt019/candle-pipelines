#![cfg(feature = "cuda")]

use candle_pipelines::error::Result;
use candle_pipelines::fill_mask::{FillMaskPipelineBuilder, ModernBertSize};
use std::time::Instant;

#[test]
fn fill_mask_basic() -> Result<()> {
    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda(0)
        .build()?;

    let output = pipeline.run("The capital of France is [MASK].")?;
    assert!(!output.prediction.token.trim().is_empty());
    assert!(output.prediction.score >= 0.0 && output.prediction.score <= 1.0);
    Ok(())
}

#[test]
fn fill_mask_empty_input_errors() -> Result<()> {
    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda(0)
        .build()?;

    assert!(pipeline.run("").is_err());
    Ok(())
}

#[test]
fn fill_mask_batch_faster_than_sequential() -> Result<()> {
    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda(0)
        .build()?;

    let texts: &[&str] = &[
        "The capital of France is [MASK].",
        "Water boils at 100 degrees [MASK].",
        "The sun rises in the [MASK].",
        "Dogs are man's best [MASK].",
        "The largest planet is [MASK].",
        "Roses are red, violets are [MASK].",
        "The Earth orbits the [MASK].",
        "Coffee contains [MASK].",
    ];

    // Warmup
    let _ = pipeline.run(texts[0]);

    let start = Instant::now();
    let sequential_results: Vec<_> = texts.iter().map(|t| pipeline.run(*t)).collect();
    let sequential_time = start.elapsed();

    let start = Instant::now();
    let batched_output = pipeline.run(texts)?;
    let batched_time = start.elapsed();

    for (seq, batch) in sequential_results.into_iter().zip(batched_output.results) {
        let seq = seq.unwrap().prediction;
        let batch = batch.prediction.unwrap();
        assert_eq!(seq.token, batch.token, "Predicted words should match");
    }

    assert!(
        batched_time < sequential_time,
        "Batching should be faster: batched={:?}, sequential={:?}",
        batched_time,
        sequential_time
    );

    Ok(())
}
