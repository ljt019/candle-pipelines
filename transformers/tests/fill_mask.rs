#![cfg(feature = "integration")]

use std::time::Instant;
use transformers::error::Result;
use transformers::fill_mask::{FillMaskPipelineBuilder, ModernBertSize};

#[test]
fn fill_mask_basic() -> Result<()> {
    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda_device(0)
        .build()?;

    let res = pipeline.predict("The capital of France is [MASK].")?;
    assert!(!res.word.trim().is_empty());
    assert!(res.score >= 0.0 && res.score <= 1.0);
    Ok(())
}

#[test]
fn fill_mask_empty_input_errors() -> Result<()> {
    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda_device(0)
        .build()?;

    assert!(pipeline.predict("").is_err());
    Ok(())
}

#[test]
fn fill_mask_batch_faster_than_sequential() -> Result<()> {
    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda_device(0)
        .build()?;

    let texts: Vec<&str> = vec![
        "The capital of France is [MASK].",
        "Water boils at 100 degrees [MASK].",
        "The sun rises in the [MASK].",
        "Dogs are man's best [MASK].",
        "The largest planet is [MASK].",
        "Roses are red, violets are [MASK].",
        "The Earth orbits the [MASK].",
        "Coffee contains [MASK].",
    ];

    let _ = pipeline.predict(texts[0]);

    let start = Instant::now();
    let sequential_results: Vec<_> = texts.iter().map(|t| pipeline.predict(t)).collect();
    let sequential_time = start.elapsed();

    let start = Instant::now();
    let batched_results = pipeline.predict_batch(&texts)?;
    let batched_time = start.elapsed();

    for (seq, batch) in sequential_results.iter().zip(batched_results.iter()) {
        let seq = seq.as_ref().unwrap();
        let batch = batch.as_ref().unwrap();
        assert_eq!(seq.word, batch.word, "Predicted words should match");
    }

    assert!(
        batched_time < sequential_time,
        "Batching should be faster: batched={:?}, sequential={:?}",
        batched_time,
        sequential_time
    );

    Ok(())
}
