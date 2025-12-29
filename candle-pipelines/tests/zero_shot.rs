#![cfg(feature = "cuda")]

use candle_pipelines::error::Result;
use candle_pipelines::zero_shot::{ModernBertSize, ZeroShotClassificationPipelineBuilder};
use std::time::Instant;

#[test]
fn zero_shot_basic() -> Result<()> {
    let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda(0)
        .build()?;

    let labels = &["politics", "sports"];
    let output = pipeline.run("The election results were surprising", labels)?;
    assert_eq!(output.predictions.len(), 2);
    Ok(())
}

#[test]
fn zero_shot_batch_faster_than_sequential() -> Result<()> {
    let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda(0)
        .build()?;

    let texts: &[&str] = &[
        "The election results were surprising.",
        "The team won the championship game.",
        "New research shows promising results.",
        "The stock market reached new highs.",
        "The concert was absolutely amazing.",
        "The recipe calls for fresh ingredients.",
        "Climate change affects global weather.",
        "The movie broke box office records.",
    ];
    let labels = &["politics", "sports", "science", "business", "entertainment"];

    // Warmup
    let _ = pipeline.run(texts[0], labels);

    let start = Instant::now();
    let sequential_results: Vec<_> = texts.iter().map(|t| pipeline.run(*t, labels)).collect();
    let sequential_time = start.elapsed();

    let start = Instant::now();
    let batched_output = pipeline.run(texts, labels)?;
    let batched_time = start.elapsed();

    for (seq, batch) in sequential_results
        .into_iter()
        .zip(batched_output.predictions.into_iter())
    {
        let seq = &seq.unwrap().predictions[0];
        let batch = &batch.unwrap()[0];
        assert_eq!(
            seq.label, batch.label,
            "Top predicted label should match"
        );
    }

    assert!(
        batched_time < sequential_time,
        "Batching should be faster: batched={:?}, sequential={:?}",
        batched_time,
        sequential_time
    );

    Ok(())
}
