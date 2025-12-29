#![cfg(feature = "cuda")]

use candle_pipelines::error::Result;
use candle_pipelines::sentiment::{ModernBertSize, SentimentAnalysisPipelineBuilder};
use std::time::Instant;

#[test]
fn sentiment_basic() -> Result<()> {
    let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda(0)
        .build()?;

    let output = pipeline.run("I love Rust!")?;
    assert!(!output.prediction.label.trim().is_empty());
    assert!(output.prediction.score >= 0.0 && output.prediction.score <= 1.0);
    Ok(())
}

#[test]
fn sentiment_batch_faster_than_sequential() -> Result<()> {
    let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda(0)
        .build()?;

    let texts: &[&str] = &[
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
    let _ = pipeline.run(texts[0]);

    let start = Instant::now();
    let sequential_results: Vec<_> = texts.iter().map(|t| pipeline.run(*t)).collect();
    let sequential_time = start.elapsed();

    let start = Instant::now();
    let batched_output = pipeline.run(texts)?;
    let batched_time = start.elapsed();

    for (seq, batch) in sequential_results
        .into_iter()
        .zip(batched_output.predictions.into_iter())
    {
        let seq = seq.unwrap().prediction;
        let batch = batch.unwrap();
        assert_eq!(seq.label, batch.label, "Labels should match");
    }

    assert!(
        batched_time < sequential_time,
        "Batching should be faster: batched={:?}, sequential={:?}",
        batched_time,
        sequential_time
    );

    Ok(())
}
