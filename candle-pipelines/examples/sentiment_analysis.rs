use candle_pipelines::error::Result;
use candle_pipelines::sentiment::{ModernBertSize, SentimentAnalysisPipelineBuilder};

fn main() -> Result<()> {
    println!("Building pipeline...");

    let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base).build()?;

    println!("Pipeline built successfully.");

    let text = "I love my new car";

    // Single text - direct access!
    let output = pipeline.run(text)?;

    println!("\n=== Sentiment Analysis Result ===");
    println!("Text: \"{}\"", text);
    println!(
        "Sentiment: {} (confidence: {:.4})",
        output.prediction.label, output.prediction.score
    );
    println!(
        "Completed in {:.2}ms",
        output.stats.total_time.as_secs_f64() * 1000.0
    );

    // Batch inference
    println!("\n=== Batch Inference ===");
    let texts = &[
        "This product is amazing!",
        "Terrible experience, would not recommend.",
        "It's okay, nothing special.",
    ];

    let output = pipeline.run(texts)?;

    for (text, pred) in texts.iter().zip(output.predictions) {
        let p = pred?;
        println!("{} → {} ({:.2})", text, p.label, p.score);
    }

    Ok(())
}
