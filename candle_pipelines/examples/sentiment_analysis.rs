use candle_pipelines::error::Result;
use candle_pipelines::sentiment::{ModernBertSize, SentimentAnalysisPipelineBuilder};

fn main() -> Result<()> {
    println!("Building pipeline...");

    let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base).build()?;

    println!("Pipeline built successfully.");

    let text = "I love my new car";

    let result = pipeline.predict(text)?;

    println!("\n=== Sentiment Analysis Result ===");
    println!("Text: \"{}\"", text);
    println!(
        "Sentiment: {} (confidence: {:.4})",
        result.label, result.score
    );

    Ok(())
}
