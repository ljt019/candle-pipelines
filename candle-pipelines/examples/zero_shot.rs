use candle_pipelines::error::Result;
use candle_pipelines::zero_shot::{ModernBertSize, ZeroShotClassificationPipelineBuilder};

fn main() -> Result<()> {
    println!("Building pipeline...");

    let pipeline =
        ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;

    println!("Pipeline built successfully.");

    let text = "I love my new car";
    let candidate_labels = &["coding", "reading", "writing", "speaking", "cars"];

    // Single-label classification - direct access to predictions!
    println!("\n=== Single-label Classification ===");
    println!("Probabilities sum to 1.0.\n");

    let output = pipeline.run(text, candidate_labels)?;

    println!("Text: \"{}\"", text);
    println!("Results:");
    for pred in &output.predictions {
        println!("  - {}: {:.4}", pred.label, pred.score);
    }
    println!(
        "Completed in {:.2}ms",
        output.stats.total_time.as_secs_f64() * 1000.0
    );

    // Verify probabilities sum to 1
    let sum: f32 = output.predictions.iter().map(|p| p.score).sum();
    println!("  Total probability: {:.4}\n", sum);

    // Multi-label classification
    println!("=== Multi-label Classification ===");
    println!("Independent probabilities.\n");

    let output = pipeline.run_multi_label(text, candidate_labels)?;

    println!("Text: \"{}\"", text);
    println!("Results:");
    for pred in &output.predictions {
        println!("  - {}: {:.4}", pred.label, pred.score);
    }

    // Batch inference - results include input text!
    println!("\n=== Batch Inference ===");
    let texts = &[
        "The team won the championship!",
        "New smartphone released today",
        "Senate passes new legislation",
    ];
    let labels = &["sports", "technology", "politics"];

    let output = pipeline.run(texts, labels)?;

    for r in output.results {
        let top = &r.predictions?[0];
        println!("{} → {} ({:.2})", r.text, top.label, top.score);
    }

    Ok(())
}
