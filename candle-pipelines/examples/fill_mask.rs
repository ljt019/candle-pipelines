use candle_pipelines::error::Result;
use candle_pipelines::fill_mask::{FillMaskPipelineBuilder, ModernBertSize};

fn main() -> Result<()> {
    println!("Building pipeline...");

    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base).build()?;

    println!("Pipeline built successfully.");

    let text = "I love my [MASK] car.";

    // Single prediction - direct access!
    let output = pipeline.run(text)?;

    println!("\n=== Fill Mask Results ===");
    println!("Text: \"{}\"", text);
    println!(
        "Prediction: \"{}\" (confidence: {:.4})",
        output.prediction.token, output.prediction.score
    );
    println!(
        "Completed in {:.2}ms",
        output.stats.total_time.as_secs_f64() * 1000.0
    );

    // Top-k predictions - also direct access
    let output = pipeline.run_top_k(text, 3)?;

    println!("\nTop 3 predictions:");
    for (i, pred) in output.predictions.iter().enumerate() {
        println!(
            "  {}. \"{}\" (confidence: {:.4})",
            i + 1,
            pred.token,
            pred.score
        );
    }

    // Batch inference - results include input text!
    println!("\n=== Batch Inference ===");
    let texts = &[
        "The capital of France is [MASK].",
        "Water boils at 100 degrees [MASK].",
        "The [MASK] rises in the east.",
    ];

    let output = pipeline.run(texts)?;

    for r in output.results {
        let p = r.prediction?;
        println!("{} → \"{}\"", r.text, p.token);
    }

    Ok(())
}
