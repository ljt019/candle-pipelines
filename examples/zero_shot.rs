use anyhow::Result;
use transformers::pipelines::utils::BasePipelineBuilder;
use transformers::pipelines::zero_shot::*;

fn main() -> Result<()> {
    println!("Building pipeline...");

    let pipeline =
        ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;

    println!("Pipeline built successfully.");

    let text = "I love my new car";
    let candidate_labels = vec!["coding", "reading", "writing", "speaking", "cars"];

    // Single-label classification: probabilities sum to 1 (mutually exclusive)
    println!("\n=== Single-label Classification (classify) ===");
    println!("Use this when you want to classify text into one of several mutually exclusive categories.");
    println!("Probabilities will sum to 1.0, representing confidence that the text belongs to each category.\n");

    let single_label_result = pipeline.classify(text, &candidate_labels)?;
    println!("Text: \"{}\"", text);
    println!("Single-label results:");
    for result in &single_label_result {
        println!("  - {}: {:.4}", result.label, result.score);
    }

    // Verify probabilities sum to 1
    let sum: f32 = single_label_result.iter().map(|r| r.score).sum();
    println!("  Total probability: {:.4}\n", sum);

    // Multi-label classification: raw entailment probabilities (independent labels)
    println!("=== Multi-label Classification (classify_multi_label) ===");
    println!("Use this when labels can be independent and multiple labels could apply.");
    println!("Returns raw entailment probabilities for each label independently.\n");

    let multi_label_result = pipeline.classify_multi_label(text, &candidate_labels)?;
    println!("Text: \"{}\"", text);
    println!("Multi-label results:");
    for result in &multi_label_result {
        println!("  - {}: {:.4}", result.label, result.score);
    }

    Ok(())
}
