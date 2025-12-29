use candle_pipelines::error::Result;
use candle_pipelines::text_generation::{Message, Qwen3Size, TextGenerationPipelineBuilder};

fn main() -> Result<()> {
    println!("Building pipeline...");

    // Start by creating the pipeline, using the builder to configure any generation parameters.
    // Parameters are optional, defaults are set to good values for each model.
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .temperature(0.7)
        .top_k(40)
        .max_len(1024)
        .build()?;

    println!("Pipeline built successfully.");

    // Get a completion from a prompt - returns Output { text, stats }
    let output = pipeline.run("Explain the concept of Large Language Models in simple terms.")?;

    println!("\n=== Generated Text ===");
    println!("{}", output.text);
    println!(
        "\n[{} tokens in {:.2}s ({:.1} tok/s)]",
        output.stats.tokens_generated,
        output.stats.total_time.as_secs_f64(),
        output.stats.tokens_per_second
    );

    // Create and use messages for your completions to keep a conversation going.
    let mut messages = vec![
        Message::system("You are a helpful pirate assistant."),
        Message::user("What is the capital of France?"),
    ];

    let output = pipeline.run(&messages)?;

    println!("\n=== Generated Text 2 ===");
    println!("{}", output.text);

    // To continue the conversation, add the response to the messages
    messages.push(Message::assistant(&output.text));
    messages.push(Message::user("What are some fun things to do there?"));

    // Now ask a follow-up question.
    let output = pipeline.run(&messages)?;

    println!("\n=== Generated Text 3 (Follow-up) ===");
    println!("{}", output.text);

    Ok(())
}
