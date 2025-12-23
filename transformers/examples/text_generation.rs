use transformers::error::Result;
use transformers::text_generation::{Message, Qwen3Size, TextGenerationPipelineBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    println!("Building pipeline...");

    // Start by creating the pipeline, using the builder to configure any generation parameters.
    // Parameters are optional, defaults are set to good values for each model.
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .temperature(0.7)
        .top_k(40)
        .max_len(1024)
        .build()
        .await?;

    println!("Pipeline built successfully.");

    // Get a completion from a prompt.
    let completion = pipeline
        .completion("Explain the concept of Large Language Models in simple terms.")
        .await?;

    println!("\n=== Generated Text ===");
    println!("{}", completion);

    // Create and use messages for your completions to keep a conversation going.
    let mut messages = vec![
        Message::system("You are a helpful pirate assistant."),
        Message::user("What is the capital of France?"),
    ];

    let completion = pipeline.completion(&messages).await?;

    println!("\n=== Generated Text 2 ===");
    println!("{}", completion);

    // To continue the conversation, add the response to the messages
    messages.push(Message::assistant(&completion));
    messages.push(Message::user("What are some fun things to do there?"));

    // Now ask a follow-up question.
    let completion = pipeline.completion(&messages).await?;

    println!("\n=== Generated Text 3 (Follow-up) ===");
    println!("{}", completion);

    // Batch inference: process multiple prompts at once
    println!("\n=== Batch Inference ===");
    let prompts = &[
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "Write a haiku about Rust programming.",
    ];

    let results = pipeline.completion_batch(prompts).await?;

    for (i, result) in results.into_iter().enumerate() {
        println!("\n[Prompt {}]: {}", i + 1, prompts[i]);
        match result {
            Ok(completion) => println!("[Response]: {}", completion),
            Err(e) => println!("[Error]: {}", e),
        }
    }

    Ok(())
}
