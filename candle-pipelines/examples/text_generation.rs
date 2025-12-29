use candle_pipelines::error::Result;
use candle_pipelines::text_generation::{Message, Qwen3Size, TextGenerationPipelineBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    println!("Building pipeline...");

    // Start by creating the pipeline, using the builder to configure any generation parameters.
    // Parameters are optional, defaults are set to good values for each model.
    // Use .build() for sync or .build_async() for async model loading.
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .temperature(0.7)
        .top_k(40)
        .max_len(1024)
        .build()?;

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

    Ok(())
}
