use std::io::Write;

use candle_pipelines::error::Result;
use candle_pipelines::text_generation::{Qwen3Size, TextGenerationPipelineBuilder};

fn main() -> Result<()> {
    // Start by creating the pipeline, using the builder to configure any generation parameters.
    // Streaming is fully sync - no async runtime needed.
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(1024)
        .cuda(0)
        .min_p(0.1)
        .build()?;

    let stream = pipeline
        .completion_stream("Explain the concept of Large Language Models in simple terms.")?;

    println!("\n--- Generated Text ---");
    for tok in stream {
        print!("{}", tok?);
        std::io::stdout().flush().unwrap();
    }

    /*
    // Also supports messages obviously
    let messages = vec![
        Message::system("You are a helpful pirate assistant."),
        Message::user("What is the capital of France?"),
    ];

    let stream_two = pipeline.completion_stream(&messages)?;

    println!("\n--- Generated Text 2 ---");
    for tok in stream_two {
        print!("{}", tok?);
        std::io::stdout().flush().unwrap();
    }
    */

    Ok(())
}
