use transformers::error::{Result, TransformersError};
use transformers::text_generation::{Qwen3Size, TextGenerationPipelineBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .cuda_device(0)
        .build()
        .await?;

    let response = pipeline.count_tokens("How are you doing today?");

    match response {
        Ok(response) => println!("Token count: {}", response),
        Err(e) => println!("Error: {}", e),
    }

    Ok(())
}
