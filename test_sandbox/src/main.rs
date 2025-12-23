use transformers::error::{Result, TransformersError};
use transformers::text_generation::{Qwen3Size, TextGenerationPipelineBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .cuda_device(0)
        .build()
        .await?;

    let response = pipeline.completion("How are you doing today?").await;

    match response {
        Ok(response) => println!("Response: {}", response),
        Err(e) => match e {
            TransformersError::Download(e) => {
                println!("Download failed: {}", e);
            }
            TransformersError::ChatTemplate(e) => {
                println!("Chat template error: {}", e);
            }
            _ => println!("Error: {}", e),
        },
    };

    Ok(())
}
