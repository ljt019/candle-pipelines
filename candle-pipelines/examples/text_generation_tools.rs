use std::io::Write;

use candle_pipelines::error::Result;
use candle_pipelines::text_generation::{tool, tools, ErrorStrategy};
use candle_pipelines::text_generation::{Qwen3Size, TextGenerationPipelineBuilder};

#[tool(retries = 5)]
/// Get the weather for a given city.
fn get_humidity(city: String) -> Result<String> {
    Ok(format!("The humidity is 1% in {}.", city))
}

#[tool] // defaults to 3 retries
/// Get the weather for a given city in degrees celsius.
fn get_temperature(city: String) -> Result<String> {
    Ok(format!(
        "The temperature is 20 degrees celsius in {}.",
        city
    ))
}

fn main() -> Result<()> {
    println!("Building pipeline...");

    // Fully sync - no async runtime needed for streaming
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(8192)
        .cuda(0)
        .tool_error_strategy(ErrorStrategy::ReturnToModel) // let model handle tool errors
        .build()?;

    println!("Pipeline built successfully.");

    pipeline.register_tools(tools![get_temperature, get_humidity]);

    let stream = pipeline.completion_stream("What's the temp and humidity like in Tokyo?")?;

    println!("\n=== Generation with Both Tools ===");

    for tok in stream {
        print!("{}", tok?);
        std::io::stdout().flush().unwrap();
    }

    pipeline.unregister_tools(tools![get_temperature]);

    let stream = pipeline.completion_stream("What's the temp and humidity like in Tokyo?")?;

    println!("\n\n=== Generation with Only Humidity Tool ===");

    for tok in stream {
        print!("{}", tok?);
        std::io::stdout().flush().unwrap();
    }

    Ok(())
}
