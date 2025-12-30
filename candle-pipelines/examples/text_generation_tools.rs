use candle_pipelines::error::Result;
use candle_pipelines::text_generation::{
    tool, tools, Message, Olmo3Size, TextGenerationPipelineBuilder,
};

#[tool(retries = 5)]
/// Get the realtime humidity data for a given city.
fn get_humidity(city: String) -> Result<String> {
    Ok(format!("The humidity is 1% in {}.", city))
}

#[tool] // defaults to 3 retries
/// Get the realtime temperature data for a given city in degrees celsius.
fn get_temperature(city: String) -> Result<String> {
    Ok(format!(
        "The temperature is 20 degrees celsius in {}.",
        city
    ))
}

fn main() -> Result<()> {
    println!("Building pipeline...");

    // Lower temperature for more reliable tool calling
    let pipeline = TextGenerationPipelineBuilder::olmo3(Olmo3Size::Size7B)
        .max_len(8192)
        .temperature(0.3) // Lower temp = more deterministic tool calls
        .cuda(0)
        .build()?;

    println!("Pipeline built successfully.");

    pipeline.register_tools(tools![get_temperature, get_humidity]);

    // Custom system message - tool instructions will be appended automatically
    let system_message = Message::system("You are a helpful weather assistant.");

    let example_1_messages = vec![
        system_message.clone(),
        Message::user("What is the temperature in Tokyo?"),
    ];

    println!("\n=== Generation with Both Tools ===");
    let output = pipeline.run(&example_1_messages)?;
    println!("{}", output.text);
    println!(
        "[{} tokens in {:.2}s]",
        output.stats.tokens_generated,
        output.stats.total_time.as_secs_f64()
    );

    pipeline.unregister_tools(tools![get_temperature]);

    let example_2_messages = vec![
        system_message,
        Message::user("What is the humidity in Tokyo?"),
    ];

    println!("\n=== Generation with Only Humidity Tool ===");
    let output = pipeline.run(&example_2_messages)?;
    println!("{}", output.text);
    println!(
        "[{} tokens in {:.2}s]",
        output.stats.tokens_generated,
        output.stats.total_time.as_secs_f64()
    );

    Ok(())
}
