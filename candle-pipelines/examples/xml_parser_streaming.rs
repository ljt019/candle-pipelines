use std::io::Write;

use candle_pipelines::error::Result;
use candle_pipelines::text_generation::{
    tool, tools, Qwen3Size, TagParts, TextGenerationPipelineBuilder, XmlParserBuilder,
};

#[tool]
/// Calculates the average speed given distance and time
fn calculate_average_speed(distance_in_miles: u64, time_in_minutes: u64) -> Result<String> {
    Ok(format!(
        "Average speed: {} mph",
        distance_in_miles / time_in_minutes
    ))
}

#[tool]
/// Gets the current weather in a given city
fn get_weather(city: String) -> Result<String> {
    Ok(format!("The weather in {} is sunny.", city))
}

fn main() -> Result<()> {
    // Build a regular pipeline - fully sync, no async runtime needed
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(1024)
        .build()?;

    pipeline.register_tools(tools![get_weather]);

    // Create XML parser for specific tags
    let parser = XmlParserBuilder::new()
        .register_tag("think")
        .register_tag("tool_result")
        .register_tag("tool_call")
        .build();

    // Stream completion (auto-uses tools if enabled)
    let stream = pipeline.completion_stream("What's the weather like in Tokyo?")?;

    // Wrap iterator with XML parser
    let event_iter = parser.wrap_iterator(stream);

    println!("\n--- Streaming Events ---");

    for event in event_iter {
        match event.tag() {
            Some("think") => match event.part() {
                TagParts::Start => println!("[THINKING]"),
                TagParts::Content => print!("{}", event.get_content()),
                TagParts::End => println!("[DONE THINKING]\n"),
            },
            Some("tool_result") => match event.part() {
                TagParts::Start => println!("[START TOOL RESULT]"),
                TagParts::Content => print!("{}", event.get_content()),
                TagParts::End => println!("[END TOOL RESULT]\n"),
            },
            Some("tool_call") => match event.part() {
                TagParts::Start => println!("[TOOL CALL]"),
                TagParts::Content => print!("{}", event.get_content()),
                TagParts::End => println!("[END TOOL CALL]\n"),
            },
            Some(_) => { /* ignore unknown tags */ }
            None => match event.part() {
                TagParts::Start => println!("[OUTPUT]"),
                TagParts::Content => print!("{}", event.get_content()),
                TagParts::End => println!("[END OUTPUT]\n"),
            },
        }
        std::io::stdout().flush().unwrap();
    }

    Ok(())
}
