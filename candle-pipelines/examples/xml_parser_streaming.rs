use std::io::Write;

use candle_pipelines::error::Result;
use candle_pipelines::text_generation::{
    tool, tools, Event, Qwen3, TagPart, TextGenerationPipelineBuilder, XmlTag,
};

/// Tags we want to parse from the model output.
#[derive(Debug, Clone, PartialEq, XmlTag)]
enum Tags {
    Think,      // matches <think>
    ToolResult, // matches <tool_result>
    ToolCall,   // matches <tool_call>
}

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
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3::Size0_6B)
        .max_len(1024)
        .build()?;

    pipeline.register_tools(tools![get_weather]);

    // Create XML parser for specific tags
    let parser = Tags::parser();

    // Get token iterator
    let tokens = pipeline.run_iter("What's the weather like in Tokyo?")?;

    // Wrap with XML parser for streaming parsing
    let events = parser.parse_iter(tokens);

    println!("\n--- Events ---");

    for event in events {
        match event {
            Ok(event) => {
                match event {
                    Event::Tag { tag: Tags::Think, part } => match part {
                        TagPart::Opened { .. } => println!("[THINKING]"),
                        TagPart::Content { text } => print!("{}", text),
                        TagPart::Closed { .. } => println!("[DONE THINKING]\n"),
                    },
                    Event::Tag { tag: Tags::ToolResult, part } => match part {
                        TagPart::Opened { .. } => println!("[START TOOL RESULT]"),
                        TagPart::Content { text } => print!("{}", text),
                        TagPart::Closed { .. } => println!("[END TOOL RESULT]\n"),
                    },
                    Event::Tag { tag: Tags::ToolCall, part } => match part {
                        TagPart::Opened { .. } => println!("[TOOL CALL]"),
                        TagPart::Content { text } => print!("{}", text),
                        TagPart::Closed { .. } => println!("[END TOOL CALL]\n"),
                    },
                    Event::Content { text } => print!("{}", text),
                }
                std::io::stdout().flush().unwrap();
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }

    Ok(())
}
