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
    let mut parser = Tags::parser();

    // Generate completion
    let completion = pipeline.run("What's the weather like in Tokyo?")?;

    // Parse the text for XML events
    let events = parser.parse(&completion.text);

    println!("\n--- Generated Events ---");
    for event in events {
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
                TagPart::Opened { .. } => println!("[START TOOL CALL]"),
                TagPart::Content { text } => print!("{}", text),
                TagPart::Closed { .. } => println!("[END TOOL CALL]\n"),
            },
            Event::Content { text } => print!("{}", text),
        }
    }

    Ok(())
}
