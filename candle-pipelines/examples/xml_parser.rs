use candle_pipelines::error::Result;
use candle_pipelines::text_generation::{
    tool, tools, Qwen3, TagParts, TextGenerationPipelineBuilder, XmlTag,
};

/// Tags we want to parse from the model output.
#[derive(Debug, Clone, PartialEq, XmlTag)]
enum Tags {
    #[tag("think")]
    Think,
    #[tag("tool_result")]
    ToolResult,
    #[tag("tool_call")]
    ToolCall,
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
            // Exhaustive matching on tag variants - no catch-all needed!
            candle_pipelines::text_generation::Event::Tagged {
                tag: Tags::Think,
                part,
                content,
                ..
            } => match part {
                TagParts::Start => println!("[THINKING]"),
                TagParts::Content => print!("{}", content),
                TagParts::End => println!("[DONE THINKING]\n"),
            },
            candle_pipelines::text_generation::Event::Tagged {
                tag: Tags::ToolResult,
                part,
                content,
                ..
            } => match part {
                TagParts::Start => println!("[START TOOL RESULT]"),
                TagParts::Content => print!("{}", content),
                TagParts::End => println!("[END TOOL RESULT]\n"),
            },
            candle_pipelines::text_generation::Event::Tagged {
                tag: Tags::ToolCall,
                part,
                content,
                ..
            } => match part {
                TagParts::Start => println!("[START TOOL CALL]"),
                TagParts::Content => print!("{}", content),
                TagParts::End => println!("[END TOOL CALL]\n"),
            },
            candle_pipelines::text_generation::Event::Output { content } => {
                print!("{}", content);
            }
        }
    }

    Ok(())
}
