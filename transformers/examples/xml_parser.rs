use transformers::error::Result;
use transformers::text_generation::{
    tool, tools, Qwen3Size, TagParts, TextGenerationPipelineBuilder,
};

#[tool]
/// Gets the current weather in a given city
fn get_weather(city: String) -> Result<String> {
    Ok(format!("The weather in {} is sunny.", city))
}

#[tokio::main]
async fn main() -> Result<()> {
    // Build a pipeline with XML parsing capabilities
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(1024)
        .build_xml(&["think", "tool_result", "tool_call"])
        .await?;

    pipeline.register_tools(tools![get_weather]).await?;

    // Generate completion - this will return Vec<Event>
    let events = pipeline
        .completion_with_tools("What's the weather like in Tokyo?")
        .await?;

    println!("\n--- Generated Events ---");
    for event in events {
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
                TagParts::Start => println!("[START TOOL CALL]"),
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
    }

    Ok(())
}
