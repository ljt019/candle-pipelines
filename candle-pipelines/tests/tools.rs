#![cfg(feature = "cuda")]

use candle_pipelines::error::{PipelineError, Result};
use candle_pipelines::text_generation::{
    tool, tools, ErrorStrategy, Message, Olmo3Size, Qwen3Size, TextGenerationPipelineBuilder,
};

#[tool]
fn get_weather(city: String) -> Result<String> {
    Ok(format!("The weather in {city} is sunny."))
}

#[tokio::test]
async fn tool_calling_basic() -> Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .cuda(0)
        .seed(42)
        .max_len(150)
        .tool_error_strategy(ErrorStrategy::ReturnToModel)
        .build_async()
        .await?;

    pipeline.register_tools(tools![get_weather]);
    let out = pipeline.run("What's the weather like in Paris today?")?;

    // Tool result is now JSON format
    assert!(
        out.text.contains("<tool_result>"),
        "Expected tool_result tag in output: {}",
        out.text
    );
    assert!(
        out.text.contains("get_weather"),
        "Expected tool name in output: {}",
        out.text
    );
    assert!(
        out.text.contains("sunny"),
        "Expected 'sunny' in output: {}",
        out.text
    );
    Ok(())
}

#[tool]
fn echo(msg: String) -> String {
    msg
}

#[tokio::test]
async fn tool_registration() -> Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .cuda(0)
        .seed(0)
        .max_len(20)
        .build_async()
        .await?;

    // Clear any tools from previous tests (models are cached and shared)
    pipeline.clear_tools();

    pipeline.register_tools(tools![echo]);
    assert_eq!(pipeline.registered_tools().len(), 1);

    pipeline.unregister_tool("echo");
    assert!(pipeline.registered_tools().is_empty());

    pipeline.register_tools(tools![echo]);
    pipeline.clear_tools();
    assert!(pipeline.registered_tools().is_empty());
    Ok(())
}

#[tool(retries = 1)]
fn fail_tool() -> Result<String> {
    Err(PipelineError::Tool("fail_tool failed: boom".to_string()))
}

#[tokio::test]
async fn tool_error_fail_strategy() -> Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .cuda(0)
        .seed(0)
        .max_len(200)
        .tool_error_strategy(ErrorStrategy::Fail)
        .build_async()
        .await?;

    pipeline.register_tools(tools![fail_tool]);
    let res = pipeline.run("call fail_tool");
    assert!(res.is_err());
    Ok(())
}

// ============ OLMo-3 Tool Tests ============

#[tool]
/// Get the temperature in a city.
fn get_temperature(city: String) -> Result<String> {
    Ok(format!("The temperature in {city} is 22°C."))
}

#[tokio::test]
async fn olmo3_tool_calling_run() -> Result<()> {
    let pipeline = TextGenerationPipelineBuilder::olmo3(Olmo3Size::Size7B)
        .cuda(0)
        .temperature(0.3)
        .max_len(512)
        .tool_error_strategy(ErrorStrategy::ReturnToModel)
        .build_async()
        .await?;

    pipeline.register_tools(tools![get_temperature]);

    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("What is the temperature in Tokyo?"),
    ];

    let out = pipeline.run(&messages)?;

    // Check that tool was called and result included
    assert!(
        out.text.contains("<tool_call>"),
        "Expected tool_call tag in output: {}",
        out.text
    );
    assert!(
        out.text.contains("<tool_result>"),
        "Expected tool_result tag in output: {}",
        out.text
    );
    assert!(
        out.text.contains("get_temperature"),
        "Expected tool name in output: {}",
        out.text
    );
    assert!(
        out.text.contains("Tokyo") || out.text.contains("tokyo"),
        "Expected Tokyo in output: {}",
        out.text
    );
    Ok(())
}

#[tokio::test]
async fn olmo3_tool_calling_run_iter() -> Result<()> {
    let pipeline = TextGenerationPipelineBuilder::olmo3(Olmo3Size::Size7B)
        .cuda(0)
        .temperature(0.3)
        .max_len(512)
        .tool_error_strategy(ErrorStrategy::ReturnToModel)
        .build_async()
        .await?;

    pipeline.register_tools(tools![get_temperature]);

    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("What is the temperature in Paris?"),
    ];

    // Use run_iter and collect all output
    let mut output = String::new();
    for token in pipeline.run_iter(&messages)? {
        output.push_str(&token?);
    }

    // Check that tool was called and result included
    assert!(
        output.contains("<tool_call>"),
        "Expected tool_call tag in streaming output: {}",
        output
    );
    assert!(
        output.contains("<tool_result>"),
        "Expected tool_result tag in streaming output: {}",
        output
    );
    assert!(
        output.contains("get_temperature"),
        "Expected tool name in streaming output: {}",
        output
    );
    assert!(
        output.contains("Paris") || output.contains("paris"),
        "Expected Paris in streaming output: {}",
        output
    );
    Ok(())
}
