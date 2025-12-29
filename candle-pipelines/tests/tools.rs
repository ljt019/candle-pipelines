#![cfg(feature = "cuda")]

use candle_pipelines::error::{PipelineError, Result};
use candle_pipelines::text_generation::{
    tool, tools, ErrorStrategy, Qwen3Size, TextGenerationPipelineBuilder,
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

    assert!(out.text.contains(
        "<tool_result name=\"get_weather\">\nThe weather in Paris is sunny.\n</tool_result>"
    ));
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
