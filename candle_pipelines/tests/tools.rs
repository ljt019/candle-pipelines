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
        .build()
        .await?;

    pipeline.register_tools(tools![get_weather]).await;
    let out = pipeline
        .completion_with_tools("What's the weather like in Paris today?")
        .await?;

    assert!(out.contains(
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
        .build()
        .await?;

    pipeline.register_tools(tools![echo]).await;
    assert_eq!(pipeline.registered_tools().await.len(), 1);

    pipeline.unregister_tool("echo").await;
    assert!(pipeline.registered_tools().await.is_empty());

    pipeline.register_tools(tools![echo]).await;
    pipeline.clear_tools().await;
    assert!(pipeline.registered_tools().await.is_empty());
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
        .build()
        .await?;

    pipeline.register_tools(tools![fail_tool]).await;
    let res = pipeline.completion_with_tools("call fail_tool").await;
    assert!(res.is_err());
    Ok(())
}
