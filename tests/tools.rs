//! Integration tests for tool calling functionality
//! Run with: cargo test --features integration

#![cfg(feature = "integration")]

use transformers::pipelines::text_generation::*;
use transformers::pipelines::utils::DeviceSelectable;
use transformers::{Result, ToolError};

#[tool]
fn get_weather(city: String) -> Result<String> {
    Ok(format!("The weather in {city} is sunny."))
}

#[tokio::test]
async fn tool_calling_basic() -> transformers::Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .cuda_device(0)
        .seed(42)
        .max_len(150)
        .build()
        .await?;

    pipeline.register_tools(tools![get_weather]).await?;
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
async fn tool_registration() -> transformers::Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .cuda_device(0)
        .seed(0)
        .max_len(20)
        .build()
        .await?;

    pipeline.register_tools(tools![echo]).await?;
    assert_eq!(pipeline.registered_tools().await.len(), 1);

    pipeline.unregister_tool("echo").await?;
    assert!(pipeline.registered_tools().await.is_empty());

    pipeline.register_tools(tools![echo]).await?;
    pipeline.clear_tools().await?;
    assert!(pipeline.registered_tools().await.is_empty());
    Ok(())
}

#[tool(on_error = ErrorStrategy::Fail, retries = 1)]
fn fail_tool() -> Result<String> {
    Err(ToolError::ExecutionFailed {
        name: "fail_tool".into(),
        attempts: 1,
        reason: "boom".into(),
    }
    .into())
}

#[tokio::test]
async fn tool_error_fail_strategy() -> transformers::Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .cuda_device(0)
        .seed(0)
        .max_len(200)
        .build()
        .await?;

    pipeline.register_tools(tools![fail_tool]).await?;
    let res = pipeline.completion_with_tools("call fail_tool").await;
    assert!(res.is_err());
    Ok(())
}
