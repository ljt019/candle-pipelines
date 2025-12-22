use crate::error::ToolError;
use crate::Result;
use futures::future::BoxFuture;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum ErrorStrategy {
    Fail,
    ReturnToModel,
}

impl Default for ErrorStrategy {
    fn default() -> Self {
        Self::Fail
    }
}

pub trait ToolCalling {
    fn register_tool(&mut self, tool: Tool) -> Result<()>;
    fn unregister_tool(&mut self, name: &str) -> Result<()>;
    fn clear_tools(&mut self) -> Result<()>;
    fn registered_tools(&self) -> Vec<Tool>;
}

pub type ToolFuture = BoxFuture<'static, Result<String>>;

#[derive(serde::Serialize)]
pub struct Tool {
    pub(crate) name: String,
    pub(crate) description: String,
    #[serde(rename = "parameters")]
    pub(crate) schema: schemars::schema::RootSchema,
    #[serde(skip_serializing)]
    pub(crate) function: Arc<dyn Fn(serde_json::Value) -> ToolFuture + Send + Sync>,
    #[serde(skip_serializing)]
    pub(crate) error_strategy: ErrorStrategy,
    #[serde(skip_serializing)]
    pub(crate) max_retries: u32,
}

impl Clone for Tool {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            schema: self.schema.clone(),
            function: Arc::clone(&self.function),
            error_strategy: self.error_strategy.clone(),
            max_retries: self.max_retries,
        }
    }
}

impl Tool {
    pub fn new(
        name: String,
        description: String,
        schema: schemars::schema::RootSchema,
        function: impl Fn(serde_json::Value) -> ToolFuture + Send + Sync + 'static,
        error_strategy: ErrorStrategy,
        max_retries: u32,
    ) -> Self {
        Self {
            name,
            description,
            schema,
            function: Arc::new(function),
            error_strategy,
            max_retries,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub async fn call(&self, parameters: serde_json::Value) -> Result<String> {
        self.validate(&parameters)?;
        (self.function)(parameters).await
    }

    pub fn schema(&self) -> &schemars::schema::RootSchema {
        &self.schema
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn error_strategy(&self) -> &ErrorStrategy {
        &self.error_strategy
    }

    pub fn max_retries(&self) -> u32 {
        self.max_retries
    }

    pub fn validate(&self, params: &serde_json::Value) -> Result<()> {
        let schema = serde_json::to_value(&self.schema).map_err(|e| ToolError::SchemaError {
            name: self.name.clone(),
            reason: format!("schema serialization failed: {e}"),
        })?;
        let compiled = jsonschema::JSONSchema::options()
            .with_draft(jsonschema::Draft::Draft7)
            .compile(&schema)
            .map_err(|e| ToolError::SchemaError {
                name: self.name.clone(),
                reason: format!("invalid schema: {e}"),
            })?;

        let validation_result = compiled.validate(params).map_err(|errors| {
            errors
                .map(|error| error.to_string())
                .collect::<Vec<String>>()
        });

        match validation_result {
            Ok(_) => Ok(()),
            Err(messages) => {
                let error_msg = messages.join(", ");
                Err(ToolError::InvalidParams {
                    name: self.name.clone(),
                    reason: error_msg,
                }
                .into())
            }
        }
    }
}

pub trait IntoTool {
    fn into_tool(self) -> Tool;
}

impl IntoTool for Tool {
    fn into_tool(self) -> Tool {
        self
    }
}
