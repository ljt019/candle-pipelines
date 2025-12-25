use crate::error::Result;
use futures::future::BoxFuture;
use std::sync::Arc;

/// How to handle tool execution errors during generation.
#[derive(Debug, Clone, Default)]
pub enum ErrorStrategy {
    /// Stop generation and return the error to the caller.
    #[default]
    Fail,
    /// Return the error message to the model so it can attempt to recover or retry. (Default)
    ReturnToModel,
}

/// Trait for types that support tool registration and execution.
pub trait ToolCalling {
    /// Register a tool for use during generation.
    fn register_tool(&mut self, tool: Tool);
    /// Remove a tool by name. No-op if not found.
    fn unregister_tool(&mut self, name: &str);
    /// Remove all registered tools.
    fn clear_tools(&mut self);
    /// Returns a list of all registered tools.
    fn registered_tools(&self) -> Vec<Tool>;
}

/// Future type returned by tool functions.
pub type ToolFuture = BoxFuture<'static, Result<String>>;

/// A tool that can be invoked by the model during generation.
///
/// Tools are created via the `#[tool]` attribute macro rather than
/// constructed directly.
#[derive(serde::Serialize)]
pub struct Tool {
    pub(crate) name: String,
    pub(crate) description: String,
    #[serde(rename = "parameters")]
    pub(crate) schema: schemars::schema::RootSchema,
    #[serde(skip_serializing)]
    pub(crate) function: Arc<dyn Fn(serde_json::Value) -> ToolFuture + Send + Sync>,
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
            max_retries: self.max_retries,
        }
    }
}

impl Tool {
    /// Creates a new Tool. This should only be used by the `#[tool]` macro.
    #[doc(hidden)]
    pub fn new(
        name: String,
        description: String,
        schema: schemars::schema::RootSchema,
        function: impl Fn(serde_json::Value) -> ToolFuture + Send + Sync + 'static,
        max_retries: u32,
    ) -> Self {
        Self {
            name,
            description,
            schema,
            function: Arc::new(function),
            max_retries,
        }
    }

    /// Returns the tool's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    pub(crate) async fn call(&self, parameters: serde_json::Value) -> Result<String> {
        self.validate(&parameters)?;
        (self.function)(parameters).await
    }

    /// Returns the JSON schema describing the tool's parameters.
    pub fn schema(&self) -> &schemars::schema::RootSchema {
        &self.schema
    }

    /// Returns the tool's description (shown to the model). Constructed using the tool function's doc string.
    pub fn description(&self) -> &str {
        &self.description
    }

    pub(crate) fn max_retries(&self) -> u32 {
        self.max_retries
    }

    fn validate(&self, params: &serde_json::Value) -> Result<()> {
        let schema = serde_json::to_value(&self.schema).map_err(|e| {
            crate::error::TransformersError::Tool(format!(
                "Schema error for '{}': schema serialization failed: {}",
                self.name, e
            ))
        })?;
        let compiled = jsonschema::JSONSchema::options()
            .with_draft(jsonschema::Draft::Draft7)
            .compile(&schema)
            .map_err(|e| {
                crate::error::TransformersError::Tool(format!(
                    "Schema error for '{}': invalid schema: {}",
                    self.name, e
                ))
            })?;

        let validation_result = compiled.validate(params);
        match validation_result {
            Ok(_) => Ok(()),
            Err(errors) => {
                let messages: Vec<String> = errors.map(|e| e.to_string()).collect();
                let error_msg = messages.join(", ");
                Err(crate::error::TransformersError::Tool(format!(
                    "Invalid parameters for '{}': {}",
                    self.name, error_msg
                )))
            }
        }
    }
}
