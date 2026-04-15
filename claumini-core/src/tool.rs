use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{ToolContext, ToolError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDescriptor {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub output_schema: Option<Value>,
}

impl ToolDescriptor {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema: Value::Object(Default::default()),
            output_schema: None,
        }
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    type Input: Send + Sync + Serialize + DeserializeOwned + 'static;
    type Output: Send + Sync + Serialize + DeserializeOwned + 'static;

    fn descriptor(&self) -> ToolDescriptor;

    async fn call(
        &self,
        input: Self::Input,
        ctx: &mut ToolContext,
    ) -> Result<Self::Output, ToolError>;
}
