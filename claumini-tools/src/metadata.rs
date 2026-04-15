use claumini_core::ToolDescriptor;
use schemars::{JsonSchema, schema_for};
use serde::Serialize;
use serde::de::DeserializeOwned;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolMetadata {
    pub name: &'static str,
    pub description: &'static str,
}

impl ToolMetadata {
    pub const fn new(name: &'static str, description: &'static str) -> Self {
        Self { name, description }
    }

    pub fn descriptor(self) -> ToolDescriptor {
        ToolDescriptor::new(self.name, self.description)
    }

    pub fn descriptor_for<I, O>(self) -> ToolDescriptor
    where
        I: JsonSchema,
        O: JsonSchema,
    {
        let mut descriptor = self.descriptor();
        descriptor.input_schema = schema_value::<I>();
        descriptor.output_schema = Some(schema_value::<O>());
        descriptor
    }

    pub fn descriptor_for_registration<R>(self) -> ToolDescriptor
    where
        R: ToolRegistration,
    {
        self.descriptor_for::<R::Input, R::Output>()
    }
}

pub trait ToolRegistration {
    type Input: Send + Sync + Serialize + DeserializeOwned + JsonSchema + 'static;
    type Output: Send + Sync + Serialize + DeserializeOwned + JsonSchema + 'static;

    fn tool_metadata() -> ToolMetadata;

    fn tool_descriptor() -> ToolDescriptor {
        Self::tool_metadata().descriptor_for::<Self::Input, Self::Output>()
    }
}

fn schema_value<T>() -> serde_json::Value
where
    T: JsonSchema,
{
    serde_json::to_value(schema_for!(T)).expect("schemars schemas should serialize to JSON")
}
