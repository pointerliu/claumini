use claumini_core::{SessionMetadata, Tool, ToolContext};
use claumini_tools::AsyncToolAdapter;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct AsyncInput {
    name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct AsyncOutput {
    greeting: String,
}

#[derive(Debug, claumini_tools::ToolRegistration)]
#[tool(
    name = "async_greeter",
    description = "Greets asynchronously",
    input = AsyncInput,
    output = AsyncOutput
)]
struct AsyncGreeterTool;

fn tool_context() -> ToolContext {
    ToolContext::new(SessionMetadata::root("test-session"), 0)
}

#[tokio::test]
async fn async_adapter_executes_and_exposes_registered_schemas() {
    let tool = AsyncToolAdapter::<AsyncGreeterTool, _>::new(|input: AsyncInput| async move {
        Ok(AsyncOutput {
            greeting: format!("hello {}", input.name),
        })
    });

    let descriptor = tool.descriptor();
    let output = tool
        .call(
            AsyncInput {
                name: String::from("claumini"),
            },
            &mut tool_context(),
        )
        .await
        .expect("async adapter should execute");

    assert_eq!(descriptor.name, "async_greeter");
    assert_eq!(descriptor.description, "Greets asynchronously");
    assert_eq!(descriptor.input_schema["type"], "object");
    assert!(descriptor.input_schema["properties"].get("name").is_some());
    assert_eq!(
        descriptor.output_schema.as_ref().expect("output schema")["type"],
        "object"
    );
    assert!(
        descriptor.output_schema.as_ref().expect("output schema")["properties"]
            .get("greeting")
            .is_some()
    );
    assert_eq!(output.greeting, "hello claumini");
}
