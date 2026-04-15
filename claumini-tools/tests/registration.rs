use claumini_tools::ToolRegistration;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct MacroInput {
    value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct MacroOutput {
    echoed: String,
}

#[derive(Debug, claumini_tools::ToolRegistration)]
#[tool(
    name = "macro_echo",
    description = "Echoes structured input",
    input = MacroInput,
    output = MacroOutput
)]
struct MacroEchoTool;

#[test]
fn derive_registration_exposes_expected_descriptor() {
    let descriptor = MacroEchoTool::tool_descriptor();

    assert_eq!(MacroEchoTool::tool_metadata().name, "macro_echo");
    assert_eq!(descriptor.name, "macro_echo");
    assert_eq!(descriptor.description, "Echoes structured input");
    assert_eq!(descriptor.input_schema["type"], "object");
    assert!(descriptor.input_schema["properties"].get("value").is_some());
    assert_eq!(
        descriptor.output_schema.as_ref().expect("output schema")["type"],
        "object"
    );
    assert!(
        descriptor.output_schema.as_ref().expect("output schema")["properties"]
            .get("echoed")
            .is_some()
    );
}
