#![forbid(unsafe_code)]

mod agent;
mod error;
mod model;
mod payload;
mod runtime;
mod tool;

pub use agent::Agent;
pub use error::{AgentError, PayloadError, ProviderError, RuntimeError, ToolError};
pub use model::{
    FinishReason, Message, MessageRole, ModelProvider, ModelRequest, ModelResponse,
    ProviderCapabilities, ToolCall, ToolSchema,
};
pub use payload::{ArtifactId, Payload};
pub use runtime::{AgentContext, MaxTurnsPolicy, RuntimeLimits, SessionMetadata, ToolContext};
pub use tool::{Tool, ToolDescriptor};

#[cfg(test)]
mod tests {
    use super::{
        ArtifactId, Message, MessageRole, ModelRequest, Payload, RuntimeLimits, ToolCall,
        ToolSchema,
    };
    use serde_json::json;

    #[test]
    fn payload_json_round_trips() {
        let payload = Payload::json(json!({"kind": "bug"})).expect("payload should serialize");

        assert_eq!(
            payload.as_json().expect("payload should expose json")["kind"],
            "bug"
        );
    }

    #[test]
    fn model_request_preserves_system_prompt_and_tools() {
        let tool = ToolSchema {
            name: "read_file".into(),
            description: "Read a file from disk".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        };
        let request = ModelRequest::new(vec![Message::new(
            MessageRole::User,
            Payload::artifact(ArtifactId::new(7)),
        )])
        .with_system_prompt("system prompt")
        .with_tool(tool.clone());

        assert_eq!(request.system_prompt.as_deref(), Some("system prompt"));
        assert_eq!(request.tools, vec![tool]);
        assert_eq!(request.tool_names().collect::<Vec<_>>(), vec!["read_file"]);
    }

    #[test]
    fn model_request_serializes_tools_without_redundant_tool_names() {
        let request = ModelRequest::new(vec![Message::new(MessageRole::User, Payload::text("hi"))])
            .with_tool(ToolSchema {
                name: "read_file".into(),
                description: "Read a file from disk".into(),
                input_schema: json!({"type": "object"}),
            });

        let value = serde_json::to_value(&request).expect("request should serialize");

        assert!(value.get("tool_names").is_none());
        assert_eq!(value["tools"][0]["name"], "read_file");
        assert_eq!(value["tools"][0]["description"], "Read a file from disk");
    }

    #[test]
    fn message_serialization_omits_tool_calls_when_empty_and_keeps_them_when_present() {
        let plain = serde_json::to_value(Message::new(MessageRole::Assistant, Payload::text("hi")))
            .expect("plain message should serialize");
        assert!(plain.get("tool_calls").is_none());

        let with_tool_calls = serde_json::to_value(
            Message::new(MessageRole::Assistant, Payload::text("")).with_tool_calls(vec![
                ToolCall {
                    id: "call-1".into(),
                    name: "lookup".into(),
                    arguments: json!({"city": "sf"}),
                },
            ]),
        )
        .expect("tool call message should serialize");

        assert_eq!(with_tool_calls["tool_calls"][0]["name"], "lookup");
    }

    #[test]
    fn runtime_limits_have_non_zero_defaults() {
        let limits = RuntimeLimits::default();

        assert!(limits.max_turns_per_session > 0);
        assert!(limits.model_request_timeout_ms > 0);
        assert!(limits.tool_call_timeout_ms > 0);
    }
}
