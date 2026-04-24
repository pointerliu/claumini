use claumini_core::{
    Message, MessageRole, ModelProvider, ModelRequest, Payload, ProviderError, ToolCall,
    ToolSchema,
};
use claumini_models::{
    ClaudeConfig, ClaudeProvider, ConfigError, MockProvider, OpenAiCompatibleConfig,
    OpenAiCompatibleProvider, OpenAiConfig, OpenAiProvider,
};
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path},
};

fn openai_config() -> OpenAiCompatibleConfig {
    OpenAiCompatibleConfig {
        base_url: "https://api.example.com/v1".into(),
        api_key: "test-key".into(),
        model: "gpt-test".into(),
        max_tokens: None,
    }
}

fn official_openai_config() -> OpenAiConfig {
    OpenAiConfig {
        api_key: "openai-key".into(),
        model: "gpt-4o-mini".into(),
    }
}

fn claude_config() -> ClaudeConfig {
    ClaudeConfig {
        api_key: "claude-key".into(),
        model: "claude-sonnet-test".into(),
    }
}

fn sample_request() -> ModelRequest {
    ModelRequest::new(vec![Message::new(
        MessageRole::User,
        Payload::text("hello"),
    )])
}

fn tool_schema() -> ToolSchema {
    ToolSchema {
        name: "lookup".into(),
        description: "Look up a record".into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "id": { "type": "string" }
            },
            "required": ["id"]
        }),
    }
}

#[test]
fn openai_compatible_provider_reports_native_tools_and_structured_output() {
    let provider = OpenAiCompatibleProvider::new(openai_config()).expect("provider should build");

    let capabilities = provider.capabilities();
    assert!(capabilities.native_tool_calling);
    assert!(capabilities.structured_output);
    assert!(!capabilities.reasoning_control);
    assert!(!capabilities.image_input);
}

#[test]
fn openai_provider_uses_official_base_url_and_reports_expected_capabilities() {
    let provider = OpenAiProvider::new(official_openai_config()).expect("provider should build");

    let capabilities = provider.capabilities();
    assert!(capabilities.native_tool_calling);
    assert!(capabilities.structured_output);
    assert!(!capabilities.reasoning_control);
    assert!(!capabilities.image_input);

    assert_eq!(provider.config(), &official_openai_config());
}

#[test]
fn claude_provider_reports_expected_capabilities() {
    let provider = ClaudeProvider::new(claude_config()).expect("provider should build");

    let capabilities = provider.capabilities();
    assert!(capabilities.native_tool_calling);
    assert!(capabilities.structured_output);
    assert!(!capabilities.reasoning_control);
    assert!(!capabilities.image_input);
}

#[test]
fn mock_provider_reports_no_native_capabilities() {
    let capabilities = MockProvider::new_text("mock response").capabilities();

    assert!(!capabilities.native_tool_calling);
    assert!(!capabilities.structured_output);
    assert!(!capabilities.reasoning_control);
    assert!(!capabilities.image_input);
}

#[test]
fn openai_config_rejects_blank_fields_and_invalid_url() {
    let err = OpenAiCompatibleProvider::new(OpenAiCompatibleConfig {
        base_url: "not-a-url".into(),
        api_key: " ".into(),
        model: "".into(),
        max_tokens: None,
    })
    .expect_err("invalid config should be rejected");

    assert_eq!(
        err,
        ConfigError::InvalidField {
            field: "base_url",
            message: "must be an absolute http or https URL",
        }
    );
}

#[test]
fn openai_provider_rejects_blank_fields() {
    let err = OpenAiProvider::new(OpenAiConfig {
        api_key: " ".into(),
        model: "".into(),
    })
    .expect_err("invalid config should be rejected");

    assert_eq!(
        err,
        ConfigError::InvalidField {
            field: "api_key",
            message: "must not be blank",
        }
    );
}

#[test]
fn claude_config_rejects_blank_fields() {
    let err = ClaudeProvider::new(ClaudeConfig {
        api_key: " ".into(),
        model: "".into(),
    })
    .expect_err("invalid config should be rejected");

    assert_eq!(
        err,
        ConfigError::InvalidField {
            field: "api_key",
            message: "must not be blank",
        }
    );
}

#[tokio::test]
async fn openai_provider_posts_chat_completions_and_normalizes_response() {
    let server = MockServer::start().await;
    let provider = OpenAiCompatibleProvider::new(OpenAiCompatibleConfig {
        base_url: server.uri(),
        api_key: "test-key".into(),
        model: "gpt-test".into(),
        max_tokens: None,
    })
    .expect("provider should build");

    let request = ModelRequest::new(vec![
        Message::new(MessageRole::User, Payload::text("Need weather")),
        Message::new(MessageRole::Assistant, Payload::text("Checking.")),
        Message::new(
            MessageRole::Tool,
            Payload::json(json!({"temperature_c": 21})).expect("json payload"),
        )
        .named("lookup"),
    ])
    .with_system_prompt("You are helpful.")
    .with_tool(tool_schema())
    .with_max_output_tokens(256)
    .with_response_schema(json!({
        "type": "object",
        "properties": {
            "answer": { "type": "string" }
        },
        "required": ["answer"]
    }));

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .and(body_json(json!({
            "model": "gpt-test",
            "messages": [
                { "role": "system", "content": "You are helpful." },
                { "role": "user", "content": "Need weather" },
                { "role": "assistant", "content": "Checking." },
                { "role": "tool", "tool_call_id": "lookup", "content": "{\"temperature_c\":21}" }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up a record",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" }
                            },
                            "required": ["id"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "__final_answer",
                        "description": "Return the final answer as a structured object matching the requested schema. Call this exactly once, after all other investigation is complete, with the definitive answer. Do not call other tools after this one.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "answer": { "type": "string" }
                            },
                            "required": ["answer"]
                        }
                    }
                }
            ],
            "max_tokens": 256
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "I need to look that up.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": "{\"id\":\"sf\"}"
                                }
                            }
                        ]
                    }
                }
            ]
        })))
        .mount(&server)
        .await;

    let response = provider
        .complete(request)
        .await
        .expect("request should succeed");

    assert_eq!(
        response
            .message
            .expect("assistant message")
            .content
            .as_text(),
        Some("I need to look that up.")
    );
    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].id, "call_1");
    assert_eq!(response.tool_calls[0].name, "lookup");
    assert_eq!(response.tool_calls[0].arguments, json!({"id": "sf"}));
}

#[tokio::test]
async fn official_openai_provider_posts_to_official_chat_completions_shape() {
    let server = MockServer::start().await;
    let provider = OpenAiProvider::new_with_base_url(official_openai_config(), server.uri())
        .expect("provider should build");

    let request = ModelRequest::new(vec![Message::new(
        MessageRole::User,
        Payload::text("Say hi"),
    )]);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer openai-key"))
        .and(body_json(json!({
            "model": "gpt-4o-mini",
            "messages": [
                { "role": "user", "content": "Say hi" }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Hello!"
                    }
                }
            ]
        })))
        .mount(&server)
        .await;

    let response = provider
        .complete(request)
        .await
        .expect("request should succeed");

    assert_eq!(
        response
            .message
            .expect("assistant message")
            .content
            .as_text(),
        Some("Hello!")
    );
    assert!(response.tool_calls.is_empty());
}

#[tokio::test]
async fn openai_provider_round_trips_assistant_tool_call_turns_in_follow_up_requests() {
    let server = MockServer::start().await;
    let provider = OpenAiCompatibleProvider::new(OpenAiCompatibleConfig {
        base_url: server.uri(),
        api_key: "test-key".into(),
        model: "gpt-test".into(),
        max_tokens: None,
    })
    .expect("provider should build");

    let request = ModelRequest::new(vec![
        Message::new(MessageRole::User, Payload::text("Need weather")),
        Message::new(MessageRole::Assistant, Payload::text("")).with_tool_calls(vec![ToolCall {
            id: "call_1".into(),
            name: "lookup".into(),
            arguments: json!({"id": "sf"}),
        }]),
        Message::new(
            MessageRole::Tool,
            Payload::json(json!({"temperature_c": 21})).expect("json payload"),
        )
        .named("call_1"),
    ]);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_json(json!({
            "model": "gpt-test",
            "messages": [
                { "role": "user", "content": "Need weather" },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "{\"id\":\"sf\"}"
                            }
                        }
                    ]
                },
                { "role": "tool", "tool_call_id": "call_1", "content": "{\"temperature_c\":21}" }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Final answer"
                    }
                }
            ]
        })))
        .mount(&server)
        .await;

    let response = provider
        .complete(request)
        .await
        .expect("request should succeed");

    assert_eq!(
        response
            .message
            .expect("assistant message")
            .content
            .as_text(),
        Some("Final answer")
    );
}

#[tokio::test]
async fn openai_provider_round_trips_assistant_reasoning_content_in_follow_up_requests() {
    let server = MockServer::start().await;
    let provider = OpenAiCompatibleProvider::new(OpenAiCompatibleConfig {
        base_url: server.uri(),
        api_key: "test-key".into(),
        model: "gpt-test".into(),
        max_tokens: None,
    })
    .expect("provider should build");

    let request = ModelRequest::new(vec![
        Message::new(MessageRole::User, Payload::text("Investigate mismatch")),
        Message::new(MessageRole::Assistant, Payload::text("")).with_tool_calls(vec![ToolCall {
            id: "call_1".into(),
            name: "lookup".into(),
            arguments: json!({"id": "ibex"}),
        }])
        .with_thinking("Need to inspect coverage first."),
        Message::new(
            MessageRole::Tool,
            Payload::json(json!({"module": "ibex_top"})).expect("json payload"),
        )
        .named("call_1"),
    ]);

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(body_json(json!({
            "model": "gpt-test",
            "messages": [
                { "role": "user", "content": "Investigate mismatch" },
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "Need to inspect coverage first.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "{\"id\":\"ibex\"}"
                            }
                        }
                    ]
                },
                { "role": "tool", "tool_call_id": "call_1", "content": "{\"module\":\"ibex_top\"}" }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Final answer"
                    }
                }
            ]
        })))
        .mount(&server)
        .await;

    let response = provider
        .complete(request)
        .await
        .expect("request should succeed");

    assert_eq!(
        response
            .message
            .expect("assistant message")
            .content
            .as_text(),
        Some("Final answer")
    );
}

#[tokio::test]
async fn claude_provider_posts_messages_and_normalizes_response() {
    let server = MockServer::start().await;
    let provider = ClaudeProvider::new_with_base_url(claude_config(), server.uri())
        .expect("provider should build");

    let request = ModelRequest::new(vec![
        Message::new(MessageRole::User, Payload::text("Need weather")),
        Message::new(MessageRole::Assistant, Payload::text("Checking.")),
        Message::new(
            MessageRole::Tool,
            Payload::json(json!({"temperature_c": 21})).expect("json payload"),
        )
        .named("lookup"),
    ])
    .with_system_prompt("You are helpful.")
    .with_tool(tool_schema())
    .with_max_output_tokens(512);

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "claude-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .and(body_json(json!({
            "model": "claude-sonnet-test",
            "system": "You are helpful.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "Need weather" }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        { "type": "text", "text": "Checking." }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "lookup",
                            "content": "{\"temperature_c\":21}"
                        }
                    ]
                }
            ],
            "tools": [
                {
                    "name": "lookup",
                    "description": "Look up a record",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "id": { "type": "string" }
                        },
                        "required": ["id"]
                    }
                }
            ],
            "max_tokens": 512
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "stop_reason": "tool_use",
            "content": [
                { "type": "text", "text": "I will look it up." },
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup",
                    "input": { "id": "sf" }
                }
            ]
        })))
        .mount(&server)
        .await;

    let response = provider
        .complete(request)
        .await
        .expect("request should succeed");

    assert_eq!(
        response
            .message
            .expect("assistant message")
            .content
            .as_text(),
        Some("I will look it up.")
    );
    assert_eq!(response.tool_calls.len(), 1);
    assert_eq!(response.tool_calls[0].id, "toolu_1");
    assert_eq!(response.tool_calls[0].name, "lookup");
    assert_eq!(response.tool_calls[0].arguments, json!({"id": "sf"}));
}

#[tokio::test]
async fn claude_provider_round_trips_assistant_tool_call_turns_in_follow_up_requests() {
    let server = MockServer::start().await;
    let provider = ClaudeProvider::new_with_base_url(claude_config(), server.uri())
        .expect("provider should build");

    let request = ModelRequest::new(vec![
        Message::new(MessageRole::User, Payload::text("Need weather")),
        Message::new(MessageRole::Assistant, Payload::text("")).with_tool_calls(vec![ToolCall {
            id: "call_1".into(),
            name: "lookup".into(),
            arguments: json!({"id": "sf"}),
        }]),
        Message::new(
            MessageRole::Tool,
            Payload::json(json!({"temperature_c": 21})).expect("json payload"),
        )
        .named("call_1"),
    ]);

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(body_json(json!({
            "model": "claude-sonnet-test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "Need weather" }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "lookup",
                            "input": { "id": "sf" }
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "{\"temperature_c\":21}"
                        }
                    ]
                }
            ],
            "max_tokens": 1024
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "content": [
                { "type": "text", "text": "Final answer" }
            ],
            "stop_reason": "end_turn"
        })))
        .mount(&server)
        .await;

    let response = provider
        .complete(request)
        .await
        .expect("request should succeed");

    assert_eq!(
        response
            .message
            .expect("assistant message")
            .content
            .as_text(),
        Some("Final answer")
    );
}

#[tokio::test]
async fn openai_provider_surfaces_synthetic_final_answer_as_assistant_text() {
    let server = MockServer::start().await;
    let provider = OpenAiCompatibleProvider::new(OpenAiCompatibleConfig {
        base_url: server.uri(),
        api_key: "test-key".into(),
        model: "gpt-test".into(),
        max_tokens: None,
    })
    .expect("provider should build");

    let request = ModelRequest::new(vec![Message::new(
        MessageRole::User,
        Payload::text("Where?"),
    )])
    .with_response_schema(json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"]
    }));

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {
                                "id": "call_final",
                                "type": "function",
                                "function": {
                                    "name": "__final_answer",
                                    "arguments": "{\"answer\":\"SF\"}"
                                }
                            }
                        ]
                    }
                }
            ]
        })))
        .mount(&server)
        .await;

    let response = provider
        .complete(request)
        .await
        .expect("request should succeed");

    assert!(response.tool_calls.is_empty());
    assert_eq!(response.finish_reason, claumini_core::FinishReason::Stop);
    assert_eq!(
        response
            .message
            .expect("assistant message")
            .content
            .as_text(),
        Some("{\"answer\":\"SF\"}")
    );
}

#[tokio::test]
async fn openai_provider_treats_invalid_json_body_as_temporary_error() {
    let server = MockServer::start().await;
    let provider = OpenAiCompatibleProvider::new(OpenAiCompatibleConfig {
        base_url: server.uri(),
        api_key: "test-key".into(),
        model: "gpt-test".into(),
        max_tokens: None,
    })
    .expect("provider should build");

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw("not-json", "application/json"),
        )
        .mount(&server)
        .await;

    let error = provider
        .complete(ModelRequest::new(vec![Message::new(
            MessageRole::User,
            Payload::text("Hello"),
        )]))
        .await
        .expect_err("invalid body should fail");

    assert!(matches!(error, ProviderError::Temporary(_)));
}

#[tokio::test]
async fn claude_provider_surfaces_synthetic_final_answer_as_assistant_text() {
    let server = MockServer::start().await;
    let provider = ClaudeProvider::new_with_base_url(claude_config(), server.uri())
        .expect("provider should build");

    let request = ModelRequest::new(vec![Message::new(
        MessageRole::User,
        Payload::text("Where?"),
    )])
    .with_response_schema(json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"]
    }));

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_final",
                    "name": "__final_answer",
                    "input": { "answer": "SF" }
                }
            ]
        })))
        .mount(&server)
        .await;

    let response = provider
        .complete(request)
        .await
        .expect("request should succeed");

    assert!(response.tool_calls.is_empty());
    assert_eq!(response.finish_reason, claumini_core::FinishReason::Stop);
    assert_eq!(
        response
            .message
            .expect("assistant message")
            .content
            .as_text(),
        Some("{\"answer\":\"SF\"}")
    );
}

#[tokio::test]
async fn mock_provider_returns_configured_response() {
    let provider = MockProvider::new_text("mock response");

    let response = provider
        .complete(sample_request())
        .await
        .expect("mock provider should return a response");

    let message = response
        .message
        .expect("mock response should contain a message");
    assert_eq!(message.content.as_text(), Some("mock response"));
}
