use async_trait::async_trait;
use claumini_core::{
    FinishReason, Message, MessageRole, ModelProvider, ModelRequest, ModelResponse, Payload,
    ProviderCapabilities, ProviderError, ToolCall, ToolSchema,
};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};

use crate::{
    ClaudeConfig, ConfigError, FINAL_ANSWER_TOOL_DESCRIPTION, FINAL_ANSWER_TOOL_NAME,
    config::validate_http_url,
};

const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

#[derive(Debug, Clone)]
pub struct ClaudeProvider {
    config: ClaudeConfig,
    client: Client,
    base_url: String,
}

impl ClaudeProvider {
    pub fn new(config: ClaudeConfig) -> Result<Self, ConfigError> {
        Self::new_with_base_url(config, DEFAULT_BASE_URL)
    }

    pub fn new_with_base_url(
        config: ClaudeConfig,
        base_url: impl Into<String>,
    ) -> Result<Self, ConfigError> {
        config.validate()?;
        let base_url = base_url.into();
        validate_http_url("base_url", &base_url)?;

        Ok(Self {
            config,
            client: Client::new(),
            base_url,
        })
    }

    pub fn config(&self) -> &ClaudeConfig {
        &self.config
    }
}

#[async_trait]
impl ModelProvider for ClaudeProvider {
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, ProviderError> {
        let response = self
            .client
            .post(format!(
                "{}/v1/messages",
                self.base_url.trim_end_matches('/')
            ))
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&ClaudeRequest::from_request(&self.config.model, request)?)
            .send()
            .await
            .map_err(map_transport_error)?;

        let status = response.status();
        if !status.is_success() {
            return Err(map_status_error(status, response.text().await.ok()));
        }

        let body = response.json::<ClaudeResponse>().await.map_err(|err| {
            ProviderError::Message(format!("failed to decode claude response: {err}"))
        })?;

        normalize_response(body)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            native_tool_calling: true,
            structured_output: true,
            reasoning_control: false,
            image_input: false,
        }
    }
}

#[derive(Debug, Serialize)]
struct ClaudeRequest {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<ClaudeRequestMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ClaudeTool>,
    max_tokens: u32,
}

impl ClaudeRequest {
    fn from_request(model: &str, request: ModelRequest) -> Result<Self, ProviderError> {
        let mut system_parts = Vec::new();
        if let Some(system_prompt) = request.system_prompt {
            system_parts.push(system_prompt);
        }

        let mut messages = Vec::new();
        for message in request.messages {
            match message.role {
                MessageRole::System => system_parts.push(payload_to_string(&message.content)?),
                _ => messages.push(ClaudeRequestMessage::from_message(message)?),
            }
        }

        let mut tools: Vec<ClaudeTool> = request.tools.iter().map(ClaudeTool::from_tool).collect();
        if let Some(schema) = request.response_schema {
            if request
                .tools
                .iter()
                .any(|tool| tool.name == FINAL_ANSWER_TOOL_NAME)
            {
                return Err(ProviderError::Message(format!(
                    "tool name '{FINAL_ANSWER_TOOL_NAME}' is reserved for structured output"
                )));
            }
            tools.push(ClaudeTool::final_answer(schema));
        }

        Ok(Self {
            model: model.to_owned(),
            system: (!system_parts.is_empty()).then(|| system_parts.join("\n\n")),
            messages,
            tools,
            max_tokens: request.max_output_tokens.unwrap_or(1024),
        })
    }
}

#[derive(Debug, Serialize)]
struct ClaudeRequestMessage {
    role: &'static str,
    content: Vec<ClaudeRequestContent>,
}

impl ClaudeRequestMessage {
    fn from_message(message: Message) -> Result<Self, ProviderError> {
        match message.role {
            MessageRole::User => Ok(Self {
                role: "user",
                content: vec![ClaudeRequestContent::text(payload_to_string(
                    &message.content,
                )?)],
            }),
            MessageRole::Assistant if !message.tool_calls.is_empty() => {
                let mut content = Vec::with_capacity(1 + message.tool_calls.len());
                let text = payload_to_string(&message.content)?;
                if !text.is_empty() {
                    content.push(ClaudeRequestContent::text(text));
                }
                content.extend(
                    message
                        .tool_calls
                        .into_iter()
                        .map(ClaudeRequestContent::tool_use),
                );

                Ok(Self {
                    role: "assistant",
                    content,
                })
            }
            MessageRole::Assistant => Ok(Self {
                role: "assistant",
                content: vec![ClaudeRequestContent::text(payload_to_string(
                    &message.content,
                )?)],
            }),
            MessageRole::Tool => Ok(Self {
                role: "user",
                content: vec![ClaudeRequestContent::ToolResult {
                    tool_use_id: message.name.unwrap_or_else(|| "tool".into()),
                    content: payload_to_string(&message.content)?,
                }],
            }),
            MessageRole::System => unreachable!("system messages are handled separately"),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ClaudeRequestContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

impl ClaudeRequestContent {
    fn text(text: String) -> Self {
        Self::Text { text }
    }

    fn tool_use(call: ToolCall) -> Self {
        Self::ToolUse {
            id: call.id,
            name: call.name,
            input: call.arguments,
        }
    }
}

#[derive(Debug, Serialize)]
struct ClaudeTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

impl ClaudeTool {
    fn from_tool(tool: &ToolSchema) -> Self {
        Self {
            name: tool.name.clone(),
            description: tool.description.clone(),
            input_schema: tool.input_schema.clone(),
        }
    }

    fn final_answer(schema: serde_json::Value) -> Self {
        Self {
            name: FINAL_ANSWER_TOOL_NAME.to_owned(),
            description: FINAL_ANSWER_TOOL_DESCRIPTION.to_owned(),
            input_schema: schema,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ClaudeResponse {
    #[serde(default)]
    content: Vec<ClaudeResponseContent>,
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ClaudeResponseContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(other)]
    Other,
}

fn normalize_response(body: ClaudeResponse) -> Result<ModelResponse, ProviderError> {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut final_answer: Option<serde_json::Value> = None;

    for block in body.content {
        match block {
            ClaudeResponseContent::Text { text } => {
                if !text.is_empty() {
                    text_parts.push(text);
                }
            }
            ClaudeResponseContent::ToolUse { id, name, input } => {
                if name == FINAL_ANSWER_TOOL_NAME {
                    final_answer = Some(input);
                    continue;
                }
                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments: input,
                });
            }
            ClaudeResponseContent::Other => {}
        }
    }

    if let Some(arguments) = final_answer {
        let text = serde_json::to_string(&arguments).map_err(|err| {
            ProviderError::Message(format!("failed to serialize final answer: {err}"))
        })?;
        return Ok(ModelResponse {
            message: Some(Message::new(MessageRole::Assistant, Payload::text(text))),
            tool_calls,
            finish_reason: FinishReason::Stop,
        });
    }

    Ok(ModelResponse {
        message: (!text_parts.is_empty())
            .then(|| Message::new(MessageRole::Assistant, Payload::text(text_parts.join("\n")))),
        finish_reason: map_finish_reason(body.stop_reason.as_deref(), !tool_calls.is_empty()),
        tool_calls,
    })
}

fn map_finish_reason(stop_reason: Option<&str>, has_tool_calls: bool) -> FinishReason {
    if has_tool_calls || matches!(stop_reason, Some("tool_use")) {
        return FinishReason::ToolCalls;
    }

    match stop_reason {
        Some("max_tokens") => FinishReason::Length,
        Some("end_turn") | Some("stop_sequence") | None => FinishReason::Stop,
        Some(_) => FinishReason::Error,
    }
}

fn payload_to_string(payload: &Payload) -> Result<String, ProviderError> {
    match payload {
        Payload::Text(text) => Ok(text.clone()),
        Payload::Json(value) => serde_json::to_string(value).map_err(|err| {
            ProviderError::Message(format!("failed to serialize payload json: {err}"))
        }),
        Payload::Artifact(id) => Ok(id.get().to_string()),
    }
}

fn map_transport_error(err: reqwest::Error) -> ProviderError {
    if err.is_timeout() {
        ProviderError::Timeout
    } else {
        ProviderError::Temporary(err.to_string())
    }
}

fn map_status_error(status: StatusCode, body: Option<String>) -> ProviderError {
    match status {
        StatusCode::TOO_MANY_REQUESTS => ProviderError::RateLimited,
        StatusCode::REQUEST_TIMEOUT | StatusCode::GATEWAY_TIMEOUT => ProviderError::Timeout,
        status if status.is_server_error() => {
            ProviderError::Temporary(body.unwrap_or_else(|| status.to_string()))
        }
        _ => ProviderError::Message(body.unwrap_or_else(|| status.to_string())),
    }
}
