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
    /// Extended-thinking block. Not user-facing: the `thinking` field carries
    /// the model's chain-of-thought and must not be treated as actionable
    /// output.
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(default)]
        thinking: String,
    },
    /// Redacted-thinking block returned when Anthropic's safety filter
    /// obscures the chain-of-thought. Opaque to us.
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        #[serde(default, rename = "data")]
        _data: String,
    },
    #[serde(other)]
    Other,
}

fn normalize_response(body: ClaudeResponse) -> Result<ModelResponse, ProviderError> {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut final_answer: Option<serde_json::Value> = None;
    let mut thinking_parts: Vec<String> = Vec::new();

    for block in body.content {
        match block {
            ClaudeResponseContent::Text { text } => {
                if !text.is_empty() {
                    text_parts.push(text);
                }
            }
            ClaudeResponseContent::ToolUse { id, name, input } => {
                if is_final_answer_tool_name(&name) {
                    if name != FINAL_ANSWER_TOOL_NAME {
                        eprintln!(
                            "claude provider: accepting mistyped final-answer \
                             tool name '{name}' (expected '{FINAL_ANSWER_TOOL_NAME}')"
                        );
                    }
                    final_answer = Some(input);
                    continue;
                }
                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments: input,
                });
            }
            ClaudeResponseContent::Thinking { thinking } => {
                if !thinking.is_empty() {
                    thinking_parts.push(thinking);
                }
            }
            ClaudeResponseContent::RedactedThinking { .. } => {
                thinking_parts.push("[redacted_thinking]".into());
            }
            ClaudeResponseContent::Other => {}
        }
    }

    let thinking_text = if thinking_parts.is_empty() {
        None
    } else {
        Some(thinking_parts.join("\n"))
    };

    if let Some(arguments) = final_answer {
        let text = serde_json::to_string(&arguments).map_err(|err| {
            ProviderError::Message(format!("failed to serialize final answer: {err}"))
        })?;
        let mut msg = Message::new(MessageRole::Assistant, Payload::text(text));
        if let Some(t) = thinking_text {
            msg = msg.with_thinking(t);
        }
        return Ok(ModelResponse {
            message: Some(msg),
            tool_calls,
            finish_reason: FinishReason::Stop,
        });
    }

    // If the turn carried only a thinking/redacted-thinking block and no
    // actionable output, the runtime has nothing to decode. Return a
    // temporary error so the retry layer can try again — the thinking text
    // itself is dropped here but would be lost anyway since this turn never
    // lands on the transcript.
    if text_parts.is_empty()
        && tool_calls.is_empty()
        && let Some(ref t) = thinking_text
    {
        let preview: String = t.chars().take(200).collect();
        return Err(ProviderError::Temporary(format!(
            "claude returned only a thinking/redacted_thinking block with \
             no text or tool_use; preview: {preview:?}"
        )));
    }

    let message = if text_parts.is_empty() {
        // No text, but we have tool_calls and/or thinking — still build a
        // message so the thinking is preserved on the transcript.
        thinking_text
            .as_ref()
            .map(|_| Message::new(MessageRole::Assistant, Payload::text(String::new())))
    } else {
        Some(Message::new(
            MessageRole::Assistant,
            Payload::text(text_parts.join("\n")),
        ))
    };
    let message = match (message, thinking_text) {
        (Some(msg), Some(t)) => Some(msg.with_thinking(t)),
        (Some(msg), None) => Some(msg),
        (None, _) => None,
    };

    Ok(ModelResponse {
        message,
        finish_reason: map_finish_reason(body.stop_reason.as_deref(), !tool_calls.is_empty()),
        tool_calls,
    })
}

/// Matches the reserved final-answer tool name, tolerating small typos in
/// the model's output. Claude sometimes misspells the tool name (e.g.
/// `__claumaini_final_answer` with an extra `a`); strict equality would
/// then drop the structured answer and fail the run with
/// "model called unavailable tool". We accept any name that follows the
/// reserved `__*_final_answer` shape to normalise it to the final answer.
fn is_final_answer_tool_name(name: &str) -> bool {
    if name == FINAL_ANSWER_TOOL_NAME {
        return true;
    }
    name.starts_with("__") && name.ends_with("_final_answer")
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
