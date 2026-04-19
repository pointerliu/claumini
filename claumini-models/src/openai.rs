use async_trait::async_trait;
use claumini_core::{
    FinishReason, Message, MessageRole, ModelProvider, ModelRequest, ModelResponse, Payload,
    ProviderCapabilities, ProviderError, ToolCall, ToolSchema,
};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    ConfigError, FINAL_ANSWER_TOOL_DESCRIPTION, FINAL_ANSWER_TOOL_NAME, OpenAiCompatibleConfig,
    OpenAiConfig,
};

const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    config: OpenAiConfig,
    inner: OpenAiCompatibleProvider,
}

impl OpenAiProvider {
    pub fn new(config: OpenAiConfig) -> Result<Self, ConfigError> {
        Self::new_with_base_url(config, OPENAI_BASE_URL)
    }

    pub fn new_with_base_url(
        config: OpenAiConfig,
        base_url: impl Into<String>,
    ) -> Result<Self, ConfigError> {
        config.validate()?;
        let inner =
            OpenAiCompatibleProvider::new(config.clone().into_compatible_config(base_url.into()))?;

        Ok(Self { config, inner })
    }

    pub fn config(&self) -> &OpenAiConfig {
        &self.config
    }
}

#[async_trait]
impl ModelProvider for OpenAiProvider {
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, ProviderError> {
        self.inner.complete(request).await
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.inner.capabilities()
    }
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleProvider {
    config: OpenAiCompatibleConfig,
    client: Client,
}

impl OpenAiCompatibleProvider {
    pub fn new(config: OpenAiCompatibleConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        Ok(Self {
            config,
            client: Client::new(),
        })
    }

    pub fn config(&self) -> &OpenAiCompatibleConfig {
        &self.config
    }
}

#[async_trait]
impl ModelProvider for OpenAiCompatibleProvider {
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, ProviderError> {
        let chat_request = OpenAiChatRequest::from_request(&self.config.model, request.clone())?;
        let response = self
            .client
            .post(format!(
                "{}/chat/completions",
                self.config.base_url.trim_end_matches('/')
            ))
            .bearer_auth(&self.config.api_key)
            .json(&chat_request)
            .send()
            .await
            .map_err(map_transport_error)?;

        let status = response.status();
        if !status.is_success() {
            return Err(map_status_error(status, response.text().await.ok()));
        }

        let body = response.json::<OpenAiChatResponse>().await.map_err(|err| {
            ProviderError::Message(format!("failed to decode openai response: {err}"))
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
struct OpenAiChatRequest {
    model: String,
    messages: Vec<OpenAiRequestMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

impl OpenAiChatRequest {
    fn from_request(model: &str, request: ModelRequest) -> Result<Self, ProviderError> {
        let mut messages = Vec::with_capacity(
            request.messages.len() + usize::from(request.system_prompt.is_some()),
        );
        if let Some(system_prompt) = request.system_prompt {
            messages.push(OpenAiRequestMessage::text("system", system_prompt));
        }

        for message in request.messages {
            messages.push(OpenAiRequestMessage::from_message(message)?);
        }

        let mut tools: Vec<OpenAiTool> =
            request.tools.iter().map(OpenAiTool::from_tool).collect();
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
            tools.push(OpenAiTool::final_answer(schema));
        }

        Ok(Self {
            model: model.to_owned(),
            messages,
            tools,
            max_tokens: request.max_output_tokens,
        })
    }
}

#[derive(Debug, Serialize)]
struct OpenAiRequestMessage {
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiRequestToolCall>>,
}

impl OpenAiRequestMessage {
    fn text(role: &'static str, content: String) -> Self {
        Self {
            role,
            content: Some(content),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    fn from_message(message: Message) -> Result<Self, ProviderError> {
        match message.role {
            MessageRole::System => Ok(Self::text("system", payload_to_string(&message.content)?)),
            MessageRole::User => Ok(Self::text("user", payload_to_string(&message.content)?)),
            MessageRole::Assistant if !message.tool_calls.is_empty() => Ok(Self {
                role: "assistant",
                content: Some(payload_to_string(&message.content)?),
                tool_call_id: None,
                tool_calls: Some(
                    message
                        .tool_calls
                        .into_iter()
                        .map(OpenAiRequestToolCall::from_tool_call)
                        .collect(),
                ),
            }),
            MessageRole::Assistant => Ok(Self::text(
                "assistant",
                payload_to_string(&message.content)?,
            )),
            MessageRole::Tool => Ok(Self {
                role: "tool",
                content: Some(payload_to_string(&message.content)?),
                tool_call_id: Some(message.name.unwrap_or_else(|| "tool".into())),
                tool_calls: None,
            }),
        }
    }
}

#[derive(Debug, Serialize)]
struct OpenAiRequestToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiRequestFunctionCall,
}

impl OpenAiRequestToolCall {
    fn from_tool_call(call: ToolCall) -> Self {
        Self {
            id: call.id,
            kind: "function",
            function: OpenAiRequestFunctionCall {
                name: call.name,
                arguments: serde_json::to_string(&call.arguments)
                    .expect("tool call arguments should serialize"),
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct OpenAiRequestFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct OpenAiTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiFunctionDefinition,
}

impl OpenAiTool {
    fn from_tool(tool: &ToolSchema) -> Self {
        Self {
            kind: "function",
            function: OpenAiFunctionDefinition {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.input_schema.clone(),
            },
        }
    }

    fn final_answer(schema: Value) -> Self {
        Self {
            kind: "function",
            function: OpenAiFunctionDefinition {
                name: FINAL_ANSWER_TOOL_NAME.to_owned(),
                description: FINAL_ANSWER_TOOL_DESCRIPTION.to_owned(),
                parameters: schema,
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct OpenAiFunctionDefinition {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Deserialize)]
struct OpenAiChatResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    finish_reason: Option<String>,
    message: OpenAiAssistantMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAiAssistantMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OpenAiResponseToolCall>,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseToolCall {
    id: String,
    function: OpenAiResponseFunction,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseFunction {
    name: String,
    arguments: String,
}

fn normalize_response(body: OpenAiChatResponse) -> Result<ModelResponse, ProviderError> {
    let choice =
        body.choices.into_iter().next().ok_or_else(|| {
            ProviderError::Message("openai response did not include a choice".into())
        })?;

    let OpenAiAssistantMessage {
        content,
        tool_calls: raw_tool_calls,
    } = choice.message;

    let mut tool_calls = Vec::with_capacity(raw_tool_calls.len());
    let mut final_answer: Option<Value> = None;
    for call in raw_tool_calls {
        let arguments: Value = serde_json::from_str(&call.function.arguments).map_err(|err| {
            ProviderError::Message(format!("failed to parse openai tool arguments: {err}"))
        })?;
        if call.function.name == FINAL_ANSWER_TOOL_NAME {
            final_answer = Some(arguments);
            continue;
        }
        tool_calls.push(ToolCall {
            id: call.id,
            name: call.function.name,
            arguments,
        });
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

    let message = content.and_then(|content| {
        if content.is_empty() {
            None
        } else {
            Some(Message::new(MessageRole::Assistant, Payload::text(content)))
        }
    });

    Ok(ModelResponse {
        finish_reason: map_finish_reason(choice.finish_reason.as_deref(), !tool_calls.is_empty()),
        message,
        tool_calls,
    })
}

fn map_finish_reason(finish_reason: Option<&str>, has_tool_calls: bool) -> FinishReason {
    if has_tool_calls {
        return FinishReason::ToolCalls;
    }

    match finish_reason {
        Some("length") => FinishReason::Length,
        Some("stop") | Some("content_filter") | None => FinishReason::Stop,
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
