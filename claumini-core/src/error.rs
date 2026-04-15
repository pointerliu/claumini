use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error(transparent)]
    Provider(#[from] ProviderError),
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Tool(#[from] ToolError),
    #[error("agent failed: {0}")]
    Message(String),
}

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("provider request timed out")]
    Timeout,
    #[error("provider rate limited the request")]
    RateLimited,
    #[error("temporary upstream failure: {0}")]
    Temporary(String),
    #[error("provider '{provider}' is not implemented")]
    Unimplemented { provider: &'static str },
    #[error("provider failed: {0}")]
    Message(String),
}

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("tool input is invalid: {0}")]
    InvalidInput(String),
    #[error("tool execution failed: {0}")]
    ExecutionFailed(String),
    #[error("tool call timed out")]
    Timeout,
}

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("runtime limit '{limit}' exceeded with value {value}")]
    LimitExceeded { limit: &'static str, value: usize },
    #[error("skill '{name}' was not found")]
    MissingSkill { name: String },
    #[error("runtime failed: {0}")]
    Message(String),
}

#[derive(Debug, Error)]
pub enum PayloadError {
    #[error("payload does not contain json")]
    ExpectedJson,
    #[error("failed to deserialize payload json: {0}")]
    Deserialize(#[source] serde_json::Error),
}
