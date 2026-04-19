use serde::{Deserialize, Serialize};

use crate::{ArtifactId, Payload};

/// How a `PromptAgent` loop should behave when `max_turns_per_session` is reached.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MaxTurnsPolicy {
    /// Abort the session with `RuntimeError::LimitExceeded` (default, preserves v0 behavior).
    Error,
    /// Append a nudge message telling the LLM its tool budget is exhausted and request one
    /// final response with no further tool calls. The provider is called one more time with
    /// an empty tool list, and whatever it returns is decoded as the session output.
    ForceFinal {
        /// Optional override for the nudge text. `None` uses the built-in default.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        nudge: Option<String>,
    },
}

impl Default for MaxTurnsPolicy {
    fn default() -> Self {
        Self::Error
    }
}

impl MaxTurnsPolicy {
    pub const DEFAULT_NUDGE: &'static str = "You have exhausted the tool-use budget for this session. \
         Do not call any more tools. Produce your FINAL response now in the required output format, \
         using the information you already have. If you were about to call a tool, make your best \
         judgement call based on current evidence instead.";

    pub fn nudge_text(&self) -> Option<&str> {
        match self {
            Self::Error => None,
            Self::ForceFinal { nudge } => Some(nudge.as_deref().unwrap_or(Self::DEFAULT_NUDGE)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeLimits {
    pub max_turns_per_session: usize,
    pub max_spawn_depth: usize,
    pub max_active_children: usize,
    pub model_request_timeout_ms: u64,
    pub tool_call_timeout_ms: u64,
    /// Behavior when `max_turns_per_session` is exceeded. Defaults to `Error`.
    #[serde(default)]
    pub max_turns_policy: MaxTurnsPolicy,
}

impl Default for RuntimeLimits {
    fn default() -> Self {
        Self {
            max_turns_per_session: 32,
            max_spawn_depth: 4,
            max_active_children: 8,
            model_request_timeout_ms: 30_000,
            tool_call_timeout_ms: 10_000,
            max_turns_policy: MaxTurnsPolicy::Error,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub session_id: String,
    pub depth: usize,
    pub parent_session_id: Option<String>,
}

impl SessionMetadata {
    pub fn root(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            depth: 0,
            parent_session_id: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentContext {
    pub session: SessionMetadata,
    pub limits: RuntimeLimits,
    pub transcript_len: usize,
    pub tool_call_count: usize,
    pub current_input: Option<Payload>,
    pub artifact_ids: Vec<ArtifactId>,
}

impl AgentContext {
    pub fn new(session: SessionMetadata, limits: RuntimeLimits) -> Self {
        Self {
            session,
            limits,
            transcript_len: 0,
            tool_call_count: 0,
            current_input: None,
            artifact_ids: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolContext {
    pub session: SessionMetadata,
    pub call_index: usize,
    pub current_input: Option<Payload>,
    pub artifact_ids: Vec<ArtifactId>,
}

impl ToolContext {
    pub fn new(session: SessionMetadata, call_index: usize) -> Self {
        Self {
            session,
            call_index,
            current_input: None,
            artifact_ids: Vec::new(),
        }
    }
}
