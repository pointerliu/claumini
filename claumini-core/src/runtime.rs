use serde::{Deserialize, Serialize};

use crate::{ArtifactId, Payload};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeLimits {
    pub max_turns_per_session: usize,
    pub max_spawn_depth: usize,
    pub max_active_children: usize,
    pub model_request_timeout_ms: u64,
    pub tool_call_timeout_ms: u64,
}

impl Default for RuntimeLimits {
    fn default() -> Self {
        Self {
            max_turns_per_session: 32,
            max_spawn_depth: 4,
            max_active_children: 8,
            model_request_timeout_ms: 30_000,
            tool_call_timeout_ms: 10_000,
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
