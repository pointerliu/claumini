use std::sync::Arc;

use claumini_core::{ArtifactId, Message, Payload, SessionMetadata, ToolCall};
use serde::{Deserialize, Serialize};

use crate::{ArtifactRecord, ArtifactStore};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Running,
    Waiting,
    Finished,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChildHandleRecord {
    pub id: String,
    pub session_id: String,
    pub completed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub index: usize,
    pub call: ToolCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    pub metadata: SessionMetadata,
    pub state: SessionState,
    pub current_input: Option<Payload>,
    pub transcript: Vec<Message>,
    pub tool_calls: Vec<ToolCallRecord>,
    pub artifact_ids: Vec<ArtifactId>,
    pub children: Vec<ChildHandleRecord>,
    #[serde(skip, default)]
    artifact_store: Option<Arc<ArtifactStore>>,
}

impl PartialEq for SessionRecord {
    fn eq(&self, other: &Self) -> bool {
        self.metadata == other.metadata
            && self.state == other.state
            && self.current_input == other.current_input
            && self.transcript == other.transcript
            && self.tool_calls == other.tool_calls
            && self.artifact_ids == other.artifact_ids
            && self.children == other.children
    }
}

impl SessionRecord {
    pub fn new(session_id: impl Into<String>) -> Self {
        Self::from_metadata(SessionMetadata::root(session_id))
    }

    pub(crate) fn from_metadata(metadata: SessionMetadata) -> Self {
        Self {
            metadata,
            state: SessionState::Running,
            current_input: None,
            transcript: Vec::new(),
            tool_calls: Vec::new(),
            artifact_ids: Vec::new(),
            children: Vec::new(),
            artifact_store: None,
        }
    }

    pub fn resolve_artifact(&self, id: ArtifactId) -> Option<ArtifactRecord> {
        self.artifact_store.as_ref()?.get(id)
    }

    pub(crate) fn with_artifact_store(
        mut self,
        artifact_store: Option<Arc<ArtifactStore>>,
    ) -> Self {
        self.artifact_store = artifact_store;
        self
    }
}
