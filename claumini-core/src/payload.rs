use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::PayloadError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ArtifactId(u64);

impl ArtifactId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

impl From<u64> for ArtifactId {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum Payload {
    Text(String),
    Json(Value),
    Artifact(ArtifactId),
}

impl Payload {
    pub fn text(value: impl Into<String>) -> Self {
        Self::Text(value.into())
    }

    pub fn json<T>(value: T) -> Result<Self, serde_json::Error>
    where
        T: Serialize,
    {
        serde_json::to_value(value).map(Self::Json)
    }

    pub const fn artifact(id: ArtifactId) -> Self {
        Self::Artifact(id)
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(value) => Some(value),
            Self::Json(_) | Self::Artifact(_) => None,
        }
    }

    pub fn as_json(&self) -> Option<&Value> {
        match self {
            Self::Json(value) => Some(value),
            Self::Text(_) | Self::Artifact(_) => None,
        }
    }

    pub const fn as_artifact(&self) -> Option<ArtifactId> {
        match self {
            Self::Artifact(id) => Some(*id),
            Self::Text(_) | Self::Json(_) => None,
        }
    }

    pub fn to_typed<T>(&self) -> Result<T, PayloadError>
    where
        T: DeserializeOwned,
    {
        match self {
            Self::Json(value) => {
                serde_json::from_value(value.clone()).map_err(PayloadError::Deserialize)
            }
            Self::Text(_) | Self::Artifact(_) => Err(PayloadError::ExpectedJson),
        }
    }
}
