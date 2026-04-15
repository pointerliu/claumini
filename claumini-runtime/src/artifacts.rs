use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use claumini_core::{ArtifactId, Payload, RuntimeError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ArtifactBody {
    Payload(Payload),
    Bytes(Vec<u8>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRecord {
    pub id: ArtifactId,
    pub name: Option<String>,
    pub media_type: Option<String>,
    pub body: ArtifactBody,
}

#[derive(Debug, Default)]
pub struct ArtifactStore {
    next_id: AtomicU64,
    records: Mutex<HashMap<ArtifactId, ArtifactRecord>>,
}

impl ArtifactStore {
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            records: Mutex::new(HashMap::new()),
        }
    }

    pub fn insert_payload(&self, payload: Payload) -> Result<ArtifactId, RuntimeError> {
        self.insert_record(None, None, ArtifactBody::Payload(payload))
    }

    pub fn insert_bytes(
        &self,
        name: Option<&str>,
        media_type: Option<&str>,
        bytes: impl Into<Vec<u8>>,
    ) -> ArtifactId {
        self.insert_record(name, media_type, ArtifactBody::Bytes(bytes.into()))
            .expect("in-memory artifact insertion should not fail")
    }

    pub fn get(&self, id: ArtifactId) -> Option<ArtifactRecord> {
        self.records.lock().ok()?.get(&id).cloned()
    }

    pub fn len(&self) -> usize {
        self.records
            .lock()
            .map(|records| records.len())
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn insert_record(
        &self,
        name: Option<&str>,
        media_type: Option<&str>,
        body: ArtifactBody,
    ) -> Result<ArtifactId, RuntimeError> {
        let id = ArtifactId::new(self.next_id.fetch_add(1, Ordering::Relaxed));
        let record = ArtifactRecord {
            id,
            name: name.map(ToOwned::to_owned),
            media_type: media_type.map(ToOwned::to_owned),
            body,
        };

        self.records
            .lock()
            .map_err(|_| RuntimeError::Message("artifact store lock poisoned".to_owned()))?
            .insert(id, record);

        Ok(id)
    }
}
