use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use claumini_core::RuntimeError;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillMetadata {
    pub name: String,
    pub description: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
struct SkillEntry {
    metadata: SkillMetadata,
}

#[derive(Debug, Clone, Default)]
pub struct SkillRegistry {
    entries: BTreeMap<String, SkillEntry>,
}

impl SkillRegistry {
    pub fn scan<I, P>(roots: I) -> Result<Self, RuntimeError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let mut entries = BTreeMap::new();

        for root in roots {
            let root = root.as_ref();

            for entry in WalkDir::new(root) {
                let entry = entry.map_err(|error| {
                    RuntimeError::Message(format!(
                        "failed to walk skills root '{}': {error}",
                        root.display()
                    ))
                })?;

                if !entry.file_type().is_file() || entry.file_name() != "SKILL.md" {
                    continue;
                }

                let path = entry.into_path();
                let body = fs::read_to_string(&path).map_err(|error| {
                    RuntimeError::Message(format!(
                        "failed to read skill '{}': {error}",
                        path.display()
                    ))
                })?;
                let metadata = SkillMetadata {
                    name: skill_name_from_path(&path)?,
                    description: extract_description(&body),
                    path,
                };

                if entries
                    .insert(
                        metadata.name.clone(),
                        SkillEntry {
                            metadata: metadata.clone(),
                        },
                    )
                    .is_some()
                {
                    return Err(RuntimeError::Message(format!(
                        "duplicate skill name '{}': skill directories must be unique",
                        metadata.name
                    )));
                }
            }
        }

        Ok(Self { entries })
    }

    pub fn metadata(&self) -> Vec<SkillMetadata> {
        self.entries
            .values()
            .map(|entry| entry.metadata.clone())
            .collect()
    }

    pub fn get(&self, name: &str) -> Option<&SkillMetadata> {
        self.entries.get(name).map(|entry| &entry.metadata)
    }

    pub fn load(&self, name: &str) -> Result<String, RuntimeError> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| RuntimeError::MissingSkill {
                name: name.to_owned(),
            })?;

        fs::read_to_string(&entry.metadata.path).map_err(|error| {
            RuntimeError::Message(format!(
                "failed to load skill '{}': {error}",
                entry.metadata.path.display()
            ))
        })
    }
}

fn skill_name_from_path(path: &Path) -> Result<String, RuntimeError> {
    path.parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .ok_or_else(|| {
            RuntimeError::Message(format!(
                "failed to derive skill name from '{}'",
                path.display()
            ))
        })
}

fn extract_description(body: &str) -> String {
    body.lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#'))
        .unwrap_or_default()
        .to_owned()
}
