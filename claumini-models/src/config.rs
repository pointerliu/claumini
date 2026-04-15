use serde::{Deserialize, Serialize};
use thiserror::Error;
use url::Url;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ConfigError {
    #[error("invalid provider config field '{field}': {message}")]
    InvalidField {
        field: &'static str,
        message: &'static str,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiCompatibleConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

impl OpenAiCompatibleConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        validate_http_url("base_url", &self.base_url)?;
        validate_not_blank("api_key", &self.api_key)?;
        validate_not_blank("model", &self.model)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAiConfig {
    pub api_key: String,
    pub model: String,
}

impl OpenAiConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        validate_not_blank("api_key", &self.api_key)?;
        validate_not_blank("model", &self.model)?;
        Ok(())
    }

    pub fn into_compatible_config(self, base_url: String) -> OpenAiCompatibleConfig {
        OpenAiCompatibleConfig {
            base_url,
            api_key: self.api_key,
            model: self.model,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaudeConfig {
    pub api_key: String,
    pub model: String,
}

impl ClaudeConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        validate_not_blank("api_key", &self.api_key)?;
        validate_not_blank("model", &self.model)?;
        Ok(())
    }
}

pub(crate) fn validate_not_blank(field: &'static str, value: &str) -> Result<(), ConfigError> {
    if value.trim().is_empty() {
        return Err(ConfigError::InvalidField {
            field,
            message: "must not be blank",
        });
    }

    Ok(())
}

pub(crate) fn validate_http_url(field: &'static str, value: &str) -> Result<(), ConfigError> {
    let parsed = Url::parse(value).map_err(|_| ConfigError::InvalidField {
        field,
        message: "must be an absolute http or https URL",
    })?;

    if matches!(parsed.scheme(), "http" | "https") {
        Ok(())
    } else {
        Err(ConfigError::InvalidField {
            field,
            message: "must be an absolute http or https URL",
        })
    }
}
