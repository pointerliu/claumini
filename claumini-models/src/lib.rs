#![forbid(unsafe_code)]

mod claude;
mod config;
mod mock;
mod openai;

pub use claude::ClaudeProvider;
pub use config::{ClaudeConfig, ConfigError, OpenAiCompatibleConfig, OpenAiConfig};
pub use mock::MockProvider;
pub use openai::{OpenAiCompatibleProvider, OpenAiProvider};

/// Synthetic tool name injected into provider requests when the caller sets
/// `ModelRequest::response_schema`. The model emits its final structured answer
/// by calling this tool; the provider converts that call into the assistant's
/// final message text so `decode_json_output` can parse it.
pub const FINAL_ANSWER_TOOL_NAME: &str = "__claumini_final_answer";

pub(crate) const FINAL_ANSWER_TOOL_DESCRIPTION: &str = "Return the final answer as a structured object matching the requested schema. Call this exactly once, after all other investigation is complete, with the definitive answer. Do not call other tools after this one.";
