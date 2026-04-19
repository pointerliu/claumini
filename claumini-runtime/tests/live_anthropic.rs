use std::env;
use std::fs;
use std::sync::{Arc, Once};

use async_trait::async_trait;
use claumini_core::{
    Message, MessageRole, ModelProvider, ModelRequest, Payload, Tool, ToolDescriptor, ToolError,
    ToolSchema,
};
use claumini_models::{ClaudeConfig, ClaudeProvider};
use claumini_runtime::{PromptAgentBuilder, SkillRegistry};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tempfile::tempdir;

static LOAD_ENV: Once = Once::new();

#[derive(Debug, Clone)]
struct LiveAnthropicConfig {
    api_base: String,
    api_key: String,
    model: String,
}

impl LiveAnthropicConfig {
    fn from_env() -> Self {
        LOAD_ENV.call_once(|| {
            let _ = dotenvy::from_filename(".env");
        });

        Self {
            api_base: required_env("ANTHROPIC_API_BASE"),
            api_key: required_env("ANTHROPIC_API_KEY"),
            model: required_env("ANTHROPIC_MODEL"),
        }
    }

    fn provider(&self) -> ClaudeProvider {
        ClaudeProvider::new_with_base_url(
            ClaudeConfig {
                api_key: self.api_key.clone(),
                model: self.model.clone(),
            },
            self.api_base.clone(),
        )
        .expect("live anthropic provider config should be valid")
    }
}

fn required_env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| panic!("missing required env var {name}"))
}

#[tokio::test]
#[ignore = "requires live Anthropic-compatible credentials in .env"]
async fn live_anthropic_provider_returns_basic_text_response() {
    let provider = LiveAnthropicConfig::from_env().provider();
    let response = provider
        .complete(
            ModelRequest::new(vec![Message::new(
                MessageRole::User,
                Payload::text("Reply with exactly BASIC_OK and nothing else."),
            )])
            .with_system_prompt("Follow the user's output format exactly.")
            .with_max_output_tokens(2048),
        )
        .await
        .expect("basic live completion should succeed");

    println!("basic response: {response:#?}");

    let text = response
        .message
        .expect("basic response should include a message")
        .content
        .as_text()
        .expect("basic response should be text")
        .trim()
        .to_owned();

    assert_eq!(text, "BASIC_OK");
    assert!(response.tool_calls.is_empty());
}

#[tokio::test]
#[ignore = "requires live Anthropic-compatible credentials in .env"]
async fn live_anthropic_provider_emits_native_tool_calls() {
    let provider = LiveAnthropicConfig::from_env().provider();
    let response = provider
        .complete(
            ModelRequest::new(vec![Message::new(
                MessageRole::User,
                Payload::text(
                    "Call the provider_probe tool exactly once with marker TOOL_OK. Do not answer in plain text.",
                ),
            )])
            .with_system_prompt("Use the available tool exactly as requested.")
            .with_tool(ToolSchema {
                name: "provider_probe".into(),
                description: "Smoke-test tool used to verify native tool calling.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "marker": { "type": "string" }
                    },
                    "required": ["marker"],
                    "additionalProperties": false
                }),
            })
            .with_max_output_tokens(256),
        )
        .await
        .expect("live tool-call completion should succeed");

    println!("tool-call response: {response:#?}");

    assert!(
        !response.tool_calls.is_empty(),
        "provider did not emit a tool call"
    );
    assert_eq!(response.tool_calls[0].name, "provider_probe");
    assert_eq!(
        response.tool_calls[0].arguments,
        json!({"marker": "TOOL_OK"})
    );
}

#[derive(Debug, Serialize, Deserialize)]
struct ProbeInput {
    marker: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ProbeOutput {
    marker: String,
}

struct ProbeTool;

#[async_trait]
impl Tool for ProbeTool {
    type Input = ProbeInput;
    type Output = ProbeOutput;

    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            name: "probe_tool".into(),
            description: "Returns the provided marker for runtime smoke testing.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "marker": { "type": "string" }
                },
                "required": ["marker"],
                "additionalProperties": false
            }),
            output_schema: None,
        }
    }

    async fn call(
        &self,
        input: Self::Input,
        _ctx: &mut claumini_core::ToolContext,
    ) -> Result<Self::Output, ToolError> {
        Ok(ProbeOutput {
            marker: input.marker,
        })
    }
}

#[tokio::test]
#[ignore = "requires live Anthropic-compatible credentials in .env"]
async fn live_anthropic_runtime_executes_tool_and_loads_skill() {
    let provider: Arc<dyn ModelProvider> = Arc::new(LiveAnthropicConfig::from_env().provider());
    let temp = tempdir().expect("tempdir should exist");
    let skill_dir = temp.path().join("runtime-smoke");
    fs::create_dir_all(&skill_dir).expect("skill dir should exist");
    fs::write(
        skill_dir.join("SKILL.md"),
        "# Runtime Smoke\n\nLive skill description.\n\nLive skill body marker\n",
    )
    .expect("skill file should be written");
    let registry = SkillRegistry::scan([temp.path()]).expect("skill registry should scan");

    let agent = PromptAgentBuilder::new(provider)
        .system_prompt(
            "You are running a smoke test. Call `probe_tool` exactly once with marker `RUNTIME_TOOL_OK`. Then call `load_skill` with name `runtime-smoke`. Then call `finish` with text exactly `RUNTIME_TOOL_OK | Live skill body marker`.",
        )
        .tool(ProbeTool)
        .skills(registry)
        .text_input()
        .text_output()
        .build()
        .expect("agent should build");

    let result = agent
        .run(
            "run the smoke test".to_owned(),
            "live-anthropic-runtime-skill",
        )
        .await
        .expect("runtime smoke test should complete");

    println!("runtime output: {}", result.output);
    println!("runtime tool calls: {:#?}", result.session.tool_calls);
    println!("runtime transcript: {:#?}", result.session.transcript);

    assert_eq!(
        result.output.trim(),
        "RUNTIME_TOOL_OK | Live skill body marker"
    );

    let tool_names: Vec<_> = result
        .session
        .tool_calls
        .iter()
        .map(|record| record.call.name.as_str())
        .collect();
    assert!(tool_names.contains(&"probe_tool"));
    assert!(tool_names.contains(&"load_skill"));
    assert!(tool_names.contains(&"finish"));
}
