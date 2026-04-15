use std::collections::VecDeque;
use std::fs;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use claumini_core::{
    Agent, AgentContext, AgentError, ArtifactId, FinishReason, MessageRole, ModelProvider,
    ModelRequest, ModelResponse, Payload, ProviderCapabilities, ProviderError, RuntimeLimits,
    SessionMetadata, Tool, ToolContext, ToolDescriptor, ToolError,
};
use claumini_runtime::{
    ArtifactBody, ArtifactStore, ChildContextMode, ChildRegistration, ChildToolPolicy,
    FINISH_TOOL_NAME, PromptAgentBuilder, PromptSession, ReservedRuntimeTools, RuntimeTool,
    SkillRegistry,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tempfile::tempdir;
use tokio::time::sleep;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct UserQuestion {
    question: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct AgentAnswer {
    answer: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct SchemaAnswer {
    answer: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct LookupInput {
    question: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ChildJsonInput {
    topic: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct ChildJsonOutput {
    answer: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ToolContextOutput {
    current_input: Payload,
    artifact_ids: Vec<u64>,
    call_index: usize,
}

#[derive(Debug, Clone)]
struct ScriptedProvider {
    requests: Arc<Mutex<Vec<ModelRequest>>>,
    errors: Arc<Mutex<VecDeque<ProviderError>>>,
    responses: Arc<Mutex<VecDeque<ModelResponse>>>,
    delay: Duration,
    delays: Arc<Mutex<VecDeque<Duration>>>,
    capabilities: ProviderCapabilities,
}

impl ScriptedProvider {
    fn new(responses: Vec<ModelResponse>) -> Self {
        Self {
            requests: Arc::new(Mutex::new(Vec::new())),
            errors: Arc::new(Mutex::new(VecDeque::new())),
            responses: Arc::new(Mutex::new(responses.into())),
            delay: Duration::ZERO,
            delays: Arc::new(Mutex::new(VecDeque::new())),
            capabilities: ProviderCapabilities {
                native_tool_calling: true,
                ..ProviderCapabilities::default()
            },
        }
    }

    fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    fn with_delays(mut self, delays: Vec<Duration>) -> Self {
        self.delays = Arc::new(Mutex::new(delays.into()));
        self
    }

    fn with_capabilities(mut self, capabilities: ProviderCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    fn with_errors(mut self, errors: Vec<ProviderError>) -> Self {
        self.errors = Arc::new(Mutex::new(errors.into()));
        self
    }

    fn requests(&self) -> Vec<ModelRequest> {
        self.requests.lock().expect("requests lock").clone()
    }
}

#[async_trait]
impl ModelProvider for ScriptedProvider {
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, ProviderError> {
        self.requests.lock().expect("requests lock").push(request);

        let delay = self
            .delays
            .lock()
            .expect("delays lock")
            .pop_front()
            .unwrap_or(self.delay);
        if !delay.is_zero() {
            sleep(delay).await;
        }

        if let Some(error) = self.errors.lock().expect("errors lock").pop_front() {
            return Err(error);
        }

        self.responses
            .lock()
            .expect("responses lock")
            .pop_front()
            .ok_or_else(|| ProviderError::Message("no scripted response remaining".into()))
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.capabilities
    }
}

fn fallback_tool_calls(tool_calls: serde_json::Value) -> String {
    json!({
        "claumini_runtime": {
            "type": "tool_calls",
            "tool_calls": tool_calls
        }
    })
    .to_string()
}

fn fallback_final(payload: serde_json::Value) -> String {
    json!({
        "claumini_runtime": {
            "type": "final",
            "payload": payload
        }
    })
    .to_string()
}

fn build_skill_registry() -> SkillRegistry {
    let temp = tempdir().expect("tempdir should exist");
    let skill_dir = temp.path().join("runtime-skill");
    fs::create_dir_all(&skill_dir).expect("skill dir should exist");
    fs::write(
        skill_dir.join("SKILL.md"),
        "# Runtime Skill\n\nHelpful runtime description.\n\nFull runtime skill body.\n",
    )
    .expect("skill file should exist");

    let registry = SkillRegistry::scan([temp.path()]).expect("skill registry should scan");
    std::mem::forget(temp);
    registry
}

#[derive(Debug, Clone)]
struct LookupTool;

#[async_trait]
impl Tool for LookupTool {
    type Input = LookupInput;
    type Output = AgentAnswer;

    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            name: "lookup".into(),
            description: "Look up an answer".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "question": { "type": "string" }
                },
                "required": ["question"]
            }),
            output_schema: None,
        }
    }

    async fn call(
        &self,
        input: Self::Input,
        _ctx: &mut ToolContext,
    ) -> Result<Self::Output, ToolError> {
        Ok(AgentAnswer {
            answer: format!("lookup:{}", input.question),
        })
    }
}

#[derive(Debug, Clone)]
struct ChildOnlyTool;

#[async_trait]
impl Tool for ChildOnlyTool {
    type Input = LookupInput;
    type Output = AgentAnswer;

    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            name: "child_lookup".into(),
            description: "Child-only lookup tool".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "question": { "type": "string" }
                },
                "required": ["question"]
            }),
            output_schema: None,
        }
    }

    async fn call(
        &self,
        input: Self::Input,
        _ctx: &mut ToolContext,
    ) -> Result<Self::Output, ToolError> {
        Ok(AgentAnswer {
            answer: format!("child:{}", input.question),
        })
    }
}

#[derive(Debug, Clone)]
struct SlowRuntimeTool {
    descriptor: ToolDescriptor,
    delay: Duration,
}

impl SlowRuntimeTool {
    fn new(delay: Duration) -> Self {
        Self {
            descriptor: ToolDescriptor {
                name: "slow_tool".into(),
                description: "Sleep before responding".into(),
                input_schema: json!({"type": "object"}),
                output_schema: None,
            },
            delay,
        }
    }
}

#[async_trait]
impl RuntimeTool for SlowRuntimeTool {
    fn descriptor(&self) -> ToolDescriptor {
        self.descriptor.clone()
    }

    async fn call_payload(
        &self,
        _input: Payload,
        _ctx: &mut ToolContext,
    ) -> Result<Payload, ToolError> {
        sleep(self.delay).await;
        Ok(Payload::json(json!({"ok": true})).expect("json payload"))
    }
}

#[derive(Debug, Clone)]
struct FailingRuntimeTool;

#[async_trait]
impl RuntimeTool for FailingRuntimeTool {
    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            name: "failing_tool".into(),
            description: "Always fails".into(),
            input_schema: json!({"type": "object"}),
            output_schema: None,
        }
    }

    async fn call_payload(
        &self,
        _input: Payload,
        _ctx: &mut ToolContext,
    ) -> Result<Payload, ToolError> {
        Err(ToolError::ExecutionFailed("boom".into()))
    }
}

#[derive(Debug, Clone)]
struct ArtifactRuntimeTool {
    name: &'static str,
    artifact_id: ArtifactId,
}

#[async_trait]
impl RuntimeTool for ArtifactRuntimeTool {
    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            name: self.name.into(),
            description: "Returns an artifact payload".into(),
            input_schema: json!({"type": "object"}),
            output_schema: None,
        }
    }

    async fn call_payload(
        &self,
        _input: Payload,
        _ctx: &mut ToolContext,
    ) -> Result<Payload, ToolError> {
        Ok(Payload::artifact(self.artifact_id))
    }
}

#[derive(Debug, Clone)]
struct ContextEchoTool;

#[async_trait]
impl Tool for ContextEchoTool {
    type Input = serde_json::Value;
    type Output = ToolContextOutput;

    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            name: "context_echo".into(),
            description: "Returns runtime context details".into(),
            input_schema: json!({"type": "object"}),
            output_schema: None,
        }
    }

    async fn call(
        &self,
        _input: Self::Input,
        ctx: &mut ToolContext,
    ) -> Result<Self::Output, ToolError> {
        Ok(ToolContextOutput {
            current_input: ctx
                .current_input
                .clone()
                .expect("runtime tool context should include current input"),
            artifact_ids: ctx.artifact_ids.iter().map(|id| id.get()).collect(),
            call_index: ctx.call_index,
        })
    }
}

#[tokio::test]
async fn prompt_agent_runs_turn_loop_records_session_and_decodes_typed_output() {
    let provider = ScriptedProvider::new(vec![
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-1".into(),
                name: "lookup".into(),
                arguments: json!({"question": "Where?"}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-2".into(),
                name: FINISH_TOOL_NAME.into(),
                arguments: json!({"json": {"answer": "final answer"}}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
    ]);

    let agent = PromptAgentBuilder::new(Arc::new(provider.clone()))
        .system_prompt("You are helpful.")
        .tool(LookupTool)
        .json_input::<UserQuestion>()
        .json_output::<AgentAnswer>()
        .build()
        .expect("agent should build");

    let PromptSession { output, session } = agent
        .run(
            UserQuestion {
                question: "Where?".into(),
            },
            "session-1",
        )
        .await
        .expect("prompt agent should finish");

    assert_eq!(
        output,
        AgentAnswer {
            answer: "final answer".into()
        }
    );
    assert_eq!(session.metadata.session_id, "session-1");
    assert_eq!(
        session.current_input,
        Some(Payload::json(json!({"question": "Where?"})).unwrap())
    );
    assert_eq!(session.tool_calls.len(), 2);
    assert_eq!(session.tool_calls[0].call.name, "lookup");
    assert_eq!(session.tool_calls[1].call.name, FINISH_TOOL_NAME);
    assert_eq!(session.transcript.len(), 4);
    assert_eq!(session.transcript[0].role, MessageRole::User);
    assert_eq!(session.transcript[1].role, MessageRole::Assistant);
    assert_eq!(session.transcript[2].role, MessageRole::Tool);
    assert_eq!(session.transcript[2].name.as_deref(), Some("call-1"));
    assert_eq!(
        session.transcript[2].content,
        Payload::json(json!({"answer": "lookup:Where?"})).unwrap()
    );

    let requests = provider.requests();
    assert_eq!(requests.len(), 2);
    let system_prompt = requests[0]
        .system_prompt
        .as_deref()
        .expect("system prompt should exist");
    assert!(system_prompt.contains("You are helpful."));
    assert_eq!(requests[0].tools.len(), 2);
    assert_eq!(requests[0].tools[0].name, "lookup");
    assert_eq!(requests[0].tools[1].name, FINISH_TOOL_NAME);
    assert_eq!(requests[1].messages.len(), 3);
    assert_eq!(requests[1].messages[1].role, MessageRole::Assistant);
    assert_eq!(requests[1].messages[2].role, MessageRole::Tool);
    assert_eq!(requests[1].messages[1].tool_calls.len(), 1);
    assert_eq!(requests[1].messages[1].tool_calls[0].name, "lookup");
}

#[tokio::test]
async fn prompt_agent_enforces_max_turns_per_session() {
    let provider = ScriptedProvider::new(vec![
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-1".into(),
                name: "lookup".into(),
                arguments: json!({"question": "loop"}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-2".into(),
                name: FINISH_TOOL_NAME.into(),
                arguments: json!({"json": {"answer": "never reached"}}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
    ]);

    let agent = PromptAgentBuilder::new(Arc::new(provider))
        .system_prompt("You are helpful.")
        .tool(LookupTool)
        .limits(RuntimeLimits {
            max_turns_per_session: 1,
            ..RuntimeLimits::default()
        })
        .json_input::<UserQuestion>()
        .json_output::<AgentAnswer>()
        .build()
        .expect("agent should build");

    let error = agent
        .run(
            UserQuestion {
                question: "loop".into(),
            },
            "session-2",
        )
        .await
        .expect_err("second turn should exceed the limit");

    assert!(matches!(
        error,
        AgentError::Runtime(claumini_core::RuntimeError::LimitExceeded {
            limit: "max_turns_per_session",
            value: 2,
        })
    ));
}

#[tokio::test]
async fn prompt_agent_times_out_provider_requests() {
    let provider = ScriptedProvider::new(vec![ModelResponse::text("late")])
        .with_delay(Duration::from_millis(30));

    let agent = PromptAgentBuilder::new(Arc::new(provider))
        .system_prompt("You are helpful.")
        .limits(RuntimeLimits {
            model_request_timeout_ms: 5,
            ..RuntimeLimits::default()
        })
        .text_input()
        .text_output()
        .build()
        .expect("agent should build");

    let error = agent
        .run("hello".to_owned(), "session-3")
        .await
        .expect_err("provider timeout should fail the run");

    assert!(matches!(
        error,
        AgentError::Provider(ProviderError::Timeout)
    ));
}

#[tokio::test]
async fn prompt_agent_times_out_tools() {
    let provider = ScriptedProvider::new(vec![
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-1".into(),
                name: "slow_tool".into(),
                arguments: json!({}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-2".into(),
                name: FINISH_TOOL_NAME.into(),
                arguments: json!({"text": "done"}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
    ]);

    let agent = PromptAgentBuilder::new(Arc::new(provider))
        .system_prompt("You are helpful.")
        .runtime_tool(SlowRuntimeTool::new(Duration::from_millis(30)))
        .limits(RuntimeLimits {
            tool_call_timeout_ms: 5,
            ..RuntimeLimits::default()
        })
        .text_input()
        .text_output()
        .build()
        .expect("agent should build");

    let PromptSession { output, session } = agent
        .run("hello".to_owned(), "session-4")
        .await
        .expect("tool timeout should be surfaced to the model");

    assert_eq!(output, "done");
    assert_eq!(session.transcript[1].role, MessageRole::Assistant);
    assert_eq!(
        session.transcript[2].content,
        Payload::json(json!({
            "ok": false,
            "error": {
                "type": "timeout",
                "message": "tool call timed out"
            }
        }))
        .expect("json payload")
    );
}

#[tokio::test]
async fn prompt_agent_surfaces_tool_failures_as_structured_tool_output_and_continues() {
    let provider = ScriptedProvider::new(vec![
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-1".into(),
                name: "failing_tool".into(),
                arguments: json!({}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-2".into(),
                name: FINISH_TOOL_NAME.into(),
                arguments: json!({"text": "recovered"}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
    ]);

    let agent = PromptAgentBuilder::new(Arc::new(provider.clone()))
        .runtime_tool(FailingRuntimeTool)
        .text_input()
        .text_output()
        .build()
        .expect("agent should build");

    let PromptSession { output, session } = agent
        .run("hello".to_owned(), "session-tool-failure")
        .await
        .expect("run should continue after tool failure");

    assert_eq!(output, "recovered");
    assert_eq!(session.transcript[1].role, MessageRole::Assistant);
    assert_eq!(session.transcript[2].role, MessageRole::Tool);
    assert_eq!(session.transcript[2].name.as_deref(), Some("call-1"));
    assert_eq!(
        session.transcript[2].content,
        Payload::json(json!({
            "ok": false,
            "error": {
                "type": "execution_failed",
                "message": "boom"
            }
        }))
        .expect("json payload")
    );

    let requests = provider.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[1].messages[2].content,
        session.transcript[2].content
    );
}

#[tokio::test]
async fn prompt_agent_json_output_populates_request_schema_guidance() {
    let provider = ScriptedProvider::new(vec![ModelResponse::text(r#"{"answer":"done"}"#)]);

    let agent = PromptAgentBuilder::new(Arc::new(provider.clone()))
        .json_input::<UserQuestion>()
        .json_output::<SchemaAnswer>()
        .build()
        .expect("agent should build");

    let PromptSession { output, .. } = agent
        .run(
            UserQuestion {
                question: "Where?".into(),
            },
            "session-schema-guidance",
        )
        .await
        .expect("structured output should decode");

    assert_eq!(
        output,
        SchemaAnswer {
            answer: "done".into()
        }
    );

    let request = &provider.requests()[0];
    let schema = request
        .response_schema
        .as_ref()
        .expect("response schema should be populated");
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["properties"]["answer"]["type"], "string");
    assert_eq!(schema["required"], json!(["answer"]));
}

#[tokio::test]
async fn prompt_agent_retries_transient_provider_failures_until_success() {
    let provider =
        ScriptedProvider::new(vec![ModelResponse::text("eventual answer")]).with_errors(vec![
            ProviderError::Temporary("try again".into()),
            ProviderError::RateLimited,
        ]);

    let agent = PromptAgentBuilder::new(Arc::new(provider.clone()))
        .text_input()
        .text_output()
        .build()
        .expect("agent should build");

    let PromptSession { output, .. } = agent
        .run("hello".to_owned(), "session-provider-retry")
        .await
        .expect("transient provider failures should be retried");

    assert_eq!(output, "eventual answer");
    assert_eq!(provider.requests().len(), 3);
}

#[tokio::test]
async fn prompt_agent_retries_provider_timeouts_until_success() {
    let provider = ScriptedProvider::new(vec![ModelResponse::text("eventual answer")])
        .with_delays(vec![Duration::from_millis(20), Duration::ZERO]);

    let agent = PromptAgentBuilder::new(Arc::new(provider.clone()))
        .limits(RuntimeLimits {
            model_request_timeout_ms: 5,
            ..RuntimeLimits::default()
        })
        .text_input()
        .text_output()
        .build()
        .expect("agent should build");

    let PromptSession { output, .. } = agent
        .run("hello".to_owned(), "session-provider-timeout-retry")
        .await
        .expect("timeout should be retried until success");

    assert_eq!(output, "eventual answer");
    assert_eq!(provider.requests().len(), 2);
}

#[tokio::test]
async fn prompt_agent_allows_reserved_runtime_tools_to_be_disabled() {
    let provider = ScriptedProvider::new(vec![ModelResponse::text("done")]);
    let registry = build_skill_registry();

    let agent = PromptAgentBuilder::new(Arc::new(provider.clone()))
        .skills(registry)
        .reserved_runtime_tools(
            ReservedRuntimeTools::default()
                .with_finish(false)
                .with_load_skill(false),
        )
        .text_input()
        .text_output()
        .build()
        .expect("agent should build");

    let PromptSession { output, .. } = agent
        .run("hello".to_owned(), "session-reserved-gating")
        .await
        .expect("plain text run should succeed");

    assert_eq!(output, "done");

    let requests = provider.requests();
    let tool_names: Vec<_> = requests[0]
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect();
    assert!(!tool_names.contains(&FINISH_TOOL_NAME));
    assert!(!tool_names.contains(&"load_skill"));
}

#[tokio::test]
async fn prompt_agent_tracks_artifact_refs_across_input_tools_children_and_final_output() {
    let input_artifact = ArtifactId::new(11);
    let tool_artifact = ArtifactId::new(12);
    let unselected_artifact = ArtifactId::new(13);
    let child_artifact = ArtifactId::new(14);
    let final_artifact = ArtifactId::new(15);

    let child_provider = ScriptedProvider::new(vec![ModelResponse {
        message: None,
        tool_calls: vec![claumini_core::ToolCall {
            id: "child-finish".into(),
            name: FINISH_TOOL_NAME.into(),
            arguments: json!({
                "payload": {
                    "kind": "artifact",
                    "value": child_artifact.get()
                }
            }),
        }],
        finish_reason: FinishReason::ToolCalls,
    }]);
    let child = PromptAgentBuilder::new(Arc::new(child_provider.clone()))
        .build()
        .expect("child should build");
    let parent_provider = ScriptedProvider::new(vec![
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-1".into(),
                name: "artifact_tool".into(),
                arguments: json!({}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-2".into(),
                name: "unselected_artifact_tool".into(),
                arguments: json!({}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-3".into(),
                name: "call_subagent".into(),
                arguments: json!({
                    "child": "worker",
                    "payload": {
                        "kind": "artifact",
                        "value": input_artifact.get()
                    }
                }),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-4".into(),
                name: FINISH_TOOL_NAME.into(),
                arguments: json!({
                    "payload": {
                        "kind": "artifact",
                        "value": final_artifact.get()
                    }
                }),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
    ]);

    let agent = PromptAgentBuilder::new(Arc::new(parent_provider))
        .runtime_tool(ArtifactRuntimeTool {
            name: "artifact_tool",
            artifact_id: tool_artifact,
        })
        .runtime_tool(ArtifactRuntimeTool {
            name: "unselected_artifact_tool",
            artifact_id: unselected_artifact,
        })
        .child(
            "worker",
            ChildRegistration::new(child)
                .context_mode(ChildContextMode::ParentSummary)
                .artifact_refs([tool_artifact]),
        )
        .build()
        .expect("parent should build");

    let PromptSession { output, session } = agent
        .run(Payload::artifact(input_artifact), "session-artifacts")
        .await
        .expect("artifact session should finish");

    assert_eq!(output, Payload::artifact(final_artifact));
    assert_eq!(
        session.artifact_ids,
        vec![
            input_artifact,
            tool_artifact,
            unselected_artifact,
            child_artifact,
            final_artifact
        ]
    );

    let child_requests = child_provider.requests();
    let child_input = child_requests[0].messages[0]
        .content
        .as_json()
        .expect("child should receive envelope");
    assert_eq!(child_input["artifact_refs"], json!([11, 12]));
}

#[tokio::test]
async fn prompt_session_resolves_recorded_artifacts_through_integrated_store() {
    let artifact_store = Arc::new(ArtifactStore::new());
    let artifact_id = artifact_store
        .insert_payload(Payload::text("stored artifact body"))
        .expect("payload artifact should store");
    let provider = ScriptedProvider::new(vec![ModelResponse {
        message: None,
        tool_calls: vec![claumini_core::ToolCall {
            id: "finish-1".into(),
            name: FINISH_TOOL_NAME.into(),
            arguments: json!({
                "payload": {
                    "kind": "artifact",
                    "value": artifact_id.get()
                }
            }),
        }],
        finish_reason: FinishReason::ToolCalls,
    }]);

    let PromptSession { output, session } = PromptAgentBuilder::new(Arc::new(provider))
        .artifact_store(Arc::clone(&artifact_store))
        .build()
        .expect("agent should build")
        .run(
            Payload::text("resolve artifact"),
            "session-resolve-artifact",
        )
        .await
        .expect("session should finish");

    assert_eq!(output, Payload::artifact(artifact_id));
    assert_eq!(session.artifact_ids, vec![artifact_id]);

    let artifact = session
        .resolve_artifact(artifact_id)
        .expect("recorded artifact should resolve through prompt session");
    assert_eq!(artifact.id, artifact_id);
    assert_eq!(
        artifact.body,
        ArtifactBody::Payload(Payload::text("stored artifact body"))
    );
}

#[tokio::test]
async fn prompt_agent_parses_fallback_tool_calls_when_provider_lacks_native_tool_support() {
    let provider = ScriptedProvider::new(vec![
        ModelResponse::text(fallback_tool_calls(json!([
            {
                "id": "call-1",
                "name": "lookup",
                "arguments": {"question": "Where?"}
            }
        ]))),
        ModelResponse::text(fallback_final(json!({
            "kind": "json",
            "value": {"answer": "final answer"}
        }))),
    ])
    .with_capabilities(ProviderCapabilities {
        native_tool_calling: false,
        ..ProviderCapabilities::default()
    });

    let agent = PromptAgentBuilder::new(Arc::new(provider))
        .tool(LookupTool)
        .json_input::<UserQuestion>()
        .json_output::<AgentAnswer>()
        .build()
        .expect("agent should build");

    let PromptSession { output, session } = agent
        .run(
            UserQuestion {
                question: "Where?".into(),
            },
            "session-fallback-tool",
        )
        .await
        .expect("fallback tool calling should finish");

    assert_eq!(
        output,
        AgentAnswer {
            answer: "final answer".into()
        }
    );
    assert_eq!(session.tool_calls.len(), 1);
    assert_eq!(session.tool_calls[0].call.name, "lookup");
    assert_eq!(session.transcript[1].role, MessageRole::Assistant);
    assert_eq!(session.transcript[2].role, MessageRole::Tool);
}

#[tokio::test]
async fn prompt_agent_parses_fallback_final_structured_output() {
    let provider = ScriptedProvider::new(vec![ModelResponse::text(fallback_final(json!({
        "kind": "json",
        "value": {"answer": "final answer"}
    })))])
    .with_capabilities(ProviderCapabilities {
        native_tool_calling: false,
        ..ProviderCapabilities::default()
    });

    let agent = PromptAgentBuilder::new(Arc::new(provider))
        .json_input::<UserQuestion>()
        .json_output::<AgentAnswer>()
        .build()
        .expect("agent should build");

    let PromptSession { output, session } = agent
        .run(
            UserQuestion {
                question: "Where?".into(),
            },
            "session-fallback-final",
        )
        .await
        .expect("fallback final output should finish");

    assert_eq!(
        output,
        AgentAnswer {
            answer: "final answer".into()
        }
    );
    assert_eq!(session.transcript.len(), 2);
    assert_eq!(session.transcript[1].role, MessageRole::Assistant);
}

#[tokio::test]
async fn prompt_agent_injects_skill_metadata_and_runtime_load_skill_tool() {
    let provider = ScriptedProvider::new(vec![ModelResponse::text("done")]);
    let registry = build_skill_registry();

    let agent = PromptAgentBuilder::new(Arc::new(provider.clone()))
        .system_prompt("You are helpful.")
        .skills(registry)
        .text_input()
        .text_output()
        .build()
        .expect("agent should build");

    let _ = agent
        .run("hello".to_owned(), "session-skills")
        .await
        .expect("prompt agent should finish");

    let requests = provider.requests();
    let request = &requests[0];
    let system_prompt = request
        .system_prompt
        .as_deref()
        .expect("system prompt should be present");

    assert!(system_prompt.contains("You are helpful."));
    assert!(system_prompt.contains("Available skills:"));
    assert!(system_prompt.contains("runtime-skill"));
    assert!(system_prompt.contains("Helpful runtime description."));
    assert!(request.tools.iter().any(|tool| tool.name == "load_skill"));
}

#[tokio::test]
async fn prompt_agent_skills_prompt_omits_load_skill_when_runtime_tool_is_disabled() {
    let provider = ScriptedProvider::new(vec![ModelResponse::text("done")]);
    let registry = build_skill_registry();

    let agent = PromptAgentBuilder::new(Arc::new(provider.clone()))
        .skills(registry)
        .reserved_runtime_tools(ReservedRuntimeTools::default().with_load_skill(false))
        .text_input()
        .text_output()
        .build()
        .expect("agent should build");

    let _ = agent
        .run("hello".to_owned(), "session-skills-no-load-tool")
        .await
        .expect("prompt agent should finish");

    let requests = provider.requests();
    let system_prompt = requests[0]
        .system_prompt
        .as_deref()
        .expect("system prompt should be present");

    assert!(system_prompt.contains("Available skills:"));
    assert!(!system_prompt.contains("load_skill"));
}

#[tokio::test]
async fn prompt_agent_load_skill_tool_returns_full_body() {
    let provider = ScriptedProvider::new(vec![
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-1".into(),
                name: "load_skill".into(),
                arguments: json!({"name": "runtime-skill"}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-2".into(),
                name: FINISH_TOOL_NAME.into(),
                arguments: json!({"text": "done"}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
    ]);
    let registry = build_skill_registry();

    let agent = PromptAgentBuilder::new(Arc::new(provider))
        .skills(registry)
        .text_input()
        .text_output()
        .build()
        .expect("agent should build");

    let PromptSession { session, .. } = agent
        .run("hello".to_owned(), "session-load-skill")
        .await
        .expect("load skill should finish");

    assert_eq!(session.transcript[1].role, MessageRole::Assistant);
    assert_eq!(session.transcript[2].role, MessageRole::Tool);
    assert_eq!(session.transcript[2].name.as_deref(), Some("call-1"));
    assert_eq!(
        session.transcript[2].content,
        Payload::text(
            "# Runtime Skill\n\nHelpful runtime description.\n\nFull runtime skill body.\n"
        )
    );
}

#[tokio::test]
async fn prompt_agent_runs_through_agent_trait_with_agent_context() {
    let input_artifact = ArtifactId::new(31);
    let output_artifact = ArtifactId::new(32);
    let provider = ScriptedProvider::new(vec![ModelResponse {
        message: None,
        tool_calls: vec![claumini_core::ToolCall {
            id: "call-1".into(),
            name: FINISH_TOOL_NAME.into(),
            arguments: json!({
                "payload": {
                    "kind": "artifact",
                    "value": output_artifact.get()
                }
            }),
        }],
        finish_reason: FinishReason::ToolCalls,
    }]);

    let agent = PromptAgentBuilder::new(Arc::new(provider))
        .build()
        .expect("agent should build");
    let mut ctx = AgentContext::new(
        SessionMetadata::root("trait-session"),
        RuntimeLimits::default(),
    );

    let output = Agent::run(&agent, Payload::artifact(input_artifact), &mut ctx)
        .await
        .expect("agent trait run should succeed");

    assert_eq!(output, Payload::artifact(output_artifact));
    assert_eq!(ctx.transcript_len, 2);
    assert_eq!(ctx.tool_call_count, 1);
    assert_eq!(ctx.session.session_id, "trait-session");
    assert_eq!(ctx.current_input, Some(Payload::artifact(input_artifact)));
    assert_eq!(ctx.artifact_ids, vec![input_artifact, output_artifact]);
}

#[tokio::test]
async fn prompt_agent_propagates_runtime_current_input_and_artifacts_into_tool_context() {
    let input_artifact = ArtifactId::new(41);
    let provider = ScriptedProvider::new(vec![
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-1".into(),
                name: "context_echo".into(),
                arguments: json!({}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-2".into(),
                name: FINISH_TOOL_NAME.into(),
                arguments: json!({"text": "done"}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
    ]);

    let agent = PromptAgentBuilder::new(Arc::new(provider))
        .tool(ContextEchoTool)
        .build()
        .expect("agent should build");

    let PromptSession { output, session } = agent
        .run(Payload::artifact(input_artifact), "session-tool-context")
        .await
        .expect("runtime tool call should succeed");

    assert_eq!(output, Payload::text("done"));
    let tool_output: ToolContextOutput = session.transcript[2]
        .content
        .to_typed()
        .expect("tool output should decode");
    assert_eq!(tool_output.current_input, Payload::artifact(input_artifact));
    assert_eq!(tool_output.artifact_ids, vec![input_artifact.get()]);
    assert_eq!(tool_output.call_index, 0);
}

#[tokio::test]
async fn prompt_agent_blocks_on_call_subagent_and_records_child_result() {
    let child_provider = ScriptedProvider::new(vec![ModelResponse::text("child result")]);
    let child = PromptAgentBuilder::new(Arc::new(child_provider.clone()))
        .build()
        .expect("child should build");
    let parent_provider = ScriptedProvider::new(vec![
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-1".into(),
                name: "call_subagent".into(),
                arguments: json!({
                    "child": "worker",
                    "payload": { "kind": "text", "value": "inspect this" }
                }),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-2".into(),
                name: FINISH_TOOL_NAME.into(),
                arguments: json!({"text": "parent complete"}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
    ]);

    let agent = PromptAgentBuilder::new(Arc::new(parent_provider))
        .child("worker", ChildRegistration::new(child))
        .text_input()
        .text_output()
        .build()
        .expect("parent should build");

    let PromptSession { output, session } = agent
        .run("start".to_owned(), "session-call-subagent")
        .await
        .expect("blocking subagent call should finish");

    assert_eq!(output, "parent complete");
    assert_eq!(session.children.len(), 1);
    assert!(session.children[0].completed);
    assert_eq!(session.transcript[1].role, MessageRole::Assistant);
    assert_eq!(session.transcript[2].role, MessageRole::Tool);
    assert_eq!(session.transcript[2].name.as_deref(), Some("call-1"));
    assert_eq!(session.transcript[2].content, Payload::text("child result"));

    let child_requests = child_provider.requests();
    assert_eq!(child_requests.len(), 1);
    assert_eq!(
        child_requests[0].messages[0].content,
        Payload::text("inspect this")
    );
}

#[tokio::test]
async fn prompt_agent_supports_json_typed_child_registration_end_to_end() {
    let child_provider = ScriptedProvider::new(vec![ModelResponse {
        message: None,
        tool_calls: vec![claumini_core::ToolCall {
            id: "child-finish".into(),
            name: FINISH_TOOL_NAME.into(),
            arguments: json!({"json": {"answer": "child json result"}}),
        }],
        finish_reason: FinishReason::ToolCalls,
    }]);
    let child = PromptAgentBuilder::new(Arc::new(child_provider.clone()))
        .json_input::<ChildJsonInput>()
        .json_output::<ChildJsonOutput>()
        .build()
        .expect("child should build");
    let parent_provider = ScriptedProvider::new(vec![
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-1".into(),
                name: "call_subagent".into(),
                arguments: json!({
                    "child": "worker",
                    "payload": {
                        "kind": "json",
                        "value": {"topic": "typed child"}
                    }
                }),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-2".into(),
                name: FINISH_TOOL_NAME.into(),
                arguments: json!({"text": "done"}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
    ]);

    let agent = PromptAgentBuilder::new(Arc::new(parent_provider))
        .child_json("worker", child)
        .text_input()
        .text_output()
        .build()
        .expect("parent should build");

    let PromptSession { output, session } = agent
        .run("start".to_owned(), "session-json-child")
        .await
        .expect("json child call should finish");

    assert_eq!(output, "done");
    assert_eq!(
        session.transcript[2].content,
        Payload::json(json!({"answer": "child json result"})).unwrap()
    );

    let child_requests = child_provider.requests();
    assert_eq!(child_requests.len(), 1);
    assert_eq!(
        child_requests[0].messages[0].content,
        Payload::json(json!({"topic": "typed child"})).unwrap()
    );
}

#[tokio::test]
async fn prompt_agent_supports_json_typed_child_registration_with_parent_summary() {
    let child_provider = ScriptedProvider::new(vec![ModelResponse {
        message: None,
        tool_calls: vec![claumini_core::ToolCall {
            id: "child-finish".into(),
            name: FINISH_TOOL_NAME.into(),
            arguments: json!({"json": {"answer": "child json result"}}),
        }],
        finish_reason: FinishReason::ToolCalls,
    }]);
    let child = PromptAgentBuilder::new(Arc::new(child_provider.clone()))
        .json_input::<ChildJsonInput>()
        .json_output::<ChildJsonOutput>()
        .build()
        .expect("child should build");
    let parent_provider = ScriptedProvider::new(vec![
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-1".into(),
                name: "call_subagent".into(),
                arguments: json!({
                    "child": "worker",
                    "payload": {
                        "kind": "json",
                        "value": {"topic": "typed child"}
                    }
                }),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-2".into(),
                name: FINISH_TOOL_NAME.into(),
                arguments: json!({"text": "done"}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
    ]);

    let agent = PromptAgentBuilder::new(Arc::new(parent_provider))
        .child(
            "worker",
            ChildRegistration::json(child)
                .context_mode(ChildContextMode::ParentSummary)
                .instructions("Focus on the summary"),
        )
        .text_input()
        .text_output()
        .build()
        .expect("parent should build");

    let _ = agent
        .run("start".to_owned(), "session-json-child-summary")
        .await
        .expect("json child with summary should finish");

    let child_requests = child_provider.requests();
    assert_eq!(
        child_requests[0].messages[0].content,
        Payload::json(json!({"topic": "typed child"})).unwrap()
    );
    let system_prompt = child_requests[0]
        .system_prompt
        .as_deref()
        .expect("typed child should receive context through system prompt");
    assert!(system_prompt.contains("Focus on the summary"));
    assert!(system_prompt.contains("parent session session-json-child-summary"));
}

#[tokio::test]
async fn prompt_agent_supports_text_child_registration_end_to_end() {
    let child_provider = ScriptedProvider::new(vec![ModelResponse::text("child text result")]);
    let child = PromptAgentBuilder::new(Arc::new(child_provider.clone()))
        .text_input()
        .text_output()
        .build()
        .expect("child should build");
    let parent_provider = ScriptedProvider::new(vec![
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-1".into(),
                name: "call_subagent".into(),
                arguments: json!({
                    "child": "worker",
                    "payload": {
                        "kind": "text",
                        "value": "typed text child"
                    }
                }),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-2".into(),
                name: FINISH_TOOL_NAME.into(),
                arguments: json!({"text": "done"}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
    ]);

    let agent = PromptAgentBuilder::new(Arc::new(parent_provider))
        .child_text("worker", child)
        .text_input()
        .text_output()
        .build()
        .expect("parent should build");

    let PromptSession { output, session } = agent
        .run("start".to_owned(), "session-text-child")
        .await
        .expect("text child call should finish");

    assert_eq!(output, "done");
    assert_eq!(
        session.transcript[2].content,
        Payload::text("child text result")
    );

    let child_requests = child_provider.requests();
    assert_eq!(child_requests.len(), 1);
    assert_eq!(
        child_requests[0].messages[0].content,
        Payload::text("typed text child")
    );
}

#[tokio::test]
async fn prompt_agent_supports_spawn_then_await_subagent_flow() {
    let child_provider = ScriptedProvider::new(vec![ModelResponse::text("spawned result")])
        .with_delay(Duration::from_millis(20));
    let child = PromptAgentBuilder::new(Arc::new(child_provider))
        .build()
        .expect("child should build");
    let parent_provider = ScriptedProvider::new(vec![
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-1".into(),
                name: "spawn_subagent".into(),
                arguments: json!({
                    "child": "worker",
                    "payload": { "kind": "text", "value": "queued work" }
                }),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-2".into(),
                name: "await_subagent".into(),
                arguments: json!({"handle": "child-1"}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-3".into(),
                name: FINISH_TOOL_NAME.into(),
                arguments: json!({"text": "done"}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
    ]);

    let agent = PromptAgentBuilder::new(Arc::new(parent_provider))
        .child("worker", ChildRegistration::new(child))
        .text_input()
        .text_output()
        .build()
        .expect("parent should build");

    let PromptSession { output, session } = agent
        .run("start".to_owned(), "session-spawn-await")
        .await
        .expect("spawn/await should finish");

    assert_eq!(output, "done");
    assert_eq!(session.children.len(), 1);
    assert_eq!(session.children[0].id, "child-1");
    assert!(session.children[0].completed);
    assert_eq!(session.transcript[2].content, Payload::text("child-1"));
    assert_eq!(
        session.transcript[4].content,
        Payload::text("spawned result")
    );
}

#[tokio::test]
async fn prompt_agent_handoff_returns_child_output_and_ends_parent_path() {
    let child_provider = ScriptedProvider::new(vec![ModelResponse::text("handoff result")]);
    let child = PromptAgentBuilder::new(Arc::new(child_provider))
        .build()
        .expect("child should build");
    let parent_provider = ScriptedProvider::new(vec![ModelResponse {
        message: None,
        tool_calls: vec![claumini_core::ToolCall {
            id: "call-1".into(),
            name: "handoff".into(),
            arguments: json!({
                "child": "worker",
                "payload": { "kind": "text", "value": "take over" }
            }),
        }],
        finish_reason: FinishReason::ToolCalls,
    }]);

    let agent = PromptAgentBuilder::new(Arc::new(parent_provider.clone()))
        .child("worker", ChildRegistration::new(child))
        .text_input()
        .text_output()
        .build()
        .expect("parent should build");

    let PromptSession { output, session } = agent
        .run("start".to_owned(), "session-handoff")
        .await
        .expect("handoff should finish");

    assert_eq!(output, "handoff result");
    assert_eq!(session.children.len(), 1);
    assert!(session.children[0].completed);
    assert_eq!(parent_provider.requests().len(), 1);
}

#[tokio::test]
async fn prompt_agent_enforces_max_spawn_depth_for_subagents() {
    let child_provider = ScriptedProvider::new(vec![ModelResponse::text("child result")]);
    let child = PromptAgentBuilder::new(Arc::new(child_provider))
        .build()
        .expect("child should build");
    let parent_provider = ScriptedProvider::new(vec![ModelResponse {
        message: None,
        tool_calls: vec![claumini_core::ToolCall {
            id: "call-1".into(),
            name: "call_subagent".into(),
            arguments: json!({
                "child": "worker",
                "payload": { "kind": "text", "value": "too deep" }
            }),
        }],
        finish_reason: FinishReason::ToolCalls,
    }]);

    let agent = PromptAgentBuilder::new(Arc::new(parent_provider))
        .child("worker", ChildRegistration::new(child))
        .limits(RuntimeLimits {
            max_spawn_depth: 0,
            ..RuntimeLimits::default()
        })
        .text_input()
        .text_output()
        .build()
        .expect("parent should build");

    let error = agent
        .run("start".to_owned(), "session-depth")
        .await
        .expect_err("subagent call should exceed max depth");

    assert!(matches!(
        error,
        AgentError::Runtime(claumini_core::RuntimeError::LimitExceeded {
            limit: "max_spawn_depth",
            value: 1,
        })
    ));
}

#[tokio::test]
async fn prompt_agent_enforces_max_active_children_for_spawned_subagents() {
    let child_provider = ScriptedProvider::new(vec![
        ModelResponse::text("first child"),
        ModelResponse::text("second child"),
    ])
    .with_delay(Duration::from_millis(40));
    let child = PromptAgentBuilder::new(Arc::new(child_provider))
        .build()
        .expect("child should build");
    let parent_provider = ScriptedProvider::new(vec![ModelResponse {
        message: None,
        tool_calls: vec![
            claumini_core::ToolCall {
                id: "call-1".into(),
                name: "spawn_subagent".into(),
                arguments: json!({
                    "child": "worker",
                    "payload": { "kind": "text", "value": "first" }
                }),
            },
            claumini_core::ToolCall {
                id: "call-2".into(),
                name: "spawn_subagent".into(),
                arguments: json!({
                    "child": "worker",
                    "payload": { "kind": "text", "value": "second" }
                }),
            },
        ],
        finish_reason: FinishReason::ToolCalls,
    }]);

    let agent = PromptAgentBuilder::new(Arc::new(parent_provider))
        .child("worker", ChildRegistration::new(child))
        .limits(RuntimeLimits {
            max_active_children: 1,
            ..RuntimeLimits::default()
        })
        .text_input()
        .text_output()
        .build()
        .expect("parent should build");

    let error = agent
        .run("start".to_owned(), "session-active-children")
        .await
        .expect_err("second concurrent child should exceed the active-child limit");

    assert!(matches!(
        error,
        AgentError::Runtime(claumini_core::RuntimeError::LimitExceeded {
            limit: "max_active_children",
            value: 2,
        })
    ));
}

#[tokio::test]
async fn prompt_agent_passes_parent_summary_and_child_instructions_to_registered_child() {
    let child_provider = ScriptedProvider::new(vec![ModelResponse::text("child result")]);
    let child = PromptAgentBuilder::new(Arc::new(child_provider.clone()))
        .build()
        .expect("child should build");
    let parent_provider = ScriptedProvider::new(vec![
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-1".into(),
                name: "call_subagent".into(),
                arguments: json!({
                    "child": "worker",
                    "payload": { "kind": "text", "value": "inspect this" }
                }),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-2".into(),
                name: FINISH_TOOL_NAME.into(),
                arguments: json!({"text": "done"}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
    ]);

    let agent = PromptAgentBuilder::new(Arc::new(parent_provider))
        .child(
            "worker",
            ChildRegistration::new(child)
                .context_mode(ChildContextMode::ParentSummary)
                .instructions("Focus on the summary"),
        )
        .text_input()
        .text_output()
        .build()
        .expect("parent should build");

    let _ = agent
        .run("start".to_owned(), "session-parent-summary")
        .await
        .expect("subagent call should finish");

    let child_request = &child_provider.requests()[0];
    let child_input = child_request.messages[0]
        .content
        .as_json()
        .expect("child input should include an envelope");

    assert_eq!(
        child_input["payload"],
        json!({ "kind": "text", "value": "inspect this" })
    );
    assert_eq!(child_input["instructions"], "Focus on the summary");
    assert_eq!(child_input["parent"]["mode"], "summary");
    assert_eq!(
        child_input["parent"]["session_id"],
        "session-parent-summary"
    );
}

#[tokio::test]
async fn prompt_agent_merges_inherited_parent_tools_into_registered_child() {
    let child_provider = ScriptedProvider::new(vec![ModelResponse::text("child result")]);
    let child = PromptAgentBuilder::new(Arc::new(child_provider.clone()))
        .tool(ChildOnlyTool)
        .build()
        .expect("child should build");
    let parent_provider = ScriptedProvider::new(vec![
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-1".into(),
                name: "call_subagent".into(),
                arguments: json!({
                    "child": "worker",
                    "payload": { "kind": "text", "value": "inspect this" }
                }),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
        ModelResponse {
            message: None,
            tool_calls: vec![claumini_core::ToolCall {
                id: "call-2".into(),
                name: FINISH_TOOL_NAME.into(),
                arguments: json!({"text": "done"}),
            }],
            finish_reason: FinishReason::ToolCalls,
        },
    ]);

    let agent = PromptAgentBuilder::new(Arc::new(parent_provider))
        .tool(LookupTool)
        .child(
            "worker",
            ChildRegistration::new(child)
                .tool_policy(ChildToolPolicy::MergeInheritedNamed(vec!["lookup".into()])),
        )
        .text_input()
        .text_output()
        .build()
        .expect("parent should build");

    let _ = agent
        .run("start".to_owned(), "session-tool-inheritance")
        .await
        .expect("subagent call should finish");

    let child_requests = child_provider.requests();
    let tool_names: Vec<_> = child_requests[0]
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect();

    assert_eq!(tool_names, vec!["child_lookup", "lookup", "finish"]);
}
