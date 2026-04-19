use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use claumini_core::{
    Agent, AgentContext, AgentError, ArtifactId, Message, MessageRole, ModelProvider, ModelRequest,
    ModelResponse, Payload, ProviderError, RuntimeError, RuntimeLimits, SessionMetadata, Tool,
    ToolCall, ToolContext, ToolDescriptor, ToolError, ToolSchema,
};
use schemars::{JsonSchema, schema_for};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::task::JoinHandle;
use tokio::time::{Duration, timeout};

use crate::{
    ArtifactStore, FINISH_TOOL_NAME, LOAD_SKILL_TOOL_NAME, RESERVED_TOOL_NAMES, SessionRecord,
    SessionState, SkillMetadata, SkillRegistry, ToolCallRecord,
};

type InputEncoder<I> = Arc<dyn Fn(I) -> Result<Payload, AgentError> + Send + Sync>;
type OutputDecoder<O> = Arc<dyn Fn(Payload) -> Result<O, AgentError> + Send + Sync>;
type PayloadInputDecoder<I> = Arc<dyn Fn(Payload) -> Result<I, AgentError> + Send + Sync>;
type PayloadOutputEncoder<O> = Arc<dyn Fn(O) -> Result<Payload, AgentError> + Send + Sync>;

const MAX_PROVIDER_ATTEMPTS: usize = 3;

#[async_trait]
pub trait RuntimeTool: Send + Sync {
    fn descriptor(&self) -> ToolDescriptor;

    async fn call_payload(
        &self,
        input: Payload,
        ctx: &mut ToolContext,
    ) -> Result<Payload, ToolError>;
}

struct TypedRuntimeTool<T> {
    inner: T,
}

impl<T> TypedRuntimeTool<T> {
    fn new(inner: T) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl<T> RuntimeTool for TypedRuntimeTool<T>
where
    T: Tool + Send + Sync,
{
    fn descriptor(&self) -> ToolDescriptor {
        self.inner.descriptor()
    }

    async fn call_payload(
        &self,
        input: Payload,
        ctx: &mut ToolContext,
    ) -> Result<Payload, ToolError> {
        let typed_input = input
            .to_typed::<T::Input>()
            .map_err(|error| ToolError::InvalidInput(error.to_string()))?;
        let output = self.inner.call(typed_input, ctx).await?;
        Payload::json(output).map_err(|error| {
            ToolError::ExecutionFailed(format!("failed to serialize tool output: {error}"))
        })
    }
}

#[derive(Debug, Clone)]
pub struct PromptSession<O> {
    pub output: O,
    pub session: SessionRecord,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildContextMode {
    PayloadOnly,
    ParentSummary,
    FullParentContext,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChildToolPolicy {
    ChildOnly,
    InheritAll,
    InheritNamed(Vec<String>),
    MergeInheritedAll,
    MergeInheritedNamed(Vec<String>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChildContextTransport {
    Envelope,
    SystemPrompt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReservedRuntimeTools {
    finish: bool,
    load_skill: bool,
    subagents: bool,
}

impl Default for ReservedRuntimeTools {
    fn default() -> Self {
        Self {
            finish: true,
            load_skill: true,
            subagents: true,
        }
    }
}

impl ReservedRuntimeTools {
    pub fn with_finish(mut self, enabled: bool) -> Self {
        self.finish = enabled;
        self
    }

    pub fn with_load_skill(mut self, enabled: bool) -> Self {
        self.load_skill = enabled;
        self
    }

    pub fn with_subagents(mut self, enabled: bool) -> Self {
        self.subagents = enabled;
        self
    }
}

#[derive(Clone)]
pub struct ChildRegistration {
    agent: PromptAgent<Payload, Payload>,
    context_mode: ChildContextMode,
    context_transport: ChildContextTransport,
    artifact_refs: Vec<ArtifactId>,
    instructions: Option<String>,
    tool_policy: ChildToolPolicy,
}

impl ChildRegistration {
    pub fn payload(agent: PromptAgent<Payload, Payload>) -> Self {
        Self::new(agent)
    }

    pub fn json<I, O>(agent: PromptAgent<I, O>) -> Self
    where
        I: DeserializeOwned + Send + Sync + 'static,
        O: Serialize + Send + Sync + 'static,
    {
        Self::new(agent.into_payload_agent(
            Arc::new(decode_json_input::<I>),
            Arc::new(encode_json_payload::<O>),
        ))
        .context_transport(ChildContextTransport::SystemPrompt)
    }

    pub fn text(agent: PromptAgent<String, String>) -> Self {
        Self::new(agent.into_payload_agent(
            Arc::new(decode_text_output),
            Arc::new(|value: String| Ok(Payload::text(value))),
        ))
        .context_transport(ChildContextTransport::SystemPrompt)
    }

    pub fn new(agent: PromptAgent<Payload, Payload>) -> Self {
        Self {
            agent,
            context_mode: ChildContextMode::PayloadOnly,
            context_transport: ChildContextTransport::Envelope,
            artifact_refs: Vec::new(),
            instructions: None,
            tool_policy: ChildToolPolicy::ChildOnly,
        }
    }

    fn context_transport(mut self, context_transport: ChildContextTransport) -> Self {
        self.context_transport = context_transport;
        self
    }

    pub fn context_mode(mut self, context_mode: ChildContextMode) -> Self {
        self.context_mode = context_mode;
        self
    }

    pub fn artifact_refs(mut self, artifact_refs: impl IntoIterator<Item = ArtifactId>) -> Self {
        self.artifact_refs = artifact_refs.into_iter().collect();
        self
    }

    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    pub fn tool_policy(mut self, tool_policy: ChildToolPolicy) -> Self {
        self.tool_policy = tool_policy;
        self
    }
}

struct SessionRuntimeState {
    next_child_handle: u64,
    spawned_children: HashMap<String, SpawnedChild>,
}

impl SessionRuntimeState {
    fn next_handle(&mut self) -> String {
        self.next_child_handle += 1;
        format!("child-{}", self.next_child_handle)
    }
}

struct SpawnedChild {
    session_id: String,
    task: JoinHandle<Result<PromptSession<Payload>, AgentError>>,
}

enum RuntimeToolOutcome {
    Continue(Payload),
    Finish(Payload),
}

#[derive(Debug, Deserialize)]
struct ChildInvocationInput {
    child: String,
    payload: Payload,
}

#[derive(Debug, Deserialize)]
struct AwaitSubagentInput {
    handle: String,
}

pub struct PromptAgent<I, O> {
    provider: Arc<dyn ModelProvider>,
    system_prompt: Option<String>,
    limits: RuntimeLimits,
    reserved_runtime_tools: ReservedRuntimeTools,
    artifact_store: Option<Arc<ArtifactStore>>,
    tools: HashMap<String, Arc<dyn RuntimeTool>>,
    tool_schemas: Vec<ToolSchema>,
    skills: Option<Arc<SkillRegistry>>,
    children: HashMap<String, ChildRegistration>,
    input_encoder: InputEncoder<I>,
    output_decoder: OutputDecoder<O>,
    output_schema: Option<Value>,
    max_output_tokens: Option<u32>,
}

impl<I, O> Clone for PromptAgent<I, O> {
    fn clone(&self) -> Self {
        Self {
            provider: Arc::clone(&self.provider),
            system_prompt: self.system_prompt.clone(),
            limits: self.limits.clone(),
            reserved_runtime_tools: self.reserved_runtime_tools,
            artifact_store: self.artifact_store.clone(),
            tools: self.tools.clone(),
            tool_schemas: self.tool_schemas.clone(),
            skills: self.skills.clone(),
            children: self.children.clone(),
            input_encoder: Arc::clone(&self.input_encoder),
            output_decoder: Arc::clone(&self.output_decoder),
            output_schema: self.output_schema.clone(),
            max_output_tokens: self.max_output_tokens,
        }
    }
}

impl<I, O> PromptAgent<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    pub async fn run(
        &self,
        input: I,
        session_id: impl Into<String>,
    ) -> Result<PromptSession<O>, AgentError> {
        self.run_with_context(
            input,
            SessionMetadata::root(session_id),
            self.limits.clone(),
        )
        .await
    }

    async fn run_with_context(
        &self,
        input: I,
        session_metadata: SessionMetadata,
        limits: RuntimeLimits,
    ) -> Result<PromptSession<O>, AgentError> {
        let input_payload = (self.input_encoder)(input)?;
        let mut session = SessionRecord::from_metadata(session_metadata)
            .with_artifact_store(self.artifact_store.clone());
        let mut runtime_state = SessionRuntimeState {
            next_child_handle: 0,
            spawned_children: HashMap::new(),
        };
        record_payload_artifacts(&mut session, &input_payload);
        session.current_input = Some(input_payload.clone());
        session
            .transcript
            .push(Message::new(MessageRole::User, input_payload));

        let provider_capabilities = self.provider.capabilities();
        let mut turn = 0usize;
        loop {
            turn += 1;
            if turn > limits.max_turns_per_session {
                match limits.max_turns_policy.nudge_text() {
                    None => {
                        return Err(AgentError::Runtime(RuntimeError::LimitExceeded {
                            limit: "max_turns_per_session",
                            value: turn,
                        }));
                    }
                    Some(nudge) => {
                        let nudge_msg =
                            Message::new(MessageRole::User, Payload::text(nudge.to_string()));
                        session.transcript.push(nudge_msg);

                        let mut request = self.build_request(&session, provider_capabilities);
                        request.tools.clear();
                        let response = self.complete_with_retry(request, &limits).await?;

                        if let Some(message) = assistant_response_message(&response) {
                            session.transcript.push(message);
                        }

                        let final_message = response.message.ok_or_else(|| {
                            AgentError::Runtime(RuntimeError::Message(
                                "force-final turn produced no message".into(),
                            ))
                        })?;

                        record_payload_artifacts(&mut session, &final_message.content);
                        let output = (self.output_decoder)(final_message.content)?;
                        session.state = SessionState::Finished;
                        return Ok(PromptSession { output, session });
                    }
                }
            }

            let request = self.build_request(&session, provider_capabilities);
            let response = self.complete_with_retry(request, &limits).await?;

            if let Some(message) = assistant_response_message(&response) {
                session.transcript.push(message);
            }

            if !provider_capabilities.native_tool_calling && response.tool_calls.is_empty() {
                let message = response.message.as_ref().ok_or_else(|| {
                    AgentError::Runtime(RuntimeError::Message(
                        "provider returned neither a fallback message nor tool calls".into(),
                    ))
                })?;

                match parse_fallback_response(message)? {
                    ParsedFallbackResponse::ToolCalls(tool_calls) => {
                        if let Some(finished) = self
                            .execute_tool_calls(
                                &tool_calls,
                                &mut session,
                                &limits,
                                &mut runtime_state,
                            )
                            .await?
                        {
                            let output = (self.output_decoder)(finished)?;
                            session.state = SessionState::Finished;
                            return Ok(PromptSession { output, session });
                        }
                        continue;
                    }
                    ParsedFallbackResponse::Final(payload) => {
                        record_payload_artifacts(&mut session, &payload);
                        let output = (self.output_decoder)(payload)?;
                        session.state = SessionState::Finished;
                        return Ok(PromptSession { output, session });
                    }
                }
            }

            if response.tool_calls.is_empty() {
                let message = response.message.ok_or_else(|| {
                    AgentError::Runtime(RuntimeError::Message(
                        "provider returned neither a final message nor tool calls".into(),
                    ))
                })?;

                record_payload_artifacts(&mut session, &message.content);
                let output = (self.output_decoder)(message.content)?;
                session.state = SessionState::Finished;
                return Ok(PromptSession { output, session });
            }

            if let Some(finished) = self
                .execute_tool_calls(
                    &response.tool_calls,
                    &mut session,
                    &limits,
                    &mut runtime_state,
                )
                .await?
            {
                let output = (self.output_decoder)(finished)?;
                session.state = SessionState::Finished;
                return Ok(PromptSession { output, session });
            }
        }
    }

    fn build_request(
        &self,
        session: &SessionRecord,
        capabilities: claumini_core::ProviderCapabilities,
    ) -> ModelRequest {
        let mut request = ModelRequest::new(session.transcript.clone())
            .with_tools(self.request_tool_schemas(capabilities));
        if let Some(system_prompt) = self.effective_system_prompt(capabilities) {
            request = request.with_system_prompt(system_prompt);
        }
        if let Some(output_schema) = &self.output_schema {
            request = request.with_response_schema(output_schema.clone());
        }
        request.max_output_tokens = self.max_output_tokens;
        request
    }

    async fn complete_with_retry(
        &self,
        request: ModelRequest,
        limits: &RuntimeLimits,
    ) -> Result<ModelResponse, AgentError> {
        let timeout_duration = Duration::from_millis(limits.model_request_timeout_ms);

        for attempt in 1..=MAX_PROVIDER_ATTEMPTS {
            match timeout(timeout_duration, self.provider.complete(request.clone())).await {
                Err(_) if attempt < MAX_PROVIDER_ATTEMPTS => continue,
                Err(_) => return Err(AgentError::Provider(ProviderError::Timeout)),
                Ok(Ok(response)) => return Ok(response),
                Ok(Err(error))
                    if is_transient_provider_error(&error) && attempt < MAX_PROVIDER_ATTEMPTS =>
                {
                    continue;
                }
                Ok(Err(error)) => return Err(AgentError::Provider(error)),
            }
        }

        Err(AgentError::Provider(ProviderError::Timeout))
    }

    fn request_tool_schemas(
        &self,
        capabilities: claumini_core::ProviderCapabilities,
    ) -> Vec<ToolSchema> {
        if capabilities.structured_output && self.output_schema.is_some() {
            return self
                .tool_schemas
                .iter()
                .filter(|schema| !RESERVED_TOOL_NAMES.contains(&schema.name.as_str()))
                .cloned()
                .collect();
        }

        self.tool_schemas.clone()
    }

    fn effective_system_prompt(
        &self,
        capabilities: claumini_core::ProviderCapabilities,
    ) -> Option<String> {
        let mut sections = Vec::new();

        if let Some(system_prompt) = &self.system_prompt {
            sections.push(system_prompt.clone());
        }

        if let Some(skills) = &self.skills {
            sections.push(skill_metadata_prompt(
                skills.metadata(),
                self.reserved_runtime_tools.load_skill,
            ));
        }

        if !capabilities.native_tool_calling {
            sections.push(fallback_prompt(&self.tool_schemas));
        }

        if let Some(output_schema) = &self.output_schema
            && !capabilities.structured_output
        {
            sections.push(structured_output_prompt(output_schema));
        }

        (!sections.is_empty()).then(|| sections.join("\n\n"))
    }

    async fn execute_tool_calls(
        &self,
        tool_calls: &[ToolCall],
        session: &mut SessionRecord,
        limits: &RuntimeLimits,
        runtime_state: &mut SessionRuntimeState,
    ) -> Result<Option<Payload>, AgentError> {
        for call in tool_calls {
            let record_index = session.tool_calls.len();
            session.tool_calls.push(ToolCallRecord {
                index: record_index,
                call: call.clone(),
            });

            if self.reserved_runtime_tools.finish && call.name == FINISH_TOOL_NAME {
                let payload = finish_payload(&call.arguments)?;
                record_payload_artifacts(session, &payload);
                return Ok(Some(payload));
            }

            if let Some(outcome) = self
                .execute_reserved_runtime_tool(call, session, limits, runtime_state)
                .await?
            {
                match outcome {
                    RuntimeToolOutcome::Continue(payload) => {
                        session
                            .transcript
                            .push(Message::new(MessageRole::Tool, payload).named(call.id.clone()));
                        continue;
                    }
                    RuntimeToolOutcome::Finish(payload) => {
                        return Ok(Some(payload));
                    }
                }
            }

            let tool = self.tools.get(&call.name).ok_or_else(|| {
                AgentError::Runtime(RuntimeError::Message(format!(
                    "model called unavailable tool '{}'",
                    call.name
                )))
            })?;

            let mut ctx = ToolContext::new(session.metadata.clone(), record_index);
            ctx.current_input = session.current_input.clone();
            ctx.artifact_ids = session.artifact_ids.clone();
            let payload = match timeout(
                Duration::from_millis(limits.tool_call_timeout_ms),
                tool.call_payload(Payload::Json(call.arguments.clone()), &mut ctx),
            )
            .await
            {
                Ok(Ok(payload)) => payload,
                Ok(Err(error)) => tool_error_payload(error),
                Err(_) => tool_error_payload(ToolError::Timeout),
            };
            record_payload_artifacts(session, &payload);

            session
                .transcript
                .push(Message::new(MessageRole::Tool, payload).named(call.id.clone()));
        }

        Ok(None)
    }

    async fn execute_reserved_runtime_tool(
        &self,
        call: &ToolCall,
        session: &mut SessionRecord,
        limits: &RuntimeLimits,
        runtime_state: &mut SessionRuntimeState,
    ) -> Result<Option<RuntimeToolOutcome>, AgentError> {
        if !self.reserved_runtime_tools.subagents {
            return Ok(None);
        }

        let outcome = match call.name.as_str() {
            crate::CALL_SUBAGENT_TOOL_NAME => RuntimeToolOutcome::Continue(
                self.call_child(call.arguments.clone(), session, limits, runtime_state)
                    .await?,
            ),
            crate::SPAWN_SUBAGENT_TOOL_NAME => RuntimeToolOutcome::Continue(
                self.spawn_child(call.arguments.clone(), session, limits, runtime_state)
                    .await?,
            ),
            crate::AWAIT_SUBAGENT_TOOL_NAME => RuntimeToolOutcome::Continue(
                self.await_child(call.arguments.clone(), session, runtime_state)
                    .await?,
            ),
            crate::HANDOFF_TOOL_NAME => RuntimeToolOutcome::Finish(
                self.call_child(call.arguments.clone(), session, limits, runtime_state)
                    .await?,
            ),
            _ => return Ok(None),
        };

        match &outcome {
            RuntimeToolOutcome::Continue(payload) | RuntimeToolOutcome::Finish(payload) => {
                record_payload_artifacts(session, payload);
            }
        }

        Ok(Some(outcome))
    }

    async fn call_child(
        &self,
        arguments: Value,
        session: &mut SessionRecord,
        limits: &RuntimeLimits,
        runtime_state: &mut SessionRuntimeState,
    ) -> Result<Payload, AgentError> {
        let input: ChildInvocationInput = serde_json::from_value(arguments).map_err(|error| {
            AgentError::Runtime(RuntimeError::Message(format!(
                "reserved runtime tool input is invalid: {error}"
            )))
        })?;
        let handle = runtime_state.next_handle();
        let (child, child_session_id, child_input) =
            self.prepare_child_run(&input, &handle, session, limits)?;

        if active_child_count(session) + 1 > limits.max_active_children {
            return Err(AgentError::Runtime(RuntimeError::LimitExceeded {
                limit: "max_active_children",
                value: active_child_count(session) + 1,
            }));
        }

        session.children.push(crate::ChildHandleRecord {
            id: handle,
            session_id: child_session_id,
            completed: false,
        });
        let child_record = session
            .children
            .last()
            .expect("child record should exist")
            .clone();
        let child_session = run_payload_child(
            child,
            child_input,
            child_session_metadata(session, &child_record),
            limits.clone(),
        )
        .await?;
        mark_child_completed(session, &child_record.id);
        Ok(child_session.output)
    }

    async fn spawn_child(
        &self,
        arguments: Value,
        session: &mut SessionRecord,
        limits: &RuntimeLimits,
        runtime_state: &mut SessionRuntimeState,
    ) -> Result<Payload, AgentError> {
        let input: ChildInvocationInput = serde_json::from_value(arguments).map_err(|error| {
            AgentError::Runtime(RuntimeError::Message(format!(
                "reserved runtime tool input is invalid: {error}"
            )))
        })?;
        let active_children = active_child_count(session) + 1;
        if active_children > limits.max_active_children {
            return Err(AgentError::Runtime(RuntimeError::LimitExceeded {
                limit: "max_active_children",
                value: active_children,
            }));
        }

        let handle = runtime_state.next_handle();
        let (child, child_session_id, child_input) =
            self.prepare_child_run(&input, &handle, session, limits)?;
        session.children.push(crate::ChildHandleRecord {
            id: handle.clone(),
            session_id: child_session_id.clone(),
            completed: false,
        });
        let session_metadata = child_session_metadata(
            session,
            session
                .children
                .last()
                .expect("child record should exist after spawn"),
        );
        let child_limits = limits.clone();
        let task = tokio::spawn(run_payload_child(
            child,
            child_input,
            session_metadata,
            child_limits,
        ));
        runtime_state.spawned_children.insert(
            handle.clone(),
            SpawnedChild {
                session_id: child_session_id,
                task,
            },
        );
        session.state = SessionState::Waiting;
        Ok(Payload::text(handle))
    }

    async fn await_child(
        &self,
        arguments: Value,
        session: &mut SessionRecord,
        runtime_state: &mut SessionRuntimeState,
    ) -> Result<Payload, AgentError> {
        let input: AwaitSubagentInput = serde_json::from_value(arguments).map_err(|error| {
            AgentError::Runtime(RuntimeError::Message(format!(
                "reserved runtime tool input is invalid: {error}"
            )))
        })?;
        let spawned_child = runtime_state
            .spawned_children
            .remove(&input.handle)
            .ok_or_else(|| {
                AgentError::Runtime(RuntimeError::Message(format!(
                    "unknown child handle '{}'",
                    input.handle
                )))
            })?;
        let result = spawned_child.task.await.map_err(|error| {
            AgentError::Runtime(RuntimeError::Message(format!(
                "child session '{}' failed to join: {error}",
                spawned_child.session_id
            )))
        })??;
        mark_child_completed(session, &input.handle);
        if active_child_count(session) == 0 {
            session.state = SessionState::Running;
        }
        Ok(result.output)
    }

    fn prepare_child_run(
        &self,
        input: &ChildInvocationInput,
        handle: &str,
        session: &SessionRecord,
        limits: &RuntimeLimits,
    ) -> Result<(PromptAgent<Payload, Payload>, String, Payload), AgentError> {
        let registration = self.children.get(&input.child).ok_or_else(|| {
            AgentError::Runtime(RuntimeError::Message(format!(
                "unknown registered child '{}'",
                input.child
            )))
        })?;
        let child_depth = session.metadata.depth + 1;
        if child_depth > limits.max_spawn_depth {
            return Err(AgentError::Runtime(RuntimeError::LimitExceeded {
                limit: "max_spawn_depth",
                value: child_depth,
            }));
        }

        let child_session_id = format!("{}.{}", session.metadata.session_id, handle);
        let child_input = registration.build_child_input(&input.payload, session)?;
        let child_agent = registration.prepare_agent(self, session, &input.payload)?;
        Ok((child_agent, child_session_id, child_input))
    }
}

pub struct PromptAgentBuilder<I = Payload, O = Payload> {
    provider: Arc<dyn ModelProvider>,
    system_prompt: Option<String>,
    limits: RuntimeLimits,
    reserved_runtime_tools: ReservedRuntimeTools,
    artifact_store: Option<Arc<ArtifactStore>>,
    tools: Vec<Arc<dyn RuntimeTool>>,
    skills: Option<Arc<SkillRegistry>>,
    children: Vec<(String, ChildRegistration)>,
    input_encoder: InputEncoder<I>,
    output_decoder: OutputDecoder<O>,
    output_schema: Option<Value>,
    max_output_tokens: Option<u32>,
    marker: PhantomData<(I, O)>,
}

impl PromptAgentBuilder {
    pub fn new(provider: Arc<dyn ModelProvider>) -> Self {
        Self {
            provider,
            system_prompt: None,
            limits: RuntimeLimits::default(),
            reserved_runtime_tools: ReservedRuntimeTools::default(),
            artifact_store: None,
            tools: Vec::new(),
            skills: None,
            children: Vec::new(),
            input_encoder: Arc::new(Ok::<_, AgentError>),
            output_decoder: Arc::new(Ok::<_, AgentError>),
            output_schema: None,
            max_output_tokens: Some(32000),
            marker: PhantomData,
        }
    }
}

impl<I, O> PromptAgentBuilder<I, O> {
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn limits(mut self, limits: RuntimeLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn max_output_tokens(mut self, tokens: Option<u32>) -> Self {
        self.max_output_tokens = tokens;
        self
    }

    pub fn reserved_runtime_tools(mut self, reserved_runtime_tools: ReservedRuntimeTools) -> Self {
        self.reserved_runtime_tools = reserved_runtime_tools;
        self
    }

    pub fn artifact_store(mut self, artifact_store: Arc<ArtifactStore>) -> Self {
        self.artifact_store = Some(artifact_store);
        self
    }

    pub fn skills(mut self, skills: SkillRegistry) -> Self {
        self.skills = Some(Arc::new(skills));
        self
    }

    pub fn child(mut self, name: impl Into<String>, registration: ChildRegistration) -> Self {
        self.children.push((name.into(), registration));
        self
    }

    pub fn child_payload(
        self,
        name: impl Into<String>,
        agent: PromptAgent<Payload, Payload>,
    ) -> Self {
        self.child(name, ChildRegistration::payload(agent))
    }

    pub fn child_json<CI, CO>(self, name: impl Into<String>, agent: PromptAgent<CI, CO>) -> Self
    where
        CI: DeserializeOwned + Send + Sync + 'static,
        CO: Serialize + Send + Sync + 'static,
    {
        self.child(name, ChildRegistration::json(agent))
    }

    pub fn child_text(self, name: impl Into<String>, agent: PromptAgent<String, String>) -> Self {
        self.child(name, ChildRegistration::text(agent))
    }

    pub fn runtime_tool<T>(mut self, tool: T) -> Self
    where
        T: RuntimeTool + 'static,
    {
        self.tools.push(Arc::new(tool));
        self
    }

    pub fn tool<T>(self, tool: T) -> Self
    where
        T: Tool + Send + Sync + 'static,
    {
        self.runtime_tool(TypedRuntimeTool::new(tool))
    }

    pub fn text_input(self) -> PromptAgentBuilder<String, O> {
        self.with_input_encoder(|value: String| Ok(Payload::text(value)))
    }

    pub fn json_input<T>(self) -> PromptAgentBuilder<T, O>
    where
        T: Serialize + Send + Sync + 'static,
    {
        self.with_input_encoder(|value: T| {
            Payload::json(value).map_err(|error| {
                AgentError::Runtime(RuntimeError::Message(format!(
                    "failed to encode typed input as json: {error}"
                )))
            })
        })
    }

    pub fn user_prompt<T, F>(self, formatter: F) -> PromptAgentBuilder<T, O>
    where
        T: Send + Sync + 'static,
        F: Fn(T) -> String + Send + Sync + 'static,
    {
        self.with_input_encoder(move |value: T| Ok(Payload::text(formatter(value))))
    }

    pub fn text_output(self) -> PromptAgentBuilder<I, String> {
        self.with_output_decoder(decode_text_output)
    }

    pub fn json_output<T>(self) -> PromptAgentBuilder<I, T>
    where
        T: DeserializeOwned + JsonSchema + Send + Sync + 'static,
    {
        self.with_output_decoder_and_schema(
            decode_json_output::<T>,
            serde_json::to_value(schema_for!(T))
                .map(Some)
                .map_err(json_encode_error),
        )
    }

    pub fn build(self) -> Result<PromptAgent<I, O>, AgentError> {
        let mut tools: HashMap<String, Arc<dyn RuntimeTool>> = HashMap::new();
        let mut tool_schemas = Vec::new();
        let mut children = HashMap::new();

        if self.reserved_runtime_tools.load_skill
            && let Some(skills) = &self.skills
        {
            let load_skill_tool: Arc<dyn RuntimeTool> =
                Arc::new(LoadSkillTool::new(Arc::clone(skills)));
            let descriptor = load_skill_tool.descriptor();
            tools.insert(descriptor.name.clone(), load_skill_tool);
            tool_schemas.push(ToolSchema {
                name: descriptor.name,
                description: descriptor.description,
                input_schema: descriptor.input_schema,
            });
        }

        for tool in self.tools {
            let descriptor = tool.descriptor();
            validate_tool_name(&descriptor.name)?;

            if tools.insert(descriptor.name.clone(), tool).is_some() {
                return Err(AgentError::Runtime(RuntimeError::Message(format!(
                    "tool '{}' is registered more than once",
                    descriptor.name
                ))));
            }

            tool_schemas.push(ToolSchema {
                name: descriptor.name,
                description: descriptor.description,
                input_schema: descriptor.input_schema,
            });
        }

        for (name, registration) in self.children {
            if name.trim().is_empty() {
                return Err(AgentError::Runtime(RuntimeError::Message(
                    "child registration name cannot be empty".into(),
                )));
            }

            if children.insert(name.clone(), registration).is_some() {
                return Err(AgentError::Runtime(RuntimeError::Message(format!(
                    "child '{}' is registered more than once",
                    name
                ))));
            }
        }

        if self.reserved_runtime_tools.subagents && !children.is_empty() {
            tool_schemas.extend(subagent_tool_schemas());
        }

        if self.reserved_runtime_tools.finish {
            tool_schemas.push(finish_tool_schema());
        }

        Ok(PromptAgent {
            provider: self.provider,
            system_prompt: self.system_prompt,
            limits: self.limits,
            reserved_runtime_tools: self.reserved_runtime_tools,
            artifact_store: self.artifact_store,
            tools,
            tool_schemas,
            skills: self.skills,
            children,
            input_encoder: self.input_encoder,
            output_decoder: self.output_decoder,
            output_schema: self.output_schema,
            max_output_tokens: self.max_output_tokens,
        })
    }

    fn with_input_encoder<T>(
        self,
        encoder: impl Fn(T) -> Result<Payload, AgentError> + Send + Sync + 'static,
    ) -> PromptAgentBuilder<T, O> {
        PromptAgentBuilder {
            provider: self.provider,
            system_prompt: self.system_prompt,
            limits: self.limits,
            reserved_runtime_tools: self.reserved_runtime_tools,
            artifact_store: self.artifact_store,
            tools: self.tools,
            skills: self.skills,
            children: self.children,
            input_encoder: Arc::new(encoder),
            output_decoder: self.output_decoder,
            output_schema: self.output_schema,
            max_output_tokens: self.max_output_tokens,
            marker: PhantomData,
        }
    }

    fn with_output_decoder<T>(
        self,
        decoder: impl Fn(Payload) -> Result<T, AgentError> + Send + Sync + 'static,
    ) -> PromptAgentBuilder<I, T> {
        self.with_output_decoder_and_schema(decoder, Ok(None))
    }

    fn with_output_decoder_and_schema<T>(
        self,
        decoder: impl Fn(Payload) -> Result<T, AgentError> + Send + Sync + 'static,
        output_schema: Result<Option<Value>, AgentError>,
    ) -> PromptAgentBuilder<I, T> {
        PromptAgentBuilder {
            provider: self.provider,
            system_prompt: self.system_prompt,
            limits: self.limits,
            reserved_runtime_tools: self.reserved_runtime_tools,
            artifact_store: self.artifact_store,
            tools: self.tools,
            skills: self.skills,
            children: self.children,
            input_encoder: self.input_encoder,
            output_decoder: Arc::new(decoder),
            output_schema: output_schema.expect("output schema should serialize"),
            max_output_tokens: self.max_output_tokens,
            marker: PhantomData,
        }
    }
}

impl ChildRegistration {
    fn build_child_input(
        &self,
        payload: &Payload,
        parent: &SessionRecord,
    ) -> Result<Payload, AgentError> {
        let artifact_refs = child_artifact_refs(self, parent, payload);
        if matches!(self.context_transport, ChildContextTransport::SystemPrompt) {
            return Ok(payload.clone());
        }
        let needs_envelope = !matches!(self.context_mode, ChildContextMode::PayloadOnly)
            || !artifact_refs.is_empty()
            || self.instructions.is_some();
        if !needs_envelope {
            return Ok(payload.clone());
        }

        let mut envelope = serde_json::Map::new();
        envelope.insert(
            "payload".into(),
            serde_json::to_value(payload).map_err(json_encode_error)?,
        );

        if let Some(instructions) = &self.instructions {
            envelope.insert("instructions".into(), Value::String(instructions.clone()));
        }

        if !artifact_refs.is_empty() {
            envelope.insert(
                "artifact_refs".into(),
                serde_json::to_value(&artifact_refs).map_err(json_encode_error)?,
            );
        }

        match self.context_mode {
            ChildContextMode::PayloadOnly => {}
            ChildContextMode::ParentSummary => {
                envelope.insert(
                    "parent".into(),
                    json!({
                        "mode": "summary",
                        "session_id": parent.metadata.session_id,
                        "summary": parent_summary(parent),
                    }),
                );
            }
            ChildContextMode::FullParentContext => {
                envelope.insert(
                    "parent".into(),
                    json!({
                        "mode": "full_context",
                        "session_id": parent.metadata.session_id,
                        "context": {
                            "metadata": parent.metadata,
                            "current_input": parent.current_input,
                            "transcript": parent.transcript,
                            "tool_calls": parent.tool_calls,
                            "artifact_ids": parent.artifact_ids,
                        }
                    }),
                );
            }
        }

        Ok(Payload::Json(Value::Object(envelope)))
    }

    fn prepare_agent(
        &self,
        parent: &PromptAgent<impl Send + 'static, impl Send + 'static>,
        parent_session: &SessionRecord,
        payload: &Payload,
    ) -> Result<PromptAgent<Payload, Payload>, AgentError> {
        let mut child = self.agent.clone();
        let inherited = parent.inherited_tool_entries(self.tool_policy.inherited_names())?;

        match &self.tool_policy {
            ChildToolPolicy::ChildOnly => {}
            ChildToolPolicy::InheritAll | ChildToolPolicy::InheritNamed(_) => {
                child.apply_tool_policy_replace(inherited);
            }
            ChildToolPolicy::MergeInheritedAll | ChildToolPolicy::MergeInheritedNamed(_) => {
                child.apply_tool_policy_merge(inherited);
            }
        }

        if let Some(context_prompt) = self.context_prompt(parent_session, payload)? {
            child.append_system_prompt_section(context_prompt);
        }

        Ok(child)
    }

    fn context_prompt(
        &self,
        parent: &SessionRecord,
        payload: &Payload,
    ) -> Result<Option<String>, AgentError> {
        if !matches!(self.context_transport, ChildContextTransport::SystemPrompt) {
            return Ok(None);
        }

        let artifact_refs = child_artifact_refs(self, parent, payload);
        let mut sections = Vec::new();

        if let Some(instructions) = &self.instructions {
            sections.push(instructions.clone());
        }

        if !artifact_refs.is_empty() {
            let artifact_refs = serde_json::to_string(&artifact_refs).map_err(json_encode_error)?;
            sections.push(format!("Artifact references: {artifact_refs}"));
        }

        match self.context_mode {
            ChildContextMode::PayloadOnly => {}
            ChildContextMode::ParentSummary => {
                sections.push(format!("Parent summary: {}", parent_summary(parent)));
            }
            ChildContextMode::FullParentContext => {
                let context = serde_json::to_string(&json!({
                    "metadata": parent.metadata,
                    "current_input": parent.current_input,
                    "transcript": parent.transcript,
                    "tool_calls": parent.tool_calls,
                    "artifact_ids": parent.artifact_ids,
                }))
                .map_err(json_encode_error)?;
                sections.push(format!("Parent context JSON: {context}"));
            }
        }

        Ok((!sections.is_empty()).then(|| sections.join("\n\n")))
    }
}

impl ChildToolPolicy {
    fn inherited_names(&self) -> Option<&[String]> {
        match self {
            ChildToolPolicy::ChildOnly
            | ChildToolPolicy::MergeInheritedAll
            | ChildToolPolicy::InheritAll => None,
            ChildToolPolicy::InheritNamed(names) | ChildToolPolicy::MergeInheritedNamed(names) => {
                Some(names)
            }
        }
    }
}

type ToolEntry = (String, Arc<dyn RuntimeTool>, ToolSchema);

impl<I, O> PromptAgent<I, O>
where
    I: 'static,
    O: 'static,
{
    fn into_payload_agent(
        self,
        input_decoder: PayloadInputDecoder<I>,
        output_encoder: PayloadOutputEncoder<O>,
    ) -> PromptAgent<Payload, Payload> {
        let PromptAgent {
            provider,
            system_prompt,
            limits,
            reserved_runtime_tools,
            artifact_store,
            tools,
            tool_schemas,
            skills,
            children,
            input_encoder,
            output_decoder,
            output_schema,
            max_output_tokens,
        } = self;

        PromptAgent {
            provider,
            system_prompt,
            limits,
            reserved_runtime_tools,
            artifact_store,
            tools,
            tool_schemas,
            skills,
            children,
            input_encoder: Arc::new(move |payload: Payload| {
                let typed_input = input_decoder(payload)?;
                input_encoder(typed_input)
            }),
            output_decoder: Arc::new(move |payload: Payload| {
                let typed_output = output_decoder(payload)?;
                output_encoder(typed_output)
            }),
            output_schema,
            max_output_tokens,
        }
    }

    fn inherited_tool_entries(
        &self,
        selected_names: Option<&[String]>,
    ) -> Result<Vec<ToolEntry>, AgentError> {
        let selected = selected_names.map(|names| names.iter().cloned().collect::<HashSet<_>>());
        let mut entries = Vec::new();

        for schema in &self.tool_schemas {
            if RESERVED_TOOL_NAMES.contains(&schema.name.as_str())
                || schema.name == FINISH_TOOL_NAME
            {
                continue;
            }

            if let Some(selected) = &selected
                && !selected.contains(&schema.name)
            {
                continue;
            }

            let tool = self.tools.get(&schema.name).ok_or_else(|| {
                AgentError::Runtime(RuntimeError::Message(format!(
                    "tool '{}' is missing from runtime tool map",
                    schema.name
                )))
            })?;
            entries.push((schema.name.clone(), Arc::clone(tool), schema.clone()));
        }

        Ok(entries)
    }
}

impl PromptAgent<Payload, Payload> {
    fn append_system_prompt_section(&mut self, section: String) {
        self.system_prompt = Some(match self.system_prompt.take() {
            Some(existing) if !existing.is_empty() => format!("{existing}\n\n{section}"),
            Some(_) | None => section,
        });
    }

    fn apply_tool_policy_replace(&mut self, inherited: Vec<ToolEntry>) {
        self.tools
            .retain(|name, _| RESERVED_TOOL_NAMES.contains(&name.as_str()));
        self.tool_schemas.retain(|schema| {
            RESERVED_TOOL_NAMES.contains(&schema.name.as_str()) || schema.name == FINISH_TOOL_NAME
        });

        for (name, tool, schema) in inherited {
            self.tools.insert(name, tool);
            let insert_at = self
                .tool_schemas
                .iter()
                .position(|entry| entry.name == FINISH_TOOL_NAME)
                .unwrap_or(self.tool_schemas.len());
            self.tool_schemas.insert(insert_at, schema);
        }
    }

    fn apply_tool_policy_merge(&mut self, inherited: Vec<ToolEntry>) {
        let mut known = self
            .tool_schemas
            .iter()
            .map(|schema| schema.name.clone())
            .collect::<HashSet<_>>();

        for (name, tool, schema) in inherited {
            if !known.insert(name.clone()) {
                continue;
            }
            self.tools.insert(name, tool);
            let insert_at = self
                .tool_schemas
                .iter()
                .position(|entry| entry.name == FINISH_TOOL_NAME)
                .unwrap_or(self.tool_schemas.len());
            self.tool_schemas.insert(insert_at, schema);
        }
    }
}

fn active_child_count(session: &SessionRecord) -> usize {
    session
        .children
        .iter()
        .filter(|child| !child.completed)
        .count()
}

fn mark_child_completed(session: &mut SessionRecord, handle: &str) {
    if let Some(child) = session.children.iter_mut().find(|child| child.id == handle) {
        child.completed = true;
    }
}

fn child_session_metadata(
    parent: &SessionRecord,
    child: &crate::ChildHandleRecord,
) -> SessionMetadata {
    SessionMetadata {
        session_id: child.session_id.clone(),
        depth: parent.metadata.depth + 1,
        parent_session_id: Some(parent.metadata.session_id.clone()),
    }
}

fn run_payload_child(
    child: PromptAgent<Payload, Payload>,
    input: Payload,
    session_metadata: SessionMetadata,
    limits: RuntimeLimits,
) -> Pin<Box<dyn Future<Output = Result<PromptSession<Payload>, AgentError>> + Send>> {
    Box::pin(async move {
        child
            .run_with_context(input, session_metadata, limits)
            .await
    })
}

fn record_payload_artifacts(session: &mut SessionRecord, payload: &Payload) {
    if let Some(artifact_id) = payload.as_artifact()
        && !session.artifact_ids.contains(&artifact_id)
    {
        session.artifact_ids.push(artifact_id);
    }
}

fn child_artifact_refs(
    registration: &ChildRegistration,
    _parent: &SessionRecord,
    payload: &Payload,
) -> Vec<ArtifactId> {
    let mut artifact_refs = Vec::new();
    if let Some(artifact_id) = payload.as_artifact()
        && !artifact_refs.contains(&artifact_id)
    {
        artifact_refs.push(artifact_id);
    }
    for artifact_id in &registration.artifact_refs {
        if !artifact_refs.contains(artifact_id) {
            artifact_refs.push(*artifact_id);
        }
    }
    artifact_refs
}

fn is_transient_provider_error(error: &ProviderError) -> bool {
    matches!(
        error,
        ProviderError::Timeout | ProviderError::RateLimited | ProviderError::Temporary(_)
    )
}

fn tool_error_payload(error: ToolError) -> Payload {
    let (kind, message) = match error {
        ToolError::InvalidInput(message) => ("invalid_input", message),
        ToolError::ExecutionFailed(message) => ("execution_failed", message),
        ToolError::Timeout => ("timeout", ToolError::Timeout.to_string()),
    };

    Payload::json(json!({
        "ok": false,
        "error": {
            "type": kind,
            "message": message,
        }
    }))
    .expect("tool error payload should serialize")
}

fn parent_summary(session: &SessionRecord) -> String {
    format!(
        "parent session {} at depth {} with {} transcript messages, {} tool calls, and {} artifacts",
        session.metadata.session_id,
        session.metadata.depth,
        session.transcript.len(),
        session.tool_calls.len(),
        session.artifact_ids.len()
    )
}

fn subagent_tool_schemas() -> [ToolSchema; 4] {
    [
        ToolSchema {
            name: crate::CALL_SUBAGENT_TOOL_NAME.into(),
            description: "Call a registered child agent and wait for its result.".into(),
            input_schema: child_invocation_schema(),
        },
        ToolSchema {
            name: crate::SPAWN_SUBAGENT_TOOL_NAME.into(),
            description: "Spawn a registered child agent and return a child handle.".into(),
            input_schema: child_invocation_schema(),
        },
        ToolSchema {
            name: crate::AWAIT_SUBAGENT_TOOL_NAME.into(),
            description: "Wait for a previously spawned child handle and return its result.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "handle": { "type": "string" }
                },
                "required": ["handle"]
            }),
        },
        ToolSchema {
            name: crate::HANDOFF_TOOL_NAME.into(),
            description: "Transfer control to a registered child and end with the child output."
                .into(),
            input_schema: child_invocation_schema(),
        },
    ]
}

fn child_invocation_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "child": { "type": "string" },
            "payload": {}
        },
        "required": ["child", "payload"]
    })
}

fn validate_tool_name(name: &str) -> Result<(), AgentError> {
    if RESERVED_TOOL_NAMES.contains(&name) {
        return Err(AgentError::Runtime(RuntimeError::Message(format!(
            "tool '{name}' uses a reserved runtime tool name"
        ))));
    }
    Ok(())
}

fn finish_tool_schema() -> ToolSchema {
    ToolSchema {
        name: FINISH_TOOL_NAME.into(),
        description: "Finish the session and return the final output payload.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "text": { "type": "string" },
                "json": {},
                "payload": {
                    "type": "object",
                    "properties": {
                        "kind": { "type": "string" },
                        "value": {}
                    },
                    "required": ["kind", "value"]
                }
            }
        }),
    }
}

#[derive(Debug, Deserialize)]
struct LoadSkillInput {
    name: String,
}

struct LoadSkillTool {
    skills: Arc<SkillRegistry>,
}

impl LoadSkillTool {
    fn new(skills: Arc<SkillRegistry>) -> Self {
        Self { skills }
    }
}

#[async_trait]
impl RuntimeTool for LoadSkillTool {
    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            name: LOAD_SKILL_TOOL_NAME.into(),
            description: "Load the full body of a configured skill by name.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" }
                },
                "required": ["name"]
            }),
            output_schema: None,
        }
    }

    async fn call_payload(
        &self,
        input: Payload,
        _ctx: &mut ToolContext,
    ) -> Result<Payload, ToolError> {
        let input = input
            .to_typed::<LoadSkillInput>()
            .map_err(|error| ToolError::InvalidInput(error.to_string()))?;
        let body = self
            .skills
            .load(&input.name)
            .map_err(|error| ToolError::ExecutionFailed(error.to_string()))?;
        Ok(Payload::text(body))
    }
}

#[derive(Debug, Deserialize)]
struct FallbackEnvelope {
    claumini_runtime: FallbackDirective,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum FallbackDirective {
    ToolCalls { tool_calls: Vec<ToolCall> },
    Final { payload: Payload },
}

enum ParsedFallbackResponse {
    ToolCalls(Vec<ToolCall>),
    Final(Payload),
}

fn parse_fallback_response(message: &Message) -> Result<ParsedFallbackResponse, AgentError> {
    let envelope = match &message.content {
        Payload::Text(text) => serde_json::from_str::<FallbackEnvelope>(text),
        Payload::Json(value) => serde_json::from_value::<FallbackEnvelope>(value.clone()),
        Payload::Artifact(_) => {
            return Err(AgentError::Runtime(RuntimeError::Message(
                "fallback response must be text or json".into(),
            )));
        }
    }
    .map_err(|error| {
        AgentError::Runtime(RuntimeError::Message(format!(
            "failed to parse fallback response envelope: {error}"
        )))
    })?;

    match envelope.claumini_runtime {
        FallbackDirective::ToolCalls { tool_calls } => {
            Ok(ParsedFallbackResponse::ToolCalls(tool_calls))
        }
        FallbackDirective::Final { payload } => Ok(ParsedFallbackResponse::Final(payload)),
    }
}

fn skill_metadata_prompt(skills: Vec<SkillMetadata>, load_skill_enabled: bool) -> String {
    let mut lines = vec!["Available skills:".to_owned()];

    if load_skill_enabled {
        lines.push(
            "Use the runtime-owned `load_skill` tool to load the full body of a skill when needed."
                .to_owned(),
        );
    }

    for skill in skills {
        lines.push(format!("- {}: {}", skill.name, skill.description));
    }

    lines.join("\n")
}

fn assistant_response_message(response: &ModelResponse) -> Option<Message> {
    if response.tool_calls.is_empty() {
        return response.message.clone();
    }

    Some(
        response
            .message
            .clone()
            .unwrap_or_else(|| Message::new(MessageRole::Assistant, Payload::text("")))
            .with_tool_calls(response.tool_calls.clone()),
    )
}

fn fallback_prompt(tools: &[ToolSchema]) -> String {
    let mut lines = vec![
        "This provider does not support native tool calling.".to_owned(),
        "Every assistant response must be a single JSON object with no extra text.".to_owned(),
        "Use exactly one of these envelopes:".to_owned(),
        "{".to_owned(),
        "  \"claumini_runtime\": {".to_owned(),
        "    \"type\": \"tool_calls\",".to_owned(),
        "    \"tool_calls\": [{\"id\": \"call-1\", \"name\": \"tool_name\", \"arguments\": {}}]"
            .to_owned(),
        "  }".to_owned(),
        "}".to_owned(),
        "or".to_owned(),
        "{".to_owned(),
        "  \"claumini_runtime\": {".to_owned(),
        "    \"type\": \"final\",".to_owned(),
        "    \"payload\": {\"kind\": \"text\", \"value\": \"done\"}".to_owned(),
        "  }".to_owned(),
        "}".to_owned(),
        "The `payload` field must be a valid serialized `Payload` object.".to_owned(),
        "Available tools:".to_owned(),
    ];

    for tool in tools {
        let schema = serde_json::to_string(&tool.input_schema).unwrap_or_else(|_| "{}".to_owned());
        lines.push(format!(
            "- {}: {} | input_schema={schema}",
            tool.name, tool.description
        ));
    }

    lines.join("\n")
}

fn structured_output_prompt(schema: &Value) -> String {
    let schema = serde_json::to_string(schema).unwrap_or_else(|_| "{}".to_owned());
    format!(
        "Return the final answer as JSON matching this schema exactly with no surrounding prose: {schema}"
    )
}

#[async_trait]
impl<I, O> Agent for PromptAgent<I, O>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    type Input = I;
    type Output = O;

    async fn run(
        &self,
        input: Self::Input,
        ctx: &mut AgentContext,
    ) -> Result<Self::Output, AgentError> {
        let PromptSession { output, session } = self
            .run_with_context(input, ctx.session.clone(), ctx.limits.clone())
            .await?;
        ctx.transcript_len = session.transcript.len();
        ctx.tool_call_count = session.tool_calls.len();
        ctx.current_input = session.current_input.clone();
        ctx.artifact_ids = session.artifact_ids.clone();
        Ok(output)
    }
}

fn finish_payload(arguments: &Value) -> Result<Payload, AgentError> {
    if let Some(payload) = arguments.get("payload") {
        return serde_json::from_value(payload.clone()).map_err(|error| {
            AgentError::Runtime(RuntimeError::Message(format!(
                "finish payload is invalid: {error}"
            )))
        });
    }

    if let Some(text) = arguments.get("text").and_then(Value::as_str) {
        return Ok(Payload::text(text));
    }

    if let Some(value) = arguments.get("json") {
        return Ok(Payload::Json(value.clone()));
    }

    Err(AgentError::Runtime(RuntimeError::Message(
        "finish requires one of 'text', 'json', or 'payload'".into(),
    )))
}

fn decode_text_output(payload: Payload) -> Result<String, AgentError> {
    match payload {
        Payload::Text(text) => Ok(text),
        Payload::Json(Value::String(text)) => Ok(text),
        other => Err(AgentError::Runtime(RuntimeError::Message(format!(
            "expected text output payload, received {other:?}"
        )))),
    }
}

fn decode_json_input<T>(payload: Payload) -> Result<T, AgentError>
where
    T: DeserializeOwned,
{
    payload.to_typed().map_err(|error| {
        AgentError::Runtime(RuntimeError::Message(format!(
            "failed to decode child json input: {error}"
        )))
    })
}

fn encode_json_payload<T>(value: T) -> Result<Payload, AgentError>
where
    T: Serialize,
{
    Payload::json(value).map_err(|error| {
        AgentError::Runtime(RuntimeError::Message(format!(
            "failed to encode child json output: {error}"
        )))
    })
}

fn decode_json_output<T>(payload: Payload) -> Result<T, AgentError>
where
    T: DeserializeOwned,
{
    match payload {
        Payload::Json(value) => serde_json::from_value(value).map_err(json_decode_error),
        Payload::Text(text) => serde_json::from_str(&text).map_err(json_decode_error),
        Payload::Artifact(_) => Err(AgentError::Runtime(RuntimeError::Message(
            "expected json output payload, received artifact".into(),
        ))),
    }
}

fn json_decode_error(error: serde_json::Error) -> AgentError {
    AgentError::Runtime(RuntimeError::Message(format!(
        "failed to decode structured output json: {error}"
    )))
}

fn json_encode_error(error: serde_json::Error) -> AgentError {
    AgentError::Runtime(RuntimeError::Message(format!(
        "failed to encode runtime json payload: {error}"
    )))
}

#[cfg(test)]
mod tests {
    use claumini_core::Payload;
    use serde_json::json;

    use super::finish_payload;

    #[test]
    fn finish_payload_prefers_explicit_payload_shape() {
        let payload = finish_payload(&json!({
            "payload": {
                "kind": "text",
                "value": "done"
            }
        }))
        .expect("finish payload should decode");

        assert_eq!(payload, Payload::text("done"));
    }
}
