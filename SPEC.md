# claumini

> claumini = claude + mini

`claumini` is a Rust agent framework for building agentic systems with swappable components. The core API stays small and typed. Higher-level crates provide runtime orchestration, model adapters, tool integration, and skill loading.

## Design goals

- Support multiple model providers through one common interface.
- Keep agents, tools, providers, and workflows composable through traits and typed data.
- Let users write orchestration in plain async Rust.
- Support nested subagents, agent handoff, and parallel worker patterns.
- Support both text payloads and user-defined Rust structs at every boundary.
- Keep v1 in-memory and simple.

## Non-goals for v1

- No session persistence or resume from disk.
- No automatic context compaction or summarization.
- No workflow DSL.
- No streaming model output.
- No built-in semantic code analysis or parser-backed repo tools.
- No token or cost accounting.

## Workspace layout

### `claumini-core`

Stable traits and shared types.

- `Agent`
- `ModelProvider`
- `Tool`
- payload and artifact types
- runtime limits and policy types
- common errors

### `claumini-runtime`

Session execution and orchestration.

- agent turn loop
- subagent spawning and waiting
- in-memory artifact store
- reserved runtime tools
- provider retry handling

### `claumini-models`

Provider adapters.

- Claude adapter
- OpenAI adapter
- custom OpenAI-compatible adapter via base URL and API key

### `claumini-tools`

Tool traits, proc macros, and minimal built-in repo tools.

- tool metadata and schema generation
- sync and async tool adapters
- text-first repo inspection tools

## Core runtime model

### Agent forms

`claumini` supports two agent styles.

1. Low-level `Agent` trait implementations for custom behavior.
2. `PromptAgentBuilder -> PromptAgent` for the common prompt-driven case.

The builder only creates the standard prompt-driven agent. Users who want custom control flow or non-LLM agents implement `Agent` directly.

### Workflow model

Workflows are plain async Rust. The framework does not define a separate DSL.

Common runtime operations:

- run an agent with an input payload
- call a subagent and wait for the result
- spawn subagents and await them later
- join multiple child results
- finish a session

This model supports:

- sequential pipelines
- fan-out and fan-in worker patterns
- nested debug or repair loops
- user-defined traversal logic such as graph walking

## Sessions and state

Sessions are in-memory only in v1.

Each session owns:

- transcript
- tool call history
- current input
- child-agent handles
- artifact references
- workflow-local state owned by user code

Runtime session states:

- `Running`
- `Waiting`
- `Finished`

If the process exits, the session is lost. Persistence is out of scope for v1.

## Payload model

The framework supports both plain text and structured data.

Users can choose per boundary:

- user -> agent
- agent -> agent
- agent -> user

Structured mode uses JSON and `serde` by default. Text mode remains first-class.

Suggested runtime payload shape:

```rust
enum Payload {
    Text(String),
    Json(serde_json::Value),
    Artifact(ArtifactId),
}
```

Typed Rust APIs should wrap this with `T: Serialize + DeserializeOwned` helpers so users can pass Rust structs without hand-writing JSON plumbing.

## Artifact model

Large reports, file lists, code snippets, and child-agent results should be stored as in-memory artifacts and passed by reference when possible.

This keeps prompts smaller and avoids repeated payload copying.

Suggested uses:

- parsed bug report
- suspicious file list
- suspicious function ranking
- child-agent experiment results
- code context slices

## Agent I/O behavior

Prompt-driven agents may operate over text or typed inputs and outputs.

- Text output is allowed.
- Structured output is allowed.
- Structured output parsing should inject schema guidance into the model prompt.
- Field descriptions should be visible to the model when structured mode is enabled.

This lets users mix styles. One agent may return text. Another may return `Vec<SuspiciousFunction>`.

## Tool system

### Tool definition

Tools are user-defined Rust types with typed input and output.

Requirements:

- user defines input and output types
- input and output fields include descriptions for model-facing schemas
- user implements tool behavior
- framework uses traits plus proc macros for schema generation and registration
- sync and async tool implementations are both supported

The runtime-facing tool layer should still be async so sync tools can be adapted into the same dispatch path.

### Tool behavior

From the agent's point of view, tool calls are synchronous request-response operations.

The agent waits until a tool returns a result or error before continuing its turn.

### Built-in repo tools

`claumini-tools` should ship a small text-first toolkit.

- `list_files`
- `search_text`
- `read_file`
- `read_range`
- optional guarded command execution as an opt-in tool

No parser-backed or semantic repo tools are required in v1.

## Reserved runtime tools

Runtime control operations should appear to prompt-driven agents as reserved built-in tools owned by the runtime.

Examples:

- `call_subagent`
- `spawn_subagent`
- `await_subagent`
- `handoff`
- `finish`

These are not user-defined tools. The runtime intercepts them before normal tool dispatch.

Users still control whether a given agent is allowed to use these runtime tools.

## Subagent model

### Invocation styles

The runtime supports both:

1. blocking child calls
2. handle-based spawn and await

The blocking form is a convenience wrapper over spawn plus await.

### Recursive subagents

Child agents may spawn their own children only if the user enables that behavior and runtime limits allow it.

### Tool and context passing

The user controls how a child agent is configured.

Allowed patterns:

- inherit parent tools
- pass a subset of parent tools
- pass only custom child tools
- merge inherited and custom tools

The same applies to context and instructions. The framework should let users choose whether a parent can:

- pass explicit payload only
- pass selected artifacts
- pass a parent summary
- pass full parent context
- add child-specific instructions

The framework does not enforce one default handoff style beyond what the user configures.

## Skills

Skills follow a two-layer loading model.

### Layer 1: cheap metadata

At startup, the runtime scans skill directories for `SKILL.md` files and collects lightweight metadata such as:

- name
- description
- optional tags or location

These descriptions are injected into the system prompt so the agent knows which skills exist.

### Layer 2: full skill body

When needed, the agent loads the full skill body on demand through a runtime capability such as `load_skill(name)`.

This keeps baseline prompt size small while still supporting domain-specific instructions.

In v1, skills are instruction bundles. They are not active runtime plugins.

## Provider abstraction

Providers use a common interface plus capability flags.

Common behavior should cover:

- chat or completion requests for prompt-driven agents
- system prompt support
- multi-turn tool use
- optional structured output via JSON

Capability flags may include:

- native tool calling support
- structured output support
- reasoning-control support
- image input support

### Tool calling strategy

If a provider supports native tool or function calling, use it.

If a provider does not support native tool calling, fall back to a prompt-level structured tool-call format that the runtime parses.

The runtime should present one uniform tool-dispatch model above the provider layer.

## Error handling

### Provider errors

Transient provider failures may be retried inside the framework.

Examples:

- timeout
- rate limit
- temporary upstream failure

Terminal provider failures should fail fast.

### Tool errors

Tool errors should be returned to the agent as structured failures inside the turn loop.

That lets the agent decide whether to retry with different arguments, use another tool, or stop.

### Workflow errors

Workflow or pipeline errors remain the responsibility of user Rust code.

Examples:

- aggregation failure
- invalid workflow branch
- business-rule violation

## Runtime limits

V1 should enforce only core safety limits.

- `max_turns_per_session`
- `max_spawn_depth`
- `max_active_children`
- `model_request_timeout`
- `tool_call_timeout`

If a limit is exceeded, the runtime should surface a structured failure or terminate the session with a clear runtime error.

No token or cost budgeting is required in v1.

## Context model

The runtime should not silently summarize, trim, or compact context.

Users own context assembly. The framework should expose session history and artifacts cleanly so the user or agent configuration can decide what to include in the next request.

This keeps behavior explicit and easier to debug.

## Response model

V1 does not require streaming model output.

Each model request returns a complete response. The runtime then decides whether the response contains:

- final text
- structured output
- native tool calls
- prompt-fallback tool calls
- reserved runtime tool calls

## Example usage patterns

### 1. Sequential bug-analysis pipeline

Agent A reads a test report and produces bug analysis.

Agent B reads the analysis and produces suspicious functions.

Agent C reads suspicious functions, inspects code context, and returns ranked function scores plus explanations.

### 2. Fan-out suspicious-file analysis

Agent A reads the repo structure and a bug report, then produces suspicious files.

The workflow spawns multiple Agent B workers, one per file, and then merges their scored root-cause candidates.

### 3. User-defined graph walking

The framework does not understand graphs directly.

Users expose graph navigation as tools such as:

- `get_node`
- `list_neighbors`
- `read_node_context`

The agent decides which node to inspect next. Workflow code validates traversal rules and stores suspicious nodes as artifacts or typed state.

### 4. Nested debug and repair loop

Agent A investigates a bug report over multiple turns.

When it wants to test a repair hypothesis, it invokes Agent B with scoped edit and run tools. Agent B edits code, runs checks, returns results, and Agent A resumes its investigation. When Agent A decides the task is complete, it calls `finish`.

## Suggested public API shape

The exact trait layout can change during implementation, but the surface should look roughly like this.

```rust
trait Agent {
    type Input;
    type Output;

    async fn run(&self, input: Self::Input, ctx: &mut AgentContext) -> Result<Self::Output, AgentError>;
}

trait ModelProvider {
    async fn complete(&self, request: ModelRequest) -> Result<ModelResponse, ProviderError>;
    fn capabilities(&self) -> ProviderCapabilities;
}

trait Tool {
    type Input;
    type Output;

    async fn call(&self, input: Self::Input, ctx: &mut ToolContext) -> Result<Self::Output, ToolError>;
}
```

The final design should keep these interfaces small and stable.

## V1 summary

`claumini` v1 is a typed, in-memory Rust agent framework with:

- swappable providers
- swappable tools
- prompt-driven and custom agents
- plain Rust orchestration
- nested subagents
- text and structured payload support
- in-memory artifacts
- text-first repo tools
- skill loading by metadata plus on-demand body retrieval

That is enough to support debugging workflows, multi-agent analysis pipelines, and user-defined traversal or repair loops without committing to a larger execution platform too early.
