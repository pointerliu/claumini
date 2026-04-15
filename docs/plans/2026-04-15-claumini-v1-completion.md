# Claumini V1 Completion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Finish the remaining `SPEC.md` work so the workspace supports prompt-driven agents, typed payload boundaries, runtime orchestration, reserved runtime tools, subagent execution, skill loading, and real provider adapters.

**Architecture:** Keep `claumini-core` as the stable contract layer and put execution behavior in `claumini-runtime`. First lock the request/response, schema, and prompt-agent APIs. Then implement the runtime turn loop and provider normalization. After that, add subagent orchestration and finish with end-to-end tests that exercise the workflows promised in `SPEC.md`.

**Tech Stack:** Rust 2024, `async-trait`, `serde`, `serde_json`, `thiserror`, `tokio`, `reqwest`, `schemars`, `regex`, `walkdir`, `tempfile`, `wiremock`

---

### Task 1: Extend `claumini-core` for typed prompt agents

**Files:**
- Modify: `claumini-core/Cargo.toml`
- Modify: `claumini-core/src/lib.rs`
- Modify: `claumini-core/src/agent.rs`
- Modify: `claumini-core/src/model.rs`
- Modify: `claumini-core/src/payload.rs`
- Modify: `claumini-core/src/runtime.rs`
- Modify: `claumini-core/src/tool.rs`
- Create: `claumini-core/src/prompt_agent.rs`

**Step 1: Write failing tests for typed payload helpers, prompt-agent builder defaults, and schema-bearing tool descriptors**

Run: `cargo test -p claumini-core`
Expected: FAIL because `PromptAgentBuilder`, typed payload helper coverage, or schema metadata APIs are missing.

**Step 2: Implement minimal typed boundary helpers**

Add:
- `Payload::from_typed<T: Serialize>`
- `Payload::to_typed<T: DeserializeOwned>` coverage for prompt-agent inputs and outputs
- `ToolDescriptor` fields for input and output schemas plus field-description-bearing JSON schema payloads

**Step 3: Implement `PromptAgentBuilder<I, O>` and `PromptAgent<I, O>`**

Add:
- builder configuration for system prompt, provider key/name, allowed tools, reserved runtime tool policy, skill roots, typed output schema, and retry options
- prompt agent type that still implements the low-level `Agent` trait

**Step 4: Re-run core tests**

Run: `cargo test -p claumini-core`
Expected: PASS

### Task 2: Add tool schema generation support

**Files:**
- Modify: `claumini-tools/Cargo.toml`
- Modify: `claumini-tools/src/lib.rs`
- Modify: `claumini-tools/src/metadata.rs`
- Modify: `claumini-tools/src/adapters.rs`
- Create: `claumini-tools/src/schema.rs`

**Step 1: Write failing tests for schema generation from typed input/output structs**

Run: `cargo test -p claumini-tools`
Expected: FAIL because tool metadata cannot derive JSON schema with descriptions yet.

**Step 2: Implement schema helpers**

Add:
- `ToolSchemaProvider` trait backed by `schemars`
- helpers that convert typed input/output structs into `ToolDescriptor`
- sync adapter support for descriptors generated from typed structs

**Step 3: Re-run tools tests**

Run: `cargo test -p claumini-tools`
Expected: PASS

### Task 3: Implement the prompt-driven runtime turn loop

**Files:**
- Modify: `claumini-runtime/Cargo.toml`
- Modify: `claumini-runtime/src/lib.rs`
- Modify: `claumini-runtime/src/session.rs`
- Modify: `claumini-runtime/src/artifacts.rs`
- Create: `claumini-runtime/src/engine.rs`
- Create: `claumini-runtime/src/provider_loop.rs`
- Create: `claumini-runtime/src/tool_dispatch.rs`
- Create: `claumini-runtime/tests/prompt_agent_loop.rs`

**Step 1: Write failing runtime integration tests for a prompt agent that calls tools and then finishes**

Run: `cargo test -p claumini-runtime prompt_agent`
Expected: FAIL because there is no engine executing model turns, tool dispatch, or finish behavior.

**Step 2: Implement the runtime engine**

Add:
- session-backed transcript updates
- model request assembly from context, skills metadata, tools, and typed output schema
- synchronous tool-call/response loop from the agent’s point of view
- `finish` support for prompt-driven agents
- structured tool-error return into the turn loop

**Step 3: Enforce limits and timeouts in the loop**

Add:
- `max_turns_per_session`
- `model_request_timeout_ms`
- `tool_call_timeout_ms`

**Step 4: Re-run runtime tests**

Run: `cargo test -p claumini-runtime`
Expected: PASS

### Task 4: Implement provider retry handling and prompt fallback tool calling

**Files:**
- Modify: `claumini-runtime/src/provider_loop.rs`
- Create: `claumini-runtime/src/fallback.rs`
- Create: `claumini-runtime/tests/provider_fallback.rs`

**Step 1: Write failing tests for transient retry behavior and prompt-level tool-call fallback parsing**

Run: `cargo test -p claumini-runtime provider_fallback`
Expected: FAIL because retry classification and fallback parsing do not exist.

**Step 2: Implement provider retry behavior**

Add:
- retry wrapper for timeout, rate-limit, and temporary upstream failures
- fail-fast behavior for terminal provider errors

**Step 3: Implement one strict JSON fallback envelope for providers without native tool calling**

Add runtime parsing for:
- final text
- structured output
- prompt-fallback tool calls
- reserved runtime tool calls

**Step 4: Re-run runtime tests**

Run: `cargo test -p claumini-runtime`
Expected: PASS

### Task 5: Implement real provider adapters in `claumini-models`

**Files:**
- Modify: `claumini-models/Cargo.toml`
- Modify: `claumini-models/src/lib.rs`
- Modify: `claumini-models/src/config.rs`
- Modify: `claumini-models/src/openai.rs`
- Modify: `claumini-models/src/claude.rs`
- Modify: `claumini-models/src/mock.rs`
- Create: `claumini-models/src/http.rs`
- Create: `claumini-models/tests/openai.rs`
- Create: `claumini-models/tests/claude.rs`

**Step 1: Write failing tests against mocked HTTP endpoints**

Run: `cargo test -p claumini-models`
Expected: FAIL because the providers still return `ProviderError::Unimplemented`.

**Step 2: Implement OpenAI-compatible transport**

Add:
- request mapping for system prompt, messages, tools, and structured-output hints
- response parsing for final text and tool calls
- base URL and API key auth support

**Step 3: Implement Claude transport**

Add:
- Anthropic-style message request mapping
- response parsing for assistant text and tool-use blocks

**Step 4: Re-run model tests**

Run: `cargo test -p claumini-models`
Expected: PASS

### Task 6: Implement reserved runtime tools and subagent orchestration

**Files:**
- Modify: `claumini-runtime/src/lib.rs`
- Modify: `claumini-runtime/src/session.rs`
- Modify: `claumini-runtime/src/runtime_tools.rs`
- Modify: `claumini-runtime/src/engine.rs`
- Create: `claumini-runtime/src/subagents.rs`
- Create: `claumini-runtime/tests/subagents.rs`

**Step 1: Write failing tests for `call_subagent`, `spawn_subagent`, `await_subagent`, `handoff`, and nested-child limits**

Run: `cargo test -p claumini-runtime subagents`
Expected: FAIL because runtime tool names have no behavior yet.

**Step 2: Implement child execution and handles**

Add:
- blocking call convenience wrapper over spawn plus await
- handle-based spawn/await records
- runtime limit checks for `max_spawn_depth` and `max_active_children`
- parent-to-child configuration rules for tools, artifacts, transcript, summaries, and child instructions

**Step 3: Implement `handoff` semantics**

Define `handoff` as transferring control to a child configuration and ending the current session path.

**Step 4: Re-run runtime tests**

Run: `cargo test -p claumini-runtime`
Expected: PASS

### Task 7: Wire skills into prompt execution

**Files:**
- Modify: `claumini-runtime/src/skills.rs`
- Modify: `claumini-runtime/src/engine.rs`
- Create: `claumini-runtime/tests/skills.rs`

**Step 1: Write failing tests for metadata injection and on-demand `load_skill(name)` access**

Run: `cargo test -p claumini-runtime skills`
Expected: FAIL because skills are scanned but not used in the prompt/runtime path.

**Step 2: Implement runtime skill plumbing**

Add:
- cheap metadata injection into the prompt-agent system prompt
- on-demand skill body loading as a runtime capability/tool path

**Step 3: Re-run runtime tests**

Run: `cargo test -p claumini-runtime`
Expected: PASS

### Task 8: Add end-to-end workflow coverage and final verification

**Files:**
- Create: `claumini-runtime/tests/workflows.rs`
- Modify: crate exports as needed

**Step 1: Write failing end-to-end tests for the example usage patterns in `SPEC.md`**

Cover:
- sequential bug-analysis pipeline
- fan-out suspicious-file analysis
- user-defined graph walking with tools
- nested debug and repair loop with `finish`

**Step 2: Implement minimal glue needed to satisfy those tests**

Only add code required by the failing tests.

**Step 3: Run full verification**

Run: `cargo fmt --all --check`
Expected: PASS

Run: `cargo check --workspace`
Expected: PASS

Run: `cargo test --workspace`
Expected: PASS
