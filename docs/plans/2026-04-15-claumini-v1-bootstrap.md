# Claumini V1 Bootstrap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bootstrap a compilable Rust workspace that implements the v1 core contracts, initial runtime primitives, text-first repo tools, and provider scaffolding described in `SPEC.md`.

**Architecture:** Build a four-crate Cargo workspace. `claumini-core` owns the stable traits and shared types. `claumini-runtime`, `claumini-tools`, and `claumini-models` depend on `claumini-core` and can be implemented largely in parallel once the core crate compiles.

**Tech Stack:** Rust 2024, `async-trait`, `serde`, `serde_json`, `thiserror`, `tokio`, `walkdir`, `regex`, `tempfile`

---

### Task 1: Bootstrap Cargo Workspace

**Files:**
- Create: `Cargo.toml`
- Create: `claumini-core/Cargo.toml`
- Create: `claumini-core/src/lib.rs`
- Create: `claumini-runtime/Cargo.toml`
- Create: `claumini-runtime/src/lib.rs`
- Create: `claumini-tools/Cargo.toml`
- Create: `claumini-tools/src/lib.rs`
- Create: `claumini-models/Cargo.toml`
- Create: `claumini-models/src/lib.rs`

**Step 1: Create the workspace manifest and crate skeletons**

```toml
[workspace]
members = [
    "claumini-core",
    "claumini-runtime",
    "claumini-tools",
    "claumini-models",
]
resolver = "2"
```

**Step 2: Run workspace tests to verify the empty skeleton fails usefully**

Run: `cargo test --workspace`
Expected: FAIL with unresolved imports or missing public items referenced by later crate code.

**Step 3: Add minimal public crate entrypoints**

```rust
#![forbid(unsafe_code)]
```

**Step 4: Run workspace tests to verify the workspace now compiles as empty crates**

Run: `cargo test --workspace`
Expected: PASS with zero or placeholder tests.

### Task 2: Implement `claumini-core`

**Files:**
- Modify: `claumini-core/Cargo.toml`
- Create: `claumini-core/src/agent.rs`
- Create: `claumini-core/src/error.rs`
- Create: `claumini-core/src/model.rs`
- Create: `claumini-core/src/payload.rs`
- Create: `claumini-core/src/runtime.rs`
- Create: `claumini-core/src/tool.rs`
- Modify: `claumini-core/src/lib.rs`

**Step 1: Write failing tests for payloads, limits, and trait-facing request types**

```rust
#[test]
fn payload_json_round_trips() {
    let payload = Payload::json(serde_json::json!({"kind": "bug"})).unwrap();
    assert_eq!(payload.as_json().unwrap()["kind"], "bug");
}
```

**Step 2: Run the core tests to verify they fail for the expected missing APIs**

Run: `cargo test -p claumini-core`
Expected: FAIL because `Payload`, runtime limits, or request helper methods do not exist yet.

**Step 3: Implement the minimal core API**

```rust
#[async_trait::async_trait]
pub trait Agent {
    type Input;
    type Output;

    async fn run(
        &self,
        input: Self::Input,
        ctx: &mut AgentContext,
    ) -> Result<Self::Output, AgentError>;
}
```

Add:
- `Payload`, `ArtifactId`, `ProviderCapabilities`, `ModelRequest`, `ModelResponse`
- `AgentContext` and `ToolContext` shells with session metadata
- `RuntimeLimits`
- `AgentError`, `ProviderError`, `ToolError`, `RuntimeError`
- `Tool` trait

**Step 4: Run the core tests to verify they pass**

Run: `cargo test -p claumini-core`
Expected: PASS

### Task 3: Implement `claumini-tools`

**Files:**
- Modify: `claumini-tools/Cargo.toml`
- Create: `claumini-tools/src/adapters.rs`
- Create: `claumini-tools/src/metadata.rs`
- Create: `claumini-tools/src/repo.rs`
- Modify: `claumini-tools/src/lib.rs`

**Step 1: Write failing tests for text-first repo tools**

```rust
#[tokio::test]
async fn read_range_returns_requested_lines_only() {
    let tool = ReadRangeTool::new();
    let output = tool.call(
        ReadRangeInput { path: file_path, start_line: 2, end_line: 3 },
        &mut ctx,
    ).await.unwrap();

    assert_eq!(output.lines.len(), 2);
}
```

**Step 2: Run tools tests to verify they fail before implementation**

Run: `cargo test -p claumini-tools`
Expected: FAIL because repo tools and adapters are not implemented.

**Step 3: Implement minimal tool metadata, sync adapter, and repo tools**

```rust
pub struct ToolMetadata {
    pub name: &'static str,
    pub description: &'static str,
}
```

Add:
- `SyncToolAdapter<T>` for wrapping sync logic behind the async `Tool` trait
- `ListFilesTool`, `SearchTextTool`, `ReadFileTool`, `ReadRangeTool`
- tests that use `tempfile` directories instead of the live repo

**Step 4: Run tools tests to verify they pass**

Run: `cargo test -p claumini-tools`
Expected: PASS

### Task 4: Implement `claumini-runtime`

**Files:**
- Modify: `claumini-runtime/Cargo.toml`
- Create: `claumini-runtime/src/artifacts.rs`
- Create: `claumini-runtime/src/session.rs`
- Create: `claumini-runtime/src/skills.rs`
- Create: `claumini-runtime/src/runtime_tools.rs`
- Modify: `claumini-runtime/src/lib.rs`

**Step 1: Write failing tests for artifact storage, skill scanning, and reserved runtime tool names**

```rust
#[test]
fn reserved_runtime_tools_are_stable() {
    assert!(reserved_tool_names().contains(&"finish"));
}
```

**Step 2: Run runtime tests to verify they fail before implementation**

Run: `cargo test -p claumini-runtime`
Expected: FAIL because the artifact store, session state, or skill registry APIs are missing.

**Step 3: Implement minimal runtime primitives**

```rust
pub enum SessionState {
    Running,
    Waiting,
    Finished,
}
```

Add:
- in-memory `ArtifactStore`
- `SessionState`, `SessionRecord`, and child-handle metadata shells
- `SkillRegistry` that scans directories for `SKILL.md`, caches metadata, and loads full bodies on demand
- reserved runtime tool constants for `call_subagent`, `spawn_subagent`, `await_subagent`, `handoff`, and `finish`

**Step 4: Run runtime tests to verify they pass**

Run: `cargo test -p claumini-runtime`
Expected: PASS

### Task 5: Implement `claumini-models`

**Files:**
- Modify: `claumini-models/Cargo.toml`
- Create: `claumini-models/src/config.rs`
- Create: `claumini-models/src/openai.rs`
- Create: `claumini-models/src/claude.rs`
- Create: `claumini-models/src/mock.rs`
- Modify: `claumini-models/src/lib.rs`

**Step 1: Write failing tests for provider capability reporting and config validation**

```rust
#[test]
fn openai_compatible_provider_reports_tool_capability() {
    let provider = OpenAiCompatibleProvider::new(config()).unwrap();
    assert!(provider.capabilities().native_tool_calling);
}
```

**Step 2: Run models tests to verify they fail before implementation**

Run: `cargo test -p claumini-models`
Expected: FAIL because provider structs and config validation are not implemented.

**Step 3: Implement minimal provider scaffolding**

```rust
pub struct OpenAiCompatibleConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}
```

Add:
- config validation for OpenAI-compatible and Claude providers
- `MockProvider` for runtime and integration tests
- placeholder provider structs that implement `ModelProvider` and return a clear `ProviderError::Unimplemented` from `complete`

**Step 4: Run models tests to verify they pass**

Run: `cargo test -p claumini-models`
Expected: PASS

### Task 6: Integrate and verify the bootstrap slice

**Files:**
- Modify: `Cargo.toml`
- Modify: crate `src/lib.rs` files as needed for exports

**Step 1: Run the full workspace suite**

Run: `cargo test --workspace`
Expected: PASS

**Step 2: Run formatting and lint-level checks**

Run: `cargo fmt --all --check`
Expected: PASS

Run: `cargo check --workspace`
Expected: PASS

**Step 3: Clean up naming and exports without widening scope**

Keep only the public items needed by the spec-facing bootstrap.

**Step 4: Re-run verification**

Run: `cargo test --workspace && cargo fmt --all --check && cargo check --workspace`
Expected: PASS
