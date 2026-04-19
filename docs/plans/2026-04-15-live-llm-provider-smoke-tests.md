# Live LLM Provider Smoke Tests Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a live smoke-test target that reads provider credentials from `.env` and verifies basic completion, tool calling, and runtime skill loading against an OpenAI-compatible endpoint.

**Architecture:** Keep production code unchanged. Add an ignored integration test under `claumini-runtime/tests` because it can exercise both the provider crate and the runtime tool loop in one place. Load secrets from a repo-local `.env` file only for this live test path.

**Tech Stack:** Rust 2024, `tokio`, `dotenvy`, `tempfile`, `claumini-core`, `claumini-models`, `claumini-runtime`

---

### Task 1: Wire Live Test Configuration

**Files:**
- Modify: `.gitignore`
- Create: `.env`
- Modify: `claumini-runtime/Cargo.toml`

**Step 1: Add `.env` to gitignore**

```gitignore
.env
```

**Step 2: Add the live provider variables**

```dotenv
CLAUMINI_LIVE_API_KEY=...
CLAUMINI_LIVE_API_BASE=https://api.kksj.org/v1
CLAUMINI_LIVE_MODEL=gpt-5-mini
```

**Step 3: Add test-only dependencies**

```toml
claumini-models = { path = "../claumini-models" }
dotenvy = "0.15.7"
```

**Step 4: Verify the runtime crate still compiles**

Run: `cargo test -p claumini-runtime --no-run`
Expected: PASS

### Task 2: Add Live Smoke Tests

**Files:**
- Create: `claumini-runtime/tests/live_openai_compatible.rs`

**Step 1: Add a helper that loads `.env` and builds `OpenAiCompatibleProvider`**

```rust
dotenvy::from_filename(".env").ok();
let provider = OpenAiCompatibleProvider::new(OpenAiCompatibleConfig { ... })?;
```

**Step 2: Add a basic text completion test**

```rust
let request = ModelRequest::new(vec![Message::new(MessageRole::User, Payload::text("..."))]);
```

**Step 3: Add a native tool-calling smoke test**

```rust
let request = request.with_tool(ToolSchema { name: "provider_probe".into(), ... });
```

**Step 4: Add a runtime smoke test that exercises both a custom tool and `load_skill`**

```rust
let agent = PromptAgentBuilder::new(Arc::new(provider))
    .tool(ProbeTool)
    .skills(registry)
    .text_input()
    .text_output()
    .build()?;
```

**Step 5: Run the ignored live tests explicitly**

Run: `cargo test -p claumini-runtime --test live_openai_compatible -- --ignored --nocapture`
Expected: PASS against the configured endpoint, or return actionable provider errors.
