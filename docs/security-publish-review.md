# Public repository security review

Date: 2026-04-19

Scope: local working tree and Git history for this repository.

## Result

No known-format secrets were found in tracked source files or committed history during this review.

The local workspace does contain private files that must not be published:

| Path | Status | Finding |
| --- | --- | --- |
| `.env` | Ignored | Contains live-looking API key variables for `CLAUMINI_LIVE_API_KEY`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY`. |
| `agentless_demo/.env` | Ignored | Contains `CLAUMINI_LIVE_API_KEY`. |
| `.oc` | Ignored after this review | Contains one token-like local value. |
| `.claude/settings.local.json` | Ignored after this review | Contains local tool permission settings. |
| `.codex` | Ignored after this review | Empty local agent file. |
| `target/` | Ignored | Rust build output. |

## Git history

Checked history for the env and local-agent files:

- `.env`
- `agentless_demo/.env`
- `.oc`
- `.codex`
- `.claude/settings.local.json`

No commits were found for those paths.

Checked all reachable Git history for common secret formats:

- OpenAI-style keys
- Anthropic-style keys
- GitHub personal access tokens
- AWS access key IDs
- Slack tokens
- Google API keys
- PEM private key headers

No committed matches were found.

## Local Git remote

The local `origin` URL contains token-like GitHub credentials in `.git/config`.

This does not get published with the repository, but it is still a local secret exposure risk through shell history, logs, screenshots, and copied command output.

Recommended fix:

```sh
git remote set-url origin https://github.com/pointerliu/claumini.git
```

Rotate the GitHub token if it was used outside this machine or appeared in logs.

## Changes made

`.gitignore` was hardened so local private files are not staged accidentally:

```gitignore
.oc
.claude/
.codex/
.codex
.env.*
!.env.example
```

Verification confirmed these paths are ignored:

- `.env`
- `agentless_demo/.env`
- `.oc`
- `.codex`
- `.claude/settings.local.json`
- `target/`

## Untracked publish candidates

These files are still untracked and would need an explicit `git add` before publishing:

- `SPEC.md`
- `agentless_demo/Cargo.toml`
- `agentless_demo/src/lib.rs`
- `agentless_demo/src/main.rs`
- `agentless_demo/tests/fanout_flow.rs`
- `claumini-runtime/tests/live_anthropic.rs`
- `docs/plans/2026-04-15-live-llm-provider-smoke-tests.md`

They were scanned for common secret patterns during this review. No matches were found.

## Before making the repo public

1. Rotate the API keys currently present in `.env` and `agentless_demo/.env`.
2. Remove the token-like credential from the local Git remote URL.
3. Review untracked publish candidates before staging them.
4. Push only after `git status --short --ignored` shows private files under ignored entries, not untracked entries.
