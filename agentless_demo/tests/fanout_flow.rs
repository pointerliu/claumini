use std::collections::VecDeque;
use std::fs;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use claumini_core::{
    FinishReason, ModelProvider, ModelRequest, ModelResponse, ProviderCapabilities, ProviderError,
    ToolCall,
};
use claumini_runtime::FINISH_TOOL_NAME;
use serde_json::json;
use tempfile::tempdir;

#[derive(Debug, Clone)]
struct ScriptedProvider {
    responses: Arc<Mutex<VecDeque<ModelResponse>>>,
}

impl ScriptedProvider {
    fn new(responses: Vec<ModelResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses.into())),
        }
    }
}

#[async_trait]
impl ModelProvider for ScriptedProvider {
    async fn complete(&self, _request: ModelRequest) -> Result<ModelResponse, ProviderError> {
        self.responses
            .lock()
            .expect("responses lock")
            .pop_front()
            .ok_or_else(|| ProviderError::Message("no scripted response remaining".into()))
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            native_tool_calling: true,
            ..ProviderCapabilities::default()
        }
    }
}

fn tool_response(id: &str, name: &str, arguments: serde_json::Value) -> ModelResponse {
    ModelResponse {
        message: None,
        tool_calls: vec![ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments,
        }],
        finish_reason: FinishReason::ToolCalls,
    }
}

fn final_response(json_payload: serde_json::Value) -> ModelResponse {
    tool_response("finish", FINISH_TOOL_NAME, json!({ "json": json_payload }))
}

fn tool_names(session: &claumini_runtime::SessionRecord) -> Vec<&str> {
    session
        .tool_calls
        .iter()
        .filter(|record| record.call.name != claumini_runtime::FINISH_TOOL_NAME)
        .map(|record| record.call.name.as_str())
        .collect()
}

#[tokio::test]
async fn run_demo_fanouts_suspicious_file_analysis_and_ranks_root_cause() {
    let temp = tempdir().expect("temp dir should exist");
    let repo_root = temp.path().join("sample-repo");
    fs::create_dir_all(repo_root.join("src")).expect("src dir should exist");
    fs::write(
        repo_root.join("src/cache.rs"),
        "pub fn cache_value(input: &str) -> String {\n    input.to_owned()\n}\n",
    )
    .expect("cache file should exist");
    fs::write(
        repo_root.join("src/config.rs"),
        "pub fn load_env() -> bool {\n    true\n}\n",
    )
    .expect("config file should exist");
    fs::write(
        repo_root.join("src/retry.rs"),
        "pub fn should_retry(status: u16) -> bool {\n    status >= 500\n}\n",
    )
    .expect("retry file should exist");
    fs::write(
        repo_root.join("BUG_REPORT.md"),
        "Cache entries stay stale after a failed retry sequence. Inspect cache invalidation and retry flow.\n",
    )
    .expect("bug report should exist");

    let triage_provider = Arc::new(ScriptedProvider::new(vec![
        tool_response("list", "list_files", json!({ "root": repo_root.clone() })),
        tool_response(
            "search",
            "search_text",
            json!({ "root": repo_root.clone(), "pattern": "cache|retry" }),
        ),
        final_response(json!({
            "suspicious_files": [
                {
                    "path": "src/cache.rs",
                    "reason": "Cache code owns the stale entry behavior."
                },
                {
                    "path": "src/retry.rs",
                    "reason": "Retry code controls failure sequencing."
                }
            ]
        })),
    ]));
    let worker_file = repo_root.join("src/cache.rs");
    let worker_provider_factory: Arc<dyn Fn() -> Arc<dyn ModelProvider> + Send + Sync> =
        Arc::new(move || {
            Arc::new(ScriptedProvider::new(vec![
                tool_response("read", "read_file", json!({ "path": worker_file.clone() })),
                final_response(json!({
                    "path": "src/cache.rs",
                    "score": 9,
                    "explanation": "Cache implementation has no refresh or invalidation path."
                })),
            ])) as Arc<dyn ModelProvider>
        });

    let result = agentless_demo::run_demo_with_provider_factory(
        repo_root.join("BUG_REPORT.md"),
        triage_provider,
        worker_provider_factory,
    )
    .await
    .expect("demo should succeed");

    let suspicious_paths = result
        .suspicious_files
        .iter()
        .map(|file| file.path.as_str())
        .collect::<Vec<_>>();
    assert!(!suspicious_paths.is_empty());
    assert!(tool_names(&result.triage_session).contains(&"list_files"));
    assert!(tool_names(&result.triage_session).contains(&"search_text"));
    assert_eq!(result.worker_runs.len(), result.suspicious_files.len());
    assert!(
        result
            .worker_runs
            .iter()
            .all(|run| tool_names(&run.session).contains(&"read_file"))
    );
    assert_eq!(result.candidates.len(), result.suspicious_files.len());
    assert!(
        result
            .candidates
            .iter()
            .all(|candidate| suspicious_paths.contains(&candidate.path.as_str()))
    );
    assert!(
        result
            .candidates
            .windows(2)
            .all(|pair| pair[0].score >= pair[1].score)
    );
}
