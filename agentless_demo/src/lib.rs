#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use claumini_core::{AgentError, ModelProvider};
use claumini_models::{OpenAiCompatibleConfig, OpenAiCompatibleProvider};
use claumini_runtime::{PromptAgentBuilder, PromptSession, SessionRecord};
use claumini_tools::{ListFilesTool, ReadFileTool, SearchTextTool};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::env;
use tempfile::tempdir;
use thiserror::Error;
use tokio::task::JoinSet;
use tracing::info;

const TRIAGE_SYSTEM_PROMPT: &str = concat!(
    "You are triaging a small Rust repository against a bug report. ",
    "Call `list_files` before deciding, then use `search_text` to inspect the repo, and return exactly the two most suspicious implementation files. ",
    "Prefer files that directly implement cache behavior, retry behavior, failure handling, or invalidation. ",
    "Do not include configuration or environment stubs unless the bug report explicitly points to configuration. ",
    "Each reason must be grounded in the observed file names or code signals and stay to one sentence."
);

const WORKER_SYSTEM_PROMPT: &str = concat!(
    "You are scoring one suspicious file for a cache-staleness-after-retry bug. ",
    "You must call `read_file` exactly once on the provided path before answering. ",
    "Read only that file, then return the same path, an integer score from 0 to 10, and a concise explanation. ",
    "Give high scores to files that directly control cache state, retry termination, or invalidation after failures. ",
    "When choosing between a cache implementation and a retry predicate, prefer the cache implementation if it lacks invalidation or refresh logic because that is the most direct way stale entries persist. ",
    "Give low scores to config or unrelated support code."
);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SuspiciousFile {
    pub path: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RootCauseCandidate {
    pub path: String,
    pub score: u32,
    pub explanation: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerRun {
    pub file: SuspiciousFile,
    pub candidate: RootCauseCandidate,
    pub session: SessionRecord,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DemoResult {
    pub suspicious_files: Vec<SuspiciousFile>,
    pub triage_session: SessionRecord,
    pub worker_runs: Vec<WorkerRun>,
    pub candidates: Vec<RootCauseCandidate>,
}

#[derive(Debug, Error)]
pub enum DemoError {
    #[error("bug report path '{0}' has no parent directory")]
    MissingRepoRoot(PathBuf),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Agent(#[from] AgentError),
    #[error("worker task failed to join: {0}")]
    Join(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TriageInput {
    repo_root: PathBuf,
    bug_report: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
struct TriageOutput {
    suspicious_files: Vec<SuspiciousFile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct WorkerInput {
    repo_root: PathBuf,
    bug_report: String,
    path: String,
}

fn load_env() {
    dotenvy::from_filename(".env").ok();
    dotenvy::from_filename("../.env").ok();
}

fn live_provider_config_from_lookup(
    mut lookup: impl FnMut(&str) -> Result<String, env::VarError>,
) -> OpenAiCompatibleConfig {
    let base_url = lookup("CLAUMINI_LIVE_API_BASE")
        .unwrap_or_else(|_| panic!("CLAUMINI_LIVE_API_BASE not set"));
    let api_key =
        lookup("CLAUMINI_LIVE_API_KEY").unwrap_or_else(|_| panic!("CLAUMINI_LIVE_API_KEY not set"));
    let model = lookup("CLAUMINI_LIVE_MODEL").unwrap_or_else(|_| "gpt-5-mini".to_string());

    OpenAiCompatibleConfig {
        base_url,
        api_key,
        model,
        max_tokens: None,
    }
}

fn live_provider_config_from_env() -> OpenAiCompatibleConfig {
    load_env();
    live_provider_config_from_lookup(|name| env::var(name))
}

fn triage_provider_from_env() -> OpenAiCompatibleProvider {
    OpenAiCompatibleProvider::new(live_provider_config_from_env())
        .expect("triage provider should be valid")
}

fn worker_provider_from_env() -> OpenAiCompatibleProvider {
    OpenAiCompatibleProvider::new(live_provider_config_from_env())
        .expect("worker provider should be valid")
}

pub async fn run_demo(bug_report_path: impl AsRef<Path>) -> Result<DemoResult, DemoError> {
    let worker_provider_factory: Arc<dyn Fn() -> Arc<dyn ModelProvider> + Send + Sync> =
        Arc::new(|| Arc::new(worker_provider_from_env()) as Arc<dyn ModelProvider>);
    run_demo_with_provider_factory(
        bug_report_path,
        Arc::new(triage_provider_from_env()),
        worker_provider_factory,
    )
    .await
}

pub async fn run_demo_with_provider_factory(
    bug_report_path: impl AsRef<Path>,
    triage_provider: Arc<dyn ModelProvider>,
    worker_provider_factory: Arc<dyn Fn() -> Arc<dyn ModelProvider> + Send + Sync>,
) -> Result<DemoResult, DemoError> {
    info!("[STAGE 0] Starting agentless demo");
    let bug_report_path = bug_report_path.as_ref().to_path_buf();
    let repo_root = bug_report_path
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| DemoError::MissingRepoRoot(bug_report_path.clone()))?;
    let bug_report = fs::read_to_string(&bug_report_path)?;
    info!("[STAGE 0] Bug report loaded, repo_root: {:?}", repo_root);

    info!("[STAGE 1] Triage provider ready");

    info!("[STAGE 2] Building triage agent with list_files and search_text tools");
    let triage_agent = PromptAgentBuilder::new(triage_provider)
        .system_prompt(TRIAGE_SYSTEM_PROMPT)
        .tool(ListFilesTool::new())
        .tool(SearchTextTool::new())
        .json_input::<TriageInput>()
        .json_output::<TriageOutput>()
        .build()?;
    info!("[STAGE 2] Triage agent built successfully");

    info!("[STAGE 3] Running triage agent to identify suspicious files");
    let PromptSession {
        output: triage_output,
        session: triage_session,
    } = triage_agent
        .run(
            TriageInput {
                repo_root: repo_root.clone(),
                bug_report: bug_report.clone(),
            },
            "agentless-demo-triage",
        )
        .await?;
    info!(
        "[STAGE 3] Triage completed. Found {} suspicious files",
        triage_output.suspicious_files.len()
    );

    info!(
        "[STAGE 4] Spawning {} worker tasks",
        triage_output.suspicious_files.len()
    );
    let mut workers = JoinSet::new();
    for file in triage_output.suspicious_files.iter().cloned() {
        let repo_root = repo_root.clone();
        let bug_report = bug_report.clone();
        let worker_provider = worker_provider_factory();
        workers
            .spawn(async move { run_worker(file, repo_root, bug_report, worker_provider).await });
    }

    info!("[STAGE 5] Collecting worker results");
    let mut worker_runs = Vec::new();
    while let Some(result) = workers.join_next().await {
        let run = result.map_err(|error| DemoError::Join(error.to_string()))??;
        info!("[STAGE 5] Worker completed for file: {}", run.file.path);
        worker_runs.push(run);
    }

    worker_runs.sort_by(|left, right| {
        right
            .candidate
            .score
            .cmp(&left.candidate.score)
            .then_with(|| left.candidate.path.cmp(&right.candidate.path))
    });
    let candidates: Vec<RootCauseCandidate> = worker_runs
        .iter()
        .map(|run| run.candidate.clone())
        .collect();

    info!(
        "[STAGE 6] Demo completed. Found {} root cause candidates",
        candidates.len()
    );
    Ok(DemoResult {
        suspicious_files: triage_output.suspicious_files,
        triage_session,
        worker_runs,
        candidates,
    })
}

pub async fn run_mock_demo() -> Result<DemoResult, DemoError> {
    let temp = tempdir()?;
    let bug_report_path = write_mock_repo(temp.path())?;
    let result = run_demo(&bug_report_path).await?;
    Ok(result)
}

async fn run_worker(
    file: SuspiciousFile,
    repo_root: PathBuf,
    bug_report: String,
    worker_provider: Arc<dyn ModelProvider>,
) -> Result<WorkerRun, DemoError> {
    info!(
        "[WORKER {}] Starting worker for file: {}",
        sanitize_session_id(&file.path),
        file.path
    );

    info!(
        "[WORKER {}] Building worker agent with read_file tool",
        sanitize_session_id(&file.path)
    );
    let worker_agent = PromptAgentBuilder::new(worker_provider)
        .system_prompt(WORKER_SYSTEM_PROMPT)
        .tool(ReadFileTool::new())
        .json_input::<WorkerInput>()
        .json_output::<RootCauseCandidate>()
        .build()?;
    info!(
        "[WORKER {}] Worker agent built",
        sanitize_session_id(&file.path)
    );

    info!(
        "[WORKER {}] Running worker agent",
        sanitize_session_id(&file.path)
    );
    let PromptSession { output, session } = worker_agent
        .run(
            WorkerInput {
                repo_root,
                bug_report,
                path: file.path.clone(),
            },
            format!("agentless-demo-worker-{}", sanitize_session_id(&file.path)),
        )
        .await?;
    info!(
        "[WORKER {}] Worker completed. Score: {}, path: {}",
        sanitize_session_id(&file.path),
        output.score,
        output.path
    );

    Ok(WorkerRun {
        file,
        candidate: output,
        session,
    })
}

fn sanitize_session_id(path: &str) -> String {
    path.chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => ch,
            _ => '-',
        })
        .collect()
}

fn write_mock_repo(root: &Path) -> Result<PathBuf, DemoError> {
    let repo_root = root.join("mock-repo");
    fs::create_dir_all(repo_root.join("src"))?;
    fs::write(
        repo_root.join("src/cache.rs"),
        "pub fn cache_value(input: &str) -> String {\n    input.to_owned()\n}\n",
    )?;
    fs::write(
        repo_root.join("src/config.rs"),
        "pub fn load_env() -> bool {\n    true\n}\n",
    )?;
    fs::write(
        repo_root.join("src/retry.rs"),
        "pub fn should_retry(status: u16) -> bool {\n    status >= 500\n}\n",
    )?;

    let bug_report_path = repo_root.join("BUG_REPORT.md");
    fs::write(
        &bug_report_path,
        "Cache entries stay stale after a failed retry sequence. Inspect cache invalidation and retry flow.\n",
    )?;
    Ok(bug_report_path)
}

#[cfg(test)]
mod tests {
    use std::env::VarError;

    use super::live_provider_config_from_lookup;

    #[test]
    #[should_panic(expected = "CLAUMINI_LIVE_API_BASE not set")]
    fn live_provider_config_requires_api_base() {
        let _ = live_provider_config_from_lookup(|name| match name {
            "CLAUMINI_LIVE_API_KEY" => Ok("test-key".to_string()),
            "CLAUMINI_LIVE_MODEL" => Ok("test-model".to_string()),
            _ => Err(VarError::NotPresent),
        });
    }
}
