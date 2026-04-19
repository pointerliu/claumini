use agentless_demo::run_mock_demo;
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), agentless_demo::DemoError> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive("agentless_demo=info".parse().unwrap()))
        .init();

    let result = run_mock_demo().await?;

    println!("Suspicious files:");
    for file in &result.suspicious_files {
        println!("- {}: {}", file.path, file.reason);
    }

    println!("\nRanked root-cause candidates:");
    for candidate in &result.candidates {
        println!(
            "- {} (score={}): {}",
            candidate.path, candidate.score, candidate.explanation
        );
    }

    Ok(())
}
