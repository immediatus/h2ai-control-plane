//! Real LLM integration tests — all `#[ignore]` by default.
//!
//! Run all with devcontainer env (NATS + local llama.server):
//! ```bash
//! cargo nextest run --workspace --run-ignored all --nocapture
//! ```
//!
//! Run a specific provider:
//! ```bash
//! ANTHROPIC_API_KEY=sk-ant-... cargo nextest run -p h2ai-adapters --test integration_test \
//!     --run-ignored all --nocapture
//! ```
//!
//! Tests that cannot reach their endpoint skip gracefully with an eprintln.

use h2ai_adapters::anthropic::AnthropicAdapter;
use h2ai_adapters::openai::OpenAIAdapter;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::sizing::TauValue;

fn task_request() -> ComputeRequest {
    ComputeRequest {
        system_context: "You are a concise technical assistant. Reply in at most 2 sentences."
            .into(),
        task: "What is a stateless authentication token? Answer in one sentence.".into(),
        tau: TauValue::new(0.3).unwrap(),
        max_tokens: 80,
    }
}

fn skip_if_no_key(env_name: &str) -> bool {
    if std::env::var(env_name).is_err() {
        eprintln!("SKIP: {env_name} not set");
        true
    } else {
        false
    }
}

#[tokio::test]
#[ignore]
async fn anthropic_real_call_returns_non_empty_output() {
    if skip_if_no_key("ANTHROPIC_API_KEY") {
        return;
    }

    let adapter = AnthropicAdapter::new(
        "https://api.anthropic.com".into(),
        "ANTHROPIC_API_KEY".into(),
        "claude-3-5-haiku-20241022".into(),
    );

    let resp = adapter
        .execute(task_request())
        .await
        .expect("Anthropic API call failed");

    eprintln!("output:      {}", resp.output);
    eprintln!("token_cost:  {}", resp.token_cost);

    assert!(!resp.output.is_empty(), "output must not be empty");
    assert!(resp.token_cost > 0, "token_cost must be > 0");
    assert!(
        resp.output.len() > 10,
        "output suspiciously short: {}",
        resp.output
    );
}

#[tokio::test]
#[ignore]
async fn openai_real_call_returns_non_empty_output() {
    if skip_if_no_key("OPENAI_API_KEY") {
        return;
    }

    let adapter = OpenAIAdapter::new(
        "https://api.openai.com/v1".into(),
        "OPENAI_API_KEY".into(),
        "gpt-4o-mini".into(),
    );

    let resp = adapter
        .execute(task_request())
        .await
        .expect("OpenAI API call failed");

    eprintln!("output:      {}", resp.output);
    eprintln!("token_cost:  {}", resp.token_cost);

    assert!(!resp.output.is_empty(), "output must not be empty");
    assert!(resp.token_cost > 0, "token_cost must be > 0");
}

#[tokio::test]
#[ignore]
async fn llamacpp_real_call_returns_non_empty_output() {
    let endpoint = std::env::var("LLAMACPP_BASE_URL")
        .unwrap_or_else(|_| "http://host.docker.internal:8080/v1".into());
    let model = std::env::var("LLAMACPP_MODEL").unwrap_or_else(|_| "local".into());

    // llama.server accepts any bearer token — LLAMACPP_API_KEY defaults to "local"
    if std::env::var("LLAMACPP_API_KEY").is_err() {
        std::env::set_var("LLAMACPP_API_KEY", "local");
    }

    let adapter = OpenAIAdapter::new(endpoint.clone(), "LLAMACPP_API_KEY".into(), model.clone());

    match adapter.execute(task_request()).await {
        Ok(resp) => {
            eprintln!("endpoint:    {endpoint}");
            eprintln!("model:       {model}");
            eprintln!("output:      {}", resp.output);
            eprintln!("token_cost:  {}", resp.token_cost);
            assert!(!resp.output.is_empty(), "output must not be empty");
            assert!(resp.token_cost > 0, "token_cost must be > 0");
        }
        Err(e) => {
            eprintln!("SKIP: llama.server at {endpoint} not reachable: {e}");
        }
    }
}
