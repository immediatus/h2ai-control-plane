//! Real LLM integration tests.
//!
//! These are `#[ignore]` by default. To run:
//!
//! ```bash
//! H2AI_INTEGRATION_TEST=true \
//! ANTHROPIC_API_KEY=sk-ant-... \
//! cargo test -p h2ai-adapters --test integration_test -- --ignored --nocapture
//! ```
//!
//! Each test skips gracefully if the required API key env var is not set.

use h2ai_adapters::anthropic::AnthropicAdapter;
use h2ai_adapters::ollama::OllamaAdapter;
use h2ai_adapters::openai::OpenAIAdapter;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::physics::TauValue;

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
async fn ollama_real_call_returns_non_empty_output() {
    let endpoint =
        std::env::var("H2AI_EXPLORER_ENDPOINT").unwrap_or_else(|_| "http://localhost:11434".into());
    let model = std::env::var("H2AI_EXPLORER_MODEL").unwrap_or_else(|_| "llama3.2".into());

    let adapter = OllamaAdapter::new(endpoint.clone(), model.clone());

    match adapter.execute(task_request()).await {
        Ok(resp) => {
            eprintln!("output:      {}", resp.output);
            eprintln!("token_cost:  {}", resp.token_cost);
            assert!(!resp.output.is_empty(), "output must not be empty");
        }
        Err(e) => {
            eprintln!("SKIP: Ollama at {endpoint} with model {model} not reachable: {e}");
        }
    }
}
