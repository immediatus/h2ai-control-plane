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
        max_tokens: 2048,
    }
}

#[tokio::test]
async fn anthropic_real_call_returns_non_empty_output() {
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        eprintln!("SKIP: ANTHROPIC_API_KEY not set");
        return;
    }

    let adapter = AnthropicAdapter::new(
        "https://api.anthropic.com".into(),
        "ANTHROPIC_API_KEY".into(),
        "claude-3-5-haiku-20241022".into(),
    );

    match adapter.execute(task_request()).await {
        Ok(resp) => {
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
        Err(e) => {
            eprintln!("SKIP: Anthropic API unavailable or auth failed: {e}");
        }
    }
}

#[tokio::test]
async fn openai_real_call_returns_non_empty_output() {
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("SKIP: OPENAI_API_KEY not set");
        return;
    }

    let adapter = OpenAIAdapter::new(
        "https://api.openai.com/v1".into(),
        "OPENAI_API_KEY".into(),
        "gpt-4o-mini".into(),
    );

    match adapter.execute(task_request()).await {
        Ok(resp) => {
            eprintln!("output:      {}", resp.output);
            eprintln!("token_cost:  {}", resp.token_cost);
            assert!(!resp.output.is_empty(), "output must not be empty");
            assert!(resp.token_cost > 0, "token_cost must be > 0");
        }
        Err(e) => {
            eprintln!("SKIP: OpenAI API unavailable or auth failed: {e}");
        }
    }
}

#[tokio::test]
async fn llamacpp_real_call_returns_non_empty_output() {
    let endpoint = match std::env::var("LLAMACPP_BASE_URL") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("SKIP: LLAMACPP_BASE_URL not set");
            return;
        }
    };
    let model = std::env::var("LLAMACPP_MODEL").unwrap_or_else(|_| "local".into());

    if std::env::var("LLAMACPP_API_KEY").is_err() {
        std::env::set_var("LLAMACPP_API_KEY", "local");
    }

    let adapter = OpenAIAdapter::new(endpoint.clone(), "LLAMACPP_API_KEY".into(), model.clone());

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        adapter.execute(task_request()),
    )
    .await;

    match result {
        Err(_elapsed) => {
            eprintln!("SKIP: llama.cpp server at {endpoint} timed out after 30s — model too slow for unit test");
        }
        Ok(Ok(resp)) => {
            eprintln!("endpoint:    {endpoint}");
            eprintln!("model:       {model}");
            eprintln!("output:      {}", resp.output);
            eprintln!("token_cost:  {}", resp.token_cost);
            assert!(!resp.output.is_empty(), "output must not be empty");
            assert!(resp.token_cost > 0, "token_cost must be > 0");
        }
        Ok(Err(e)) => {
            eprintln!("SKIP: llama.cpp server at {endpoint} not reachable: {e}");
        }
    }
}
