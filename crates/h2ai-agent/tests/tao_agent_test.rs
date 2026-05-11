use async_trait::async_trait;
use h2ai_adapters::mock::{MockAdapter, SequencedMockAdapter};
use h2ai_agent::tao_agent::{TaoAgent, TaoAgentInput};
use h2ai_config::H2AIConfig;
use h2ai_tools::registry::ToolRegistry;
use h2ai_tools::shell::ShellExecutor;
use h2ai_types::adapter::IComputeAdapter;
use h2ai_types::sizing::TauValue;
use std::sync::{Arc, Mutex};

fn cfg() -> H2AIConfig {
    H2AIConfig::default()
}

fn agent_input(instructions: &str, context: &str) -> TaoAgentInput {
    TaoAgentInput {
        instructions: instructions.into(),
        system_context: context.into(),
        tau: TauValue::new(0.5).unwrap(),
        max_tokens: 128,
    }
}

// ── Test 1: direct output when no tool call ───────────────────────────────────

#[tokio::test]
async fn tao_agent_returns_direct_output_when_no_tool_call() {
    let adapter = MockAdapter::new("final answer".into());
    let registry = ToolRegistry::new();
    let result = TaoAgent::new(&adapter as &dyn IComputeAdapter, registry, &cfg())
        .run(agent_input("do something", "context"))
        .await;
    assert_eq!(result.output, "final answer");
    assert!(result.tool_calls.is_empty());
    assert_eq!(result.total_token_cost, 0); // MockAdapter returns token_cost: 0
}

// ── Test 2: single tool call then final answer ────────────────────────────────

#[tokio::test]
async fn tao_agent_executes_tool_and_feeds_observation() {
    let adapter = SequencedMockAdapter::new(vec![
        r#"{"tool":"shell","input":{"command":"echo","args":["hello"]}}"#.into(),
        "final answer after observation".into(),
    ]);
    let mut registry = ToolRegistry::new();
    registry.register_shell(ShellExecutor::new(vec!["echo".into()], 5));

    let result = TaoAgent::new(&adapter as &dyn IComputeAdapter, registry, &cfg())
        .run(agent_input("run echo", ""))
        .await;

    assert_eq!(result.output, "final answer after observation");
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(result.tool_calls[0].iteration, 1);
    assert!(
        result.tool_calls[0].output.contains("hello"),
        "observation must contain echo output; got: {:?}",
        result.tool_calls[0].output
    );
}

// ── Test 3: iteration cap ─────────────────────────────────────────────────────

#[tokio::test]
async fn tao_agent_stops_at_max_iterations() {
    let responses: Vec<String> = (0..20)
        .map(|_| r#"{"tool":"shell","input":{"command":"echo","args":["loop"]}}"#.into())
        .collect();
    let adapter = SequencedMockAdapter::new(responses);
    let mut registry = ToolRegistry::new();
    registry.register_shell(ShellExecutor::new(vec!["echo".into()], 5));

    let mut cfg = cfg();
    cfg.agent_max_tool_iterations = 3;

    let result = TaoAgent::new(&adapter as &dyn IComputeAdapter, registry, &cfg)
        .run(agent_input("loop forever", ""))
        .await;

    assert_eq!(
        result.tool_calls.len(),
        3,
        "must stop at agent_max_tool_iterations=3; got {} calls",
        result.tool_calls.len()
    );
    assert!(
        result.truncated,
        "truncated must be true when cap is reached while still calling tools"
    );
}

// ── Test 4: zero max_iterations clamped to 1 ─────────────────────────────────

#[tokio::test]
async fn tao_agent_zero_max_iterations_treated_as_one() {
    let adapter = MockAdapter::new("direct answer".into());
    let registry = ToolRegistry::new();

    let mut cfg = cfg();
    cfg.agent_max_tool_iterations = 0;

    let result = TaoAgent::new(&adapter as &dyn IComputeAdapter, registry, &cfg)
        .run(agent_input("anything", ""))
        .await;

    // Should still produce output (one iteration ran).
    assert_eq!(result.output, "direct answer");
}

// ── Test 5: failed tool call recorded, loop continues ────────────────────────

#[tokio::test]
async fn tao_agent_records_tool_error_and_continues() {
    let adapter = SequencedMockAdapter::new(vec![
        r#"{"tool":"shell","input":{"command":"rm","args":["-rf","/"]}}"#.into(),
        "final answer".into(),
    ]);
    let mut registry = ToolRegistry::new();
    // allowlist only has echo — rm is blocked
    registry.register_shell(ShellExecutor::new(vec!["echo".into()], 5));

    let result = TaoAgent::new(&adapter as &dyn IComputeAdapter, registry, &cfg())
        .run(agent_input("delete everything", ""))
        .await;

    assert_eq!(result.output, "final answer");
    assert_eq!(result.tool_calls.len(), 1);
    assert!(
        result.tool_calls[0].output.contains("not permitted")
            || result.tool_calls[0].output.contains("NotPermitted")
            || result.tool_calls[0].output.contains("tool error"),
        "error must be in observation; got: {:?}",
        result.tool_calls[0].output
    );
}

// ── Test 6: unknown tool name treated as final answer ─────────────────────────

#[tokio::test]
async fn tao_agent_unknown_tool_name_treated_as_final_answer() {
    // LLM outputs JSON that looks like a tool call but uses an unknown tool name.
    let adapter = MockAdapter::new(r#"{"tool":"nonexistent_tool","input":{}}"#.into());
    let registry = ToolRegistry::new();

    let result = TaoAgent::new(&adapter as &dyn IComputeAdapter, registry, &cfg())
        .run(agent_input("do something", ""))
        .await;

    // Should be treated as final answer, not a tool call.
    assert!(result.tool_calls.is_empty());
    assert_eq!(result.output, r#"{"tool":"nonexistent_tool","input":{}}"#);
}

// ── Test 7: partial tool JSON (no input field) treated as final answer ─────────

#[tokio::test]
async fn tao_agent_partial_tool_json_without_input_treated_as_final_answer() {
    // JSON has a valid tool name but no `input` field — must NOT dispatch as a tool call.
    let payload = r#"{"tool":"shell","reasoning":"I should run echo"}"#;
    let adapter = MockAdapter::new(payload.into());
    let mut registry = ToolRegistry::new();
    registry.register_shell(ShellExecutor::new(vec!["echo".into()], 5));

    let result = TaoAgent::new(&adapter as &dyn IComputeAdapter, registry, &cfg())
        .run(agent_input("do something", ""))
        .await;

    assert!(
        result.tool_calls.is_empty(),
        "partial JSON must not dispatch a tool call"
    );
    assert_eq!(result.output, payload);
}

// ── Test 8: truncated flag set when iteration cap hit while in tool-call mode ──

#[tokio::test]
async fn tao_agent_truncated_flag_set_when_cap_reached() {
    let responses: Vec<String> = (0..5)
        .map(|_| r#"{"tool":"shell","input":{"command":"echo","args":["x"]}}"#.into())
        .collect();
    let adapter = SequencedMockAdapter::new(responses);
    let mut registry = ToolRegistry::new();
    registry.register_shell(ShellExecutor::new(vec!["echo".into()], 5));

    let mut cfg = cfg();
    cfg.agent_max_tool_iterations = 3;

    let result = TaoAgent::new(&adapter as &dyn IComputeAdapter, registry, &cfg)
        .run(agent_input("loop", ""))
        .await;

    assert!(
        result.truncated,
        "truncated must be true when cap reached mid-loop"
    );
    assert!(!result.adapter_failed);
    assert_eq!(result.tool_calls.len(), 3);
}

// ── Test 9: tool block injected into system context ───────────────────────────

/// A test adapter that records the last ComputeRequest it received.
#[derive(Debug, Clone)]
struct RecordingAdapter {
    response: String,
    last_context: Arc<Mutex<Option<String>>>,
}

impl RecordingAdapter {
    fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
            last_context: Arc::new(Mutex::new(None)),
        }
    }
    fn captured_context(&self) -> Option<String> {
        self.last_context.lock().unwrap().clone()
    }
}

#[async_trait]
impl IComputeAdapter for RecordingAdapter {
    async fn execute(
        &self,
        request: h2ai_types::adapter::ComputeRequest,
    ) -> Result<h2ai_types::adapter::ComputeResponse, h2ai_types::adapter::AdapterError> {
        *self.last_context.lock().unwrap() = Some(request.system_context);
        Ok(h2ai_types::adapter::ComputeResponse {
            output: self.response.clone(),
            token_cost: 0,
            adapter_kind: h2ai_types::config::AdapterKind::CloudGeneric {
                endpoint: "mock://recording".into(),
                api_key_env: "NONE".into(),
                model: None,
            },
            tokens_used: None,
        })
    }
    fn kind(&self) -> &h2ai_types::config::AdapterKind {
        unreachable!()
    }
}

#[tokio::test]
async fn tao_agent_injects_tool_block_into_system_context() {
    let adapter = RecordingAdapter::new("final answer");
    let mut registry = ToolRegistry::new();
    registry.register_shell(ShellExecutor::new(vec![], 5));

    TaoAgent::new(&adapter as &dyn IComputeAdapter, registry, &cfg())
        .run(agent_input("do something", "base context"))
        .await;

    let ctx = adapter
        .captured_context()
        .expect("adapter was never called");
    assert!(
        ctx.contains("[TOOLS]"),
        "system context must contain [TOOLS] block"
    );
    assert!(
        ctx.contains("shell"),
        "system context must advertise the shell tool"
    );
    assert!(
        ctx.contains("base context"),
        "original system context must be preserved"
    );
}

// ── Test 10: adapter error path produces non-empty output ─────────────────────

/// Adapter that always returns an error.
#[derive(Debug)]
struct ErrorAdapter;

#[async_trait]
impl IComputeAdapter for ErrorAdapter {
    async fn execute(
        &self,
        _request: h2ai_types::adapter::ComputeRequest,
    ) -> Result<h2ai_types::adapter::ComputeResponse, h2ai_types::adapter::AdapterError> {
        Err(h2ai_types::adapter::AdapterError::NetworkError(
            "simulated failure".into(),
        ))
    }
    fn kind(&self) -> &h2ai_types::config::AdapterKind {
        unreachable!()
    }
}

#[tokio::test]
async fn tao_agent_adapter_error_produces_error_output() {
    let adapter = ErrorAdapter;
    let registry = ToolRegistry::new();

    let result = TaoAgent::new(&adapter as &dyn IComputeAdapter, registry, &cfg())
        .run(agent_input("do something", ""))
        .await;

    assert!(
        !result.output.is_empty(),
        "output must not be empty on adapter error"
    );
    assert!(
        result.output.contains("adapter error") || result.output.contains("simulated failure"),
        "output must describe the error; got: {:?}",
        result.output
    );
    assert!(result.tool_calls.is_empty());
    assert!(
        result.adapter_failed,
        "adapter_failed must be true when adapter errors"
    );
    assert!(
        !result.truncated,
        "truncated must be false on adapter error"
    );
}

// ── Test 11: tool known to name map but not registered in registry ────────────

#[tokio::test]
async fn tao_agent_unregistered_tool_records_error_and_continues() {
    // LLM emits a valid tool name (web_search is in agent_tool_from_name) but
    // the registry has no WebSearch executor — hits ToolError::NotRegistered.
    let adapter = SequencedMockAdapter::new(vec![
        r#"{"tool":"web_search","input":{"query":"test"}}"#.into(),
        "final answer".into(),
    ]);
    // Registry has only shell registered — web_search is absent.
    let mut registry = ToolRegistry::new();
    registry.register_shell(ShellExecutor::new(vec!["echo".into()], 5));

    let result = TaoAgent::new(&adapter as &dyn IComputeAdapter, registry, &cfg())
        .run(agent_input("search something", ""))
        .await;

    assert_eq!(result.output, "final answer");
    assert_eq!(result.tool_calls.len(), 1);
    assert!(
        result.tool_calls[0].output.contains("tool error")
            || result.tool_calls[0].output.contains("NotRegistered")
            || result.tool_calls[0].output.contains("not registered"),
        "NotRegistered error must be in observation; got: {:?}",
        result.tool_calls[0].output
    );
    assert!(!result.adapter_failed);
}

// ── Test 12: total_token_cost accumulates across multiple adapter calls ────────

#[tokio::test]
async fn tao_agent_accumulates_token_cost_across_iterations() {
    // SequencedMockAdapter returns token_cost: 10 per call.
    // Two adapter calls (one tool call + one final answer) → total 20.
    let adapter = SequencedMockAdapter::new(vec![
        r#"{"tool":"shell","input":{"command":"echo","args":["hi"]}}"#.into(),
        "final answer".into(),
    ]);
    let mut registry = ToolRegistry::new();
    registry.register_shell(ShellExecutor::new(vec!["echo".into()], 5));

    let result = TaoAgent::new(&adapter as &dyn IComputeAdapter, registry, &cfg())
        .run(agent_input("run echo", ""))
        .await;

    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(
        result.total_token_cost, 20,
        "two adapter calls at token_cost=10 each must total 20; got {}",
        result.total_token_cost
    );
}
