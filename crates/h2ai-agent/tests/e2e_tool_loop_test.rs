use h2ai_adapters::mock::SequencedMockAdapter;
use h2ai_agent::tao_agent::{TaoAgent, TaoAgentInput};
use h2ai_config::H2AIConfig;
use h2ai_tools::mcp::{McpExecutor, MockMcpBackend};
use h2ai_tools::registry::ToolRegistry;
use h2ai_tools::wasm::{MockWasmBackend, WasmExecutor};
use h2ai_tools::web_search::{MockSearchBackend, WebSearchExecutor};
use h2ai_types::adapter::IComputeAdapter;
use h2ai_types::agent::AgentTool;
use h2ai_types::sizing::TauValue;
use std::collections::HashMap;

fn cfg() -> H2AIConfig {
    H2AIConfig::default()
}

/// Scripted 3-tool traversal:
/// 1. LLM emits web_search call
/// 2. LLM emits file_system read_file call
/// 3. LLM emits code_execution javascript call
/// 4. LLM emits final answer
#[tokio::test]
async fn tao_agent_traverses_three_tools_and_produces_final_answer() {
    let adapter = SequencedMockAdapter::new(vec![
        r#"{"tool":"web_search","input":{"query":"h2ai agent_max_tool_iterations default"}}"#
            .into(),
        r#"{"tool":"file_system","input":{"op":"read_file","path":"reference.toml"}}"#.into(),
        r#"{"tool":"code_execution","input":{"language":"javascript","script":"5*5"}}"#.into(),
        "The default is 5, confirmed in reference.toml. Computed square: 25.".into(),
    ]);

    let search_backend =
        MockSearchBackend::new("agent_max_tool_iterations: default is 5 (source: h2ai docs)");

    let mut mcp_files = HashMap::new();
    mcp_files.insert(
        "reference.toml".to_string(),
        "agent_max_tool_iterations = 5".to_string(),
    );
    let mcp_backend = MockMcpBackend::new(mcp_files);
    let wasm_backend = MockWasmBackend::new("25");

    let mut registry = ToolRegistry::new();
    registry.register_web_search(WebSearchExecutor::new(Box::new(search_backend), 3));
    registry.register_mcp(McpExecutor::new(Box::new(mcp_backend)));
    registry.register_wasm(WasmExecutor::new(Box::new(wasm_backend)));

    let result = TaoAgent::new(&adapter as &dyn IComputeAdapter, registry, &cfg())
        .run(TaoAgentInput {
            instructions: "Find the default value of agent_max_tool_iterations, confirm it in the config file, then compute its square.".into(),
            system_context: "".into(),
            tau: TauValue::new(0.5).unwrap(),
            max_tokens: 256,
        })
        .await;

    assert_eq!(
        result.tool_calls.len(),
        3,
        "expected 3 tool calls, got {}",
        result.tool_calls.len()
    );
    assert_eq!(
        result.tool_calls[0].tool,
        AgentTool::WebSearch,
        "first call must be web_search"
    );
    assert_eq!(
        result.tool_calls[1].tool,
        AgentTool::FileSystem,
        "second call must be file_system"
    );
    assert_eq!(
        result.tool_calls[2].tool,
        AgentTool::CodeExecution,
        "third call must be code_execution"
    );

    assert!(
        result.tool_calls[0].output.contains("5"),
        "web_search observation must mention the default value; got: {:?}",
        result.tool_calls[0].output
    );
    assert!(
        result.tool_calls[1]
            .output
            .contains("agent_max_tool_iterations"),
        "file_system observation must contain the config key; got: {:?}",
        result.tool_calls[1].output
    );
    assert_eq!(
        result.tool_calls[2].output, "25",
        "code_execution observation must be '25'"
    );

    assert!(
        result.output.contains("25"),
        "final answer must mention the computed square; got: {:?}",
        result.output
    );
    assert_eq!(
        result.total_token_cost, 40,
        "4 adapter calls × 10 each must total 40; got {}",
        result.total_token_cost
    );
    assert!(!result.truncated, "must not be truncated");
    assert!(!result.adapter_failed, "adapter must not have failed");
}

/// Verify the [TOOLS] block in system context advertises all three executors.
#[tokio::test]
async fn tao_agent_three_tool_registry_injects_all_schemas_into_system_context() {
    use async_trait::async_trait;
    use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse};
    use h2ai_types::config::AdapterKind;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Debug)]
    struct ContextCapture(Arc<Mutex<Option<String>>>);

    #[async_trait]
    impl IComputeAdapter for ContextCapture {
        async fn execute(&self, req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
            *self.0.lock().unwrap() = Some(req.system_context);
            Ok(ComputeResponse {
                output: "done".into(),
                token_cost: 0,
                adapter_kind: AdapterKind::CloudGeneric {
                    endpoint: "mock://capture".into(),
                    api_key_env: "NONE".into(),
                    model: None,
                },
                tokens_used: None,
            })
        }
        fn kind(&self) -> &AdapterKind {
            unreachable!()
        }
    }

    let capture = ContextCapture(Arc::new(Mutex::new(None)));

    let mut registry = ToolRegistry::new();
    registry.register_web_search(WebSearchExecutor::new(
        Box::new(MockSearchBackend::new("x")),
        3,
    ));
    registry.register_mcp(McpExecutor::new(Box::new(MockMcpBackend::new(
        HashMap::new(),
    ))));
    registry.register_wasm(WasmExecutor::new(Box::new(MockWasmBackend::new("x"))));

    TaoAgent::new(&capture as &dyn IComputeAdapter, registry, &cfg())
        .run(TaoAgentInput {
            instructions: "anything".into(),
            system_context: "base".into(),
            tau: TauValue::new(0.5).unwrap(),
            max_tokens: 64,
        })
        .await;

    let ctx = capture
        .0
        .lock()
        .unwrap()
        .clone()
        .expect("adapter never called");
    assert!(ctx.contains("[TOOLS]"), "context must have [TOOLS] block");
    assert!(
        ctx.contains("web_search"),
        "context must advertise web_search"
    );
    assert!(
        ctx.contains("file_system"),
        "context must advertise file_system"
    );
    assert!(
        ctx.contains("code_execution"),
        "context must advertise code_execution"
    );
    assert!(
        ctx.contains("base"),
        "original system context must be preserved"
    );
}
