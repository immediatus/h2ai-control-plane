use h2ai_config::H2AIConfig;
use h2ai_tools::registry::ToolRegistry;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::agent::{AgentTool, ToolCallRecord};
use h2ai_types::sizing::TauValue;
use serde::Deserialize;

/// Input bundle for a single `TaoAgent::run` invocation.
pub struct TaoAgentInput {
    pub instructions: String,
    pub system_context: String,
    pub tau: TauValue,
    pub max_tokens: u64,
}

/// Output produced by a completed `TaoAgent` run.
///
/// `output` holds the LLM's final answer when `truncated` is false, or the last
/// tool observation when `truncated` is true (the iteration cap was reached before
/// the LLM produced a plain-text response). Callers must check `truncated` to
/// distinguish the two cases.
///
/// `adapter_failed` is set when the underlying `IComputeAdapter` returned an error.
/// In that case `output` contains the error description and should be treated as a
/// failure signal rather than a valid answer.
///
/// **Context budget note:** each tool observation is appended to the system context
/// on every iteration. `max_tokens` in `TaoAgentInput` governs LLM *output* tokens
/// only; callers are responsible for keeping total observation volume within the
/// LLM's *input* context window.
pub struct TaoAgentResult {
    pub output: String,
    pub total_token_cost: u64,
    pub tool_calls: Vec<ToolCallRecord>,
    /// `true` when the loop exited because `max_iterations` was reached while the
    /// LLM was still emitting tool calls. `false` when the LLM produced a final answer.
    pub truncated: bool,
    /// `true` when the adapter returned an error. `output` contains the error description.
    pub adapter_failed: bool,
}

/// Wire format the LLM must output to invoke a tool:
/// `{"tool": "shell", "input": {…}}`
///
/// `input` must be a JSON object (`{…}`). Absent or null `input` is rejected so
/// that partial JSON (e.g. `{"tool":"shell","reasoning":"…"}`) is not silently
/// dispatched as a tool call.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolCallRequest {
    tool: String,
    input: serde_json::Map<String, serde_json::Value>,
}

/// Maps lowercase wire name (from `ToolSchema::name`) to `AgentTool`.
fn agent_tool_from_name(name: &str) -> Option<AgentTool> {
    match name {
        "shell" => Some(AgentTool::Shell),
        "web_search" => Some(AgentTool::WebSearch),
        "code_execution" => Some(AgentTool::CodeExecution),
        "file_system" => Some(AgentTool::FileSystem),
        _ => None,
    }
}

/// Builds the tool capability block injected into the system prompt.
fn tool_system_block(registry: &ToolRegistry) -> String {
    let schemas = registry.all_schemas();
    if schemas.is_empty() {
        return String::new();
    }
    let mut block = String::from(
        "\n\n[TOOLS]\nTo call a tool output ONLY a JSON object — no prose, no markdown fences:\n\
         {\"tool\": \"<name>\", \"input\": <input_object>}\n\nAvailable tools:\n",
    );
    for s in &schemas {
        block.push_str(&format!(
            "- {}: {}\n  Input schema: {}\n",
            s.name, s.description, s.parameters
        ));
    }
    block
        .push_str("\nWhen you have a final answer (no tool call needed), output it as plain text.");
    block
}

pub struct TaoAgent<'a> {
    adapter: &'a dyn IComputeAdapter,
    registry: ToolRegistry,
    max_iterations: u8,
}

impl<'a> TaoAgent<'a> {
    pub fn new(adapter: &'a dyn IComputeAdapter, registry: ToolRegistry, cfg: &H2AIConfig) -> Self {
        Self {
            adapter,
            registry,
            // Guard: 0 is invalid — treat as 1 so the agent always runs at least once.
            max_iterations: cfg.agent_max_tool_iterations.max(1),
        }
    }

    pub async fn run(self, input: TaoAgentInput) -> TaoAgentResult {
        let tool_block = tool_system_block(&self.registry);
        let base_context = format!("{}{}", input.system_context, tool_block);

        let mut context = base_context;
        let mut total_token_cost: u64 = 0;
        let mut tool_calls: Vec<ToolCallRecord> = Vec::new();
        let mut last_output = String::new();
        let mut truncated = false;
        let mut adapter_failed = false;

        for iteration in 1..=self.max_iterations {
            let req = ComputeRequest {
                system_context: context.clone(),
                task: input.instructions.clone(),
                tau: input.tau,
                max_tokens: input.max_tokens,
            };

            let resp = match self.adapter.execute(req).await {
                Ok(r) => r,
                Err(e) => {
                    let msg = format!("adapter error: {e}");
                    tracing::warn!(iteration, error = %e, "TaoAgent adapter call failed");
                    last_output = msg;
                    adapter_failed = true;
                    break;
                }
            };

            total_token_cost += resp.token_cost;
            let output = resp.output.trim().to_owned();

            // Detect tool call: deserialize as ToolCallRequest (requires object `input`
            // field — null or absent `input` is rejected by the Map deserialiser) and
            // verify the tool name is registered.
            let call: Option<ToolCallRequest> = serde_json::from_str::<ToolCallRequest>(&output)
                .ok()
                .and_then(|r| {
                    if agent_tool_from_name(&r.tool).is_some() {
                        Some(r)
                    } else {
                        None
                    }
                });

            match call {
                None => {
                    // Not a tool call — final answer.
                    last_output = output;
                    break;
                }
                Some(req) => {
                    let tool = agent_tool_from_name(&req.tool).unwrap();
                    let input_json = serde_json::Value::Object(req.input).to_string();

                    tracing::debug!(
                        iteration,
                        tool = ?tool,
                        "TaoAgent dispatching tool call"
                    );

                    let observation = match self.registry.execute(tool.clone(), &input_json).await {
                        Ok(out) => out,
                        Err(e) => format!("tool error: {e}"),
                    };

                    tool_calls.push(ToolCallRecord {
                        tool,
                        input_json,
                        output: observation.clone(),
                        iteration,
                    });

                    context.push_str(&format!(
                        "\n\n[TOOL RESULT — iteration {iteration}]\n{observation}"
                    ));
                    last_output = observation;

                    // Mark truncated if this was the last iteration and still in tool-call mode.
                    if iteration == self.max_iterations {
                        truncated = true;
                        tracing::warn!(
                            max_iterations = self.max_iterations,
                            tool_calls = tool_calls.len(),
                            "TaoAgent reached iteration cap while still in tool-call mode; \
                             result is last tool observation, not a final LLM answer"
                        );
                    }
                }
            }
        }

        TaoAgentResult {
            output: last_output,
            total_token_cost,
            tool_calls,
            truncated,
            adapter_failed,
        }
    }
}
