use h2ai_types::agent::AgentTool;

/// Returns the list of tools supported by this agent.
#[must_use]
pub fn agent_tools() -> Vec<AgentTool> {
    vec![AgentTool::Shell, AgentTool::FileSystem]
}
