use chrono::Utc;
use h2ai_telemetry::direct_log::DirectLogProvider;
use h2ai_telemetry::provider::AuditProvider;
use h2ai_types::agent::AgentTelemetryEvent;
use h2ai_types::identity::{AgentId, TaskId};

fn agent_id() -> AgentId {
    AgentId::from("test-agent-direct")
}

fn task_id() -> TaskId {
    TaskId::new()
}

#[tokio::test]
async fn direct_log_record_llm_prompt_sent() {
    let provider = DirectLogProvider;
    let event = AgentTelemetryEvent::LlmPromptSent {
        task_id: task_id(),
        agent_id: agent_id(),
        prompt: "Hello, world!".into(),
        timestamp: Utc::now(),
    };
    let result = provider.record_event(event).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn direct_log_record_llm_response_received() {
    let provider = DirectLogProvider;
    let event = AgentTelemetryEvent::LlmResponseReceived {
        task_id: task_id(),
        agent_id: agent_id(),
        response: "42".into(),
        token_cost: 5,
        timestamp: Utc::now(),
    };
    let result = provider.record_event(event).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn direct_log_record_shell_command_executed() {
    let provider = DirectLogProvider;
    let event = AgentTelemetryEvent::ShellCommandExecuted {
        task_id: task_id(),
        agent_id: agent_id(),
        command: "echo".into(),
        args: vec!["hello".into()],
        stdout: "hello\n".into(),
        stderr: String::new(),
        exit_code: 0,
        timestamp: Utc::now(),
    };
    let result = provider.record_event(event).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn direct_log_record_system_error() {
    let provider = DirectLogProvider;
    let event = AgentTelemetryEvent::SystemError {
        task_id: task_id(),
        agent_id: agent_id(),
        error: "something went wrong".into(),
        timestamp: Utc::now(),
    };
    let result = provider.record_event(event).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn direct_log_flush_succeeds() {
    let provider = DirectLogProvider;
    let result = provider.flush().await;
    assert!(result.is_ok());
}
