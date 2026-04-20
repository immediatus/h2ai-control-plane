use chrono::Utc;
use h2ai_telemetry::redaction::redact_event;
use h2ai_types::agent::AgentTelemetryEvent;
use h2ai_types::identity::{AgentId, TaskId};

fn task_id() -> TaskId {
    TaskId::new()
}
fn agent_id() -> AgentId {
    AgentId::from("agent-1")
}

#[test]
fn redact_removes_bearer_token_from_prompt() {
    let event = AgentTelemetryEvent::LlmPromptSent {
        task_id: task_id(),
        agent_id: agent_id(),
        prompt: "Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig".into(),
        timestamp: Utc::now(),
    };
    let redacted = redact_event(event);
    if let AgentTelemetryEvent::LlmPromptSent { prompt, .. } = redacted {
        assert!(prompt.contains("[REDACTED]"));
        assert!(!prompt.contains("eyJ"));
    }
}

#[test]
fn redact_removes_api_key_from_shell_command() {
    let event = AgentTelemetryEvent::ShellCommandExecuted {
        task_id: task_id(),
        agent_id: agent_id(),
        command:
            "curl -H 'Authorization: sk-abcdefghijklmnopqrstuvwxyz1234567' https://api.example.com"
                .into(),
        stdout: String::new(),
        stderr: String::new(),
        exit_code: 0,
        timestamp: Utc::now(),
    };
    let redacted = redact_event(event);
    if let AgentTelemetryEvent::ShellCommandExecuted { command, .. } = redacted {
        assert!(command.contains("[REDACTED]"));
        assert!(!command.contains("sk-"));
    }
}

#[test]
fn redact_leaves_clean_text_unchanged() {
    let event = AgentTelemetryEvent::SystemError {
        task_id: task_id(),
        agent_id: agent_id(),
        error: "connection refused to localhost:4222".into(),
        timestamp: Utc::now(),
    };
    let redacted = redact_event(event);
    if let AgentTelemetryEvent::SystemError { error, .. } = redacted {
        assert_eq!(error, "connection refused to localhost:4222");
    }
}
