use chrono::Utc;
use h2ai_telemetry::redaction::redact_event;
use h2ai_types::agent::AgentTelemetryEvent;
use h2ai_types::identity::{AgentId, TaskId};

fn task_id() -> TaskId {
    TaskId::new()
}

fn agent_id() -> AgentId {
    AgentId::from("agent-extra")
}

#[test]
fn redact_removes_secret_from_llm_response() {
    let event = AgentTelemetryEvent::LlmResponseReceived {
        task_id: task_id(),
        agent_id: agent_id(),
        response: "The token is sk-abcdefghijklmnopqrstuvwxyz1234567 — keep it safe.".into(),
        token_cost: 20,
        timestamp: Utc::now(),
    };
    let redacted = redact_event(event);
    if let AgentTelemetryEvent::LlmResponseReceived {
        response,
        token_cost,
        ..
    } = redacted
    {
        assert!(
            response.contains("[REDACTED]"),
            "expected redaction in response: {response}"
        );
        assert!(
            !response.contains("sk-"),
            "sk- key should be removed from response"
        );
        // token_cost and other fields are preserved
        assert_eq!(token_cost, 20);
    } else {
        panic!("wrong event variant");
    }
}

#[test]
fn redact_preserves_token_cost_in_llm_response() {
    let event = AgentTelemetryEvent::LlmResponseReceived {
        task_id: task_id(),
        agent_id: agent_id(),
        response: "No secrets here, just plain text.".into(),
        token_cost: 42,
        timestamp: Utc::now(),
    };
    let redacted = redact_event(event);
    if let AgentTelemetryEvent::LlmResponseReceived {
        response,
        token_cost,
        ..
    } = redacted
    {
        assert_eq!(response, "No secrets here, just plain text.");
        assert_eq!(token_cost, 42);
    } else {
        panic!("wrong event variant");
    }
}

#[test]
fn redact_removes_bearer_token_from_llm_response() {
    let event = AgentTelemetryEvent::LlmResponseReceived {
        task_id: task_id(),
        agent_id: agent_id(),
        response: "Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig".into(),
        token_cost: 0,
        timestamp: Utc::now(),
    };
    let redacted = redact_event(event);
    if let AgentTelemetryEvent::LlmResponseReceived { response, .. } = redacted {
        assert!(response.contains("[REDACTED]"));
        assert!(!response.contains("eyJ"));
    } else {
        panic!("wrong event variant");
    }
}

#[test]
fn redact_removes_secret_from_shell_stdout() {
    let event = AgentTelemetryEvent::ShellCommandExecuted {
        task_id: task_id(),
        agent_id: agent_id(),
        command: "cat".into(),
        args: vec!["/etc/secrets".into()],
        stdout: "API_KEY=sk-abcdefghijklmnopqrstuvwxyz1234567".into(),
        stderr: String::new(),
        exit_code: 0,
        timestamp: Utc::now(),
    };
    let redacted = redact_event(event);
    if let AgentTelemetryEvent::ShellCommandExecuted { stdout, .. } = redacted {
        assert!(
            stdout.contains("[REDACTED]"),
            "stdout should be redacted: {stdout}"
        );
        assert!(!stdout.contains("sk-"), "sk- should be removed from stdout");
    } else {
        panic!("wrong event variant");
    }
}

#[test]
fn redact_removes_secret_from_shell_stderr() {
    let event = AgentTelemetryEvent::ShellCommandExecuted {
        task_id: task_id(),
        agent_id: agent_id(),
        command: "some-tool".into(),
        args: vec![],
        stdout: String::new(),
        stderr: "Warning: token Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig exposed".into(),
        exit_code: 1,
        timestamp: Utc::now(),
    };
    let redacted = redact_event(event);
    if let AgentTelemetryEvent::ShellCommandExecuted {
        stderr, exit_code, ..
    } = redacted
    {
        assert!(
            stderr.contains("[REDACTED]"),
            "stderr should be redacted: {stderr}"
        );
        assert!(
            !stderr.contains("eyJ"),
            "bearer token should be removed from stderr"
        );
        assert_eq!(exit_code, 1, "exit_code is preserved");
    } else {
        panic!("wrong event variant");
    }
}

#[test]
fn redact_removes_secret_from_system_error() {
    let event = AgentTelemetryEvent::SystemError {
        task_id: task_id(),
        agent_id: agent_id(),
        error: "failed to authenticate: token=sk-abcdefghijklmnopqrstuvwxyz1234567".into(),
        timestamp: Utc::now(),
    };
    let redacted = redact_event(event);
    if let AgentTelemetryEvent::SystemError { error, .. } = redacted {
        assert!(
            error.contains("[REDACTED]"),
            "error should be redacted: {error}"
        );
        assert!(!error.contains("sk-"), "sk- should be removed from error");
    } else {
        panic!("wrong event variant");
    }
}

#[test]
fn redact_preserves_shell_command_fields() {
    let event = AgentTelemetryEvent::ShellCommandExecuted {
        task_id: task_id(),
        agent_id: agent_id(),
        command: "ls".into(),
        args: vec!["-la".into()],
        stdout: "total 0".into(),
        stderr: String::new(),
        exit_code: 0,
        timestamp: Utc::now(),
    };
    let redacted = redact_event(event);
    if let AgentTelemetryEvent::ShellCommandExecuted {
        command,
        args,
        stdout,
        stderr,
        exit_code,
        ..
    } = redacted
    {
        assert_eq!(command, "ls");
        assert_eq!(args, vec!["-la"]);
        assert_eq!(stdout, "total 0");
        assert_eq!(stderr, "");
        assert_eq!(exit_code, 0);
    } else {
        panic!("wrong event variant");
    }
}
