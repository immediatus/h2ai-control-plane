use h2ai_types::agent::AgentTelemetryEvent;
use regex::Regex;
use std::sync::OnceLock;

static SECRET_PATTERN: OnceLock<Regex> = OnceLock::new();

fn secret_regex() -> &'static Regex {
    SECRET_PATTERN.get_or_init(|| {
        Regex::new(r"(?i)(sk-[A-Za-z0-9]{20,}|[A-Za-z0-9]{32,}|Bearer\s+[A-Za-z0-9\-._~+/]+=*)")
            .expect("static regex is valid")
    })
}

fn redact(s: &str) -> String {
    secret_regex().replace_all(s, "[REDACTED]").into_owned()
}

/// Redact secrets from an `AgentTelemetryEvent` before it reaches an audit provider.
pub fn redact_event(event: AgentTelemetryEvent) -> AgentTelemetryEvent {
    match event {
        AgentTelemetryEvent::LlmPromptSent {
            task_id,
            agent_id,
            prompt,
            timestamp,
        } => AgentTelemetryEvent::LlmPromptSent {
            task_id,
            agent_id,
            prompt: redact(&prompt),
            timestamp,
        },
        AgentTelemetryEvent::LlmResponseReceived {
            task_id,
            agent_id,
            response,
            token_cost,
            timestamp,
        } => AgentTelemetryEvent::LlmResponseReceived {
            task_id,
            agent_id,
            response: redact(&response),
            token_cost,
            timestamp,
        },
        AgentTelemetryEvent::ShellCommandExecuted {
            task_id,
            agent_id,
            command,
            stdout,
            stderr,
            exit_code,
            timestamp,
        } => AgentTelemetryEvent::ShellCommandExecuted {
            task_id,
            agent_id,
            command: redact(&command),
            stdout: redact(&stdout),
            stderr: redact(&stderr),
            exit_code,
            timestamp,
        },
        AgentTelemetryEvent::SystemError {
            task_id,
            agent_id,
            error,
            timestamp,
        } => AgentTelemetryEvent::SystemError {
            task_id,
            agent_id,
            error: redact(&error),
            timestamp,
        },
    }
}
