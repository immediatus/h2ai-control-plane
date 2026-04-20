use chrono::Utc;
use h2ai_types::agent::{
    AgentDescriptor, AgentState, AgentTelemetryEvent, AgentTool, TaskPayload, TaskResult,
};
use h2ai_types::identity::{AgentId, TaskId};
use h2ai_types::physics::TauValue;

fn task_id() -> TaskId {
    TaskId::new()
}

// --- AgentState ---

#[test]
fn agent_state_variants_exist() {
    let _idle = AgentState::Idle;
    let _exec = AgentState::Executing;
    let _wait = AgentState::AwaitingApproval;
    let _fail = AgentState::Failed("timeout".into());
}

#[test]
fn agent_state_serde_roundtrip_idle() {
    let state = AgentState::Idle;
    let json = serde_json::to_string(&state).unwrap();
    let back: AgentState = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, AgentState::Idle));
}

#[test]
fn agent_state_serde_roundtrip_failed() {
    let state = AgentState::Failed("oom".into());
    let json = serde_json::to_string(&state).unwrap();
    let back: AgentState = serde_json::from_str(&json).unwrap();
    match back {
        AgentState::Failed(msg) => assert_eq!(msg, "oom"),
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn agent_state_serde_roundtrip_executing() {
    let state = AgentState::Executing;
    let json = serde_json::to_string(&state).unwrap();
    let back: AgentState = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, AgentState::Executing));
}

#[test]
fn agent_state_serde_roundtrip_awaiting_approval() {
    let state = AgentState::AwaitingApproval;
    let json = serde_json::to_string(&state).unwrap();
    let back: AgentState = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, AgentState::AwaitingApproval));
}

#[test]
fn agent_state_json_shape_idle_has_no_message_key() {
    let json = serde_json::to_string(&AgentState::Idle).unwrap();
    assert!(json.contains("\"state\":\"Idle\""));
    assert!(!json.contains("message"));
}

#[test]
fn agent_state_json_shape_failed_has_message_key() {
    let json = serde_json::to_string(&AgentState::Failed("oom".into())).unwrap();
    assert!(json.contains("\"state\":\"Failed\""));
    assert!(json.contains("\"message\":\"oom\""));
}

// --- TaskPayload ---

#[test]
fn task_payload_serde_roundtrip() {
    let id = task_id();
    let payload = TaskPayload {
        task_id: id.clone(),
        agent: AgentDescriptor {
            model: "gpt-4o".into(),
            tools: vec![AgentTool::Shell, AgentTool::WebSearch],
        },
        instructions: "summarise the doc".into(),
        context: "system: you are a helpful assistant".into(),
        tau: TauValue::new(0.4).unwrap(),
        max_tokens: 512,
    };
    let json = serde_json::to_string(&payload).unwrap();
    let back: TaskPayload = serde_json::from_str(&json).unwrap();
    assert_eq!(back.task_id, id);
    assert_eq!(back.agent.model, "gpt-4o");
    assert_eq!(
        back.agent.tools,
        vec![AgentTool::Shell, AgentTool::WebSearch]
    );
    assert_eq!(back.instructions, "summarise the doc");
    assert_eq!(back.tau.value(), 0.4);
    assert_eq!(back.max_tokens, 512);
}

// --- TaskResult ---

#[test]
fn task_result_serde_roundtrip_success() {
    let id = task_id();
    let result = TaskResult {
        task_id: id.clone(),
        agent_id: "agent-42".into(),
        output: "The answer is 42.".into(),
        token_cost: 120,
        error: None,
    };
    let json = serde_json::to_string(&result).unwrap();
    let back: TaskResult = serde_json::from_str(&json).unwrap();
    assert_eq!(back.task_id, id);
    assert_eq!(back.output, "The answer is 42.");
    assert_eq!(back.token_cost, 120);
    assert!(back.error.is_none());
}

#[test]
fn task_result_serde_roundtrip_failure() {
    let id = task_id();
    let result = TaskResult {
        task_id: id.clone(),
        agent_id: "agent-7".into(),
        output: String::new(),
        token_cost: 0,
        error: Some("adapter timed out".into()),
    };
    let json = serde_json::to_string(&result).unwrap();
    let back: TaskResult = serde_json::from_str(&json).unwrap();
    assert_eq!(back.error.as_deref(), Some("adapter timed out"));
}

// --- AgentTelemetryEvent ---

#[test]
fn agent_telemetry_llm_prompt_sent_serde_roundtrip() {
    let tid = task_id();
    let ts = Utc::now();
    let event = AgentTelemetryEvent::LlmPromptSent {
        task_id: tid.clone(),
        agent_id: "agent-1".into(),
        prompt: "Summarise this document.".into(),
        timestamp: ts,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("\"event_type\":\"LlmPromptSent\""));
    let back: AgentTelemetryEvent = serde_json::from_str(&json).unwrap();
    match back {
        AgentTelemetryEvent::LlmPromptSent {
            task_id,
            agent_id,
            prompt,
            timestamp,
        } => {
            assert_eq!(task_id, tid);
            assert_eq!(agent_id, AgentId::from("agent-1"));
            assert_eq!(prompt, "Summarise this document.");
            assert_eq!(timestamp, ts);
        }
        other => panic!("unexpected variant: {other:?}"),
    }
}

#[test]
fn agent_telemetry_llm_response_received_serde_roundtrip() {
    let tid = task_id();
    let ts = Utc::now();
    let event = AgentTelemetryEvent::LlmResponseReceived {
        task_id: tid.clone(),
        agent_id: "agent-2".into(),
        response: "The answer is 42.".into(),
        token_cost: 256,
        timestamp: ts,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("\"event_type\":\"LlmResponseReceived\""));
    let back: AgentTelemetryEvent = serde_json::from_str(&json).unwrap();
    match back {
        AgentTelemetryEvent::LlmResponseReceived {
            task_id,
            agent_id,
            response,
            token_cost,
            timestamp,
        } => {
            assert_eq!(task_id, tid);
            assert_eq!(agent_id, AgentId::from("agent-2"));
            assert_eq!(response, "The answer is 42.");
            assert_eq!(token_cost, 256);
            assert_eq!(timestamp, ts);
        }
        other => panic!("unexpected variant: {other:?}"),
    }
}

#[test]
fn agent_telemetry_shell_command_executed_serde_roundtrip() {
    let tid = task_id();
    let ts = Utc::now();
    let event = AgentTelemetryEvent::ShellCommandExecuted {
        task_id: tid.clone(),
        agent_id: "agent-3".into(),
        command: "ls -la".into(),
        stdout: "total 8\ndrwxr-xr-x 2 user user 4096 Jan 1 00:00 .".into(),
        stderr: String::new(),
        exit_code: 0,
        timestamp: ts,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("\"event_type\":\"ShellCommandExecuted\""));
    let back: AgentTelemetryEvent = serde_json::from_str(&json).unwrap();
    match back {
        AgentTelemetryEvent::ShellCommandExecuted {
            task_id,
            agent_id,
            command,
            stdout,
            stderr,
            exit_code,
            timestamp,
        } => {
            assert_eq!(task_id, tid);
            assert_eq!(agent_id, AgentId::from("agent-3"));
            assert_eq!(command, "ls -la");
            assert_eq!(stdout, "total 8\ndrwxr-xr-x 2 user user 4096 Jan 1 00:00 .");
            assert_eq!(stderr, "");
            assert_eq!(exit_code, 0);
            assert_eq!(timestamp, ts);
        }
        other => panic!("unexpected variant: {other:?}"),
    }
}

#[test]
fn agent_telemetry_system_error_serde_roundtrip() {
    let tid = task_id();
    let ts = Utc::now();
    let event = AgentTelemetryEvent::SystemError {
        task_id: tid.clone(),
        agent_id: "agent-4".into(),
        error: "connection refused".into(),
        timestamp: ts,
    };
    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("\"event_type\":\"SystemError\""));
    let back: AgentTelemetryEvent = serde_json::from_str(&json).unwrap();
    match back {
        AgentTelemetryEvent::SystemError {
            task_id,
            agent_id,
            error,
            timestamp,
        } => {
            assert_eq!(task_id, tid);
            assert_eq!(agent_id, AgentId::from("agent-4"));
            assert_eq!(error, "connection refused");
            assert_eq!(timestamp, ts);
        }
        other => panic!("unexpected variant: {other:?}"),
    }
}
