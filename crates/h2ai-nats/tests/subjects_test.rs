use h2ai_nats::nkey::generate_agent_credentials;
use h2ai_nats::subjects::{
    agent_telemetry_subject, agent_terminate_subject, audit_event_subject, ephemeral_task_subject,
    task_result_subject,
};
use h2ai_types::identity::{AgentId, TaskId};

#[test]
fn ephemeral_task_subject_contains_task_id() {
    let tid = TaskId::new();
    let subj = ephemeral_task_subject(&tid);
    assert!(subj.starts_with("h2ai.tasks.ephemeral."));
    assert!(subj.contains(&tid.to_string()));
}

#[test]
fn task_result_subject_format() {
    let id = TaskId::new();
    let subj = task_result_subject(&id);
    assert!(subj.starts_with("h2ai.results."), "got: {subj}");
    assert!(subj.ends_with(&id.to_string()));
}

#[test]
fn agent_telemetry_subject_format() {
    let aid = AgentId::from("agent-1");
    assert_eq!(agent_telemetry_subject(&aid), "h2ai.telemetry.agent-1");
}

#[test]
fn agent_terminate_subject_format() {
    let aid = AgentId::from("agent-2");
    assert_eq!(
        agent_terminate_subject(&aid),
        "h2ai.control.terminate.agent-2"
    );
}

#[test]
fn audit_event_subject_format() {
    let aid = AgentId::from("agent-x");
    assert_eq!(audit_event_subject(&aid), "audit.events.agent-x");
}

#[test]
fn generate_credentials_seed_starts_with_s() {
    let aid = AgentId::from("agent-1");
    let tid = TaskId::new();
    let task_subj = ephemeral_task_subject(&tid);
    let creds = generate_agent_credentials(&aid, &tid, &task_subj).unwrap();
    assert!(creds.nkey_seed.starts_with('S'));
    assert!(!creds.allowed_publish.is_empty());
    assert!(!creds.allowed_subscribe.is_empty());
}

#[test]
fn generate_credentials_subscribe_includes_task_subject() {
    let aid = AgentId::from("agent-1");
    let tid = TaskId::new();
    let task_subj = ephemeral_task_subject(&tid);
    let creds = generate_agent_credentials(&aid, &tid, &task_subj).unwrap();
    assert!(creds.allowed_subscribe.iter().any(|s| s == &task_subj));
}

#[test]
fn agent_credentials_allow_result_publish() {
    let agent_id = AgentId::from("agent-42");
    let task_id = TaskId::new();
    let task_subject = ephemeral_task_subject(&task_id);

    let creds =
        generate_agent_credentials(&agent_id, &task_id, &task_subject).expect("credentials");

    let result_subj = task_result_subject(&task_id);
    assert!(
        creds.allowed_publish.contains(&result_subj),
        "allowed_publish missing {result_subj}: {:?}",
        creds.allowed_publish
    );
}
