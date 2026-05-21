use h2ai_nats::nkey::{generate_agent_credentials, NKeyError, ScopedAgentCredentials};
use h2ai_nats::subjects::{
    agent_telemetry_subject, audit_event_subject, ephemeral_task_subject, task_result_subject,
};
use h2ai_types::identity::{AgentId, TaskId};

fn make_creds() -> ScopedAgentCredentials {
    let aid = AgentId::from("agent-nkey-test");
    let tid = TaskId::new();
    let task_subj = ephemeral_task_subject(&tid);
    generate_agent_credentials(&aid, &tid, &task_subj).expect("credentials")
}

#[test]
fn nkey_error_display_generation() {
    let err = NKeyError::Generation("some nkeys error".to_string());
    let msg = err.to_string();
    assert!(
        msg.contains("nkey generation failed"),
        "unexpected message: {msg}"
    );
    assert!(
        msg.contains("some nkeys error"),
        "should include inner error: {msg}"
    );
}

#[test]
fn nkey_error_debug_impl() {
    let err = NKeyError::Generation("debug test".to_string());
    let debug_str = format!("{err:?}");
    assert!(debug_str.contains("Generation"), "debug: {debug_str}");
}

#[test]
fn credentials_nkey_seed_is_valid_user_seed() {
    let creds = make_creds();
    // User NKey seeds start with SU
    assert!(
        creds.nkey_seed.starts_with("SU"),
        "expected SU prefix, got: {}",
        &creds.nkey_seed[..2.min(creds.nkey_seed.len())]
    );
}

#[test]
fn credentials_allowed_publish_count() {
    let aid = AgentId::from("agent-pub-count");
    let tid = TaskId::new();
    let task_subj = ephemeral_task_subject(&tid);
    let creds = generate_agent_credentials(&aid, &tid, &task_subj).expect("credentials");
    // telemetry + audit + result = 3
    assert_eq!(
        creds.allowed_publish.len(),
        3,
        "expected exactly 3 allowed_publish subjects: {:?}",
        creds.allowed_publish
    );
}

#[test]
fn credentials_allowed_publish_includes_telemetry_subject() {
    let aid = AgentId::from("agent-telem");
    let tid = TaskId::new();
    let task_subj = ephemeral_task_subject(&tid);
    let creds = generate_agent_credentials(&aid, &tid, &task_subj).expect("credentials");
    let telem_subj = agent_telemetry_subject(&aid);
    assert!(
        creds.allowed_publish.contains(&telem_subj),
        "allowed_publish missing {telem_subj}: {:?}",
        creds.allowed_publish
    );
}

#[test]
fn credentials_allowed_publish_includes_audit_subject() {
    let aid = AgentId::from("agent-audit");
    let tid = TaskId::new();
    let task_subj = ephemeral_task_subject(&tid);
    let creds = generate_agent_credentials(&aid, &tid, &task_subj).expect("credentials");
    let audit_subj = audit_event_subject(&aid);
    assert!(
        creds.allowed_publish.contains(&audit_subj),
        "allowed_publish missing {audit_subj}: {:?}",
        creds.allowed_publish
    );
}

#[test]
fn credentials_allowed_subscribe_count() {
    let aid = AgentId::from("agent-sub");
    let tid = TaskId::new();
    let task_subj = ephemeral_task_subject(&tid);
    let creds = generate_agent_credentials(&aid, &tid, &task_subj).expect("credentials");
    assert_eq!(
        creds.allowed_subscribe.len(),
        1,
        "expected exactly 1 subscribe subject: {:?}",
        creds.allowed_subscribe
    );
}

#[test]
fn credentials_nkey_seed_is_unique_per_call() {
    let aid = AgentId::from("agent-unique");
    let tid = TaskId::new();
    let task_subj = ephemeral_task_subject(&tid);
    let creds1 = generate_agent_credentials(&aid, &tid, &task_subj).expect("creds1");
    let creds2 = generate_agent_credentials(&aid, &tid, &task_subj).expect("creds2");
    assert_ne!(
        creds1.nkey_seed, creds2.nkey_seed,
        "each call should generate a fresh key pair"
    );
}

#[test]
fn scoped_agent_credentials_clone() {
    let creds = make_creds();
    let creds2 = creds.clone();
    assert_eq!(creds.nkey_seed, creds2.nkey_seed);
    assert_eq!(creds.allowed_publish, creds2.allowed_publish);
    assert_eq!(creds.allowed_subscribe, creds2.allowed_subscribe);
}

#[test]
fn credentials_result_subject_format() {
    let aid = AgentId::from("agent-res");
    let tid = TaskId::new();
    let task_subj = ephemeral_task_subject(&tid);
    let creds = generate_agent_credentials(&aid, &tid, &task_subj).expect("credentials");
    let expected_result_subj = task_result_subject(&tid);
    assert!(
        creds.allowed_publish.contains(&expected_result_subj),
        "allowed_publish missing result subject {expected_result_subj}: {:?}",
        creds.allowed_publish
    );
}
