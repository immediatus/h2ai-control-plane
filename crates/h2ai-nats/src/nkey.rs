use crate::subjects::{agent_telemetry_subject, audit_event_subject, task_result_subject};
use h2ai_types::identity::{AgentId, TaskId};
use nkeys::KeyPair;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NKeyError {
    #[error("nkey generation failed: {0}")]
    Generation(String),
}

#[derive(Debug, Clone)]
pub struct ScopedAgentCredentials {
    pub nkey_seed: String,
    pub allowed_publish: Vec<String>,
    pub allowed_subscribe: Vec<String>,
}

pub fn generate_agent_credentials(
    agent_id: &AgentId,
    task_id: &TaskId,
    task_subject: &str,
) -> Result<ScopedAgentCredentials, NKeyError> {
    let kp = KeyPair::new_user();
    let seed = kp
        .seed()
        .map_err(|e| NKeyError::Generation(e.to_string()))?;

    let allowed_publish = vec![
        agent_telemetry_subject(agent_id),
        audit_event_subject(agent_id),
        task_result_subject(task_id),
    ];

    let allowed_subscribe = vec![task_subject.to_string()];

    Ok(ScopedAgentCredentials {
        nkey_seed: seed,
        allowed_publish,
        allowed_subscribe,
    })
}
