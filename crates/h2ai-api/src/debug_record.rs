use h2ai_orchestrator::engine::EngineOutput;
use h2ai_types::events::VerificationScoredEvent;
use serde::Serialize;
use std::io::Write;

#[derive(Serialize, Default)]
struct VerificationRecord {
    explorer_id: String,
    score: f64,
    passed: bool,
    cache_hit: bool,
    reason: String,
}

#[derive(Serialize, Default)]
pub struct TaskDebugRecord {
    task_id: String,
    timestamp: String,
    description: String,
    verification_events: Vec<VerificationRecord>,
    q_confidence: f64,
    waste_ratio: f64,
    resolved_output: String,
}

impl TaskDebugRecord {
    #[must_use]
    pub fn build(description: &str, output: &EngineOutput) -> Self {
        let verification_events: Vec<VerificationRecord> = output
            .verification_events
            .iter()
            .map(|e: &VerificationScoredEvent| VerificationRecord {
                explorer_id: e.explorer_id.to_string(),
                score: e.score,
                passed: e.passed,
                cache_hit: e.cache_hit,
                reason: e.reason.clone(),
            })
            .collect();

        Self {
            task_id: output.task_id.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            description: description.to_owned(),
            verification_events,
            q_confidence: output.attribution.q_confidence,
            waste_ratio: output.waste_ratio,
            resolved_output: output.resolved_output.clone(),
        }
    }
}

/// Append a single `TaskDebugRecord` as one JSON line to `path`.
/// Opens the file in append mode; creates it if absent. Best-effort — errors are logged, not propagated.
pub fn append_debug_record(path: &str, record: &TaskDebugRecord) {
    let line = match serde_json::to_string(record) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(target: "h2ai.debug_log", "failed to serialize debug record: {e}");
            return;
        }
    };
    let result = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| writeln!(f, "{line}").map(|()| ()));
    if let Err(e) = result {
        tracing::warn!(target: "h2ai.debug_log", path = %path, "failed to write debug record: {e}");
    }
}
