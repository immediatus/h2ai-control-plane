use h2ai_config::H2AIConfig;
use h2ai_orchestrator::engine::EngineOutput;
use h2ai_types::events::{CorrelatedFabricationEvent, VerificationScoredEvent};
use serde::Serialize;
use std::io::Write;

#[derive(Serialize)]
struct SraniConfigSnapshot {
    adaptive: bool,
    ema_alpha: f64,
    temperature: f64,
    gate_threshold: f64,
    warn_threshold: f64,
    inject_threshold: f64,
}

#[derive(Serialize)]
struct VerificationRecord {
    explorer_id: String,
    score: f64,
    passed: bool,
    cache_hit: bool,
    reason: String,
}

#[derive(Serialize)]
struct SraniRecord {
    cfi: f64,
    injection_pressure: f64,
    shared_ungrounded_entities: Vec<String>,
    proposal_count: usize,
    hint_injected: bool,
}

#[derive(Serialize)]
pub struct TaskDebugRecord {
    task_id: String,
    timestamp: String,
    description: String,
    srani_config: SraniConfigSnapshot,
    srani_ema_before: f64,
    srani_count_before: usize,
    srani_ema_after: f64,
    srani_count_after: usize,
    verification_events: Vec<VerificationRecord>,
    srani_events: Vec<SraniRecord>,
    q_confidence: f64,
    waste_ratio: f64,
    resolved_output: String,
}

impl TaskDebugRecord {
    pub fn build(
        description: &str,
        srani_ema_before: f64,
        srani_count_before: usize,
        output: &EngineOutput,
        cfg: &H2AIConfig,
    ) -> Self {
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

        let srani_events: Vec<SraniRecord> = output
            .srani_events
            .iter()
            .map(|e: &CorrelatedFabricationEvent| SraniRecord {
                cfi: e.cfi,
                injection_pressure: e.injection_pressure,
                shared_ungrounded_entities: e.shared_ungrounded_entities.clone(),
                proposal_count: e.proposal_count,
                hint_injected: e.hint_injected,
            })
            .collect();

        Self {
            task_id: output.task_id.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            description: description.to_owned(),
            srani_config: SraniConfigSnapshot {
                adaptive: cfg.srani.adaptive,
                ema_alpha: cfg.srani.ema_alpha,
                temperature: cfg.srani.temperature,
                gate_threshold: cfg.srani.gate_threshold,
                warn_threshold: cfg.srani.warn_threshold,
                inject_threshold: cfg.srani.inject_threshold,
            },
            srani_ema_before,
            srani_count_before,
            srani_ema_after: output.srani_ema_cfi_updated,
            srani_count_after: output.srani_count_updated,
            verification_events,
            srani_events,
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
        .and_then(|mut f| writeln!(f, "{line}").map(|_| ()));
    if let Err(e) = result {
        tracing::warn!(target: "h2ai.debug_log", path = %path, "failed to write debug record: {e}");
    }
}
