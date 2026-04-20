use chrono::Utc;
use h2ai_config::H2AIConfig;
use h2ai_types::events::MultiplicationConditionFailedEvent;
use h2ai_types::identity::TaskId;
use h2ai_types::physics::{CoherencyCoefficients, CoordinationThreshold, MultiplicationCondition};

pub struct MultiplicationChecker;

impl MultiplicationChecker {
    pub fn check(
        task_id: &TaskId,
        cc: &CoherencyCoefficients,
        theta: &CoordinationThreshold,
        baseline_competence: f64,
        error_correlation: f64,
        retry_count: u32,
        cfg: &H2AIConfig,
    ) -> Result<(), MultiplicationConditionFailedEvent> {
        MultiplicationCondition::evaluate(
            baseline_competence,
            error_correlation,
            cc.cg_mean(),
            theta.value(),
            cfg.min_baseline_competence,
            cfg.max_error_correlation,
        )
        .map_err(|failure| MultiplicationConditionFailedEvent {
            task_id: task_id.clone(),
            failure,
            retry_count,
            timestamp: Utc::now(),
        })
    }
}
