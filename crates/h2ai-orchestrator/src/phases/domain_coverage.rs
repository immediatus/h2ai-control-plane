use crate::engine::{EngineError, EngineInput};
use h2ai_types::events::DiversityGuardDegradedEvent;

pub struct Output {
    pub diversity_degraded_event: Option<DiversityGuardDegradedEvent>,
}

pub fn run(input: &EngineInput<'_>) -> Result<Output, EngineError> {
    let task_id = input.task_id.clone();

    // ── Domain Coverage Pre-check ──────────────────────────────────
    // Slot domain assignments don't change between retries, so check once here.
    // Fires DiversityGuardDegradedEvent when coverage < domain_coverage_threshold.
    // Fails the task immediately when require_bivariate_cg = true.
    let corpus_tags = crate::domain_coverage::corpus_domain_tags(&input.constraint_corpus);
    let coverage_score = crate::domain_coverage::compute_coverage_score(
        &input.manifest.explorers.slot_configs,
        &corpus_tags,
    );

    let diversity_degraded_event = if coverage_score < input.cfg.domain_coverage_threshold {
        let slot_domains: Vec<String> = input
            .manifest
            .explorers
            .slot_configs
            .iter()
            .flat_map(|s| s.constraint_domains.iter().cloned())
            .collect();
        if input.cfg.safety.require_bivariate_cg {
            input.store.mark_failed(&task_id);
            return Err(EngineError::MultiplicationConditionFailed(format!(
                "domain_coverage {coverage_score:.2} < threshold {:.2} (require_bivariate_cg=true)",
                input.cfg.domain_coverage_threshold
            )));
        }
        Some(DiversityGuardDegradedEvent {
            task_id,
            reason: format!(
                "slot domain coverage {coverage_score:.2} below threshold {:.2}",
                input.cfg.domain_coverage_threshold
            ),
            coverage_score,
            slot_domains,
        })
    } else {
        None
    };

    Ok(Output {
        diversity_degraded_event,
    })
}
