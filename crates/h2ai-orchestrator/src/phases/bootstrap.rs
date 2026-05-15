use crate::engine::{EngineError, EngineInput};
use chrono::Utc;
use h2ai_config::FamilyConstraint;
use h2ai_constraints::types::ConstraintDoc;
use h2ai_context::compaction::{compact, CompactionConfig};
use h2ai_context::compiler;
use h2ai_types::adapter::IComputeAdapter;
use h2ai_types::config::AdapterKind;
use h2ai_types::events::TaskBootstrappedEvent;

pub struct Output {
    pub system_context: String,
    pub system_context_with_rubric: String,
    pub explorer_adapter_kind: AdapterKind,
    pub bootstrapped_event: TaskBootstrappedEvent,
}

pub async fn run(input: &EngineInput<'_>) -> Result<Output, EngineError> {
    let task_id = input.task_id.clone();

    let description = &input.manifest.description;
    let compiled = compiler::compile(description, &input.constraint_corpus, false);
    let compiled_with_rubric = compiler::compile(description, &input.constraint_corpus, true);

    let adr_keywords: Vec<String> = input
        .constraint_corpus
        .iter()
        .flat_map(|d: &ConstraintDoc| d.vocabulary().into_iter())
        .chain(input.manifest.constraints.iter().cloned())
        .collect();
    let compaction_cfg = CompactionConfig {
        max_tokens: input.cfg.max_context_tokens.unwrap_or(usize::MAX / 4),
        preserve_keywords: adr_keywords,
    };
    let system_context = compact(&compiled.system_context, &compaction_cfg);
    let system_context_with_rubric = compact(&compiled_with_rubric.system_context, &compaction_cfg);

    let bootstrapped_event = TaskBootstrappedEvent {
        task_id: task_id.clone(),
        system_context: system_context.clone(),
        pareto_weights: input.manifest.pareto_weights.clone(),
        timestamp: Utc::now(),
    };

    let explorer_adapter_kind = input
        .explorer_adapters
        .first()
        .map(|a| a.kind().clone())
        .unwrap_or_else(|| input.auditor_config.adapter.clone());

    // ── Verifier/Explorer Family Conflict Gate ──────────────────────────
    if input.calibration.explorer_verification_family_match {
        match input.cfg.safety.family_constraint {
            FamilyConstraint::RequireDiverse => {
                let distinct: std::collections::HashSet<usize> = input
                    .explorer_adapters
                    .iter()
                    .map(|a| *a as *const dyn IComputeAdapter as *const () as usize)
                    .collect();
                if distinct.len() == 1 && input.explorer_adapters.len() > 1 {
                    input.store.mark_failed(&task_id);
                    return Err(EngineError::MultiplicationConditionFailed(
                        "all explorer slots map to the same adapter — increase adapter pool or use distinct diversity IDs".into()
                    ));
                }
            }
            FamilyConstraint::SingleFamilyOk => {
                tracing::warn!(
                    "single-family adapter pool: correlated hallucination protection degraded"
                );
            }
            FamilyConstraint::Disabled => {}
        }
    }

    Ok(Output {
        system_context,
        system_context_with_rubric,
        explorer_adapter_kind,
        bootstrapped_event,
    })
}
