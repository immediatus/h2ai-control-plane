use crate::engine::EngineInput;
use crate::phases::StepResult;
use crate::verification::{EvalCache, VerificationInput, VerificationPhase};
use chrono::Utc;
use h2ai_types::config::VerificationConfig;
use h2ai_types::events::{
    BranchPrunedEvent, ProposalEvent, TopologyProvisionedEvent, VerificationScoredEvent,
    VerifierComparisonEvent,
};
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::RoleErrorCost;
use std::collections::HashMap;

pub struct Input<'a> {
    pub engine_input: &'a EngineInput<'a>,
    pub task_id: &'a TaskId,
    pub verification_config: VerificationConfig,
    pub provisioned: &'a TopologyProvisionedEvent,
    pub task_eval_cache: EvalCache,
    /// Turn-1 output map from the generation phase (explorer_id → turn-1 raw_output).
    pub turn1_map: HashMap<h2ai_types::identity::ExplorerId, String>,
    /// τ values from this generation wave; carried in the ZeroSurvival early exit so the
    /// MAPE-K loop can push them to `tau_values_tried` before calling `RetryPolicy::decide`.
    pub tau_values: Vec<f64>,
}

pub struct Output {
    pub proposals: Vec<ProposalEvent>,
    pub pruned: Vec<BranchPrunedEvent>,
    pub iteration_verification_events: Vec<VerificationScoredEvent>,
    /// Turn-1 re-wrapped proposals for the TaoMultiplierEstimator Option B feed.
    pub turn1_proposals_for_scoring: Vec<ProposalEvent>,
    /// Comparison events (populated only when `record_adversarial_comparison` is set).
    pub all_comparison_events: Vec<VerifierComparisonEvent>,
}

/// Run Phase 3.5: Verification Loop (LLM-as-Judge).
///
/// Evaluates each proposal against the constraint corpus via `VerificationPhase::run`,
/// scores passing and failing proposals, constructs `BranchPrunedEvent`s for failures,
/// and performs a post-verification diversity gate.  When the diversity gate collapses,
/// returns `EarlyExit(ZeroSurvival { … })` so the MAPE-K loop can retry.
pub async fn run(proposals: Vec<ProposalEvent>, input: Input<'_>) -> StepResult<Output> {
    let engine_input = input.engine_input;
    let task_id = input.task_id;
    let provisioned = input.provisioned;

    // ── Phase 3.5: Verification Loop (LLM-as-Judge) ──────────────────
    let mut pruned: Vec<BranchPrunedEvent> = Vec::new();
    let mut iteration_verification_events: Vec<VerificationScoredEvent> = Vec::new();
    let ver_out = VerificationPhase::run(VerificationInput {
        proposals,
        constraint_corpus: &engine_input.constraint_corpus,
        evaluator: engine_input.verification_adapter,
        config: input.verification_config.clone(),
        eval_cache: std::sync::Arc::clone(&input.task_eval_cache),
        consensus_passes: engine_input.cfg.verifier_consensus_passes,
    })
    .await;
    let all_comparison_events: Vec<VerifierComparisonEvent> = ver_out.comparison_events.clone();

    // Diversity gate: post-verification — check constraint-satisfaction profile entropy.
    // Collapsed fingerprints signal collective hallucination; trigger MAPE-K retry.
    if matches!(
        crate::diversity::DiversityGuard::check(
            &ver_out.passed,
            engine_input.cfg.safety.diversity_threshold
        ),
        crate::diversity::DiversityResult::Collapsed
    ) {
        let coherence = crate::coherence::CoherenceState::default();
        return StepResult::EarlyExit(crate::phases::ExitReason::ZeroSurvival {
            failure_mode: None,
            coherence,
            n_eff_cosine: None,
            filter_ratio: 0.0,
            tau_values: input.tau_values,
        });
    }

    let mut proposals: Vec<ProposalEvent> = Vec::new();
    for (prop, results, any_cache_hit) in ver_out.passed {
        let score = h2ai_constraints::types::aggregate_compliance_score(&results);
        iteration_verification_events.push(VerificationScoredEvent {
            task_id: task_id.clone(),
            explorer_id: prop.explorer_id.clone(),
            score,
            reason: String::new(),
            passed: true,
            cache_hit: any_cache_hit,
            timestamp: Utc::now(),
        });
        engine_input.store.record_validation(task_id, true);
        proposals.push(prop);
    }
    for (prop, results, violations, any_cache_hit) in ver_out.failed {
        let hard_gate = results.iter().all(|r| r.hard_passes());
        let soft = h2ai_constraints::types::aggregate_compliance_score(&results);
        let compliance = if hard_gate { soft } else { 0.0 };
        let score = compliance;
        iteration_verification_events.push(VerificationScoredEvent {
            task_id: task_id.clone(),
            explorer_id: prop.explorer_id.clone(),
            score,
            reason: violations
                .iter()
                .map(|v| v.constraint_id.clone())
                .collect::<Vec<_>>()
                .join(", "),
            passed: false,
            cache_hit: any_cache_hit,
            timestamp: Utc::now(),
        });
        let error_cost = RoleErrorCost::new((1.0 - compliance).clamp(0.0, 1.0)).unwrap();
        let cost = provisioned
            .explorer_configs
            .iter()
            .position(|ec| ec.explorer_id == prop.explorer_id)
            .and_then(|idx| provisioned.role_error_costs.get(idx))
            .cloned()
            .unwrap_or(error_cost);
        tracing::info!(
            target: "h2ai.engine",
            explorer_id = %prop.explorer_id,
            compliance = compliance,
            hard_gate = hard_gate,
            violated = ?violations.iter().map(|v| &v.constraint_id).collect::<Vec<_>>(),
            "proposal pruned"
        );
        pruned.push(BranchPrunedEvent {
            task_id: task_id.clone(),
            explorer_id: prop.explorer_id,
            reason: format!("verification compliance {compliance:.2}"),
            constraint_error_cost: cost,
            violated_constraints: violations,
            timestamp: Utc::now(),
        });
        engine_input.store.record_validation(task_id, false);
    }

    // Build turn-1 proposals for Option B estimator feed.
    // Only accepted (passed) proposals that ran multiple TAO turns.
    let turn1_proposals_for_scoring: Vec<ProposalEvent> = proposals
        .iter()
        .filter_map(|prop| {
            input
                .turn1_map
                .get(&prop.explorer_id)
                .map(|t1_output| ProposalEvent {
                    raw_output: t1_output.clone(),
                    ..prop.clone()
                })
        })
        .collect();

    StepResult::Done(Output {
        proposals,
        pruned,
        iteration_verification_events,
        turn1_proposals_for_scoring,
        all_comparison_events,
    })
}
