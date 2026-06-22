use crate::engine::EngineInput;
use crate::judge_panel::JudgePanel;
use crate::phases::StepResult;
use crate::verification::{EvalCache, VerificationInput, VerificationPhase};
use chrono::Utc;
use h2ai_constraints::types::{
    beta_credible_interval, count_check_verdicts, fractional_compliance_score,
};
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
    /// Turn-1 output map from the generation phase (`explorer_id` → turn-1 `raw_output`).
    pub turn1_map: HashMap<h2ai_types::identity::ExplorerId, String>,
    /// τ values from this generation wave; carried in the `ZeroSurvival` early exit so the
    /// MAPE-K loop can push them to `tau_values_tried` before calling `RetryPolicy::decide`.
    pub tau_values: Vec<f64>,
    /// Constraint IDs whose verifier judgment is bypassed (from MAPE-K decide).
    /// When all failing Hard constraints for a proposal are in this set,
    /// the proposal passes pruning with bypass active.
    pub bypassed_constraint_ids: std::collections::HashSet<String>,
}

pub struct Output {
    pub proposals: Vec<ProposalEvent>,
    pub pruned: Vec<BranchPrunedEvent>,
    pub iteration_verification_events: Vec<VerificationScoredEvent>,
    /// Turn-1 re-wrapped proposals for the `TaoMultiplierEstimator` Option B feed.
    pub turn1_proposals_for_scoring: Vec<ProposalEvent>,
    /// Comparison events (populated only when `record_adversarial_comparison` is set).
    pub all_comparison_events: Vec<VerifierComparisonEvent>,
    /// Mean pairwise constraint-conflict rate computed from raw proposal texts.
    pub conflict_rate: Option<f64>,
    /// Per-constraint verifier reasons from the best-scoring passing proposal this wave.
    /// Key = constraint_id, Value = verifier_reason text (non-empty reasons only).
    /// Empty when no proposals passed verification.
    pub best_passing_constraint_reasons: std::collections::HashMap<String, String>,
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

    // ── Conflict rate: computed before proposals are consumed by VerificationPhase ─
    let conflict_rate = {
        let texts: Vec<&str> = proposals.iter().map(|p| p.raw_output.as_str()).collect();
        h2ai_autonomic::calibration::compute_conflict_rate(&texts, &engine_input.constraint_corpus)
    };

    // ── Phase 3.5: Verification Loop (LLM-as-Judge) ──────────────────
    let mut pruned: Vec<BranchPrunedEvent> = Vec::new();
    let mut iteration_verification_events: Vec<VerificationScoredEvent> = Vec::new();

    // Build the judge panel: primary = verification adapter; add cross-family explorer adapters.
    let panel_cfg = &engine_input.cfg.judge_panel;
    let primary_family = engine_input.verification_adapter.kind().family();
    let mut seen_panel_families = std::collections::HashSet::new();
    seen_panel_families.insert(primary_family);
    let additional: Vec<_> = engine_input
        .explorer_adapters
        .iter()
        .filter_map(|a| {
            let fam = a.kind().family();
            if seen_panel_families.insert(fam) {
                Some((*a as &dyn h2ai_types::adapter::IComputeAdapter, a.kind()))
            } else {
                None
            }
        })
        .collect();
    let panel = JudgePanel::build(engine_input.verification_adapter, &additional, panel_cfg);

    let ver_input = VerificationInput {
        proposals,
        constraint_corpus: &engine_input.constraint_corpus,
        evaluator: engine_input.verification_adapter,
        config: input.verification_config.clone(),
        eval_cache: std::sync::Arc::clone(&input.task_eval_cache),
        consensus_passes: engine_input.cfg.verifier_consensus_passes,
    };
    let (ver_out, uncertain_map) =
        VerificationPhase::run_with_panel(ver_input, &panel, panel_cfg).await;
    let all_comparison_events: Vec<VerifierComparisonEvent> = ver_out.comparison_events.clone();

    // Log ConstraintAmbiguityEvent when a constraint's uncertain vote count reaches threshold.
    {
        let mut uncertain_counts: HashMap<String, usize> = HashMap::new();
        for ids in uncertain_map.values() {
            for id in ids {
                *uncertain_counts.entry(id.clone()).or_insert(0) += 1;
            }
        }
        let ambiguous: Vec<String> = uncertain_counts
            .iter()
            .filter(|(_, &count)| count >= panel_cfg.ambiguity_threshold)
            .map(|(id, _)| id.clone())
            .collect();
        if !ambiguous.is_empty() {
            tracing::info!(
                target: "h2ai.engine",
                task_id = %task_id,
                ambiguous_constraints = ?ambiguous,
                uncertain_counts = ?uncertain_counts,
                "ConstraintAmbiguityEvent: constraints with uncertain judge panel votes"
            );
        }
    }

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

    let mut best_pass_score = -1.0_f64;
    let mut best_passing_constraint_reasons: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    let mut proposals: Vec<ProposalEvent> = Vec::new();
    for (prop, results, any_cache_hit) in ver_out.passed {
        let score = h2ai_constraints::types::aggregate_compliance_score(&results);
        // Strict `>` means ties resolve to first-seen (common when all proposals share score 1.0
        // on soft-constraint-only tasks). Deterministic but unordered within a tie.
        if score > best_pass_score {
            best_pass_score = score;
            best_passing_constraint_reasons = results
                .iter()
                .filter_map(|r| {
                    r.verifier_reason
                        .as_ref()
                        .filter(|s| !s.is_empty())
                        .map(|reason| (r.constraint_id.clone(), reason.clone()))
                })
                .collect();
        }
        let (passed_checks, total_checks) = count_check_verdicts(&results);
        let (score_lower, score_upper) = if total_checks > 0 {
            let (lo, hi) = beta_credible_interval(passed_checks, total_checks);
            (Some(lo), Some(hi))
        } else {
            (None, None)
        };
        iteration_verification_events.push(VerificationScoredEvent {
            task_id: task_id.clone(),
            explorer_id: prop.explorer_id.clone(),
            score,
            reason: String::new(),
            passed: true,
            cache_hit: any_cache_hit,
            passed_checks: Some(passed_checks),
            total_checks: Some(total_checks),
            score_lower,
            score_upper,
            timestamp: Utc::now(),
        });
        engine_input.store.record_validation(task_id, true);
        proposals.push(prop);
    }
    let ct_scale = input.verification_config.constraint_threshold_scale;
    for (prop, results, violations, any_cache_hit) in ver_out.failed {
        let hard_gate = results.iter().all(|r| {
            r.hard_passes_scaled(ct_scale)
                || input.bypassed_constraint_ids.contains(&r.constraint_id)
        });
        let soft = h2ai_constraints::types::aggregate_compliance_score(&results);
        let compliance = if hard_gate {
            soft
        } else {
            fractional_compliance_score(&results)
        };
        let score = compliance;
        let (passed_checks, total_checks) = count_check_verdicts(&results);
        let (score_lower, score_upper) = if total_checks > 0 {
            let (lo, hi) = beta_credible_interval(passed_checks, total_checks);
            (Some(lo), Some(hi))
        } else {
            (None, None)
        };
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
            passed_checks: Some(passed_checks),
            total_checks: Some(total_checks),
            score_lower,
            score_upper,
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
            raw_output: prop.raw_output.clone(),
            constraint_error_cost: cost,
            violated_constraints: violations,
            timestamp: Utc::now(),
            retry_count: 0,
            bypass_reason: None,
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
        conflict_rate,
        best_passing_constraint_reasons,
    })
}
