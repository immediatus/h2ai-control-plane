use crate::diagnostics::TalagrandDiagnostic;
use crate::engine::{EngineError, EngineOutput};
use crate::phases::ExitReason;
use crate::self_optimizer::{OptimizerParams, QualityMeasurement, SelfOptimizer, SuggestInput};
use h2ai_autonomic::retry::{RetryAction, RetryPolicy};
use h2ai_types::config::{TaoConfig, TopologyKind, VerificationConfig};
use h2ai_types::events::{
    ConstraintFrontierEvent, CorrelatedEnsembleWarning, ProposalFailedEvent,
    ResearcherGroundingEvent, ShadowAuditorResultEvent, TopologyProvisionedEvent,
    VerificationScoredEvent, ZeroSurvivalEvent,
};
use h2ai_types::identity::{ExplorerId, TaskId};

/// Immutable snapshot of MAPE-K params for one wave.
#[derive(Clone, Debug)]
pub struct PipelineParams {
    pub optimizer: OptimizerParams,
    pub force_topology: Option<TopologyKind>,
    pub tau_reduction_factor: f64,
    pub tau_spread_factor: f64,
    pub adapter_rotation_offset: usize,
    pub retry_context: Option<String>,
    pub tao_config: TaoConfig,
    pub verification_config: VerificationConfig,
    pub pending_tombstone: Option<String>,
    /// Leader context snapshot for per-slot context injection in generation.
    pub leader_context: Option<crate::leader::LeaderContextSnapshot>,
    /// Assembled contexts from the previous wave for cross-wave delta encoding.
    pub prev_assembled_contexts: Vec<Option<crate::context_assembler::AssembledContext>>,
    /// Budget hint suffix appended to `active_ctx` when cost conservation is active.
    /// Computed by `MapeKController::params()`. `None` when cost guard is disabled.
    pub budget_hint: Option<String>,
    /// Constraint IDs whose verifier judgment is currently bypassed.
    /// Populated from MapeKController.bypassed_verifier_constraints by params().
    /// Passed to phases/verify.rs Input to modify hard-gate behavior.
    pub bypassed_constraint_ids: std::collections::HashSet<String>,
}

/// Talagrand feedback stored in `WaveEvents`.
#[derive(Clone, Debug)]
pub struct TalagrandFeedback {
    pub tau_spread_next: f64,
}

/// `TaoEstimator` update stored in `WaveEvents`.
#[derive(Clone, Debug)]
pub struct TaoEstimatorUpdate {
    pub t1_score: f64,
    pub final_score: f64,
}

/// All events produced in one pipeline wave, collected for `observe()`.
#[derive(Clone, Debug)]
pub struct WaveEvents {
    pub verification_events: Vec<VerificationScoredEvent>,
    pub failed_proposals: Vec<ProposalFailedEvent>,
    pub shadow_audit_events: Vec<ShadowAuditorResultEvent>,
    pub correlated_warnings: Vec<CorrelatedEnsembleWarning>,
    pub researcher_grounding_events: Vec<ResearcherGroundingEvent>,
    pub quality_measurement: Option<crate::self_optimizer::QualityMeasurement>,
    pub talagrand_feedback: Option<TalagrandFeedback>,
    pub tao_estimator_update: Option<TaoEstimatorUpdate>,
    pub topology_retry_event: Option<TopologyProvisionedEvent>,
    pub frontier_event: Option<ConstraintFrontierEvent>,
    /// Verification filter ratio from this wave's merge phase (surviving / total evaluated).
    /// 1.0 when no merge ran (early-exit before merge). Used by the coordinator to call `decide()`.
    pub filter_ratio: f64,
    /// Pruned branch events from this wave's audit phase.
    /// Accumulated by `observe()` into `all_pruned` so `RetryPolicy::decide` can
    /// collect `remediation_hint` strings for `RetryWithHints`.
    pub pruned_events: Vec<h2ai_types::events::BranchPrunedEvent>,
    /// Mean pairwise constraint-conflict rate across all proposals in this wave.
    /// `None` when fewer than 2 proposals were generated or corpus is empty.
    pub conflict_rate: Option<f64>,
    /// Proposal texts keyed by explorer ID — populated in pipeline.rs before verification.
    pub wave_proposal_texts: std::collections::HashMap<h2ai_types::identity::ExplorerId, String>,
    /// `AssembledContexts` from this wave's generation phase, for next-wave delta encoding.
    pub assembled_contexts: Vec<Option<crate::context_assembler::AssembledContext>>,
    /// Sum of `token_cost` from all ProposalEvents generated this wave.
    /// Zero for waves where no proposals were generated (e.g. EarlyExit).
    pub wave_token_cost: u64,
    /// Per-constraint verifier reasons from the best-scoring passing proposal this wave.
    /// Populated from `phases::verify::Output.best_passing_constraint_reasons` in pipeline.rs.
    /// Empty on waves with no passing proposals.
    pub best_passing_constraint_reasons: std::collections::HashMap<String, String>,
}

impl Default for WaveEvents {
    fn default() -> Self {
        Self {
            verification_events: Vec::new(),
            failed_proposals: Vec::new(),
            shadow_audit_events: Vec::new(),
            correlated_warnings: Vec::new(),
            researcher_grounding_events: Vec::new(),
            quality_measurement: None,
            talagrand_feedback: None,
            tao_estimator_update: None,
            topology_retry_event: None,
            frontier_event: None,
            filter_ratio: 1.0,
            pruned_events: Vec::new(),
            conflict_rate: None,
            wave_proposal_texts: std::collections::HashMap::new(),
            assembled_contexts: Vec::new(),
            wave_token_cost: 0,
            best_passing_constraint_reasons: std::collections::HashMap::new(),
        }
    }
}

/// Successful merge result — passed from pipeline to controller via `PipelineOutcome::Resolved`.
pub struct MergeOutput {
    pub task_id: TaskId,
    pub resolved_output: String,
    /// `true` = merge resolved successfully (always true when `Done` is returned).
    pub selection_resolved: bool,
    /// Full `SelectionResolvedEvent` produced by `MergeEngine::resolve`.
    /// Carried here so engine.rs can publish it without reconstructing timing data.
    pub selection_resolved_event: h2ai_types::events::SelectionResolvedEvent,
    pub attribution: crate::attribution::HarnessAttribution,
    pub attribution_interval: Option<crate::attribution::AttributionInterval>,
    pub talagrand: Option<crate::diagnostics::TalagrandDiagnostic>,
    pub suggested_next_params: Option<OptimizerParams>,
    pub waste_ratio: f64,
    pub applied_optimizations: Vec<h2ai_types::events::AppliedOptimization>,
    pub epistemic_yield: Option<f64>,
    pub frontier_event: Option<ConstraintFrontierEvent>,
    pub adapter_correctness: Vec<(ExplorerId, bool)>,
    pub coherence_state: crate::coherence::CoherenceState,
    pub comparison_events: Vec<h2ai_types::events::VerifierComparisonEvent>,
    pub oracle_gate_passed: Option<bool>,
    /// τ values from this wave's generation phase; pushed into `tau_values_tried` by engine.rs.
    pub tau_values: Vec<f64>,
    /// Per-iteration verification events; appended to `all_verification_events` by engine.rs.
    pub iteration_verification_events: Vec<h2ai_types::events::VerificationScoredEvent>,
    /// Generation token cost for this wave (sum of all proposal token costs).
    pub wave_token_cost: u64,
    /// Mean pairwise cosine similarity across surviving verified proposal texts.
    /// `None` when < 2 proposals survived or embedding model unavailable.
    pub pairwise_cosine_mean: Option<f64>,
}

/// Pipeline outcome after one wave.
pub enum PipelineOutcome {
    Resolved(Box<MergeOutput>),
    EarlyExit(ExitReason),
    Fatal(EngineError),
}

/// What the pipeline returns after one wave.
pub struct PipelineWaveResult {
    pub outcome: PipelineOutcome,
    pub events: WaveEvents,
}

/// Controller decision after observing a wave.
pub enum MapeKDecision {
    Return(Box<EngineOutput>),
    Retry,
    Fail(EngineError, crate::engine::EngineRunContext),
    /// Constraint check text is ambiguous — SpecRepairAdvisor was triggered.
    /// The engine should reload constraints and restart the task from wave 0.
    SpecAmbiguous {
        constraint_id: String,
        check_index: usize,
        instability_score: f64,
        divergent_reasons: Vec<String>,
        ambiguity_evidence: Vec<String>,
        ambiguity_score: f32,
    },
    /// Task complexity exceeds the LLM's computation budget — retries are futile.
    /// `graft_first = true` → route to H1 grafting on first failure.
    /// `graft_first = false` → route to HITL immediately.
    ComplexityOverflow {
        /// Probe score 1–5, or 0 if fired by intra-retry detector.
        probe_score: u8,
        /// Human-readable rationale.
        rationale: String,
        /// true = H1 grafting; false = HITL immediately.
        graft_first: bool,
    },
}

/// Per-corpus-constraint cached fields needed for repair signal enrichment.
pub(crate) struct ConstraintPassEntry {
    pub pass_criteria: Option<String>,
    pub remediation_hint: Option<String>,
}

/// MAPE-K controller — owns all retry state. Full impl added in Task 9.
#[allow(dead_code)] // bandit_state / tao_estimator / tao_multiplier reserved for future pipeline use
pub struct MapeKController {
    // Optimizer
    pub(crate) current_params: OptimizerParams,
    pub(crate) quality_history: Vec<QualityMeasurement>,
    pub(crate) n_max_ceiling: u32,
    pub(crate) cg_mean: f64,

    // Topology
    pub(crate) force_topology: Option<TopologyKind>,
    pub(crate) tried_topologies: Vec<TopologyKind>,

    // τ feedback
    pub(crate) tau_reduction_factor: f64,
    pub(crate) tau_spread_factor: f64,
    pub(crate) tau_values_tried: Vec<Vec<f64>>,

    // Retry routing
    pub(crate) retry_context: Option<String>,
    pub(crate) adapter_rotation_offset: usize,
    pub(crate) mode_collapse_count: usize,
    pub(crate) last_multiplication_failure:
        Option<h2ai_types::sizing::MultiplicationConditionFailure>,
    pub(crate) pending_tombstone: Option<String>,
    pub(crate) system_context_with_rubric: String,
    pub(crate) max_retries: usize,

    // Per-wave config overrides
    pub(crate) tao_config: TaoConfig,
    pub(crate) verification_config: VerificationConfig,

    // Cross-wave aggregation
    pub(crate) all_verification_events: Vec<VerificationScoredEvent>,
    pub(crate) all_failed_proposals: Vec<ProposalFailedEvent>,
    pub(crate) all_shadow_audit_events: Vec<ShadowAuditorResultEvent>,
    pub(crate) all_correlated_warnings: Vec<CorrelatedEnsembleWarning>,
    pub(crate) all_researcher_grounding_events: Vec<ResearcherGroundingEvent>,
    pub(crate) all_pruned: Vec<h2ai_types::events::BranchPrunedEvent>,
    /// Pruned events from the most recent wave only — used for tombstone synthesis
    /// so the LLM receives violations from the immediately preceding wave rather
    /// than the full historical accumulator (which causes attention dilution).
    pub(crate) last_wave_pruned: Vec<h2ai_types::events::BranchPrunedEvent>,
    /// Pruned events from the wave before last — used with `last_wave_pruned` for
    /// cross-wave instability detection. Rotated in `observe()` on each new wave.
    pub(crate) prev_wave_pruned: Vec<h2ai_types::events::BranchPrunedEvent>,
    pub(crate) topology_retry_events: Vec<TopologyProvisionedEvent>,

    // Immutable fields
    pub(crate) task_id: TaskId,
    pub(crate) assessed_quadrant: h2ai_types::sizing::TaskQuadrant,
    pub(crate) complexity_event: h2ai_types::events::TaskComplexityAssessedEvent,
    pub(crate) diversity_degraded_event: Option<h2ai_types::events::DiversityGuardDegradedEvent>,
    pub(crate) bandit_state:
        Option<std::sync::Arc<tokio::sync::RwLock<crate::bandit::BanditState>>>,
    pub(crate) tao_estimator:
        std::sync::Arc<tokio::sync::RwLock<crate::tao_loop::TaoMultiplierEstimator>>,
    pub(crate) tao_multiplier: f64,
    pub(crate) calibration_ensemble: Option<h2ai_types::sizing::EnsembleCalibration>,
    pub(crate) cfg_ref: std::sync::Arc<h2ai_config::H2AIConfig>,
    pub(crate) task_deadline: Option<std::time::Instant>,

    // ── Epistemic Leader ─────────────────────────────────────────────────────
    pub leader: Option<crate::leader::LeaderState>,
    pub(crate) last_wave_verification_events: Vec<h2ai_types::events::VerificationScoredEvent>,
    pub(crate) last_wave_proposal_texts:
        std::collections::HashMap<h2ai_types::identity::ExplorerId, String>,
    pub(crate) pending_leader_elected_events: Vec<h2ai_types::events::LeaderElectedEvent>,
    pub(crate) pending_socratic_diagnosis_events: Vec<h2ai_types::events::SocraticDiagnosisEvent>,
    pub(crate) ambiguity_scorecards:
        std::collections::HashMap<(String, usize), h2ai_constraints::ambiguity::AmbiguityScorecard>,
    pub(crate) pending_ambiguity_events: Vec<h2ai_types::events::ConstraintAmbiguityDetectedEvent>,
    pub(crate) last_wave_violated_constraint_ids: Vec<String>,
    /// `AssembledContexts` from the most recently completed wave.
    /// Passed as `prev_assembled_contexts` to the next wave's generation phase.
    pub(crate) prev_assembled_contexts: Vec<Option<crate::context_assembler::AssembledContext>>,

    // ── CSPR-v2: Conflict-Aware Constraint-Scoped Prior Repair ──────────────
    /// Cross-wave global best proposal: (score, `proposal_text`).
    /// Updated by `observe()` each wave; used by CSPR-v2 repair context builder.
    /// Only set when the proposal passed the hard gate — used with constraint reasons.
    pub(crate) global_best_proposal: Option<(f64, String)>,
    /// Best proposal text seen across all waves, regardless of pass status.
    /// Used as CSPR repair prior when `global_best_proposal` is None (i.e. all
    /// wave proposals failed the hard gate). Prevents wave 2 from starting cold
    /// in zero-pass scenarios by providing the best partial match as a repair basis.
    pub(crate) global_best_partial_proposal: Option<(f64, String)>,
    /// Dynamic per-constraint verifier reasons from the global-best passing proposal.
    /// Populated in `observe()` when a new best proposal score is recorded.
    /// Used as `passing_constraint_pins` hints in CSPR repair context to anchor the
    /// LLM on what the verifier accepted, not just what the corpus requires.
    pub(crate) global_best_constraint_reasons: std::collections::HashMap<String, String>,
    /// Mean compliance scores from each wave, used for plateau detection.
    pub(crate) compliance_score_history: Vec<f64>,
    /// Static conflict graph built from the task's constraint corpus.
    /// Passed from engine.rs at construction time.
    pub(crate) conflict_graph: h2ai_constraints::conflict::ConstraintConflictGraph,

    // ── Constraint-Informed Synthesis ────────────────────────────────
    /// Binary check strings from the constraint corpus, collected for B1/B2 injection.
    pub(crate) binary_checks: Vec<String>,
    /// Offset map: (constraint_id, start_idx_in_binary_checks, count).
    /// Used by partial_pass_from_event to attribute per-check verdicts to the right constraint.
    pub(crate) constraint_check_offsets: Vec<(String, usize, usize)>,

    // ── Knowledge-Gap Detection + Domain Synthesis ───────────────────
    /// Per (constraint_id, check_idx) → validated DomainSynthesis from gap researcher.
    /// Populated lazily by `run_gap_i1_research`; cleared on task completion.
    pub(crate) domain_synthesis_cache:
        std::collections::HashMap<(String, usize), h2ai_types::gap_i1::DomainSynthesis>,
    /// Optional LLM adapter used by the researcher (e.g. `researcher_adapter`).
    /// `None` when `gap_i1.enabled = false` or no researcher adapter is configured.
    pub(crate) gap_researcher_adapter:
        Option<std::sync::Arc<dyn h2ai_types::adapter::IComputeAdapter>>,
    /// Grounding chain gap researcher: DDG search + LLM distiller.
    pub(crate) gap_grounding_chain:
        Option<std::sync::Arc<crate::grounding_chain::GapResearchChain>>,
    /// Maps constraint_id → cached corpus fields for repair signal enrichment.
    pub(crate) constraint_pass_map: std::collections::HashMap<String, ConstraintPassEntry>,

    // ── Complexity-Ceiling Routing ───────────────────────────────────
    /// Pre-dispatch complexity probe result. `None` when probe is disabled or not yet run.
    pub(crate) probe_result: Option<h2ai_autonomic::complexity_probe::ComplexityProbeResult>,
    /// True when the constraint corpus contains at least one `binary_checks` entry.
    /// Computed at construction time; gates `ComplexityOverflow{graft_first:true}` routing.
    pub(crate) corpus_synthesis_viable: bool,

    // ── Pipeline Resilience: Frozen Verifier Detection ─────────────────────────
    /// Per-constraint, per-wave mean violation scores. Key = constraint_id.
    /// Populated in observe() from BranchPrunedEvent.violated_constraints.
    pub(crate) per_constraint_wave_scores: std::collections::HashMap<String, Vec<f64>>,
    /// Rolling window of verifier reasons per constraint_id.
    /// Bounded by cfg_ref.verifier_freeze.reason_window_size.
    pub(crate) verifier_reason_history:
        std::collections::HashMap<String, std::collections::VecDeque<String>>,
    /// Constraint IDs whose verifier judgment is currently bypassed.
    /// Set by decide() when detect_frozen_verifier fires with bypass_hard_gate_on_freeze=true.
    pub(crate) bypassed_verifier_constraints: std::collections::HashSet<String>,
    /// Frozen verifier events queued during decide() for engine.rs to emit.
    pub(crate) pending_frozen_verifier_events: Vec<h2ai_types::events::VerifierFrozenEvent>,

    // ── AgentDropout N-reduction ─────────────────────────────────────
    /// N_eff (participation ratio) from the most recently completed ZeroSurvival wave.
    /// Initialized to `1.0` (no dropout). Updated in `handle_exit_reason` on
    /// every `ZeroSurvival` arm. Used by engine.rs to reduce `n_agents` on retry ≥ 2.
    pub(crate) last_wave_n_eff: f64,

    // ── GAP-L1: Tiered Early Exit ─────────────────────────────────────────
    /// Set by `decide()` when the TEE acceptance gate fires on `Resolved`.
    /// Cleared by `take_tee_event()` so engine.rs can publish it.
    pub(crate) tee_event: Option<h2ai_types::events::TieredExitEvent>,

    // ── GAP-H3: Cost Guard ────────────────────────────────────────────────────
    /// Cumulative generation token cost across all waves for this task.
    pub(crate) tokens_used: u64,
    /// Set to true when budget abort threshold is crossed; blocks further Retry returns.
    pub(crate) budget_exhausted: bool,
    /// Populated by decide() when convergence gate fires; taken by engine.rs to publish.
    pub(crate) convergence_gate_event: Option<h2ai_types::events::ConvergenceGateEvent>,

    // ── GAP-F5: Retroactive Induction Trigger ────────────────────────────────
    /// Domain tags from the task manifest (constraint_tags). Stored at construction
    /// for use in InductionContext without access to EngineInput inside decide().
    pub(crate) task_class_tags: Vec<String>,
    /// Optional retroactive induction scheduler — set when GAP-F5 is enabled.
    pub(crate) induction_scheduler:
        Option<std::sync::Arc<dyn crate::induction::InductionScheduler>>,
    /// Handle to a running induction task spawned on ZeroSurvival.
    pub(crate) pending_induction:
        Option<tokio::task::JoinHandle<Option<crate::induction::InductionResult>>>,
    /// Count of ZeroSurvival events on the current task class (for min_prior_tasks gate).
    pub(crate) zero_survival_count: u32,
    /// Hint texts that were injected into `retry_context` via `apply_induction_result`.
    /// Used to call `record_success` on the scheduler when the task resolves successfully.
    pub applied_hint_texts: Vec<String>,
}

// ── Cross-wave instability detection ──────────────────────────────────

/// Returns true when the last two compliance scores differ by less than `threshold`
/// AND `retry_count` meets the minimum for reliable detection.
/// This detects MUS oscillation: the model cannot exit the current compliance level
/// through sequential single-constraint repair.
pub fn is_compliance_plateau(history: &[f64], retry_count: u32, threshold: f64) -> bool {
    if retry_count < 2 || history.len() < 2 {
        return false;
    }
    let n = history.len();
    (history[n - 1] - history[n - 2]).abs() < threshold
}

/// Returns true when:
/// 1. ≥2 partials exist
/// 2. The union of their passed check indices covers `total_checks`
/// 3. No single partial covers all `total_checks`
///
/// This is the MUS integration failure signature: each constraint can be satisfied
/// in isolation but not jointly in any single proposal.
pub fn has_isolation_evidence(
    partials: &[h2ai_autonomic::repair::PartialPass],
    total_checks: usize,
) -> bool {
    if partials.len() < 2 || total_checks == 0 {
        return false;
    }
    // Condition 3: no single partial covers everything.
    if partials
        .iter()
        .any(|p| p.passed_check_indices().len() == total_checks)
    {
        return false;
    }
    // Condition 2: union of all passed indices covers everything.
    let union: std::collections::HashSet<usize> = partials
        .iter()
        .flat_map(|p| p.passed_check_indices())
        .collect();
    union.len() == total_checks
}

/// Mean word-bag Jaccard between two lists of reason strings.
/// Returns 1.0 when either list is empty (no divergence signal).
pub fn constraint_reasons_jaccard(reasons_a: &[String], reasons_b: &[String]) -> f64 {
    if reasons_a.is_empty() || reasons_b.is_empty() {
        return 1.0;
    }
    let combined_a = reasons_a.join(" ");
    let combined_b = reasons_b.join(" ");
    let bag_a: std::collections::HashSet<&str> = combined_a.split_whitespace().collect();
    let bag_b: std::collections::HashSet<&str> = combined_b.split_whitespace().collect();
    let union = bag_a.union(&bag_b).count();
    if union == 0 {
        return 1.0;
    }
    bag_a.intersection(&bag_b).count() as f64 / union as f64
}

#[derive(Debug)]
pub struct InstabilitySignal {
    pub constraint_id: String,
    pub check_index: usize,
    pub score: f64,
    pub reasons: Vec<String>,
    pub ambiguity_evidence: Vec<String>,
    pub ambiguity_score: f32,
}

// ── GAP-H3: Cost Guard free functions ─────────────────────────────────────────

/// Compute the budget hint suffix for the explorer system prompt.
///
/// Returns `Some(hint)` when:
/// - `cfg.enabled && cfg.budget_prompt_injection_enabled`
/// - `fraction_used ∈ [cfg.budget_injection_warn_fraction, 0.85)` (TALE elasticity paradox guard)
/// - `complexity <= cfg.budget_injection_max_complexity`
pub fn build_budget_hint_if_needed(
    cfg: &h2ai_config::CostGuardConfig,
    tokens_used: u64,
    complexity: u8,
) -> Option<String> {
    if !cfg.enabled || !cfg.budget_prompt_injection_enabled {
        return None;
    }
    if complexity > cfg.budget_injection_max_complexity {
        return None;
    }
    let frac = cfg.fraction_used(tokens_used);
    if frac < cfg.budget_injection_warn_fraction || frac >= 0.85 {
        return None;
    }
    let remaining = cfg.remaining(tokens_used);
    Some(format!(
        "\n\n[Token budget: approximately {} tokens remain for this response. \
         Prioritize the most critical compliance findings. \
         State your conclusion first, then minimum supporting evidence. \
         Omit elaboration, preamble, and redundant citations.]",
        remaining.max(0)
    ))
}

/// Check whether the convergence gate should fire.
///
/// Fires when ALL hold:
/// 1. `cfg.enabled`
/// 2. `wave >= cfg.min_wave`
/// 3. `budget_fraction_used >= cfg.budget_floor_fraction`
/// 4. `n_live >= 1` and `min_score >= cfg.score_floor`
/// 5. `cosine_mean >= cfg.theta_converge`
pub fn check_convergence_gate(
    cfg: &h2ai_config::ConvergenceGateConfig,
    cosine_mean: Option<f64>,
    min_score: f64,
    wave: u32,
    n_live: usize,
    budget_fraction_used: f64,
) -> bool {
    if !cfg.enabled {
        return false;
    }
    if wave < cfg.min_wave {
        return false;
    }
    if budget_fraction_used < cfg.budget_floor_fraction {
        return false;
    }
    if n_live == 0 {
        return false;
    }
    if min_score < cfg.score_floor {
        return false;
    }
    let Some(mean) = cosine_mean else {
        return false;
    };
    mean >= cfg.theta_converge
}

/// Return the best available hint for a passing constraint's pin.
///
/// Prefers the dynamic verifier reason from the global-best proposal (grounded in
/// what the verifier actually accepted) over the static corpus hint (which is generic).
/// Falls back to corpus hint when no dynamic reason exists. Returns `None` when neither is available.
#[must_use]
pub fn build_best_passing_pin_hint(
    constraint_id: &str,
    dynamic_reasons: &std::collections::HashMap<String, String>,
    corpus_hint: Option<String>,
) -> Option<String> {
    if let Some(dynamic) = dynamic_reasons.get(constraint_id) {
        if !dynamic.is_empty() {
            return Some(dynamic.clone());
        }
    }
    corpus_hint.filter(|s| !s.is_empty())
}

impl MapeKController {
    // ── Constructor ────────────────────────────────────────────────────────────

    /// Build the controller from the engine input and pre-loop phase outputs.
    ///
    /// Reads the bandit asynchronously so `new` is `async`.
    pub async fn new(
        input: &crate::engine::EngineInput<'_>,
        bootstrap_out: &crate::phases::bootstrap::Output,
        complexity_out: &crate::phases::complexity::Output,
        conflict_graph: h2ai_constraints::conflict::ConstraintConflictGraph,
    ) -> Self {
        let task_id = input.task_id.clone();
        let assessed_quadrant = complexity_out.assessed_quadrant;
        let complexity_event = complexity_out.complexity_event.clone();
        let cg_mean = complexity_out.cg_mean;
        let n_max_ceiling = complexity_out.n_max_ceiling;

        let manifest_count = input.manifest.explorers.count as u32;
        let n_optimal_hint = input
            .calibration
            .ensemble
            .as_ref()
            .map_or(manifest_count, |ec| {
                (ec.n_optimal as u32).min(manifest_count)
            });

        let bandit_n = if let Some(ref bandit_arc) = input.bandit_state {
            let bandit = bandit_arc.read().await;
            Some(bandit.select(input.cfg))
        } else {
            None
        };
        let initial_n_agents = bandit_n
            .unwrap_or(n_optimal_hint)
            .max(1)
            .min(n_max_ceiling.max(1));

        let current_params = OptimizerParams {
            n_agents: initial_n_agents,
            max_turns: u32::from(input.tao_config.max_turns),
            verify_threshold: input.verification_config.threshold,
        };

        let task_deadline = input
            .cfg
            .task_deadline_secs
            .map(|s| std::time::Instant::now() + std::time::Duration::from_secs(s));

        Self {
            current_params,
            quality_history: Vec::new(),
            n_max_ceiling,
            cg_mean,
            force_topology: None,
            tried_topologies: Vec::new(),
            tau_reduction_factor: 1.0,
            tau_spread_factor: 1.0,
            tau_values_tried: Vec::new(),
            retry_context: None,
            adapter_rotation_offset: 0,
            mode_collapse_count: 0,
            last_multiplication_failure: None,
            pending_tombstone: None,
            system_context_with_rubric: bootstrap_out.system_context_with_rubric.clone(),
            max_retries: input.cfg.max_autonomic_retries as usize,
            tao_config: input.tao_config.clone(),
            verification_config: input.verification_config.clone(),
            all_verification_events: Vec::new(),
            all_failed_proposals: Vec::new(),
            all_shadow_audit_events: Vec::new(),
            all_correlated_warnings: Vec::new(),
            all_researcher_grounding_events: Vec::new(),
            all_pruned: Vec::new(),
            last_wave_pruned: Vec::new(),
            prev_wave_pruned: Vec::new(),
            topology_retry_events: Vec::new(),
            task_id,
            assessed_quadrant,
            complexity_event,
            diversity_degraded_event: None,
            bandit_state: input.bandit_state.clone(),
            tao_estimator: std::sync::Arc::clone(&input.tao_estimator),
            tao_multiplier: input.tao_multiplier,
            calibration_ensemble: input.calibration.ensemble.clone(),
            cfg_ref: std::sync::Arc::new(input.cfg.clone()),
            task_deadline,
            leader: None,
            last_wave_verification_events: Vec::new(),
            last_wave_proposal_texts: std::collections::HashMap::new(),
            pending_leader_elected_events: Vec::new(),
            pending_socratic_diagnosis_events: Vec::new(),
            ambiguity_scorecards: h2ai_constraints::ambiguity::seed_scorecards(
                &input.constraint_corpus,
                &input.cfg.ambiguity_detection,
            ),
            pending_ambiguity_events: Vec::new(),
            last_wave_violated_constraint_ids: Vec::new(),
            prev_assembled_contexts: Vec::new(),
            global_best_proposal: None,
            global_best_partial_proposal: None,
            global_best_constraint_reasons: std::collections::HashMap::new(),
            compliance_score_history: Vec::new(),
            conflict_graph,
            binary_checks: input
                .constraint_corpus
                .iter()
                .flat_map(|d| d.binary_checks.iter().cloned())
                .collect(),
            constraint_check_offsets: {
                let mut offsets = Vec::new();
                let mut start = 0usize;
                for doc in input.constraint_corpus.iter() {
                    let count = doc.binary_checks.len();
                    if count > 0 {
                        offsets.push((doc.id.clone(), start, count));
                        start += count;
                    }
                }
                offsets
            },
            domain_synthesis_cache: std::collections::HashMap::new(),
            gap_researcher_adapter: input.researcher_adapter.clone(),
            gap_grounding_chain: None, // wired by engine.rs after controller construction
            constraint_pass_map: input
                .constraint_corpus
                .iter()
                .map(|d| {
                    (
                        d.id.clone(),
                        ConstraintPassEntry {
                            pass_criteria: d.pass_criteria.clone(),
                            remediation_hint: d.remediation_hint.clone(),
                        },
                    )
                })
                .collect(),
            probe_result: None,
            corpus_synthesis_viable: input
                .constraint_corpus
                .iter()
                .any(|d| !d.binary_checks.is_empty()),
            per_constraint_wave_scores: std::collections::HashMap::new(),
            verifier_reason_history: std::collections::HashMap::new(),
            bypassed_verifier_constraints: std::collections::HashSet::new(),
            pending_frozen_verifier_events: vec![],
            last_wave_n_eff: 1.0,
            tee_event: None,
            tokens_used: 0,
            budget_exhausted: false,
            convergence_gate_event: None,
            task_class_tags: input.manifest.constraint_tags.clone(),
            induction_scheduler: None,
            pending_induction: None,
            zero_survival_count: 0,
            applied_hint_texts: vec![],
        }
    }

    // ── Snapshot ───────────────────────────────────────────────────────────────

    /// Return an immutable snapshot of the current MAPE-K parameters for one wave.
    #[must_use]
    pub fn params(&self) -> PipelineParams {
        PipelineParams {
            optimizer: self.current_params.clone(),
            force_topology: self.force_topology.clone(),
            tau_reduction_factor: self.tau_reduction_factor,
            tau_spread_factor: self.tau_spread_factor,
            adapter_rotation_offset: self.adapter_rotation_offset,
            retry_context: self.retry_context.clone(),
            tao_config: self.tao_config.clone(),
            verification_config: self.verification_config.clone(),
            pending_tombstone: self.pending_tombstone.clone(),
            leader_context: self
                .leader
                .as_ref()
                .map(|ls| ls.to_snapshot(self.last_wave_violated_constraint_ids.clone())),
            prev_assembled_contexts: self.prev_assembled_contexts.clone(),
            budget_hint: {
                let complexity = self.probe_result.as_ref().map_or(0, |p| p.complexity);
                build_budget_hint_if_needed(&self.cfg_ref.cost_guard, self.tokens_used, complexity)
            },
            bypassed_constraint_ids: self.bypassed_verifier_constraints.clone(),
        }
    }

    // ── GAP-F5: Retroactive Induction Scheduler builder ──────────────────────

    /// Attach a retroactive induction scheduler. Must be called after construction.
    pub fn with_induction_scheduler(
        mut self,
        scheduler: std::sync::Arc<dyn crate::induction::InductionScheduler>,
    ) -> Self {
        self.induction_scheduler = Some(scheduler);
        self
    }

    // ── WaveContinue signal injection ────────────────────────────────

    /// Apply operator-supplied grounding and mandate override from a `WaveContinue` signal.
    ///
    /// `grounding` is appended to `retry_context` so the next wave's context assembler
    /// receives it as additional repair guidance. `mandate_override`, when present, is
    /// prepended with a label and appended to the same context field — it will appear in
    /// the slot context alongside the original mandate.
    pub fn inject_wave_continue(
        &mut self,
        grounding: Option<String>,
        mandate_override: Option<String>,
    ) {
        let mut parts: Vec<String> = Vec::new();
        if let Some(g) = grounding {
            if !g.trim().is_empty() {
                parts.push(g);
            }
        }
        if let Some(m) = mandate_override {
            if !m.trim().is_empty() {
                parts.push(format!("MANDATE OVERRIDE: {m}"));
            }
        }
        if parts.is_empty() {
            return;
        }
        let injection = parts.join("\n");
        self.retry_context = Some(match self.retry_context.take() {
            Some(existing) => format!("{existing}\n{injection}"),
            None => injection,
        });
    }

    // ── Complexity-Ceiling Routing ────────────────────────────────────

    /// Return N_eff (participation ratio cosine) from the most recent `ZeroSurvival` wave.
    /// Returns `1.0` before any ZeroSurvival wave has been processed (no-dropout default).
    /// Used by engine.rs for AgentDropout N-reduction on retry ≥ 2.
    #[must_use]
    pub fn last_wave_n_eff(&self) -> f64 {
        self.last_wave_n_eff
    }

    /// Returns and clears the pending `TieredExitEvent`, if any.
    pub(crate) fn take_tee_event(&mut self) -> Option<h2ai_types::events::TieredExitEvent> {
        self.tee_event.take()
    }

    // ── GAP-H3: Cost Guard accessors ──────────────────────────────────────────

    /// Charge `wave_token_cost` to the per-task token counter.
    pub fn observe_wave_tokens(&mut self, wave_token_cost: u64) {
        self.tokens_used = self.tokens_used.saturating_add(wave_token_cost);
    }

    /// Current cumulative token usage.
    pub fn tokens_used(&self) -> u64 {
        self.tokens_used
    }

    /// Take the convergence gate event if one was set, leaving `None` in its place.
    pub(crate) fn take_convergence_event(
        &mut self,
    ) -> Option<h2ai_types::events::ConvergenceGateEvent> {
        self.convergence_gate_event.take()
    }

    /// Take and reset the budget_exhausted flag.
    pub fn take_budget_exhausted(&mut self) -> bool {
        std::mem::replace(&mut self.budget_exhausted, false)
    }

    /// Mark the pipeline as OOM-aborted, routing to the BudgetExhausted exit path.
    pub fn mark_oom_abort(&mut self) {
        self.budget_exhausted = true;
    }

    /// Store the pre-dispatch complexity probe result for use in routing decisions.
    pub fn set_probe_result(
        &mut self,
        result: h2ai_autonomic::complexity_probe::ComplexityProbeResult,
    ) {
        self.probe_result = Some(result);
    }

    // ── Observe ────────────────────────────────────────────────────────────────

    /// Aggregate events from a completed wave into the cross-wave accumulators.
    pub async fn observe(&mut self, wave: &PipelineWaveResult, wave_index: u32) {
        // GAP-F5: consume a completed induction task if one is pending.
        // Awaits up to grace_period_ms before proceeding — spec-compliant bounded wait.
        // If the induction hasn't finished within the grace period it is dropped.
        // The result is applied after grounding retry_context update so the
        // induction hint appends on top of (rather than underneath) grounding output.
        let pending_induction_result: Option<crate::induction::InductionResult> =
            if let Some(handle) = self.pending_induction.take() {
                let grace_ms = self.cfg_ref.induction_trigger.grace_period_ms;
                if let Ok(Ok(Some(result))) =
                    tokio::time::timeout(std::time::Duration::from_millis(grace_ms), handle).await
                {
                    let current_tags = self.task_class_tags.clone();
                    if result.is_compatible_with(&current_tags) {
                        Some(result)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

        let e = &wave.events;
        self.all_verification_events
            .extend(e.verification_events.iter().cloned());
        self.all_failed_proposals
            .extend(e.failed_proposals.iter().cloned());
        self.all_shadow_audit_events
            .extend(e.shadow_audit_events.iter().cloned());
        self.all_correlated_warnings
            .extend(e.correlated_warnings.iter().cloned());
        self.all_researcher_grounding_events
            .extend(e.researcher_grounding_events.iter().cloned());
        if let Some(ref qm) = e.quality_measurement {
            self.quality_history.push(qm.clone());
        }
        if let Some(ref tf) = e.talagrand_feedback {
            self.tau_spread_factor = tf.tau_spread_next;
        }
        if let Some(ref retry_ev) = e.topology_retry_event {
            self.topology_retry_events.push(retry_ev.clone());
        }
        // GAP-F5: apply induction hint after grounding so it appends on top.
        if let Some(result) = pending_induction_result {
            self.apply_induction_result(&result);
        }
        // Rotate: prev_wave_pruned ← last_wave_pruned before overwriting with new wave.
        self.prev_wave_pruned = std::mem::take(&mut self.last_wave_pruned);
        let annotated_pruned: Vec<h2ai_types::events::BranchPrunedEvent> = e
            .pruned_events
            .iter()
            .map(|ev| h2ai_types::events::BranchPrunedEvent {
                retry_count: wave_index,
                ..ev.clone()
            })
            .collect();
        self.last_wave_pruned = annotated_pruned.clone();
        self.all_pruned.extend(annotated_pruned);
        // Epistemic leader: snapshot last-wave verification events and proposal texts.
        self.last_wave_verification_events = e.verification_events.clone();
        self.last_wave_proposal_texts = e.wave_proposal_texts.clone();
        // CSPR-v2: update cross-wave global-best proposal and dynamic per-constraint reasons.
        for (explorer_id, text) in &e.wave_proposal_texts {
            if text.is_empty() {
                continue;
            }
            if let Some(ev) = e
                .verification_events
                .iter()
                .find(|ev| &ev.explorer_id == explorer_id)
            {
                // Track the best proposal regardless of pass status as a zero-pass fallback.
                // When all wave proposals fail the hard gate, global_best_partial_proposal
                // provides a concrete repair basis for wave 2 via CSPR.
                let is_better_partial = self
                    .global_best_partial_proposal
                    .as_ref()
                    .is_none_or(|(best_score, _)| ev.score > *best_score);
                if is_better_partial {
                    self.global_best_partial_proposal = Some((ev.score, text.clone()));
                }

                // Only update the passing-proposal tracker and constraint reasons when ev.passed.
                if ev.passed {
                    let is_better = self
                        .global_best_proposal
                        .as_ref()
                        .is_none_or(|(best_score, _)| ev.score > *best_score);
                    if is_better {
                        self.global_best_proposal = Some((ev.score, text.clone()));
                        // Update dynamic per-constraint reasons from this wave's best passing proposal.
                        // The wave's best_passing_constraint_reasons approximates this proposal's
                        // compliance evidence — both come from the highest-scoring passing proposal.
                        if !e.best_passing_constraint_reasons.is_empty() {
                            self.global_best_constraint_reasons =
                                e.best_passing_constraint_reasons.clone();
                        }
                    }
                }
            }
        }
        // Record wave mean compliance score for plateau detection.
        let wave_events = &e.verification_events;
        if !wave_events.is_empty() {
            let mean =
                wave_events.iter().map(|ev| ev.score).sum::<f64>() / wave_events.len() as f64;
            self.compliance_score_history.push(mean);
        }
        // Populate per-constraint wave scores from this wave's pruned events.
        {
            let mut wave_constraint_scores: std::collections::HashMap<String, Vec<f64>> =
                std::collections::HashMap::new();
            for pruned_ev in &self.last_wave_pruned {
                for violation in &pruned_ev.violated_constraints {
                    wave_constraint_scores
                        .entry(violation.constraint_id.clone())
                        .or_default()
                        .push(violation.score);
                }
            }
            let window_size = self.cfg_ref.verifier_freeze.reason_window_size as usize;
            for (cid, scores) in wave_constraint_scores {
                let mean = scores.iter().sum::<f64>() / scores.len() as f64;
                self.per_constraint_wave_scores
                    .entry(cid.clone())
                    .or_default()
                    .push(mean);
                // Push most recent verifier reason into rolling window.
                if let Some(last_reason) = self
                    .last_wave_pruned
                    .iter()
                    .flat_map(|p| p.violated_constraints.iter())
                    .filter(|v| v.constraint_id == cid)
                    .filter_map(|v| v.verifier_reason.as_deref())
                    .next_back()
                {
                    let deque = self.verifier_reason_history.entry(cid).or_default();
                    if deque.len() >= window_size {
                        deque.pop_front();
                    }
                    deque.push_back(last_reason.to_string());
                }
            }
        }
        // Gap quality: update post-injection pass rates and evict ineffective syntheses.
        if self.cfg_ref.gap_i1.enabled {
            let gap_cfg = self.cfg_ref.gap_quality.clone();
            let constraint_scores = self.per_constraint_wave_scores.clone();
            let keys_to_evict: Vec<(String, usize)> = self
                .domain_synthesis_cache
                .iter_mut()
                .filter_map(|(key, synth)| {
                    synth.injected_at_wave?;
                    let post_rate = constraint_scores
                        .get(&key.0)
                        .and_then(|scores| scores.last().copied())
                        .unwrap_or(0.0);
                    synth.post_injection_pass_rates.push(post_rate);
                    let verdict = h2ai_autonomic::repair::assess_gap_quality(synth, &gap_cfg);
                    if matches!(
                        verdict,
                        h2ai_autonomic::repair::GapQualityVerdict::Ineffective
                    ) {
                        Some(key.clone())
                    } else {
                        None
                    }
                })
                .collect();
            for key in keys_to_evict {
                tracing::info!(
                    target: "h2ai.gap_i1",
                    constraint_id = %key.0,
                    check_idx = key.1,
                    "evicting ineffective gap synthesis"
                );
                self.domain_synthesis_cache.remove(&key);
            }
        }
        self.last_wave_violated_constraint_ids = self
            .last_wave_pruned
            .iter()
            .flat_map(|p| {
                p.violated_constraints
                    .iter()
                    .map(|v| v.constraint_id.clone())
            })
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        // Capture assembled contexts from this wave for use as prev_assembled_contexts
        // in the next wave's generation phase.
        self.prev_assembled_contexts = e.assembled_contexts.clone();
    }

    // ── Decide ─────────────────────────────────────────────────────────────────

    /// Evaluate the wave outcome and return the MAPE-K decision.
    ///
    /// `retry_count` is the current loop iteration (0-based) used when building
    /// the constraint-feedback hint string.  `filter_ratio` is the wave's
    /// verification pass rate; it is forwarded to `run_apply_optimizer`.
    pub fn decide(
        &mut self,
        outcome: PipelineOutcome,
        retry_count: u32,
        filter_ratio: f64,
    ) -> MapeKDecision {
        // Detect constraint spec instability across waves
        if self.cfg_ref.gap_k1.enabled {
            if let Some(instability) = self.find_instability(retry_count) {
                return MapeKDecision::SpecAmbiguous {
                    constraint_id: instability.constraint_id,
                    check_index: instability.check_index,
                    instability_score: instability.score,
                    divergent_reasons: instability.reasons,
                    ambiguity_evidence: instability.ambiguity_evidence,
                    ambiguity_score: instability.ambiguity_score,
                };
            }
        }

        // Frozen verifier detection (pipeline resilience).
        {
            let freeze_cfg = self.cfg_ref.verifier_freeze.clone();
            if freeze_cfg.enabled && retry_count >= freeze_cfg.min_waves_to_detect {
                // Snapshot scores and reasons to avoid split-borrow issues when
                // mutating bypassed_verifier_constraints inside the loop.
                let scores_snapshot: std::collections::HashMap<String, Vec<f64>> =
                    self.per_constraint_wave_scores.clone();
                let reasons_snapshot: std::collections::HashMap<String, Vec<String>> = self
                    .verifier_reason_history
                    .iter()
                    .map(|(k, dq)| (k.clone(), dq.iter().cloned().collect()))
                    .collect();

                let constraint_ids: Vec<String> = scores_snapshot.keys().cloned().collect();
                for cid in &constraint_ids {
                    if self.bypassed_verifier_constraints.contains(cid) {
                        continue;
                    }
                    let scores = match scores_snapshot.get(cid) {
                        Some(s) => s.as_slice(),
                        None => continue,
                    };
                    // other_trends: all constraints except the one under test, excluding
                    // already-bypassed ones — built fresh per iteration from the snapshot.
                    let other_trends: Vec<&[f64]> = scores_snapshot
                        .iter()
                        .filter(|(other_cid, _)| {
                            *other_cid != cid
                                && !self.bypassed_verifier_constraints.contains(*other_cid)
                        })
                        .map(|(_, s)| s.as_slice())
                        .collect();
                    let reasons: Vec<String> =
                        reasons_snapshot.get(cid).cloned().unwrap_or_default();
                    if let Some(mut signal) = h2ai_autonomic::epistemic::detect_frozen_verifier(
                        cid,
                        scores,
                        &reasons,
                        &other_trends,
                        &freeze_cfg,
                    ) {
                        signal.frozen_since_wave = retry_count;
                        let bypassed =
                            freeze_cfg.bypass_hard_gate_on_freeze && !freeze_cfg.emit_event_only;
                        self.pending_frozen_verifier_events.push(
                            h2ai_types::events::VerifierFrozenEvent {
                                constraint_id: cid.clone(),
                                frozen_since_wave: signal.frozen_since_wave,
                                per_wave_scores: signal.per_wave_scores.clone(),
                                sample_reason: signal.sample_reason.clone(),
                                bypassed,
                            },
                        );
                        if bypassed {
                            tracing::warn!(
                                target: "h2ai.engine",
                                constraint_id = %cid,
                                frozen_since_wave = signal.frozen_since_wave,
                                "verifier frozen: bypassing hard gate for this constraint"
                            );
                            self.bypassed_verifier_constraints.insert(cid.clone());
                        }
                    }
                }
            }
        }

        match outcome {
            PipelineOutcome::Resolved(merge_out) => {
                let merge_out = *merge_out;

                // ── GAP-H3: Budget Exhaustion Gate ───────────────────────────────────
                if self.cfg_ref.cost_guard.enabled
                    && self.cfg_ref.cost_guard.fraction_used(self.tokens_used)
                        >= self.cfg_ref.cost_guard.budget_abort_fraction
                {
                    self.budget_exhausted = true;
                }

                // ── GAP-H3: Convergence Gate ─────────────────────────────────────────
                if self.cfg_ref.convergence_gate.enabled {
                    let min_score = merge_out
                        .iteration_verification_events
                        .iter()
                        .filter(|e| e.passed)
                        .map(|e| e.score)
                        .fold(f64::INFINITY, f64::min);
                    let n_live = merge_out
                        .iteration_verification_events
                        .iter()
                        .filter(|e| e.passed)
                        .count();
                    let budget_fraction = self.cfg_ref.cost_guard.fraction_used(self.tokens_used);
                    if check_convergence_gate(
                        &self.cfg_ref.convergence_gate,
                        merge_out.pairwise_cosine_mean,
                        if min_score.is_infinite() {
                            0.0
                        } else {
                            min_score
                        },
                        retry_count,
                        n_live,
                        budget_fraction,
                    ) {
                        self.convergence_gate_event =
                            Some(h2ai_types::events::ConvergenceGateEvent {
                                task_id: merge_out.task_id.clone(),
                                wave: retry_count,
                                n_live,
                                convergence_fraction: merge_out.pairwise_cosine_mean.unwrap_or(0.0),
                                theta_converge: self.cfg_ref.convergence_gate.theta_converge,
                                best_score: merge_out
                                    .iteration_verification_events
                                    .iter()
                                    .filter(|e| e.passed)
                                    .map(|e| e.score)
                                    .fold(0.0_f64, f64::max),
                                timestamp: chrono::Utc::now(),
                            });
                        // Fall through — convergence gate accepts current output.
                    }
                }

                // ── GAP-L1: TEE acceptance gate ──────────────────────────────────
                if self.cfg_ref.tiered_exit.enabled {
                    let tee = &self.cfg_ref.tiered_exit;
                    let n = self.current_params.n_agents;
                    let k_required = tee.k_for_wave(n);
                    let k_accepted = merge_out
                        .iteration_verification_events
                        .iter()
                        .filter(|e| e.passed && e.score >= tee.acceptance_score)
                        .count() as u32;

                    if k_accepted < k_required
                        && retry_count < self.cfg_ref.max_autonomic_retries
                        && !self.budget_exhausted
                    {
                        return MapeKDecision::Retry;
                    }

                    self.tee_event = Some(h2ai_types::events::TieredExitEvent {
                        wave: retry_count,
                        n,
                        k_required,
                        k_accepted,
                        acceptance_score: tee.acceptance_score,
                    });
                }

                // Push tau_values from the successful wave.
                self.tau_values_tried.push(merge_out.tau_values.clone());
                // Push quality measurement from the merge result.
                self.quality_history.push(QualityMeasurement {
                    params: self.current_params.clone(),
                    q_confidence: merge_out.attribution.q_confidence,
                });
                // Extend cross-wave comparison events.
                MapeKDecision::Return(Box::new(self.finalize(merge_out)))
            }

            PipelineOutcome::Fatal(e) => {
                MapeKDecision::Fail(e, crate::engine::EngineRunContext::default())
            }

            PipelineOutcome::EarlyExit(reason) => {
                self.handle_exit_reason(reason, retry_count, filter_ratio)
            }
        }
    }

    /// Internal: map an `ExitReason` to a `MapeKDecision`.
    fn handle_exit_reason(
        &mut self,
        reason: ExitReason,
        retry_count: u32,
        filter_ratio: f64,
    ) -> MapeKDecision {
        // Probe-based routing — fires before normal retry decisions.
        // HITL path (graft_first=false): fires on any failure when probe >= hitl_threshold.
        // Graft path (graft_first=true): requires BOTH corpus_synthesis_viable AND
        //   retry_count >= min_retries_before_graft so a non-deterministic probe score
        //   cannot bypass the consensus retry guarantee or route to a non-viable path.
        if let Some(ref probe) = self.probe_result {
            let cfg = &self.cfg_ref.complexity_routing;
            if cfg.enabled {
                if probe.complexity >= cfg.hitl_threshold {
                    return MapeKDecision::ComplexityOverflow {
                        probe_score: probe.complexity,
                        rationale: probe.rationale.clone(),
                        graft_first: false,
                    };
                }
                if probe.complexity >= cfg.decompose_threshold
                    && retry_count >= cfg.min_retries_before_graft
                    && self.corpus_synthesis_viable
                {
                    return MapeKDecision::ComplexityOverflow {
                        probe_score: probe.complexity,
                        rationale: probe.rationale.clone(),
                        graft_first: true,
                    };
                }
            }
        }

        match reason {
            ExitReason::MultiplicationFailed {
                msg: _,
                tau_values,
                failure,
            } => {
                tracing::warn!(
                    target: "h2ai.mape_k",
                    failure = ?failure,
                    "multiplication condition failed"
                );
                self.last_multiplication_failure = Some(failure);
                self.tau_values_tried.push(tau_values);

                let zero_event = ZeroSurvivalEvent {
                    task_id: self.task_id.clone(),
                    retry_count,
                    timestamp: chrono::Utc::now(),
                    n_eff_cosine_actual: None,
                    failure_mode: None,
                };
                self.apply_retry_action(
                    RetryPolicy::decide(
                        &zero_event,
                        &self.tried_topologies.clone(),
                        self.all_pruned.clone(),
                        self.tau_values_tried.clone(),
                        self.last_multiplication_failure.clone(),
                    ),
                    retry_count,
                    1.0, // filter_ratio not applicable here
                )
            }

            ExitReason::DiversityFailed { n_eff, tau_values } => {
                self.last_multiplication_failure = Some(
                    h2ai_types::sizing::MultiplicationConditionFailure::InsufficientPoolDiversity {
                        n_eff,
                        threshold: self.cfg_ref.safety.diversity_threshold,
                    },
                );
                self.tau_values_tried.push(tau_values);
                let zero_event = ZeroSurvivalEvent {
                    task_id: self.task_id.clone(),
                    retry_count,
                    timestamp: chrono::Utc::now(),
                    n_eff_cosine_actual: Some(n_eff),
                    failure_mode: Some(h2ai_types::events::FailureMode::ModeCollapse),
                };
                self.apply_retry_action(
                    RetryPolicy::decide(
                        &zero_event,
                        &self.tried_topologies.clone(),
                        self.all_pruned.clone(),
                        self.tau_values_tried.clone(),
                        self.last_multiplication_failure.clone(),
                    ),
                    retry_count,
                    1.0,
                )
            }

            ExitReason::ZeroSurvival {
                failure_mode: detected_failure_mode,
                coherence: _,
                n_eff_cosine: zs_n_eff_cosine,
                filter_ratio: _wave_filter_ratio,
                tau_values: zs_tau_values,
            } => {
                self.tau_values_tried.push(zs_tau_values);

                // Apply FailureMode-specific state mutations before RetryPolicy selection.
                match &detected_failure_mode {
                    Some(h2ai_types::events::FailureMode::ModeCollapse) => {
                        let pool_len = 1usize; // conservative fallback; engine knows the pool length
                        let _ = pool_len;
                        self.mode_collapse_count += 1;
                        self.pending_tombstone = None;
                    }
                    Some(h2ai_types::events::FailureMode::ConstrainedExploration) => {
                        // Use only the immediately preceding wave's violations — not the
                        // full cross-wave accumulator — to avoid feeding the LLM constraint
                        // errors from multiple waves ago that it has already corrected.
                        let wave_violations: Vec<h2ai_types::events::ConstraintViolation> = self
                            .last_wave_pruned
                            .iter()
                            .flat_map(|p| p.violated_constraints.iter().cloned())
                            .collect();
                        self.pending_tombstone =
                            h2ai_autonomic::epistemic::synthesize_tombstone(&wave_violations);
                    }
                    Some(h2ai_types::events::FailureMode::CorrelatedHallucination { .. }) => {
                        // Handled by HallucinationDetected before Phase 3.5.
                    }
                    None => {}
                }

                // Integration wave triggers (GAP-F6): fires before ceiling detector to intercept
                // MUS oscillation that the ceiling detector would otherwise classify as a generic ceiling.
                if self.cfg_ref.integration_wave.enabled {
                    let iw_cfg = &self.cfg_ref.integration_wave;

                    // Trigger 1: Score plateau — last two waves at same compliance level.
                    let plateau = is_compliance_plateau(
                        &self.compliance_score_history,
                        retry_count,
                        iw_cfg.plateau_threshold,
                    );

                    // Trigger 2: Isolation evidence — cross-wave partial proposals cover all checks
                    // in disjoint subsets, proving no single proposal can satisfy all constraints.
                    let partials = h2ai_autonomic::repair::select_orthogonal_partials(
                        &self.all_pruned,
                        &self.binary_checks,
                        &self.constraint_check_offsets,
                        2,
                        h2ai_autonomic::repair::partial_max_chars(
                            self.cfg_ref.model_max_tokens,
                            2,
                            self.cfg_ref.partial_pass_overhead_factor,
                        ),
                    );
                    let isolation = has_isolation_evidence(&partials, self.binary_checks.len());

                    if plateau || isolation {
                        tracing::info!(
                            target: "h2ai.mape_k",
                            task_id = %self.task_id,
                            plateau,
                            isolation,
                            retry_count,
                            "integration wave triggered"
                        );
                        return MapeKDecision::ComplexityOverflow {
                            probe_score: 0,
                            rationale: format!(
                                "integration wave: plateau={plateau} isolation={isolation} at wave {retry_count}"
                            ),
                            graft_first: true,
                        };
                    }
                }

                // Intra-retry ceiling detector.
                // Fires when probe-based routing was disabled or under-classified the
                // task and ≥2/3 ceiling signals (peaked failure entropy, stalled
                // retry slope, low n_eff×cg_mean product) have accumulated.
                if self.cfg_ref.complexity_routing.intra_retry.enabled
                    && retry_count
                        >= self
                            .cfg_ref
                            .complexity_routing
                            .intra_retry
                            .min_retry_count_for_detection
                {
                    let score_history: Vec<f64> = self
                        .quality_history
                        .iter()
                        .map(|qm| qm.q_confidence)
                        .collect();
                    let n_eff = zs_n_eff_cosine.unwrap_or(1.0);
                    let signals = crate::ceiling_detector::count_ceiling_signals(
                        &self.last_wave_pruned,
                        &score_history,
                        n_eff,
                        self.cg_mean,
                        &self.cfg_ref.complexity_routing.intra_retry,
                    );
                    if signals >= 2 {
                        tracing::info!(
                            target: "h2ai.mape_k",
                            task_id = %self.task_id,
                            signals,
                            retry_count,
                            "ceiling detector fired"
                        );
                        return MapeKDecision::ComplexityOverflow {
                            probe_score: 0,
                            rationale: format!(
                                "intra-retry: {signals}/3 ceiling signals fired at wave {retry_count}"
                            ),
                            graft_first: retry_count < 2,
                        };
                    }
                }

                // Record N_eff for AgentDropout dropout decisions on the next wave.
                self.last_wave_n_eff = zs_n_eff_cosine.unwrap_or(1.0);

                // GAP-F5: Retroactive induction trigger.
                self.zero_survival_count += 1;
                if let Some(ref scheduler) = self.induction_scheduler {
                    let cfg = &self.cfg_ref.induction_trigger;
                    if cfg.enabled && self.zero_survival_count >= cfg.min_prior_tasks {
                        let sched = std::sync::Arc::clone(scheduler);
                        let ctx = crate::induction::InductionContext {
                            tenant_id: self.task_id.to_string(),
                            task_class_tags: self.task_class_tags.clone(),
                            violated_constraint_ids: self
                                .all_violated_constraint_ids()
                                .into_iter()
                                .take(5)
                                .collect(),
                        };
                        self.pending_induction =
                            Some(tokio::spawn(
                                async move { sched.run_retroactive(&ctx).await },
                            ));
                    }
                }

                let zero_event = ZeroSurvivalEvent {
                    task_id: self.task_id.clone(),
                    retry_count,
                    timestamp: chrono::Utc::now(),
                    n_eff_cosine_actual: zs_n_eff_cosine,
                    failure_mode: detected_failure_mode,
                };

                self.apply_retry_action(
                    RetryPolicy::decide(
                        &zero_event,
                        &self.tried_topologies.clone(),
                        self.all_pruned.clone(),
                        self.tau_values_tried.clone(),
                        self.last_multiplication_failure.clone(),
                    ),
                    retry_count,
                    filter_ratio,
                )
            }

            ExitReason::HallucinationDetected {
                retry_context_hint,
                tau_values,
                warning,
                researcher_grounding_events,
            } => {
                self.all_correlated_warnings.push(warning);
                self.all_researcher_grounding_events
                    .extend(researcher_grounding_events);
                self.retry_context = Some(retry_context_hint);
                self.tau_values_tried.push(tau_values);
                self.run_apply_optimizer(1.0);
                MapeKDecision::Retry
            }

            ExitReason::OraclePostSelectionBlocked {
                evicted_winner_summary,
            } => {
                tracing::warn!(
                    target: "h2ai.oracle",
                    task_id = %self.task_id,
                    "oracle post-selection gate rejected winner — rotating adapter family and retrying"
                );
                let warning = h2ai_types::events::CorrelatedEnsembleWarning {
                    task_id: self.task_id.clone(),
                    cv: 1.0,
                    mean_jaccard_distance: 0.0,
                    retry_count,
                };
                self.all_correlated_warnings.push(warning);
                self.adapter_rotation_offset = self.adapter_rotation_offset.wrapping_add(1);
                self.retry_context = Some(evicted_winner_summary);
                self.run_apply_optimizer(1.0);
                MapeKDecision::Retry
            }

            ExitReason::OracleBlocked => {
                MapeKDecision::Fail(EngineError::MaxRetriesExhausted, self.take_run_context())
            }
        }
    }

    /// Map a `RetryAction` to a `MapeKDecision`, mutating controller state.
    fn apply_retry_action(
        &mut self,
        action: RetryAction,
        retry_count: u32,
        filter_ratio: f64,
    ) -> MapeKDecision {
        match action {
            RetryAction::Retry(next_topology) => {
                self.force_topology = Some(next_topology);
                self.run_apply_optimizer(filter_ratio);
                MapeKDecision::Retry
            }
            RetryAction::RetryWithTauReduction {
                topology,
                tau_factor,
            } => {
                self.force_topology = Some(topology);
                self.tau_reduction_factor *= tau_factor;
                self.run_apply_optimizer(filter_ratio);
                MapeKDecision::Retry
            }
            RetryAction::RetryWithHints { topology, hints } => {
                self.force_topology = Some(topology);
                if !hints.is_empty() {
                    let attempts_remaining = (self.max_retries as u32).saturating_sub(retry_count);
                    // Legacy hint-only format (no RepairTarget metadata available).
                    let hint_lines = hints
                        .iter()
                        .map(|h| format!("• {h}"))
                        .collect::<Vec<_>>()
                        .join("\n\n");
                    self.retry_context = Some(format!(
                        "{ctx}\n\n--- CONSTRAINT FEEDBACK (iteration {retry_count}) ---\n\
                        The following constraints were violated. Fix ALL of these in your next response:\n\n\
                        {hint_lines}\n\n\
                        {attempts_remaining} retry attempt(s) remaining.\n\
                        ---",
                        ctx = self.system_context_with_rubric
                    ));
                }
                self.run_apply_optimizer(filter_ratio);
                MapeKDecision::Retry
            }
            RetryAction::RetryWithTargets { topology, targets } => {
                self.force_topology = Some(topology);
                if !targets.is_empty() {
                    let attempts_remaining = (self.max_retries as u32).saturating_sub(retry_count);
                    let use_cspr = self.cfg_ref.cspr.enabled
                        && (self.global_best_proposal.is_some()
                            || self.global_best_partial_proposal.is_some());
                    let prior_text = if use_cspr {
                        // Prefer the global-best *passing* proposal; fall back to the best
                        // partial (failing) proposal when all waves so far produced zero passes.
                        // This prevents wave 2 from starting cold in hard zero-pass scenarios
                        // like multi-hard-constraint tasks where the model needs a concrete
                        // repair basis rather than an empty prior.
                        self.global_best_proposal
                            .as_ref()
                            .or(self.global_best_partial_proposal.as_ref())
                            .map(|(_, t)| t.as_str())
                            .unwrap_or("")
                    } else {
                        ""
                    };

                    let partial_passes = h2ai_autonomic::repair::select_orthogonal_partials(
                        &self.all_pruned,
                        &self.binary_checks,
                        &self.constraint_check_offsets,
                        2,
                        h2ai_autonomic::repair::partial_max_chars(
                            self.cfg_ref.model_max_tokens,
                            2,
                            self.cfg_ref.partial_pass_overhead_factor,
                        ),
                    );
                    // Collect any cached domain syntheses for the violated constraints.
                    let syntheses: Vec<h2ai_types::gap_i1::DomainSynthesis> = targets
                        .iter()
                        .flat_map(|t| {
                            self.domain_synthesis_cache
                                .iter()
                                .filter(|((cid, _), _)| cid == &t.constraint_id)
                                .map(|(_, s)| s.clone())
                                .collect::<Vec<_>>()
                        })
                        .collect();
                    let failing_ids: std::collections::HashSet<&str> =
                        targets.iter().map(|t| t.constraint_id.as_str()).collect();
                    let coupled_hints: Vec<(String, Option<String>)> = targets
                        .iter()
                        .flat_map(|t| self.conflict_graph.conflicts_for(&t.constraint_id))
                        .filter(|id| !failing_ids.contains(*id))
                        .map(|id| {
                            let hint = build_best_passing_pin_hint(
                                id,
                                &self.global_best_constraint_reasons,
                                self.corpus_pass_hint_for(id),
                            );
                            (id.to_owned(), hint)
                        })
                        .collect();
                    // All constraints that passed in the global-best proposal —
                    // complement of the failing targets. Used as preservation pins.
                    let passing_pins: Vec<(String, Option<String>)> = self
                        .constraint_check_offsets
                        .iter()
                        .map(|(id, _, _)| id.as_str())
                        .filter(|id| !failing_ids.contains(*id))
                        .map(|id| {
                            let hint = build_best_passing_pin_hint(
                                id,
                                &self.global_best_constraint_reasons,
                                self.corpus_pass_hint_for(id),
                            );
                            (id.to_owned(), hint)
                        })
                        .collect();
                    self.retry_context = Some(h2ai_autonomic::repair::build_repair_context(
                        h2ai_autonomic::repair::RepairInput {
                            prior_proposal_text: prior_text,
                            targets: &targets,
                            zone3_hints: None,
                            conflict_graph: &self.conflict_graph,
                            retry_count,
                            attempts_remaining,
                            system_context_with_rubric: &self.system_context_with_rubric,
                            checks: &self.binary_checks,
                            partial_passes: &partial_passes,
                            prior_best_score: self
                                .global_best_proposal
                                .as_ref()
                                .or(self.global_best_partial_proposal.as_ref())
                                .map(|(score, _)| *score),
                            domain_syntheses: &syntheses,
                            coupled_constraint_hints: &coupled_hints,
                            passing_constraint_pins: &passing_pins,
                        },
                    ));
                }
                self.run_apply_optimizer(filter_ratio);
                MapeKDecision::Retry
            }
            RetryAction::Fail(reason) => {
                // When DPPM is enabled and the corpus has binary checks, route to
                // the integration wave instead of hard-failing. FRONTIER exhaustion
                // is the unconditional DPPM trigger when the intra-retry detector
                // didn't fire (e.g. too few partials or no isolation evidence yet).
                if self.corpus_synthesis_viable && self.cfg_ref.dppm.enabled {
                    tracing::info!(
                        target: "h2ai.mape_k",
                        task_id = %self.task_id,
                        retry_count,
                        "retry frontier exhausted with dppm enabled — routing to integration wave"
                    );
                    return MapeKDecision::ComplexityOverflow {
                        probe_score: 0,
                        rationale: "frontier exhausted; DPPM-MetaRefine synthesis triggered".into(),
                        graft_first: true,
                    };
                }
                tracing::warn!(
                    target: "h2ai.mape_k",
                    task_id = %self.task_id,
                    retry_count,
                    reason = ?reason,
                    "retry policy decided Fail — giving up"
                );
                MapeKDecision::Fail(EngineError::MaxRetriesExhausted, self.take_run_context())
            }
        }
    }

    // ── Finalize ───────────────────────────────────────────────────────────────

    /// Assemble the final `EngineOutput` from a successful merge result and the
    /// cross-wave accumulators held in the controller.
    pub fn finalize(&mut self, merge_out: MergeOutput) -> EngineOutput {
        let run_scores: Vec<f64> = self
            .all_verification_events
            .iter()
            .map(|e| e.score)
            .collect();
        let talagrand = TalagrandDiagnostic::from_verification_scores(&[run_scores]);
        EngineOutput {
            task_id: merge_out.task_id,
            resolved_output: merge_out.resolved_output,
            selection_resolved: merge_out.selection_resolved_event,
            attribution: merge_out.attribution,
            attribution_interval: merge_out.attribution_interval,
            verification_events: self.all_verification_events.clone(),
            failed_proposals: self.all_failed_proposals.clone(),
            talagrand,
            suggested_next_params: merge_out.suggested_next_params,
            waste_ratio: merge_out.waste_ratio,
            applied_optimizations: merge_out.applied_optimizations,
            topology_retry_events: self.topology_retry_events.clone(),
            mode_collapse_count: self.mode_collapse_count,
            epistemic_yield: merge_out.epistemic_yield,
            provenance_map: None,
            task_quadrant: Some(self.assessed_quadrant),
            complexity_event: self.complexity_event.clone(),
            frontier_event: merge_out.frontier_event,
            adapter_correctness: merge_out.adapter_correctness,
            coherence_state: merge_out.coherence_state,
            comparison_events: merge_out.comparison_events,
            shadow_audit_events: self.all_shadow_audit_events.clone(),
            correlated_warnings: self.all_correlated_warnings.clone(),
            researcher_grounding_events: self.all_researcher_grounding_events.clone(),
            diversity_degraded_event: self.diversity_degraded_event.clone(),
            oracle_gate_passed: merge_out.oracle_gate_passed,
            leader_elected_events: std::mem::take(&mut self.pending_leader_elected_events),
            socratic_diagnosis_events: std::mem::take(&mut self.pending_socratic_diagnosis_events),
            consensus_agreement_rate: None,
            tokens_used: self.tokens_used,
        }
    }

    // ── Self-Optimizer ─────────────────────────────────────────────────────────

    /// Update `current_params`, `tao_config`, and `verification_config` via `SelfOptimizer`.
    pub fn run_apply_optimizer(&mut self, filter_ratio: f64) {
        let (p_mean, rho_mean) = match &self.calibration_ensemble {
            Some(ec) => (ec.p_mean, ec.rho_mean),
            None => (
                0.5 + self.cg_mean / 2.0,
                (1.0 - self.cg_mean).clamp(0.0, 1.0),
            ),
        };
        let n_optimal = self
            .calibration_ensemble
            .as_ref()
            .map(|ec| ec.n_optimal as u32);
        let suggested = SelfOptimizer::suggest(SuggestInput {
            current: &self.current_params,
            history: &self.quality_history,
            n_max_ceiling: self.n_max_ceiling,
            n_optimal,
            p_mean,
            rho_mean,
            filter_ratio,
            cfg: &self.cfg_ref,
        });
        if suggested.max_turns != self.current_params.max_turns {
            self.tao_config.max_turns = suggested.max_turns as u8;
        }
        if (suggested.verify_threshold - self.current_params.verify_threshold).abs() > 1e-9 {
            self.verification_config.threshold = suggested.verify_threshold;
        }
        self.current_params = suggested;
    }

    // ── Epistemic Leader ───────────────────────────────────────────────────────

    /// Compute a `LeaderElectionPlan` from the last wave's verification events.
    /// Returns `None` when no verification events are available yet.
    #[must_use]
    pub fn prepare_leader_election(
        &self,
        cfg: &h2ai_config::H2AIConfig,
    ) -> Option<crate::leader::LeaderElectionPlan> {
        use crate::leader::{
            assign_follower_aspects, select_best_and_runner_up, should_rotate, update_credibility,
        };

        if self.last_wave_verification_events.is_empty() {
            return None;
        }

        let scores: Vec<(h2ai_types::identity::ExplorerId, f64)> = self
            .last_wave_verification_events
            .iter()
            .map(|e| (e.explorer_id.clone(), e.score))
            .collect();

        let (winner_id, runner_up_id) = select_best_and_runner_up(&scores)?;

        let do_rotate = match &self.leader {
            Some(ls) => should_rotate(
                ls,
                cfg.leader_stagnation_threshold,
                cfg.leader_stagnation_waves,
            ),
            None => false,
        };

        let leader_id = if do_rotate {
            runner_up_id.clone().unwrap_or_else(|| winner_id.clone())
        } else {
            winner_id
        };

        let prior_proposal = self
            .last_wave_proposal_texts
            .get(&leader_id)
            .cloned()
            .unwrap_or_default();

        let violated_ids = self.last_wave_violated_constraint_ids.clone();

        let n_followers = scores.len().saturating_sub(1);
        let follower_aspects = assign_follower_aspects(&violated_ids, n_followers);

        let q_confidence = scores
            .iter()
            .find(|(id, _)| *id == leader_id)
            .map_or(0.0, |(_, s)| *s);

        let term = self.leader.as_ref().map_or(1, |ls| ls.term + 1);

        let existing_credibility = if do_rotate {
            1.0
        } else {
            match &self.leader {
                Some(ls) => {
                    let improved = self.quality_history.len() >= 2 && {
                        let n = self.quality_history.len();
                        let delta = self.quality_history[n - 1].q_confidence
                            - self.quality_history[n - 2].q_confidence;
                        delta >= cfg.leader_stagnation_threshold
                    };
                    update_credibility(
                        ls.credibility_score,
                        improved,
                        cfg.leader_credibility_decay_rate,
                    )
                }
                None => 1.0,
            }
        };

        let existing_buffer = if do_rotate {
            vec![]
        } else {
            self.leader
                .as_ref()
                .map(|ls| ls.belief_buffer.clone())
                .unwrap_or_default()
        };

        Some(crate::leader::LeaderElectionPlan {
            task_id: self.task_id.clone(),
            term,
            leader_explorer_id: leader_id,
            runner_up_explorer_id: runner_up_id,
            prior_proposal,
            violated_constraint_ids: violated_ids,
            q_confidence,
            should_rotate: do_rotate,
            follower_aspects,
            existing_belief_buffer: existing_buffer,
            existing_credibility,
        })
    }

    /// Apply a completed `LeaderElectionPlan` (after async Socratic diagnosis) to
    /// update `self.leader` and push the corresponding telemetry events.
    pub fn apply_leader_result(
        &mut self,
        plan: crate::leader::LeaderElectionPlan,
        question: String,
        eig_rank: u32,
        dedup_candidates_tried: u32,
        cfg: &h2ai_config::H2AIConfig,
    ) {
        use crate::leader::{fnv1a, BeliefRecord};
        use h2ai_types::events::RotationReason;

        let rotation_reason = if plan.should_rotate {
            Some(RotationReason::Stagnation)
        } else {
            None
        };

        let elected_ev = h2ai_types::events::LeaderElectedEvent {
            task_id: plan.task_id.clone(),
            term: plan.term,
            leader_explorer_id: plan.leader_explorer_id.clone(),
            q_confidence: plan.q_confidence,
            credibility_score: plan.existing_credibility,
            rotation_reason,
            timestamp: chrono::Utc::now(),
        };
        self.pending_leader_elected_events.push(elected_ev);

        let diagnosis_ev = h2ai_types::events::SocraticDiagnosisEvent {
            task_id: plan.task_id.clone(),
            term: plan.term,
            question: question.clone(),
            violated_constraints: plan.violated_constraint_ids.clone(),
            eig_rank,
            dedup_candidates_tried,
            timestamp: chrono::Utc::now(),
        };
        self.pending_socratic_diagnosis_events.push(diagnosis_ev);

        let n = self.quality_history.len();
        let improved = n >= 2 && {
            let delta =
                self.quality_history[n - 1].q_confidence - self.quality_history[n - 2].q_confidence;
            delta >= cfg.leader_stagnation_threshold
        };
        let stagnation_count = if plan.should_rotate {
            0
        } else {
            match &self.leader {
                Some(ls) => {
                    if improved {
                        0
                    } else {
                        ls.stagnation_count + 1
                    }
                }
                None => u32::from(!improved),
            }
        };

        let mut belief_buffer = plan.existing_belief_buffer;
        let outcomes: Vec<f64> = self
            .last_wave_verification_events
            .iter()
            .map(|e| e.score)
            .collect();
        belief_buffer.push(BeliefRecord {
            question_hash: fnv1a(&question),
            question_text: question.clone(),
            outcome_scores: outcomes,
        });

        let mut confidence_history = self
            .leader
            .as_ref()
            .map(|ls| ls.confidence_history.clone())
            .unwrap_or_default();
        confidence_history.push(plan.q_confidence);

        self.leader = Some(crate::leader::LeaderState {
            term: plan.term,
            leader_explorer_id: plan.leader_explorer_id,
            prior_proposal: plan.prior_proposal,
            socratic_question: question,
            confidence_history,
            stagnation_count,
            belief_buffer,
            credibility_score: plan.existing_credibility,
            follower_aspects: plan.follower_aspects,
        });
    }

    /// Drain and return the pending leader telemetry events accumulated since the
    /// last call.  Called by `engine.rs` after each wave to publish to the event bus.
    pub fn take_leader_events(
        &mut self,
    ) -> (
        Vec<h2ai_types::events::LeaderElectedEvent>,
        Vec<h2ai_types::events::SocraticDiagnosisEvent>,
    ) {
        (
            std::mem::take(&mut self.pending_leader_elected_events),
            std::mem::take(&mut self.pending_socratic_diagnosis_events),
        )
    }

    // ── Coordinator helpers ────────────────────────────────────────────────────

    /// Returns the task deadline for the coordinator's deadline check.
    #[must_use]
    pub const fn deadline(&self) -> Option<std::time::Instant> {
        self.task_deadline
    }

    /// Returns all verification events collected — used for `MaxRetriesExhausted` error.
    #[must_use]
    pub fn take_verification_events(&self) -> Vec<h2ai_types::events::VerificationScoredEvent> {
        self.all_verification_events.clone()
    }

    /// Snapshot of accumulated run data for the failure path. Called once per failure.
    #[must_use]
    pub fn take_run_context(&self) -> crate::engine::EngineRunContext {
        crate::engine::EngineRunContext {
            verification_events: self.all_verification_events.clone(),
            topology_retry_events: self.topology_retry_events.clone(),
            best_partial_text: self
                .global_best_proposal
                .as_ref()
                .or(self.global_best_partial_proposal.as_ref())
                .map(|(_, text)| text.clone()),
            // violation_freq and last_selection_valid_count are accumulated in engine.rs
            // and injected into the context after take_run_context() returns.
            violation_freq: std::collections::HashMap::new(),
            last_selection_valid_count: None,
        }
    }

    /// Read-only view of all pruned events accumulated across waves.
    /// Used by the synthesis wave to extract the global best partial for HITL fallback.
    #[must_use]
    pub fn all_pruned(&self) -> &[h2ai_types::events::BranchPrunedEvent] {
        &self.all_pruned
    }

    /// Returns the system context with rubric string, for synthesis wave construction.
    #[must_use]
    pub fn system_context_with_rubric(&self) -> &str {
        &self.system_context_with_rubric
    }

    // ── Knowledge-Gap Detection + Domain Synthesis ───────────────────

    /// Returns per-(constraint_id, check_idx) mean pass rate across all accumulated
    /// pruned events.  Entries reflect only constraints that appeared in at least one
    /// `BranchPrunedEvent`; constraints that never violated are absent (pass-rate 1.0
    /// by definition, which is above any cold-check threshold).
    fn wave_check_rates(&self) -> Vec<((String, usize), f64)> {
        let mut rates: std::collections::HashMap<(String, usize), Vec<f64>> = Default::default();
        for pruned in &self.all_pruned {
            for violation in &pruned.violated_constraints {
                let cid = violation.constraint_id.clone();
                if violation.check_verdicts.is_empty() {
                    // No per-check verdicts — mark check 0 as failed.
                    rates.entry((cid, 0)).or_default().push(0.0);
                } else {
                    for (idx, &passed) in violation.check_verdicts.iter().enumerate() {
                        rates
                            .entry((cid.clone(), idx))
                            .or_default()
                            .push(if passed { 1.0 } else { 0.0 });
                    }
                }
            }
        }
        rates
            .into_iter()
            .map(|(k, vals)| {
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                (k, mean)
            })
            .collect()
    }

    /// Return the best corpus-supplied positive hint for a constraint.
    ///
    /// Prefers `pass_criteria` (positive framing) over `remediation_hint` (negative/repair
    /// framing), and returns `None` when neither is present or both are empty strings.
    /// Used by `run_gap_i1_research` to supply a non-tautological `correct_pattern`
    /// when web search is unavailable.
    /// Collect all accumulated gap_i1 domain syntheses for injection into DPPM solver prompts.
    ///
    /// Exposed so engine.rs can pass the full synthesis cache to every cluster solver's
    /// `RepairInput.domain_syntheses` — without this, DPPM solvers never see the
    /// incorrect→correct pattern corrections that gap researchers accumulated during
    /// prior retry waves.
    pub fn all_domain_syntheses(&self) -> Vec<h2ai_types::gap_i1::DomainSynthesis> {
        self.domain_synthesis_cache.values().cloned().collect()
    }

    /// Per-constraint, per-wave mean violation scores (read-only).
    pub fn per_constraint_wave_scores(&self) -> &std::collections::HashMap<String, Vec<f64>> {
        &self.per_constraint_wave_scores
    }

    /// True if the frozen verifier bypass is active for this constraint_id.
    pub fn is_verifier_bypassed(&self, constraint_id: &str) -> bool {
        self.bypassed_verifier_constraints.contains(constraint_id)
    }

    /// Drain and return all pending frozen verifier events.
    pub fn take_pending_frozen_verifier_events(
        &mut self,
    ) -> Vec<h2ai_types::events::VerifierFrozenEvent> {
        std::mem::take(&mut self.pending_frozen_verifier_events)
    }

    pub(crate) fn corpus_pass_hint_for(&self, constraint_id: &str) -> Option<String> {
        self.constraint_pass_map.get(constraint_id).and_then(|e| {
            e.pass_criteria
                .clone()
                .filter(|s| !s.is_empty())
                .or_else(|| e.remediation_hint.clone().filter(|s| !s.is_empty()))
        })
    }

    /// Look up the binary check text for a given (constraint_id, check_idx) pair
    /// using the flat `binary_checks` vec and the `constraint_check_offsets` index.
    fn constraint_check_text(&self, constraint_id: &str, check_idx: usize) -> String {
        self.constraint_check_offsets
            .iter()
            .find(|(cid, _, _)| cid == constraint_id)
            .and_then(|(_, start, count)| {
                if check_idx < *count {
                    self.binary_checks.get(start + check_idx).cloned()
                } else {
                    None
                }
            })
            .unwrap_or_default()
    }

    /// Run the researcher loop: detect cold checks (low pass rate) and
    /// fire `run_gap_researcher` for each gap not yet in the synthesis cache.
    ///
    /// Gated on `cfg_ref.gap_i1.enabled`.  No-op when the flag is false or when
    /// neither an LLM adapter nor a web-search grounder is configured.
    pub async fn run_gap_i1_research(&mut self) {
        if !self.cfg_ref.gap_i1.enabled {
            return;
        }
        let Some(adapter) = self.gap_researcher_adapter.clone() else {
            return;
        };
        let gap_chain = self.gap_grounding_chain.clone();

        let check_rates = self.wave_check_rates();
        let cold_gaps = h2ai_autonomic::knowledge_gap::detect_cold_checks(
            &check_rates,
            self.cfg_ref.gap_i1.cold_check_threshold,
        );
        let gaps_to_research = cold_gaps
            .into_iter()
            .take(self.cfg_ref.gap_i1.max_gap_records_per_wave)
            .collect::<Vec<_>>();

        if gaps_to_research.is_empty() {
            return;
        }
        tracing::info!(
            target: "h2ai.gap_i1",
            n_gaps = gaps_to_research.len(),
            "gap research triggered for cold checks"
        );

        for mut gap in gaps_to_research {
            let cache_key = (gap.constraint_id.clone(), gap.check_idx);
            if self.domain_synthesis_cache.contains_key(&cache_key) {
                continue;
            }
            let check_text = self.constraint_check_text(&gap.constraint_id, gap.check_idx);

            // Extract the most representative failure reason for this (constraint, check) pair
            // from verifier feedback on pruned proposals. This becomes `incorrect_concept`,
            // which drives web-search query construction.
            gap.incorrect_concept =
                self.extract_incorrect_concept(&gap.constraint_id, gap.check_idx);
            gap.gap_query = if gap.incorrect_concept.is_empty() {
                check_text.clone()
            } else {
                format!("{} — {}", check_text, &gap.incorrect_concept)
            };

            tracing::info!(
                target: "h2ai.gap_i1",
                constraint_id = %gap.constraint_id,
                check_idx = gap.check_idx,
                incorrect_concept = %gap.incorrect_concept,
                "dispatching gap researcher"
            );

            let corpus_hint = self.corpus_pass_hint_for(&gap.constraint_id);
            match crate::grounding_chain::run_gap_researcher(
                &gap,
                &check_text,
                &adapter,
                gap_chain.as_deref(),
                crate::grounding_chain::GapResearcherOpts {
                    min_confidence: self.cfg_ref.gap_i1.synthesis_min_confidence,
                    timeout_secs: self.cfg_ref.gap_i1.researcher_timeout_secs,
                    corpus_pass_hint: corpus_hint.as_deref(),
                    synthesis_max_tokens: self.cfg_ref.gap_research.gap_synthesis_max_tokens,
                },
            )
            .await
            {
                Some(synth) => {
                    tracing::info!(
                        target: "h2ai.gap_i1",
                        constraint_id = %gap.constraint_id,
                        check_idx = gap.check_idx,
                        confidence = synth.confidence,
                        "gap synthesis accepted"
                    );
                    self.domain_synthesis_cache.insert(cache_key, synth);
                }
                None => {
                    tracing::warn!(
                        target: "h2ai.gap_i1",
                        constraint_id = %gap.constraint_id,
                        check_idx = gap.check_idx,
                        "gap synthesis rejected or timed out"
                    );
                }
            }
        }
    }

    /// Extract the most representative failure reason for a given (constraint_id, check_idx)
    /// pair from all pruned proposals. Used to populate `incorrect_concept` for gap research.
    pub fn extract_incorrect_concept(&self, constraint_id: &str, check_idx: usize) -> String {
        Self::extract_incorrect_concept_from(&self.all_pruned, constraint_id, check_idx)
    }

    /// Pure helper for testing: takes the pruned events slice directly.
    pub fn extract_incorrect_concept_from(
        all_pruned: &[h2ai_types::events::BranchPrunedEvent],
        constraint_id: &str,
        check_idx: usize,
    ) -> String {
        let reasons: Vec<&str> = all_pruned
            .iter()
            .flat_map(|p| &p.violated_constraints)
            .filter(|v| v.constraint_id == constraint_id)
            .filter(|v| {
                // Only include if this check_idx is KNOWN to have failed.
                // Empty verdicts carry no per-check attribution; ignore them.
                !v.check_verdicts.is_empty()
                    && v.check_verdicts.get(check_idx).copied() == Some(false)
            })
            .filter_map(|v| v.verifier_reason.as_deref())
            .filter(|r| !r.is_empty())
            .collect();
        if reasons.is_empty() {
            return String::new();
        }
        reasons
            .into_iter()
            .min_by_key(|r| r.len())
            .unwrap_or("")
            .chars()
            .take(200)
            .collect()
    }

    // ── Cross-wave instability detection ──────────────────────────────

    /// Scan `last_wave_pruned` and `prev_wave_pruned` for the same constraint
    /// appearing in both waves with hard violations whose rejection reasons have
    /// low Jaccard similarity (indicating the verifier is flipping).
    pub fn find_instability(&mut self, wave: u32) -> Option<InstabilitySignal> {
        let mut last_reasons: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        let mut prev_reasons: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();

        for event in &self.last_wave_pruned {
            for v in &event.violated_constraints {
                if v.severity_label == "Hard" {
                    if let Some(r) = &v.verifier_reason {
                        if !r.is_empty() {
                            last_reasons
                                .entry(v.constraint_id.clone())
                                .or_default()
                                .push(r.clone());
                        }
                    }
                }
            }
        }

        for event in &self.prev_wave_pruned {
            for v in &event.violated_constraints {
                if v.severity_label == "Hard" {
                    if let Some(r) = &v.verifier_reason {
                        if !r.is_empty() {
                            prev_reasons
                                .entry(v.constraint_id.clone())
                                .or_default()
                                .push(r.clone());
                        }
                    }
                }
            }
        }

        let mut fired: Option<(String, f64, Vec<String>)> = None;
        for (cid, last_rs) in &last_reasons {
            if let Some(prev_rs) = prev_reasons.get(cid) {
                let score = constraint_reasons_jaccard(last_rs, prev_rs);
                if score < self.cfg_ref.gap_k1.instability_threshold {
                    let mut reasons = last_rs.clone();
                    reasons.extend(prev_rs.iter().cloned());
                    reasons.dedup();
                    reasons.truncate(5);
                    fired = Some((cid.clone(), score, reasons));
                    break;
                }
            }
        }
        let (cid, score, reasons) = fired?;

        if !self.cfg_ref.ambiguity_detection.enabled {
            return Some(InstabilitySignal {
                constraint_id: cid,
                check_index: 0,
                score,
                reasons,
                ambiguity_evidence: vec![],
                ambiguity_score: 0.0,
            });
        }

        self.accumulate_ambiguity(&cid, score, reasons, wave)
    }

    fn accumulate_ambiguity(
        &mut self,
        cid: &str,
        instability_score: f64,
        reasons: Vec<String>,
        wave: u32,
    ) -> Option<InstabilitySignal> {
        use h2ai_constraints::ambiguity::{
            most_divergent_pair, score_evidence, AmbiguityEvidence, AmbiguityScorecard, PatchMode,
            DYNAMIC_ONLY_CHECK_IDX,
        };
        let acfg = self.cfg_ref.ambiguity_detection.clone();

        let key = self
            .ambiguity_scorecards
            .iter()
            .filter(|((c, idx), _)| c == cid && *idx != DYNAMIC_ONLY_CHECK_IDX)
            .max_by(|a, b| {
                a.1.score
                    .partial_cmp(&b.1.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, _)| k.clone())
            .unwrap_or_else(|| (cid.to_string(), DYNAMIC_ONLY_CHECK_IDX));

        let current = self
            .ambiguity_scorecards
            .get(&key)
            .cloned()
            .unwrap_or_else(|| AmbiguityScorecard::new(cid.to_string(), key.1));

        if current.rewrite_applied {
            return None;
        }

        let ev = AmbiguityEvidence::JaccardFreezeWave {
            wave,
            cross_wave_jaccard: instability_score as f32,
        };
        let updated = score_evidence(&current, ev, &acfg);

        if updated.score < acfg.score_threshold {
            if updated.score >= 0.3 {
                tracing::error!(
                    target: "h2ai.mape_k",
                    constraint_id = %cid,
                    score = updated.score,
                    wave,
                    "ambiguity evidence accumulating — constraint spec suspected ambiguous"
                );
            } else {
                tracing::warn!(
                    target: "h2ai.mape_k",
                    constraint_id = %cid,
                    score = updated.score,
                    wave,
                    "verifier divergence recorded in ambiguity scorecard"
                );
            }
            self.ambiguity_scorecards.insert(key, updated);
            return None;
        }

        let mut fired = updated;
        fired.rewrite_applied = true;
        let patch_mode = fired.patch_mode();
        let evidence_lines: Vec<String> = fired.evidence.iter().map(ToString::to_string).collect();
        let final_score = fired.score;
        self.ambiguity_scorecards.insert(key, fired);

        let mut ordered = reasons.clone();
        if let Some((a, b)) = most_divergent_pair(&reasons) {
            let (a, b) = (a.to_string(), b.to_string());
            ordered.retain(|r| r != &a && r != &b);
            ordered.insert(0, b);
            ordered.insert(0, a);
        }

        match patch_mode {
            PatchMode::Precise { check_idx } => Some(InstabilitySignal {
                constraint_id: cid.to_string(),
                check_index: check_idx,
                score: instability_score,
                reasons: ordered,
                ambiguity_evidence: evidence_lines,
                ambiguity_score: final_score,
            }),
            PatchMode::DiagnosticOnly => {
                tracing::error!(
                    target: "h2ai.mape_k",
                    constraint_id = %cid,
                    final_score,
                    wave,
                    "constraint ambiguity threshold crossed (diagnostic-only — check index unknown)"
                );
                self.pending_ambiguity_events.push(
                    h2ai_types::events::ConstraintAmbiguityDetectedEvent {
                        task_id: self.task_id.clone(),
                        constraint_id: cid.to_string(),
                        check_idx: None,
                        original_check_text: String::new(),
                        suggested_rewrite: String::new(),
                        evidence: evidence_lines,
                        final_score,
                        wave,
                        timestamp: chrono::Utc::now(),
                    },
                );
                None
            }
        }
    }

    pub fn take_pending_ambiguity_events(
        &mut self,
    ) -> Vec<h2ai_types::events::ConstraintAmbiguityDetectedEvent> {
        std::mem::take(&mut self.pending_ambiguity_events)
    }

    // ── Test helpers ───────────────────────────────────────────────────────────

    pub fn set_n_agents(&mut self, n: u32) {
        self.current_params.n_agents = n;
    }

    pub fn tee_event_ref(&self) -> Option<&h2ai_types::events::TieredExitEvent> {
        self.tee_event.as_ref()
    }

    pub fn set_corpus_viable(&mut self, v: bool) {
        self.corpus_synthesis_viable = v;
    }

    pub fn corpus_synthesis_viable_flag(&self) -> bool {
        self.corpus_synthesis_viable
    }

    pub fn seed_pruned_waves(
        &mut self,
        last: Vec<h2ai_types::events::BranchPrunedEvent>,
        prev: Vec<h2ai_types::events::BranchPrunedEvent>,
    ) {
        self.last_wave_pruned = last;
        self.prev_wave_pruned = prev;
    }

    pub fn seed_ambiguity_scorecard(
        &mut self,
        key: (String, usize),
        card: h2ai_constraints::ambiguity::AmbiguityScorecard,
    ) {
        self.ambiguity_scorecards.insert(key, card);
    }

    pub fn ambiguity_scorecards_ref(
        &self,
    ) -> &std::collections::HashMap<(String, usize), h2ai_constraints::ambiguity::AmbiguityScorecard>
    {
        &self.ambiguity_scorecards
    }

    // ── GAP-F5: Induction helpers ─────────────────────────────────────────────

    /// Collect deduplicated constraint IDs from the full cross-wave pruned accumulator.
    fn all_violated_constraint_ids(&self) -> Vec<String> {
        self.all_pruned
            .iter()
            .flat_map(|e| {
                e.violated_constraints
                    .iter()
                    .map(|v| v.constraint_id.clone())
            })
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }

    /// Inject the top induction hint into `self.retry_context`.
    fn apply_induction_result(&mut self, result: &crate::induction::InductionResult) {
        if let Some(top_hint) = result.patterns.first() {
            let hint_text = format!(
                "\n[INDUCTION HINT — success_rate={:.2}]: {}\n",
                top_hint.success_rate(),
                top_hint.hint_text
            );
            self.retry_context =
                Some(self.retry_context.as_deref().unwrap_or("").to_string() + &hint_text);
            self.applied_hint_texts.push(top_hint.hint_text.clone());
        }
    }

    #[must_use]
    pub fn new_for_test(cfg: h2ai_config::H2AIConfig) -> Self {
        use crate::tao_loop::TaoMultiplierEstimator;
        use h2ai_types::events::TaskComplexityAssessedEvent;
        use h2ai_types::identity::TaskId;
        use h2ai_types::sizing::{ProbeSkipReason, TaskQuadrant};

        let task_id = TaskId::new();
        let complexity_event = TaskComplexityAssessedEvent {
            task_id: task_id.clone(),
            tcc_structural: 0.0,
            tcc_empirical: None,
            tcc_effective: 0.0,
            n_eff_pool: None,
            task_quadrant: TaskQuadrant::Precision,
            probe_skipped: true,
            probe_skip_reason: ProbeSkipReason::None,
            heavy_fraction: 0.0,
            tcc_mismatch: false,
            probe_cost_tokens: 0,
            n_informative_static: 0,
            timestamp: chrono::Utc::now(),
        };
        let tao_config = TaoConfig::default();
        let verification_config = VerificationConfig::default();
        let max_retries = cfg.max_autonomic_retries as usize;

        Self {
            current_params: OptimizerParams {
                n_agents: 3,
                max_turns: 4,
                verify_threshold: verification_config.threshold,
            },
            quality_history: Vec::new(),
            n_max_ceiling: 9,
            cg_mean: 0.5,
            force_topology: None,
            tried_topologies: Vec::new(),
            tau_reduction_factor: 1.0,
            tau_spread_factor: 1.0,
            tau_values_tried: Vec::new(),
            retry_context: None,
            adapter_rotation_offset: 0,
            mode_collapse_count: 0,
            last_multiplication_failure: None,
            pending_tombstone: None,
            system_context_with_rubric: String::new(),
            max_retries,
            tao_config,
            verification_config,
            all_verification_events: Vec::new(),
            all_failed_proposals: Vec::new(),
            all_shadow_audit_events: Vec::new(),
            all_correlated_warnings: Vec::new(),
            all_researcher_grounding_events: Vec::new(),
            all_pruned: Vec::new(),
            last_wave_pruned: Vec::new(),
            prev_wave_pruned: Vec::new(),
            topology_retry_events: Vec::new(),
            task_id,
            assessed_quadrant: TaskQuadrant::Precision,
            complexity_event,
            diversity_degraded_event: None,
            bandit_state: None,
            tao_estimator: std::sync::Arc::new(tokio::sync::RwLock::new(
                TaoMultiplierEstimator::new_with_alpha(0.1),
            )),
            tao_multiplier: 1.0,
            calibration_ensemble: None,
            cfg_ref: std::sync::Arc::new(cfg),
            task_deadline: None,
            leader: None,
            last_wave_verification_events: Vec::new(),
            last_wave_proposal_texts: std::collections::HashMap::new(),
            pending_leader_elected_events: Vec::new(),
            pending_socratic_diagnosis_events: Vec::new(),
            ambiguity_scorecards: std::collections::HashMap::new(),
            pending_ambiguity_events: Vec::new(),
            last_wave_violated_constraint_ids: Vec::new(),
            prev_assembled_contexts: Vec::new(),
            global_best_proposal: None,
            global_best_partial_proposal: None,
            global_best_constraint_reasons: std::collections::HashMap::new(),
            compliance_score_history: Vec::new(),
            conflict_graph: h2ai_constraints::conflict::ConstraintConflictGraph::build(&[]),
            binary_checks: Vec::new(),
            constraint_check_offsets: Vec::new(),
            domain_synthesis_cache: std::collections::HashMap::new(),
            gap_researcher_adapter: None,
            gap_grounding_chain: None,
            constraint_pass_map: std::collections::HashMap::new(),
            probe_result: None,
            corpus_synthesis_viable: false,
            per_constraint_wave_scores: std::collections::HashMap::new(),
            verifier_reason_history: std::collections::HashMap::new(),
            bypassed_verifier_constraints: std::collections::HashSet::new(),
            pending_frozen_verifier_events: vec![],
            last_wave_n_eff: 1.0,
            tee_event: None,
            tokens_used: 0,
            budget_exhausted: false,
            convergence_gate_event: None,
            task_class_tags: Vec::new(),
            induction_scheduler: None,
            pending_induction: None,
            zero_survival_count: 0,
            applied_hint_texts: vec![],
        }
    }

    #[must_use]
    pub fn new_minimal() -> Self {
        Self::new_for_test(h2ai_config::H2AIConfig::default())
    }

    pub fn seed_synthesis(
        &mut self,
        constraint_id: &str,
        check_idx: usize,
        synthesis: h2ai_types::gap_i1::DomainSynthesis,
    ) {
        self.domain_synthesis_cache
            .insert((constraint_id.to_string(), check_idx), synthesis);
    }
}
