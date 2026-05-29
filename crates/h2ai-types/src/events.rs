use crate::config::{
    AdapterKind, AuditorConfig, ExplorerConfig, ParetoWeights, ReviewGate, TopologyKind,
};
use crate::identity::{ExplorerId, SubtaskId, TaskId};
use crate::sizing::{
    CoherencyCoefficients, CoordinationThreshold, EigenCalibration, EnsembleCalibration,
    MergeStrategy, MultiplicationConditionFailure, OracleDomain, OracleSpec, PredictionBasis,
    ProbeSkipReason, RoleErrorCost, TaskQuadrant, TauValue,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Quality level of the current calibration state.
///
/// Used in Phase 1.5 bootstrap guard: when `Bootstrap`, synthetic priors are the only
/// source and the N-probe sampling is bypassed (routes to Coverage unconditionally).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CalibrationQuality {
    /// Calibration has run against real adapters; priors are empirically grounded.
    #[default]
    Domain,
    /// Only synthetic priors available (no real adapter data yet).
    Bootstrap,
}

/// How `CG(i,j)` was computed during calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CgMode {
    /// CG is mean pairwise Hamming distance between constraint satisfaction profiles.
    /// Falls back to `cfg.calibration_cg_fallback` when no constraint corpus is provided.
    #[default]
    ConstraintProfile,
    /// CG is the fraction of calibration prompts where `cosine(embed_i, embed_j)` > `θ_agree`.
    /// Semantically robust: paraphrase-insensitive, matches the theoretical specification.
    /// Requires the `fastembed-embed` feature and an `EmbeddingModel` in `AppState`.
    EmbeddingCosine,
}

/// Whether the calibration was derived from real adapter measurements or synthetic priors.
///
/// Used to signal downstream consumers (e.g. Phase 1.5 routing) how much to trust
/// the calibration coefficients. `Measured` requires at least 3 adapters for USL fit
/// and at least 2 for CG pairwise samples. `PartialFit` means one of the two conditions
/// was met. `SyntheticPriors` means neither was met (M < 3 and < 2 adapter outputs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CalibrationSource {
    /// Calibration is fully empirically grounded: USL fit from ≥3 adapters and pairwise CG from ≥2.
    #[default]
    Measured,
    /// One of USL fit or CG pairwise sample was empirical; the other fell back to config defaults.
    PartialFit,
    /// Both USL parameters and CG came from config synthetic priors (M < 3 and < 2 adapter outputs).
    SyntheticPriors,
}

/// Classifies why all proposals were pruned in a MAPE-K zero-survival wave.
///
/// Computed synchronously from cosine `N_eff` before re-provisioning.
/// Drives retry routing: `ConstrainedExploration` injects a tombstone;
/// `ModeCollapse` rotates the adapter selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FailureMode {
    /// Agents explored diverse solution areas but none satisfied constraints.
    /// Retry: same topology, inject Constraint Violation Tombstone.
    ConstrainedExploration,
    /// Agents converged on a shared hallucination (`N_eff` ≈ 1).
    /// Retry: rotate adapter selection or widen `τ_spread`.
    ModeCollapse,
    /// Proposals share a correlated assumption (low Jaccard CV).
    /// Retry: inject contradiction hint + researcher grounding.
    CorrelatedHallucination { cv: f64, mean_jaccard_distance: f64 },
}

/// Emitted asynchronously after `MergeResolvedEvent` — does not block task close.
///
/// Measures semantic independence of the surviving proposals. `yield_ratio` uses
/// `N_requested` as the denominator (not `N_responded`) — financial yield: you paid
/// for N adapters, you received `n_eff_cosine_actual` independent perspectives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpistemicYieldEvent {
    pub task_id: TaskId,
    pub n_eff_cosine_actual: f64,
    pub n_eff_prior: f64,
    /// `n_eff_actual` / `N_requested`
    pub yield_ratio: f64,
    pub adapters: Vec<String>,
}

/// Emitted when the calibration harness finishes measuring α, β₀, and CG for the adapter pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationCompletedEvent {
    pub calibration_id: TaskId,
    pub coefficients: CoherencyCoefficients,
    pub coordination_threshold: CoordinationThreshold,
    /// Condorcet-based ensemble calibration. `None` when < 2 adapters ran calibration
    /// (falls back to config defaults).
    pub ensemble: Option<EnsembleCalibration>,
    /// Eigenvalue-based calibration (from pairwise CG matrix). `None` when fewer than 2 adapters.
    pub eigen: Option<EigenCalibration>,
    pub timestamp: DateTime<Utc>,
    /// β₀ derived from timing the pairwise CG measurement loop during calibration.
    /// Captures coherence-drag baseline; the CG coupling then adjusts for divergence severity.
    /// `None` when fewer than 2 adapters ran calibration.
    #[serde(default)]
    pub pairwise_beta: Option<f64>,
    /// How CG was computed: constraint Hamming distance profile or fallback.
    /// Defaults to `ConstraintProfile` when deserialising events written before this field was added.
    #[serde(default)]
    pub cg_mode: CgMode,
    /// Distinct non-Mock adapter families present in the calibration pool (sorted).
    /// Empty when all adapters are Mock (test-only deployments).
    #[serde(default)]
    pub adapter_families: Vec<String>,
    /// True when explorer and verification adapters are from the same non-Mock family.
    /// LLM self-preference judge bias is likely; consider routing verification to a different family.
    #[serde(default)]
    pub explorer_verification_family_match: bool,
    /// True when all non-Mock adapters belong to a single family.
    /// Weiszfeld BFT correlated hallucination protection is degraded.
    #[serde(default)]
    pub single_family_warning: bool,
    /// Lower bound of `N_max` one-σ confidence interval (`CG_mean` − `cg_std_dev`).
    /// Equals `n_max()` when only one CG sample exists.
    #[serde(default)]
    pub n_max_lo: f64,
    /// Upper bound of `N_max` one-σ confidence interval (`CG_mean` + `cg_std_dev`).
    /// `n_max_lo ≤ n_max() ≤ n_max_hi`. Wide interval = high CG measurement variance.
    #[serde(default)]
    pub n_max_hi: f64,
    /// Pool-level semantic independence measured at calibration time via cosine `N_eff`.
    /// Used as the Bayesian prior at task provisioning. `0.0` when no `EmbeddingModel`
    /// is present (fallback formula: 1.0 + `cg_fallback` × (N − 1) is computed in the harness).
    #[serde(default)]
    pub n_eff_cosine_prior: f64,
    /// Whether this calibration is empirically grounded (`Domain`) or synthetic-prior only
    /// (`Bootstrap`). Phase 1.5 skips the N-probe path when `Bootstrap`.
    /// Defaults to `Domain` so existing serialised events deserialise correctly.
    #[serde(default)]
    pub calibration_quality: CalibrationQuality,
    /// Fine-grained source classification: whether USL fit and CG pairwise samples were
    /// measured from real adapters or fell back to synthetic config priors.
    /// Defaults to `Measured` so existing serialised events deserialise correctly.
    #[serde(default)]
    pub calibration_source: CalibrationSource,
    /// Conflict-rate-based β computed from Phase B pairwise violation matrix.
    /// `None` when fewer than 2 proposals were generated or corpus is empty.
    #[serde(default)]
    pub beta_quality: Option<f64>,
}

/// Point-in-time snapshot of a task's in-memory state for crash-recovery replay optimization.
///
/// Stored in NATS KV at key `snapshots/{task_id}/latest`.
/// Recovery loads this snapshot then replays only events with sequence > `last_sequence`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSnapshot {
    pub task_id: TaskId,
    /// NATS `JetStream` sequence number of the last event included in this snapshot.
    pub last_sequence: u64,
    /// Serialized `TaskState` as JSON — stored as a string to avoid a circular crate dependency.
    pub task_state_json: String,
    pub taken_at: DateTime<Utc>,
}

/// Emitted when a task is initialised: system context compiled and locked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskBootstrappedEvent {
    pub task_id: TaskId,
    pub system_context: String,
    pub pareto_weights: ParetoWeights,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the planner selects topology, explorer roles, and merge strategy for a retry iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyProvisionedEvent {
    pub task_id: TaskId,
    pub topology_kind: TopologyKind,
    pub explorer_configs: Vec<ExplorerConfig>,
    pub auditor_config: AuditorConfig,
    pub n_max: f64,
    pub interface_n_max: Option<f64>,
    #[serde(alias = "kappa_eff")]
    pub beta_eff: f64,
    pub role_error_costs: Vec<RoleErrorCost>,
    pub merge_strategy: MergeStrategy,
    pub coordination_threshold: CoordinationThreshold,
    pub review_gates: Vec<ReviewGate>,
    pub retry_count: u32,
    pub timestamp: DateTime<Utc>,
    /// Dense constraint violation summary injected on `ConstrainedExploration` retries.
    /// Contains constraint IDs and `c_i` weights only — never raw proposal text.
    /// `None` on wave 1 and on `ModeCollapse` retries.
    #[serde(default)]
    pub constraint_tombstone: Option<String>,
}

/// Emitted when the multiplication condition gate rejects the current topology on a given retry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiplicationConditionFailedEvent {
    pub task_id: TaskId,
    pub failure: MultiplicationConditionFailure,
    pub retry_count: u32,
    pub timestamp: DateTime<Utc>,
}

/// Why an explorer failed to produce a proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProposalFailureReason {
    /// The adapter did not respond within the per-turn deadline.
    Timeout,
    /// The adapter process was killed by the OOM killer; the message is the signal detail.
    OomPanic(String),
    /// The adapter returned an error; the message contains the error description.
    AdapterError(String),
}

/// Emitted when an explorer completes a TAO loop and produces a raw output proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub tau: TauValue,
    /// TAO retry-loop generation counter. First attempt = 0; each MAPE-K retry increments by 1.
    /// Used by `ProposalSet` as the primary LUB key: higher generation always supersedes lower.
    #[serde(default)]
    pub generation: u64,
    pub raw_output: String,
    pub token_cost: u64,
    pub adapter_kind: AdapterKind,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when an explorer's TAO loop terminates without producing a usable proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalFailedEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub reason: ProposalFailureReason,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when all explorers in Phase 3 have finished (or timed out), summarising success/failure counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationPhaseCompletedEvent {
    pub task_id: TaskId,
    pub total_explorers: u32,
    pub successful: u32,
    pub failed: u32,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the verification phase starts evaluating a specific explorer's proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub timestamp: DateTime<Utc>,
}

/// A single constraint that a proposal violated during the verification phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    pub constraint_id: String,
    /// Predicate score [0,1]; 0 = total violation.
    pub score: f64,
    /// "Hard", "Soft", or "Advisory"
    pub severity_label: String,
    pub remediation_hint: Option<String>,
    /// Natural-language constraint statement from ConstraintDoc.description.
    #[serde(default)]
    pub constraint_description: String,
    /// Dynamic verifier interpretation from the LLM judge for this constraint.
    /// None for static predicates or when contradiction was detected across proposals.
    #[serde(default)]
    pub verifier_reason: Option<String>,
    /// Per-check verdicts parsed from LlmJudge CoT: `true` = PRESENT, `false` = MISSING.
    /// Empty when there are no binary checks or when the reason could not be parsed.
    #[serde(default)]
    pub check_verdicts: Vec<bool>,
    /// Pass-criteria text from the constraint YAML `criteria.pass` field.
    #[serde(default)]
    pub criteria_pass: Option<String>,
}

/// Emitted when an explorer's proposal is eliminated by the verification or auditor gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchPrunedEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub reason: String,
    /// The full proposal text that was pruned. Used by synthesis to build
    /// partial-pass examples from the actual proposal rather than the status string.
    #[serde(default)]
    pub raw_output: String,
    pub constraint_error_cost: RoleErrorCost,
    pub violated_constraints: Vec<ConstraintViolation>,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when all proposals for a retry iteration were pruned, triggering MAPE-K retry logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroSurvivalEvent {
    pub task_id: TaskId,
    pub retry_count: u32,
    pub timestamp: DateTime<Utc>,
    /// Effective independent adapters computed from cosine similarity on failed proposals.
    /// `None` when no `EmbeddingModel` is present in `AppState`.
    #[serde(default)]
    pub n_eff_cosine_actual: Option<f64>,
    /// MAPE-K failure classification. `None` when no `EmbeddingModel` is available.
    #[serde(default)]
    pub failure_mode: Option<FailureMode>,
}

/// Emitted when `CG_embed` falls below `cg_collapse_threshold`.
/// The planner forces `N_max=1` — no ensemble benefit is possible when coordination quality collapses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroCoordinationQualityEvent {
    pub task_id: TaskId,
    pub cg_embed: f64,
    pub forced_n_max: u32,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the maximum role error cost exceeds the BFT threshold, signalling consensus-mode merging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusRequiredEvent {
    pub task_id: TaskId,
    pub max_role_error_cost: RoleErrorCost,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the merge engine finishes selecting surviving proposals.
///
/// The CRDT semilattice resolves to a single winning proposal by selection; content synthesis,
/// if enabled, is a separate Phase 5a operation. This event records which proposals survived
/// and which were pruned, the merge strategy used, and the merge timing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionResolvedEvent {
    pub task_id: TaskId,
    pub valid_proposals: Vec<ExplorerId>,
    pub pruned_proposals: Vec<(ExplorerId, String)>,
    pub merge_strategy: MergeStrategy,
    pub timestamp: DateTime<Utc>,
    /// Wall-clock seconds consumed by `MergeEngine::resolve()` for this event.
    /// `None` for events reconstructed from older serialised logs.
    #[serde(default)]
    pub merge_elapsed_secs: Option<f64>,
    /// Number of proposals (valid + pruned) that entered `resolve()`.
    #[serde(default)]
    pub n_input_proposals: usize,
    /// Number of non-pruned proposals that scored exactly 0.0 and were excluded
    /// from selection to prevent synthesis contamination.
    #[serde(default)]
    pub n_failed_proposals: usize,
}

/// Emitted when the merge engine produces the final resolved output string for a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResolvedEvent {
    pub task_id: TaskId,
    pub resolved_output: String,
    /// Ensemble Efficiency Index: `Q_realized` / `Q_ceiling` ∈ [0, 1].
    /// Measures what fraction of the Condorcet quality ceiling was realized.
    /// None when calibration is unavailable at merge time.
    #[serde(default)]
    pub j_eff: Option<f64>,
    pub timestamp: DateTime<Utc>,
    /// Whether the oracle gate passed for this merged output.
    /// `None` when no oracle gate was configured or evaluation has not yet completed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oracle_gate_passed: Option<bool>,
    /// Zone 3 audit-findings text from the last OSP merge attempt.
    /// Contains only `constraint_id` and `remediation_hint` — never raw proposal text.
    /// Used by the engine as retry hint injection context on `ZeroSurvival`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zone3_hints: Option<String>,
}

/// Emitted when the MAPE-K loop exhausts all retries without producing a resolved output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskFailedEvent {
    pub task_id: TaskId,
    pub pruned_events: Vec<BranchPrunedEvent>,
    pub topologies_tried: Vec<TopologyKind>,
    pub tau_values_tried: Vec<Vec<f64>>,
    pub multiplication_condition_failure: Option<MultiplicationConditionFailure>,
    pub timestamp: DateTime<Utc>,
}

/// Emitted alongside `MergeResolved` when the resolved output does not cover all
/// constraint domains — some domain had violations that no surviving proposal fixed.
///
/// Non-blocking (task still succeeds). `uncovered_domains` identifies which areas
/// of the constraint space were not closed by the surviving ensemble.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceIncompleteEvent {
    pub task_id: TaskId,
    pub uncovered_domains: Vec<String>,
    /// Surviving proposal pairs that score on opposite sides of 0.5 for the same constraint domain.
    /// Each entry is `(explorer_a_id, explorer_b_id, domain)`. Absent in payloads from older
    /// producers (handled by `#[serde(default)]`); empty when no contradictions were detected.
    #[serde(default)]
    pub active_contradictions: Vec<(String, String, String)>,
    pub retries: u32,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when `record_adversarial_comparison` is enabled in `VerificationConfig`.
///
/// Records both standard and adversarial verifier scores for the same proposal.
/// Does NOT affect pruning decisions — the configured verifier score drives those.
/// Correlate with `OracleResultEvent` by `task_id` for offline A/B analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierComparisonEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub standard_score: f64,
    pub adversarial_score: f64,
    pub standard_passed: bool,
    pub adversarial_passed: bool,
    /// Human-readable label for the verifier configuration used. Currently always "llmjudge".
    pub verifier_kind: String,
    pub timestamp: DateTime<Utc>,
}

/// Per-proposal shadow auditor outcome — published to `h2ai.audit.shadow_results`.
///
/// Both auditors ran concurrently; `disagreement = primary_approved != shadow_approved`.
/// The primary auditor's decision always controls Phase 4 pruning in shadow mode.
/// In majority-vote mode (promoted domain), both must approve for the proposal to survive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowAuditorResultEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    /// Decision from the configured primary auditor adapter.
    pub primary_approved: bool,
    /// Decision from the shadow auditor adapter.
    pub shadow_approved: bool,
    /// `true` when `primary_approved != shadow_approved`.
    pub disagreement: bool,
    /// Task domain from `constraint_tags[0]`, or `"default"` when no tags are present.
    pub domain: String,
    /// `format!("{:?}", auditor_adapter.kind())` of the primary auditor.
    pub primary_family: String,
    /// `format!("{:?}", shadow_adapter.kind())` of the shadow auditor.
    pub shadow_family: String,
    pub timestamp_ms: u64,
}

/// Emitted by `ShadowAuditorAccumulator` when a domain's rolling disagreement rate
/// exceeds `promotion_threshold` over `promotion_window` observations.
///
/// From this point the domain uses two-auditor AND vote in Phase 4.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditDomainPromotedEvent {
    pub domain: String,
    pub disagreement_rate: f64,
    pub n_observations: usize,
    pub timestamp_ms: u64,
}

/// Emitted by `ShadowAuditorAccumulator` when a promoted domain's rolling rate
/// drops below `promotion_threshold / 2` over `2 * promotion_window` observations.
///
/// Majority-vote enforcement is removed for this domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditDomainDemotedEvent {
    pub domain: String,
    pub disagreement_rate: f64,
    pub n_observations: usize,
    pub timestamp_ms: u64,
}

/// Emitted when the CV of pairwise Jaccard distances among surviving proposals
/// falls below `correlated_hallucination_cv_threshold`.
///
/// Low CV = semantic clustering.
/// Triggers a MAPE-K retry with researcher grounding injected into explorer prompts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelatedEnsembleWarning {
    pub task_id: TaskId,
    /// Coefficient of variation of pairwise Jaccard distance matrix.
    pub cv: f64,
    /// Mean pairwise Jaccard distance across all proposal pairs.
    pub mean_jaccard_distance: f64,
    /// MAPE-K iteration at which the warning fired.
    pub retry_count: u32,
}

/// Emitted when SRANI detects shared ungrounded architectural entities across proposals.
///
/// CFI (Correlated Fabrication Index) = max pairwise overlap of ungrounded entity sets.
/// 0.0 = no shared fabrication; 1.0 = all proposals share the same fabricated component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelatedFabricationEvent {
    pub task_id: TaskId,
    /// CFI value that triggered this event.
    pub cfi: f64,
    /// Sigmoid injection pressure: `sigmoid((CFI − μ) / T)`. Range [0, 1].
    /// 0.20 = warn floor; `gate_threshold` (default 0.50) = injection cutoff.
    /// Set to 0.0 when `adaptive=false` (legacy static-threshold path).
    pub injection_pressure: f64,
    /// Architectural entities present in ≥2 proposals but absent from the task specification.
    pub shared_ungrounded_entities: Vec<String>,
    /// Number of proposals analysed.
    pub proposal_count: usize,
    /// True if a grounding hint was injected into `retry_context`.
    pub hint_injected: bool,
    pub timestamp: DateTime<Utc>,
}

/// Identifies which tier of the SRANI grounding chain produced a `ResearcherGroundingEvent`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GroundingSource {
    SpecAnchor,
    #[default]
    LlmResearcher,
    WebSearch,
}

/// Emitted when the researcher adapter fetches external grounding for a C1 retry
/// or a proactive search-enabled slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearcherGroundingEvent {
    pub task_id: TaskId,
    /// Shared assumption identified among correlated proposals. Empty for proactive slots.
    pub shared_assumption: String,
    /// Summary of what external literature says about this topic.
    pub literature_summary: String,
    /// Domain slot for this grounding event. For SRANI reactive grounding, classified from
    /// fabricated entity names (e.g. `"message_broker"`, `"cache_layer"`). `None` only for
    /// events deserialised from storage predating slot classification.
    pub slot: Option<String>,
    /// Which grounding tier produced this event. Defaults to `LlmResearcher` for
    /// backward-compatible deserialisation of pre-existing events.
    #[serde(default)]
    pub source: GroundingSource,
}

/// Emitted when Phase 2.6 domain coverage falls below `domain_coverage_threshold`.
/// Non-fatal by default; fatal when `require_bivariate_cg = true`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityGuardDegradedEvent {
    pub task_id: TaskId,
    /// Human-readable explanation (e.g. "coverage 0.25 < threshold 0.40").
    pub reason: String,
    /// Fraction of corpus domains covered by the slot assignment.
    pub coverage_score: f64,
    /// All domain tags assigned across all slots (flattened).
    pub slot_domains: Vec<String>,
}

/// Emitted when a review gate fires and routes a proposal to a reviewer explorer for approval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewGateTriggeredEvent {
    pub task_id: TaskId,
    pub gate_id: String,
    pub blocked_explorer_id: ExplorerId,
    pub reviewer_explorer_id: ExplorerId,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when a reviewer explorer rejects the proposal at a review gate, blocking it from merging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewGateBlockedEvent {
    pub task_id: TaskId,
    pub gate_id: String,
    pub blocked_explorer_id: ExplorerId,
    pub reviewer_explorer_id: ExplorerId,
    pub rejection_reason: String,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when active subtask count approaches `interface_n_max`, warning of interface saturation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceSaturationWarningEvent {
    pub task_id: TaskId,
    pub active_subtasks: u32,
    pub interface_n_max: f64,
    pub saturation_ratio: f64,
    pub timestamp: DateTime<Utc>,
}

/// Emitted after each TAO loop turn, recording the observation and whether the turn passed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaoIterationEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub turn: u8,
    pub observation: String,
    pub passed: bool,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the LLM-as-Judge verifier assigns a compliance score to a proposal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationScoredEvent {
    pub task_id: TaskId,
    pub explorer_id: ExplorerId,
    pub score: f64,
    pub reason: String,
    pub passed: bool,
    /// True when the score was reused from a similar proposal via the per-task eval cache,
    /// avoiding a redundant LLM call. False for freshly computed scores.
    #[serde(default)]
    pub cache_hit: bool,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the orchestrator creates a decomposition plan for a multi-step task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskPlanCreatedEvent {
    pub task_id: TaskId,
    pub plan_id: TaskId,
    pub subtask_count: usize,
    pub timestamp: DateTime<Utc>,
}

/// Covers both approved and rejected outcomes — use `approved` field to distinguish.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskPlanReviewedEvent {
    pub task_id: TaskId,
    pub plan_id: TaskId,
    pub approved: bool,
    pub reason: String,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when an individual subtask begins execution within a wave.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskStartedEvent {
    pub task_id: TaskId,
    pub plan_id: TaskId,
    pub subtask_id: SubtaskId,
    pub description: String,
    pub wave: usize,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when an individual subtask finishes successfully, recording token cost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskCompletedEvent {
    pub task_id: TaskId,
    pub plan_id: TaskId,
    pub subtask_id: SubtaskId,
    pub token_cost: u64,
    pub timestamp: DateTime<Utc>,
}

/// Category of self-optimizer suggestion applied on a wasteful-but-successful run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OptimizationKind {
    /// `SelfOptimizer` suggested adjusting the `verify_threshold` to reduce wasted proposals.
    TauSpreadAdjusted,
    /// `SelfOptimizer` suggested switching topology (stored as a one-shot hint in `AppState`).
    TopologyHintSet,
}

/// One self-optimizer suggestion that was applied on a completed task run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedOptimization {
    pub kind: OptimizationKind,
    pub reason: String,
    /// Human-readable description of the parameter before the suggestion.
    pub before: String,
    /// Human-readable description of the parameter after the suggestion.
    pub after: String,
}

/// Quality attribution snapshot for a completed task.
///
/// Published alongside `SelectionResolved` on the success path.
/// `q_confidence` is the heuristic/empirical confidence estimate from the CG/USL/CJT chain —
/// it measures how confident the system is in its output, not whether the output is correct.
/// `q_measured` (when present) is the Tier 1 oracle result (actual correctness).
/// The interval fields are `None` when fewer than 2 CG calibration samples are available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAttributionEvent {
    pub task_id: TaskId,
    /// Heuristic or empirical confidence estimate from CG/USL/CJT chain.
    /// This is a confidence score (system's self-assessment), not oracle-grounded quality.
    /// See `prediction_basis` for whether it is `Heuristic` or `Empirical`.
    pub q_confidence: f64,
    /// Fraction of Tier 1 oracle tests passed. `None` when no oracle ran.
    #[serde(default)]
    pub q_measured: Option<f64>,
    /// 5th percentile of the bootstrap or conformal interval.
    #[serde(default)]
    pub q_interval_lo: Option<f64>,
    /// 95th percentile of the bootstrap or conformal interval.
    #[serde(default)]
    pub q_interval_hi: Option<f64>,
    /// Source of quality predictions: `Heuristic` or `Empirical`.
    pub prediction_basis: PredictionBasis,
    /// Fraction of dispatched proposals that survived verification (valid / `total_evaluated`).
    /// 1.0 = no waste; below `optimizer_waste_threshold` = wasteful run.
    #[serde(default = "default_waste_ratio")]
    pub waste_ratio: f64,
    /// `SelfOptimizer` suggestions applied on this successful-but-wasteful run.
    /// Empty when the run was not wasteful or no applicable suggestions existed.
    #[serde(default)]
    pub applied_optimizations: Vec<AppliedOptimization>,
    pub timestamp: DateTime<Utc>,
    /// When the task passed through the HITL gate, this records the reviewer's decision.
    #[serde(default)]
    pub approval_decision: Option<crate::approval::ApprovalDecision>,
    /// `CalibrationSource` active during this task's execution.
    /// `#[serde(default)]` preserves backwards compatibility with older stored events.
    #[serde(default)]
    pub calibration_source: CalibrationSource,
}

const fn default_waste_ratio() -> f64 {
    1.0
}

/// Emitted at the end of Phase 1.5 (Task Complexity Assessment).
///
/// Records the full complexity signal chain: structural prior → optional empirical probe →
/// effective TCC → quadrant classification. Always emitted even in shadow mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskComplexityAssessedEvent {
    pub task_id: TaskId,
    /// `TCC_structural`: zero-cost prior from corpus metadata (formula-based, no LLM calls).
    pub tcc_structural: f64,
    /// `TCC_empirical`: participation ratio from N-probe satisfaction matrix.
    /// `None` when probe was skipped (see `probe_skip_reason`).
    #[serde(default)]
    pub tcc_empirical: Option<f64>,
    /// `TCC_effective` = `max(tcc_structural, tcc_empirical)` + `mismatch_penalty`.
    /// Equals `tcc_structural` when probe was skipped.
    pub tcc_effective: f64,
    /// Pool-level `N_eff` from the most recent calibration (eigenvalue participation ratio).
    /// `None` when `EigenCalibration` was not available at calibration time.
    #[serde(default)]
    pub n_eff_pool: Option<f64>,
    /// Routing quadrant before `shadow_mode` override.
    pub task_quadrant: TaskQuadrant,
    /// Whether the N-probe mini-generation step was skipped.
    pub probe_skipped: bool,
    /// Reason the probe was skipped; `None` when probe ran.
    #[serde(default)]
    pub probe_skip_reason: ProbeSkipReason,
    /// Fraction of Heavy-tier constraints (`OracleExecution`) in the corpus.
    pub heavy_fraction: f64,
    /// True when `tcc_empirical` diverges from `tcc_structural` by > 0.3 (signal mismatch).
    pub tcc_mismatch: bool,
    /// Total tokens consumed by the probe mini-generation calls (0 when probe skipped).
    pub probe_cost_tokens: u64,
    /// Number of Static-tier constraints that produced informative variation in the probe.
    pub n_informative_static: usize,
    pub timestamp: DateTime<Utc>,
}

/// Emitted after Phase 3.5 verification; records the full constraint satisfaction matrix
/// across all surviving proposals for Pareto frontier coverage analysis.
///
/// Used for H3 validation in the experiment: measures whether cross-family
/// committees actually cover more of the constraint Pareto frontier than Self-MoA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintFrontierEvent {
    pub task_id: TaskId,
    /// `satisfaction_matrix[i][j]` = score of proposal i on constraint j ∈ [0, 1].
    pub satisfaction_matrix: Vec<Vec<f64>>,
    /// Constraint IDs in column order.
    pub constraint_ids: Vec<String>,
    /// Explorer IDs in row order.
    pub explorer_ids: Vec<ExplorerId>,
    /// Participation ratio (Σλ)²/Σλ² of the column-space eigenvalues.
    /// Measures how much of the constraint Pareto frontier was covered by the ensemble.
    pub pareto_coverage: f64,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when judge panel disagreement is persistent across proposals in a wave.
///
/// Fire-and-forget corpus quality signal. Indicates that a constraint's
/// ambiguity caused persistent uncertainty across the judge panel, suggesting the
/// constraint wording or scope needs refinement in the corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintAmbiguityEvent {
    pub task_id: TaskId,
    /// MAPE-K wave index in which the ambiguity was detected.
    pub wave: u32,
    /// Constraint IDs whose uncertain-vote count reached the configured threshold.
    pub ambiguous_constraints: Vec<String>,
    /// Per-constraint uncertain vote counts for this wave.
    pub uncertain_counts: std::collections::HashMap<String, usize>,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when verifier reasons for the same constraint contradict across proposals in a wave.
///
/// Distinct from `ConstraintAmbiguityEvent` (which tracks judge panel uncertain votes).
/// Fires when dynamic reason strings from different pruned proposals for the same
/// `constraint_id` have pairwise Jaccard word-bag similarity below 0.35.
/// The repair prompt falls back to static `remediation_hint` for this constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierReasonContradictionEvent {
    pub task_id: TaskId,
    /// MAPE-K wave index in which the contradiction was detected.
    pub wave: u32,
    /// Constraint whose verifier reasons contradicted across proposals.
    pub constraint_id: String,
    /// All verifier reason strings collected (one per pruned proposal with Some reason).
    pub reasons: Vec<String>,
    /// Minimum pairwise Jaccard similarity that triggered the fallback (< 0.35).
    pub min_jaccard: f64,
    /// Static remediation_hint used as fallback. None when hint was also absent.
    pub fallback_hint: Option<String>,
    /// Number of sub-claims the verifier reported as BEYOND_BUDGET.
    /// Non-zero indicates a complexity ceiling hit, not a content rejection.
    #[serde(default)]
    pub beyond_budget_count: u32,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the pre-dispatch complexity probe completes for a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityProbeEvent {
    pub task_id: TaskId,
    /// Probe score 1–5: 1 = trivial, 5 = requires multi-step proof verification.
    pub complexity: u8,
    /// One-sentence rationale from the probe model.
    pub rationale: String,
    /// Whether the probe recommended decomposition.
    pub decompose_recommended: bool,
    /// Wall-clock time the probe LLM call took, in milliseconds.
    pub probe_latency_ms: u64,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the intra-retry ceiling detector fires.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityCeilingDetectedEvent {
    pub task_id: TaskId,
    /// MAPE-K wave index at which the ceiling was detected.
    pub retry_count: u32,
    /// Shannon entropy of constraint-failure distribution in the last wave.
    pub entropy: f64,
    /// Score-improvement slope between the last two waves.
    pub retry_slope: f64,
    /// N_eff × CG_mean product from the last wave.
    pub n_eff_cg_product: f64,
    /// Number of ceiling signals that fired (2 or 3).
    pub signals_fired: u8,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when a task's cumulative generation token usage crosses the warning threshold.
/// Published by the engine; does not interrupt task execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostThresholdWarningEvent {
    pub task_id: TaskId,
    pub tokens_used: u64,
    pub budget_tokens: u64,
    pub fraction_used: f64,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when a task's cumulative generation token usage hits the abort threshold.
/// The engine stops retrying and returns the best available partial output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetExhaustedEvent {
    pub task_id: TaskId,
    pub tokens_used: u64,
    pub budget_tokens: u64,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the convergence gate fires: surviving verified proposals are semantically
/// equivalent above `theta_converge` with sufficient supermajority.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceGateEvent {
    pub task_id: TaskId,
    /// Wave index (0-based) at which the gate fired.
    pub wave: u32,
    /// Count of surviving verified proposals.
    pub n_live: usize,
    /// Mean pairwise cosine similarity of surviving proposals.
    pub convergence_fraction: f64,
    /// The cosine threshold that was checked.
    pub theta_converge: f64,
    /// Best verification score among surviving proposals.
    pub best_score: f64,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when GAP-L1 Tiered Early Exit accepts K-of-N proposals and exits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieredExitEvent {
    /// Zero-based retry wave index on which the exit fired.
    pub wave: u32,
    /// Explorer count (N) used in this wave.
    pub n: u32,
    /// Minimum K required (from `k_for_wave(n)`).
    pub k_required: u32,
    /// Actual number of proposals that met the acceptance threshold.
    pub k_accepted: u32,
    /// `acceptance_score` threshold from config.
    pub acceptance_score: f64,
}

/// Emitted when the oracle gate finishes evaluating proposed solutions before merge.
///
/// The gate runs synchronously before `MergeResolvedEvent` is emitted. When
/// `gate_passed = false` the orchestrator may trigger a MAPE-K retry rather than
/// accepting the current ensemble output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleGateResultEvent {
    pub task_id: String,
    pub gate_passed: bool,
    /// Aggregate confidence of the oracle evaluation ∈ [0.0, 1.0].
    pub confidence: f64,
    /// Brief human-readable explanation produced by the oracle.
    pub summary: String,
    /// Total number of proposals evaluated by the oracle gate.
    pub checked_proposals: u32,
    /// Number of proposals that passed the oracle gate.
    pub passed_proposals: u32,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the orchestrator requires human clarification before proceeding.
///
/// Published to `h2ai.tasks.{task_id}.pending_clarification`. The orchestrator
/// suspends the task until a clarification answer is received or `timeout_secs` elapses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingClarificationEvent {
    pub task_id: String,
    /// The clarification question directed at the human operator.
    pub question: String,
    /// Relevant task context that helps the operator answer the question.
    pub context: String,
    pub timeout_secs: u64,
    pub timestamp: DateTime<Utc>,
}

/// Async Phase 6 oracle evaluation request.
///
/// Published to `h2ai.oracle.{tenant_id}.pending` (NATS core) immediately after task completion.
/// Non-blocking — the orchestrator does not wait for the oracle result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OraclePendingEvent {
    pub task_id: TaskId,
    /// The winning merged output to evaluate.
    pub winning_output: String,
    pub q_confidence: f64,
    /// Number of explorers used in the winning ensemble (for bandit arm key).
    pub n_used: u32,
    pub oracle_spec: OracleSpec,
    pub domain: OracleDomain,
    /// Additional oracle specs for multi-oracle FUSE evaluation.
    /// When non-empty, `oracle_worker` runs all specs and applies worst-of-family
    /// reduction before publishing the final `OracleResultEvent`.
    /// `oracle_spec` is always included as the primary spec.
    #[serde(default)]
    pub oracle_specs: Vec<OracleSpec>,
    #[serde(default)]
    pub tenant_id: crate::identity::TenantId,
}

/// Published to `h2ai.oracle.results`. Consumed by `OracleAccumulator`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleResultEvent {
    pub task_id: TaskId,
    /// Echoed from [`OraclePendingEvent`] for correlation.
    pub q_confidence: f64,
    /// Echoed from [`OraclePendingEvent`] for bandit update.
    pub n_used: u32,
    pub passed: bool,
    pub score: f64,
    /// Nonconformity score: `|q_confidence − passed as f64|`.
    pub residual: f64,
    pub domain: OracleDomain,
    pub duration_ms: u64,
    pub timestamp_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verdict: Option<crate::sizing::OracleVerdict>,
    #[serde(default)]
    pub tenant_id: crate::identity::TenantId,
}

/// Emitted when ECE > 0.15 for 10 consecutive oracle observations.
///
/// Signals that the calibration residuals have drifted from the predicted confidence.
/// Operators should inspect oracle quality and adapter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationDriftWarning {
    pub n_observations: usize,
    pub ece: f64,
    pub timestamp_ms: u64,
}

/// Emitted when oracle pass rate is suspiciously low (< 0.05) or high (> 0.97)
/// for 20+ consecutive observations.
///
/// Likely indicates a broken oracle sidecar or trivially-passing test suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleSuspectEvent {
    pub pass_rate: f64,
    pub n_observations: usize,
    pub reason: String,
    pub timestamp_ms: u64,
}

/// Risk level of a pending HITL approval request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalRiskLevel {
    Low,
    Medium,
    High,
}

/// What caused the HITL approval gate to fire for a task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalTrigger {
    /// The `require_approval` flag in the task manifest was set to `true`.
    ManifestFlag,
    /// The system's `q_confidence` fell below the configured threshold.
    LowConfidence,
}

/// Emitted when a task output is held pending human review before delivery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingApprovalEvent {
    pub task_id: TaskId,
    pub proposed_output: String,
    pub q_confidence: f64,
    /// 0=Heuristic 1=Bootstrap 2=Conformal
    pub prediction_basis: u8,
    pub n_used: u32,
    pub risk_level: ApprovalRiskLevel,
    pub triggered_by: ApprovalTrigger,
    pub timeout_at_ms: u64,
    pub timestamp_ms: u64,
}

/// Emitted when a human (or system timeout) resolves a pending approval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalResolvedEvent {
    pub task_id: TaskId,
    pub approved: bool,
    pub operator_id: String,
    pub reviewer_note: Option<String>,
    pub decided_at_ms: u64,
}

/// Emitted after the pre-execution thinking loop completes (or is skipped when disabled).
/// Consumers can assert `iterations_run >= 1` to verify the loop fired.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingLoopCompletedEvent {
    pub task_id: TaskId,
    /// False when `thinking_loop.enabled = false`; all other fields are zero/empty.
    pub enabled: bool,
    pub iterations_run: u32,
    pub coverage_score: f64,
    /// Char count of the final `shared_understanding` string (keeps event small).
    pub shared_understanding_len: usize,
    /// Names of the archetypes selected in the final iteration (empty when disabled).
    pub archetypes: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Emitted inside `patch_ensemble_p_from_oracle` when the oracle has accumulated enough
/// observations (n >= 10) to replace the heuristic `p_mean` with the empirical pass rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleCalibrationPatchedEvent {
    pub task_id: TaskId,
    pub oracle_pass_rate: f64,
    pub n_observations: usize,
    pub p_mean_before: f64,
    pub p_mean_after: f64,
    pub rho_mean: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Emitted when `j_eff` EMA drops below the configured threshold, triggering OPRO.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OproTriggeredEvent {
    pub adapter_name: String,
    pub prompt_key: String,
    pub j_eff_ema: f64,
    pub n_tasks_total: u32,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when OPRO selects a winning prompt variant after sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptVariantPromotedEvent {
    pub adapter_name: String,
    pub prompt_key: String,
    pub variant_id: String,
    pub winning_score: f64,
    pub timestamp: DateTime<Utc>,
}

/// Reason for leadership rotation in a MAPE-K retry wave.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RotationReason {
    /// First election — no prior leader existed.
    FirstElection,
    /// Leader rotated due to confidence stagnation.
    Stagnation,
}

/// Emitted when a Krum-elected leader is installed or rotated for a new wave term.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderElectedEvent {
    pub task_id: TaskId,
    pub term: u32,
    pub leader_explorer_id: ExplorerId,
    pub q_confidence: f64,
    pub credibility_score: f64,
    /// `None` = first election; `Some(reason)` = rotation cause.
    pub rotation_reason: Option<RotationReason>,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the leader formulates its Socratic diagnostic question for the next wave.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocraticDiagnosisEvent {
    pub task_id: TaskId,
    pub term: u32,
    pub question: String,
    pub violated_constraints: Vec<String>,
    /// Which EIG candidate rank was selected (1 = best).
    pub eig_rank: u32,
    /// Candidates skipped due to belief-buffer deduplication.
    pub dedup_candidates_tried: u32,
    pub timestamp: DateTime<Utc>,
}

/// Emitted pre-flight when a binary check's LlmJudge pass rate on the constraint's
/// `pass` rubric falls below `gap_k1.coherence_threshold`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintCoherenceWarning {
    pub constraint_id: String,
    pub check_index: usize,
    /// Fraction of probe runs that returned Pass (0.0–1.0).
    pub consistency: f64,
    pub timestamp: DateTime<Utc>,
}

/// Emitted at runtime when Jaccard similarity between rejection reasons for the same
/// constraint across consecutive waves falls below `gap_k1.instability_threshold`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierInstabilityEvent {
    pub task_id: TaskId,
    pub constraint_id: String,
    /// Mean pairwise Jaccard of divergent rejection reasons — lower = more divergent.
    pub instability_score: f64,
    pub wave_a: u32,
    pub wave_b: u32,
    /// Up to 5 semantically distinct rejection reasons that showed divergence.
    pub divergent_reasons: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when `SpecRepairAdvisor` starts generating candidate rewrites.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintRepairAttempted {
    pub task_id: TaskId,
    pub constraint_id: String,
    pub check_index: usize,
    pub candidate_count: usize,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when `create_next_version` CAS write succeeds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintVersionCreated {
    pub task_id: TaskId,
    pub constraint_id: String,
    pub old_version: u64,
    pub new_version: u64,
    pub timestamp: DateTime<Utc>,
}

/// Emitted when the best repair candidate scored below `repair_acceptance_threshold`.
/// Causes fallthrough to HITL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintRepairFailed {
    pub task_id: TaskId,
    pub constraint_id: String,
    pub check_index: usize,
    /// Best candidate consistency score achieved.
    pub best_score: f64,
    pub timestamp: DateTime<Utc>,
}

/// Discriminated union of all events published to the NATS event stream by the orchestrator.
///
/// Serialised with an `event_type` tag and a `payload` content field for downstream consumers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type", content = "payload")]
pub enum H2AIEvent {
    /// Wraps [`CalibrationCompletedEvent`]: calibration harness finished.
    CalibrationCompleted(CalibrationCompletedEvent),
    /// Emitted when calibration fails (e.g. LLM adapter unreachable).
    CalibrationFailed {
        calibration_id: String,
        reason: String,
    },
    /// Wraps [`TaskBootstrappedEvent`]: task context compiled and `J_eff` gate passed.
    TaskBootstrapped(TaskBootstrappedEvent),
    /// Wraps [`TopologyProvisionedEvent`]: planner selected topology and explorer roles.
    TopologyProvisioned(TopologyProvisionedEvent),
    /// Wraps [`MultiplicationConditionFailedEvent`]: multiplication condition gate rejected the topology.
    MultiplicationConditionFailed(MultiplicationConditionFailedEvent),
    /// Wraps [`ProposalEvent`]: an explorer completed its TAO loop and produced output.
    Proposal(ProposalEvent),
    /// Wraps [`ProposalFailedEvent`]: an explorer's TAO loop terminated without usable output.
    ProposalFailed(ProposalFailedEvent),
    /// Wraps [`GenerationPhaseCompletedEvent`]: all explorers in Phase 3 finished.
    GenerationPhaseCompleted(GenerationPhaseCompletedEvent),
    /// Wraps [`ReviewGateTriggeredEvent`]: a review gate routed a proposal to a reviewer.
    ReviewGateTriggered(ReviewGateTriggeredEvent),
    /// Wraps [`ReviewGateBlockedEvent`]: a reviewer rejected a proposal at a review gate.
    ReviewGateBlocked(ReviewGateBlockedEvent),
    /// Wraps [`ValidationEvent`]: verifier started scoring an explorer's proposal.
    Validation(ValidationEvent),
    /// Wraps [`BranchPrunedEvent`]: an explorer's proposal was eliminated by verification or the auditor.
    BranchPruned(BranchPrunedEvent),
    /// Wraps [`ZeroSurvivalEvent`]: all proposals were pruned, triggering MAPE-K retry.
    ZeroSurvival(ZeroSurvivalEvent),
    /// Wraps [`InterfaceSaturationWarningEvent`]: active subtask count is approaching `interface_n_max`.
    InterfaceSaturationWarning(InterfaceSaturationWarningEvent),
    /// Wraps [`ConsensusRequiredEvent`]: error costs exceed the BFT threshold, switching to consensus merge.
    ConsensusRequired(ConsensusRequiredEvent),
    /// Wraps [`SelectionResolvedEvent`]: merge engine finished selecting surviving proposals.
    SelectionResolved(SelectionResolvedEvent),
    /// Wraps [`MergeResolvedEvent`]: final resolved output string produced for the task.
    MergeResolved(MergeResolvedEvent),
    /// Wraps [`TaskFailedEvent`]: MAPE-K loop exhausted retries without resolving.
    TaskFailed(TaskFailedEvent),
    /// Wraps [`CoherenceIncompleteEvent`]: resolved output has uncovered constraint domains.
    CoherenceIncomplete(CoherenceIncompleteEvent),
    /// Wraps [`TaoIterationEvent`]: one TAO loop turn completed with its observation and pass/fail status.
    TaoIteration(TaoIterationEvent),
    /// Wraps [`VerificationScoredEvent`]: LLM-as-Judge assigned a compliance score to a proposal.
    VerificationScored(VerificationScoredEvent),
    /// Wraps [`SubtaskPlanCreatedEvent`]: orchestrator created a decomposition plan.
    SubtaskPlanCreated(SubtaskPlanCreatedEvent),
    /// Wraps [`SubtaskPlanReviewedEvent`]: reviewer approved or rejected a decomposition plan.
    SubtaskPlanReviewed(SubtaskPlanReviewedEvent),
    /// Wraps [`SubtaskStartedEvent`]: an individual subtask began execution.
    SubtaskStarted(SubtaskStartedEvent),
    /// Wraps [`SubtaskCompletedEvent`]: an individual subtask finished successfully.
    SubtaskCompleted(SubtaskCompletedEvent),
    /// Wraps [`TaskAttributionEvent`]: quality attribution snapshot for a completed task.
    TaskAttribution(TaskAttributionEvent),
    /// Wraps [`EpistemicYieldEvent`]: semantic independence of surviving proposals (async, post-merge).
    EpistemicYield(EpistemicYieldEvent),
    /// Wraps [`TaskComplexityAssessedEvent`]: Phase 1.5 task complexity and routing quadrant.
    TaskComplexityAssessed(TaskComplexityAssessedEvent),
    /// Wraps [`ConstraintFrontierEvent`]: Pareto frontier coverage of constraint satisfaction matrix.
    ConstraintFrontier(ConstraintFrontierEvent),
    /// Wraps [`OraclePendingEvent`]: Phase 6 oracle evaluation dispatched.
    OraclePending(OraclePendingEvent),
    /// Wraps [`OracleResultEvent`]: oracle sidecar returned a result.
    OracleResult(OracleResultEvent),
    /// Wraps [`CalibrationDriftWarning`]: ECE exceeded 0.15 threshold.
    CalibrationDrift(CalibrationDriftWarning),
    /// Wraps [`OracleSuspectEvent`]: oracle pass rate outside healthy range.
    OracleSuspect(OracleSuspectEvent),
    /// Task output is pending human approval before delivery.
    PendingApproval(PendingApprovalEvent),
    /// Human or system timeout decision on a pending approval.
    ApprovalResolved(ApprovalResolvedEvent),
    /// Wraps [`VerifierComparisonEvent`]: dual-run verifier comparison data point.
    VerifierComparison(VerifierComparisonEvent),
    /// Per-proposal shadow auditor outcome (shadow mode).
    ShadowAudit(ShadowAuditorResultEvent),
    /// Domain promoted to two-auditor majority-vote mode by `ShadowAuditorAccumulator`.
    AuditDomainPromoted(AuditDomainPromotedEvent),
    /// Domain demoted from majority-vote mode (disagreement rate fell below threshold/2).
    AuditDomainDemoted(AuditDomainDemotedEvent),
    /// C1 correlated ensemble warning — low Jaccard CV detected post-Phase-3.
    CorrelatedEnsemble(CorrelatedEnsembleWarning),
    /// Researcher grounding event — external knowledge fetched for C1 retry or proactive slot.
    ResearcherGrounding(ResearcherGroundingEvent),
    /// C3 domain coverage degradation — slot assignment covers insufficient corpus domains.
    DiversityGuardDegraded(DiversityGuardDegradedEvent),
    /// SRANI correlated fabrication warning — shared ungrounded entities detected across proposals.
    CorrelatedFabrication(CorrelatedFabricationEvent),
    /// Pre-execution thinking loop completed (or was skipped). Always emitted per task.
    ThinkingLoopCompleted(ThinkingLoopCompletedEvent),
    /// Oracle accumulated enough observations to replace heuristic `p_mean` with `pass_rate`.
    OracleCalibrationPatched(OracleCalibrationPatchedEvent),
    /// Oracle gate finished evaluating proposals before merge; records pass/fail and confidence.
    OracleGateResult(OracleGateResultEvent),
    /// Orchestrator requires human clarification before continuing task execution.
    PendingClarification(PendingClarificationEvent),
    /// OPRO triggered: `j_eff` EMA fell below threshold.
    OproTriggered(OproTriggeredEvent),
    /// Prompt variant promoted: OPRO selected a winning variant.
    PromptVariantPromoted(PromptVariantPromotedEvent),
    /// Corpus quality signal: judge panel disagreement persistent across proposals in a wave.
    ConstraintAmbiguity(ConstraintAmbiguityEvent),
    /// Wraps [`LeaderElectedEvent`]: Krum-elected leader installed or rotated for a new wave term.
    LeaderElected(LeaderElectedEvent),
    /// Wraps [`SocraticDiagnosisEvent`]: leader's Socratic diagnostic question for the current wave.
    SocraticDiagnosis(SocraticDiagnosisEvent),
    /// Wraps [`VerifierReasonContradictionEvent`]: contradictory verifier reasons detected across proposals for the same constraint.
    VerifierReasonContradiction(VerifierReasonContradictionEvent),
    /// Pre-dispatch complexity probe completed.
    ComplexityProbe(ComplexityProbeEvent),
    /// Intra-retry ceiling detector fired.
    ComplexityCeilingDetected(ComplexityCeilingDetectedEvent),
    /// Wraps [`TieredExitEvent`]: GAP-L1 tiered exit fired — K-of-N proposals accepted.
    TieredExit(TieredExitEvent),
    /// GAP-H3: per-task budget reached warning threshold.
    CostThresholdWarning(CostThresholdWarningEvent),
    /// GAP-H3: per-task budget exhausted; retries blocked.
    BudgetExhausted(BudgetExhaustedEvent),
    /// GAP-H3: convergence gate fired; verified proposals are semantically equivalent.
    ConvergenceGate(ConvergenceGateEvent),
    /// Constraint coherence warning — binary check consistency below threshold.
    ConstraintCoherenceWarning(ConstraintCoherenceWarning),
    /// Verifier instability event — rejection reasons diverge across waves.
    VerifierInstability(VerifierInstabilityEvent),
    /// Repair attempt started — SpecRepairAdvisor generating candidates.
    ConstraintRepairAttempted(ConstraintRepairAttempted),
    /// Version created — CAS write succeeded for new constraint version.
    ConstraintVersionCreated(ConstraintVersionCreated),
    /// Repair failed — best candidate below acceptance threshold.
    ConstraintRepairFailed(ConstraintRepairFailed),
}

impl H2AIEvent {
    #[must_use]
    pub fn subject(&self, task_id: &TaskId) -> String {
        match self {
            Self::PendingApproval(e) => {
                format!("h2ai.tasks.{}.pending_approval", e.task_id)
            }
            Self::ApprovalResolved(e) => {
                format!("h2ai.tasks.{}.approval_resolved", e.task_id)
            }
            _ => format!("h2ai.tasks.{task_id}"),
        }
    }
}

#[cfg(test)]
mod gap_k1_event_tests {
    use super::*;

    #[test]
    fn constraint_coherence_warning_round_trips() {
        let e = ConstraintCoherenceWarning {
            constraint_id: "C-1".into(),
            check_index: 0,
            consistency: 0.4,
            timestamp: chrono::Utc::now(),
        };
        let json = serde_json::to_string(&e).unwrap();
        let back: ConstraintCoherenceWarning = serde_json::from_str(&json).unwrap();
        assert_eq!(back.check_index, 0);
    }

    #[test]
    fn verifier_instability_event_round_trips() {
        let e = VerifierInstabilityEvent {
            task_id: TaskId::new(),
            constraint_id: "C-1".into(),
            instability_score: 0.034,
            wave_a: 1,
            wave_b: 2,
            divergent_reasons: vec!["reason A".into(), "reason B".into()],
            timestamp: chrono::Utc::now(),
        };
        let json = serde_json::to_string(&e).unwrap();
        let back: VerifierInstabilityEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.divergent_reasons.len(), 2);
    }

    #[test]
    fn constraint_version_created_round_trips() {
        let e = ConstraintVersionCreated {
            task_id: TaskId::new(),
            constraint_id: "C-1".into(),
            old_version: 1,
            new_version: 2,
            timestamp: chrono::Utc::now(),
        };
        let json = serde_json::to_string(&e).unwrap();
        let back: ConstraintVersionCreated = serde_json::from_str(&json).unwrap();
        assert_eq!(back.old_version, 1);
        assert_eq!(back.new_version, 2);
    }

    #[test]
    fn constraint_repair_attempted_round_trips() {
        let e = ConstraintRepairAttempted {
            task_id: TaskId::new(),
            constraint_id: "C-1".into(),
            check_index: 0,
            candidate_count: 3,
            timestamp: chrono::Utc::now(),
        };
        let json = serde_json::to_string(&e).unwrap();
        let back: ConstraintRepairAttempted = serde_json::from_str(&json).unwrap();
        assert_eq!(back.candidate_count, 3);
    }

    #[test]
    fn constraint_repair_failed_round_trips() {
        let e = ConstraintRepairFailed {
            task_id: TaskId::new(),
            constraint_id: "C-1".into(),
            check_index: 0,
            best_score: 0.42,
            timestamp: chrono::Utc::now(),
        };
        let json = serde_json::to_string(&e).unwrap();
        let back: ConstraintRepairFailed = serde_json::from_str(&json).unwrap();
        assert_eq!(back.best_score, 0.42);
    }

    #[test]
    fn complexity_probe_event_roundtrip() {
        use chrono::Utc;
        let ev = ComplexityProbeEvent {
            task_id: TaskId::new(),
            complexity: 4,
            rationale: "formal proof".into(),
            decompose_recommended: true,
            probe_latency_ms: 250,
            timestamp: Utc::now(),
        };
        let json = serde_json::to_string(&ev).unwrap();
        let back: ComplexityProbeEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.complexity, 4);
        assert!(back.decompose_recommended);
    }

    #[test]
    fn verifier_reason_contradiction_beyond_budget_defaults() {
        // Existing events without the field must deserialise cleanly (backward compat).
        let json = r#"{"task_id":"00000000-0000-0000-0000-000000000001","wave":1,"constraint_id":"c1","reasons":[],"min_jaccard":0.1,"fallback_hint":null,"timestamp":"2026-01-01T00:00:00Z"}"#;
        let ev: VerifierReasonContradictionEvent = serde_json::from_str(json).unwrap();
        assert_eq!(ev.beyond_budget_count, 0);
    }
}

#[cfg(test)]
mod tiered_exit_event_tests {
    use super::*;

    #[test]
    fn tiered_exit_event_serializes() {
        let evt = TieredExitEvent {
            wave: 1,
            n: 3,
            k_required: 2,
            k_accepted: 3,
            acceptance_score: 0.85,
        };
        let json = serde_json::to_string(&evt).expect("serialize");
        assert!(json.contains("\"wave\":1"));
        assert!(json.contains("\"n\":3"));
        assert!(json.contains("\"k_required\":2"));
        assert!(json.contains("\"k_accepted\":3"));
    }

    #[test]
    fn h2ai_event_tee_roundtrip() {
        let evt = TieredExitEvent {
            wave: 0,
            n: 1,
            k_required: 1,
            k_accepted: 1,
            acceptance_score: 0.9,
        };
        let wrapped = H2AIEvent::TieredExit(evt.clone());
        let json = serde_json::to_string(&wrapped).expect("serialize");
        let back: H2AIEvent = serde_json::from_str(&json).expect("deserialize");
        if let H2AIEvent::TieredExit(inner) = back {
            assert_eq!(inner.wave, 0);
            assert_eq!(inner.n, 1);
        } else {
            panic!("wrong variant");
        }
    }
}

#[cfg(test)]
mod cost_guard_event_tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn cost_threshold_warning_event_serializes() {
        let evt = CostThresholdWarningEvent {
            task_id: TaskId::new(),
            tokens_used: 8_000,
            budget_tokens: 10_000,
            fraction_used: 0.80,
            timestamp: Utc::now(),
        };
        let s = serde_json::to_string(&evt).unwrap();
        let back: CostThresholdWarningEvent = serde_json::from_str(&s).unwrap();
        assert_eq!(back.tokens_used, 8_000);
        assert!((back.fraction_used - 0.80).abs() < 1e-9);
    }

    #[test]
    fn budget_exhausted_event_serializes() {
        let evt = BudgetExhaustedEvent {
            task_id: TaskId::new(),
            tokens_used: 10_500,
            budget_tokens: 10_000,
            timestamp: Utc::now(),
        };
        let s = serde_json::to_string(&evt).unwrap();
        let back: BudgetExhaustedEvent = serde_json::from_str(&s).unwrap();
        assert_eq!(back.tokens_used, 10_500);
    }

    #[test]
    fn convergence_gate_event_serializes() {
        let evt = ConvergenceGateEvent {
            task_id: TaskId::new(),
            wave: 1,
            n_live: 2,
            convergence_fraction: 1.0,
            theta_converge: 0.87,
            best_score: 0.83,
            timestamp: Utc::now(),
        };
        let s = serde_json::to_string(&evt).unwrap();
        let back: ConvergenceGateEvent = serde_json::from_str(&s).unwrap();
        assert_eq!(back.wave, 1);
        assert!((back.convergence_fraction - 1.0).abs() < 1e-9);
    }

    #[test]
    fn h2ai_event_wraps_cost_events() {
        let warn = CostThresholdWarningEvent {
            task_id: TaskId::new(),
            tokens_used: 8_000,
            budget_tokens: 10_000,
            fraction_used: 0.80,
            timestamp: Utc::now(),
        };
        let wrapped = H2AIEvent::CostThresholdWarning(warn);
        let s = serde_json::to_string(&wrapped).unwrap();
        let back: H2AIEvent = serde_json::from_str(&s).unwrap();
        if let H2AIEvent::CostThresholdWarning(inner) = back {
            assert_eq!(inner.tokens_used, 8_000);
        } else {
            panic!("wrong variant");
        }
    }
}
