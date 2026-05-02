use crate::diagnostics::TalagrandDiagnostic;
pub use crate::nats_dispatch_adapter::NatsDispatchConfig;
use crate::self_optimizer::{OptimizerParams, QualityMeasurement, SelfOptimizer, SuggestInput};
use crate::task_store::{TaskPhase, TaskState, TaskStore};
use chrono::Utc;
use futures::future::join_all;
use h2ai_autonomic::checker::MultiplicationChecker;
use h2ai_autonomic::merger::{MergeEngine, MergeOutcome};
use h2ai_autonomic::planner::{ProvisionInput, TopologyPlanner};
use h2ai_autonomic::retry::{RetryAction, RetryPolicy};
use h2ai_config::H2AIConfig;
use h2ai_constraints::types::ConstraintDoc;
use h2ai_context::compaction::{compact, CompactionConfig};
use h2ai_context::compiler;
use h2ai_context::embedding::EmbeddingModel;
use h2ai_state::semilattice::ProposalSet;
use h2ai_types::adapter::{AdapterRegistry, ComputeRequest, IComputeAdapter};
use h2ai_types::config::{
    AgentRole, AuditorConfig, RoleSpec, TaoConfig, TopologyKind, VerificationConfig,
};
use h2ai_types::events::{
    BranchPrunedEvent, CalibrationCompletedEvent, GenerationPhaseCompletedEvent, ProposalEvent,
    ProposalFailedEvent, ProposalFailureReason, SemilatticeCompiledEvent, TaskBootstrappedEvent,
    VerificationScoredEvent, ZeroSurvivalEvent,
};
use h2ai_types::identity::TaskId;
use h2ai_types::manifest::TaskManifest;
use h2ai_types::physics::{
    MergeStrategy, MultiplicationConditionFailure, PredictionBasis, RoleErrorCost, TauValue,
};
use thiserror::Error;

/// Errors that can abort an `ExecutionEngine::run_offline` call.
#[derive(Debug, Error)]
pub enum EngineError {
    /// The compiled system context carries too little relevant signal (J_eff below threshold).
    /// Retrying with a richer constraint corpus or broader keyword set may help.
    #[error("context underflow: J_eff={j_eff:.3} < {threshold:.1}")]
    ContextUnderflow { j_eff: f64, threshold: f64 },
    /// The multiplication condition gate rejected all topologies across all retries.
    /// Recalibrating with higher-quality or more diverse adapters may resolve this.
    #[error("multiplication condition failed: {0}")]
    MultiplicationConditionFailed(String),
    /// The MAPE-K autonomic retry loop hit `max_autonomic_retries` without resolving.
    /// Increasing the retry budget or investigating calibration data is recommended.
    #[error("max retries exhausted")]
    MaxRetriesExhausted,
    /// An adapter call failed or timed out; the message contains the error detail.
    /// May be transient — retrying at the caller level is reasonable.
    #[error("adapter error: {0}")]
    Adapter(String),
    /// Output from a step could not be parsed (e.g. invalid JSON, bad regex pattern).
    /// Indicates a configuration or adapter output format error; retrying is unlikely to help.
    #[error("parse error: {0}")]
    Parse(String),
    /// The wall-clock task deadline was exceeded before the engine resolved.
    /// Increase `task_deadline_secs` or reduce ensemble size to fit the budget.
    #[error("task deadline exceeded (budget {budget_secs}s)")]
    DeadlineExceeded { budget_secs: u64 },
    /// The provisioned ensemble is too small for the requested OutlierResistant fault bound.
    /// Either reduce `f` or provision at least `2f + 3` explorers.
    #[error("insufficient quorum for OutlierResistant f={f}: need n ≥ {required}, got n={n}")]
    InsufficientQuorum { n: usize, f: usize, required: usize },
}

/// All inputs required to run the multi-phase execution pipeline for a single task.
pub struct EngineInput<'a> {
    /// Unique identifier for the task being executed.
    pub task_id: TaskId,
    /// Task manifest containing description, constraints, Pareto weights, and explorer spec.
    pub manifest: TaskManifest,
    /// Calibration event carrying α, β₀, CG samples, and optional ensemble/eigen calibration.
    pub calibration: CalibrationCompletedEvent,
    /// Pool of compute adapters shared across explorer slots (round-robin indexed by position).
    pub explorer_adapters: Vec<&'a dyn IComputeAdapter>,
    /// Scores proposals in Phase 3.5. Must return `{"score": float, "reason": "..."}`.
    pub verification_adapter: &'a dyn IComputeAdapter,
    /// Approves/rejects proposals in Phase 4. Must return `{"approved": bool, "reason": "..."}`.
    pub auditor_adapter: &'a dyn IComputeAdapter,
    /// Configuration for the auditor adapter (prompt template, τ, token budget).
    pub auditor_config: AuditorConfig,
    /// TAO loop configuration applied to every explorer (turns, patterns, repetition threshold).
    pub tao_config: TaoConfig,
    /// Verification phase configuration (LLM-as-Judge threshold and prompt settings).
    pub verification_config: VerificationConfig,
    /// Constraint corpus loaded from the ADR/design-doc index; used for context compilation and scoring.
    pub constraint_corpus: Vec<ConstraintDoc>,
    /// Runtime configuration (retries, deadlines, context token budget, thresholds).
    pub cfg: &'a H2AIConfig,
    /// In-memory task state store for phase and validation tracking.
    pub store: TaskStore,
    /// When Some, each explorer slot gets a NatsDispatchAdapter instead of
    /// drawing from explorer_adapters. explorer_adapters may be empty.
    pub nats_dispatch: Option<NatsDispatchConfig>,
    /// Adapter registry for profile-based routing.
    pub registry: &'a AdapterRegistry,
    /// Optional embedding model for Weiszfeld geometric median and cosine similarity.
    /// When `Some`, enables the Weiszfeld path in incoherent merge clusters.
    pub embedding_model: Option<&'a dyn EmbeddingModel>,
    /// Pre-task snapshot of TaoMultiplierEstimator::multiplier().
    /// Used for tao_per_turn_factor in AttributionInput so attribution reflects
    /// what was known at dispatch time, not the mid-task update.
    pub tao_multiplier: f64,
    /// Shared estimator updated with (turn-1 score, final score) pairs after
    /// each iteration's verification. Persisted by tasks.rs after engine returns.
    pub tao_estimator: std::sync::Arc<tokio::sync::RwLock<crate::tao_loop::TaoMultiplierEstimator>>,
}

/// Successful result returned by `ExecutionEngine::run_offline` after all phases complete.
#[derive(Debug)]
pub struct EngineOutput {
    /// Identifier of the task that was resolved.
    pub task_id: TaskId,
    /// Final merged output string produced by the merge engine.
    pub resolved_output: String,
    /// Semilattice compilation event describing which proposals survived and the merge strategy used.
    pub semilattice: SemilatticeCompiledEvent,
    /// Quality attribution snapshot (Q_total, components) computed at resolve time.
    pub attribution: crate::attribution::HarnessAttribution,
    /// Bootstrap CI over Q_total from CG sample variance. `None` when < 2 CG samples.
    pub attribution_interval: Option<crate::attribution::AttributionInterval>,
    /// All verification scored events collected across every MAPE-K retry iteration.
    pub verification_events: Vec<VerificationScoredEvent>,
    /// Rank-histogram calibration diagnostic built from this task's verification scores.
    /// `None` when no verification events were produced.
    /// Typically `Some(Insufficient)` state for a single task (< 20 runs needed for calibration).
    pub talagrand: Option<crate::diagnostics::TalagrandDiagnostic>,
    /// SelfOptimizer suggestion for the next task run, computed from this run's quality.
    /// `None` only when no quality history was accumulated (should not happen on success).
    /// Callers may apply this to their next `EngineInput` to improve throughput.
    pub suggested_next_params: Option<crate::self_optimizer::OptimizerParams>,
    /// Fraction of dispatched proposals that survived verification (valid / total_evaluated).
    /// 1.0 = no waste; below `cfg.optimizer_waste_threshold` = wasteful run.
    pub waste_ratio: f64,
    /// SelfOptimizer suggestions derived from this wasteful-but-successful run.
    /// Empty when not wasteful or no applicable suggestion was found.
    /// Callers should apply these to AppState (τ spread EMA, topology hint).
    pub applied_optimizations: Vec<h2ai_types::events::AppliedOptimization>,
}

#[derive(serde::Deserialize)]
struct AuditResponse {
    approved: bool,
    #[allow(dead_code)]
    reason: String,
}

/// Stateless coordinator for the five-phase task execution pipeline.
///
/// Orchestrates context compilation (Phase 1), topology provisioning (Phase 2),
/// the multiplication condition gate (Phase 2.5), parallel generation (Phase 3),
/// verification (Phase 3.5), auditor gate (Phase 4), and merge (Phase 5).
/// Wraps all phases in a MAPE-K autonomic retry loop that adjusts topology,
/// τ spread, and optimizer parameters on each failure before giving up.
pub struct ExecutionEngine;

impl ExecutionEngine {
    /// Run all five phases with the MAPE-K autonomic retry loop; no NATS publishing.
    ///
    /// Suitable for unit tests and offline evaluation because it does not require a
    /// live NATS connection — all events remain in-process via `TaskStore`.
    /// Returns `EngineOutput` on the first successful merge, or an `EngineError` when
    /// all retries are exhausted or a non-retryable condition is encountered.
    pub async fn run_offline(input: EngineInput<'_>) -> Result<EngineOutput, EngineError> {
        let task_id = input.task_id.clone();
        input
            .store
            .insert(task_id.clone(), TaskState::new(task_id.clone()));

        // ── Phase 1: Bootstrap ──────────────────────────────────────────────
        let description = &input.manifest.description;
        let required_kw = input
            .constraint_corpus
            .iter()
            .flat_map(|d: &ConstraintDoc| d.vocabulary().into_iter())
            .chain(input.manifest.constraints.iter().cloned())
            .collect::<Vec<_>>()
            .join(" ");
        let required_kw = if required_kw.is_empty() {
            description.clone()
        } else {
            required_kw
        };

        let compiled = compiler::compile(
            description,
            &input.constraint_corpus,
            &required_kw,
            input.cfg,
            input.embedding_model,
        )
        .await
        .map_err(|e| {
            let h2ai_context::compiler::ContextError::ContextUnderflow { j_eff, threshold } = e;
            EngineError::ContextUnderflow { j_eff, threshold }
        })?;

        let adr_keywords: Vec<String> = input
            .constraint_corpus
            .iter()
            .flat_map(|d: &ConstraintDoc| d.vocabulary().into_iter())
            .chain(input.manifest.constraints.iter().cloned())
            .collect();
        let system_context = compact(
            &compiled.system_context,
            &CompactionConfig {
                max_tokens: input.cfg.max_context_tokens.unwrap_or(usize::MAX / 4),
                preserve_keywords: adr_keywords,
            },
        );

        let _bootstrapped = TaskBootstrappedEvent {
            task_id: task_id.clone(),
            system_context: system_context.clone(),
            pareto_weights: input.manifest.pareto_weights.clone(),
            j_eff: compiled.j_eff,
            timestamp: Utc::now(),
        };

        let explorer_adapter_kind = input
            .explorer_adapters
            .first()
            .map(|a| a.kind().clone())
            .unwrap_or_else(|| input.auditor_config.adapter.clone());

        let cg_mean = input.calibration.coefficients.cg_mean();
        let n_max_ceiling = input.calibration.coefficients.n_max().floor() as u32;

        // ── MAPE-K retry state ───────────────────────────────────────────────
        // When EnsembleCalibration is present use n_optimal (Condorcet-derived) as the
        // default ensemble size instead of the manifest count. n_max_ceiling (Amdahl) is
        // the hard ceiling regardless.
        let n_optimal_hint = input
            .calibration
            .ensemble
            .as_ref()
            .map(|ec| ec.n_optimal as u32)
            .unwrap_or(input.manifest.explorers.count as u32);
        let initial_n_agents = n_optimal_hint.max(1).min(n_max_ceiling.max(1));
        let mut current_params = OptimizerParams {
            n_agents: initial_n_agents,
            max_turns: input.tao_config.max_turns as u32,
            verify_threshold: input.verification_config.threshold,
        };
        let mut tao_config = input.tao_config.clone();
        let mut verification_config = input.verification_config.clone();
        let mut force_topology: Option<TopologyKind> = None;
        let mut tried_topologies: Vec<TopologyKind> = Vec::new();
        let mut tau_reduction_factor: f64 = 1.0;
        // τ-spread expansion factor driven by Talagrand U-curve detection.
        // Starts at 1.0; increases by 20% per over-confident iteration, capped at tau_spread_max_factor.
        let mut tau_spread_factor: f64 = 1.0;
        let mut all_pruned: Vec<BranchPrunedEvent> = Vec::new();
        let mut tau_values_tried: Vec<Vec<f64>> = Vec::new();
        let mut quality_history: Vec<QualityMeasurement> = Vec::new();
        let mut all_verification_events: Vec<VerificationScoredEvent> = Vec::new();
        let mut last_multiplication_failure: Option<MultiplicationConditionFailure> = None;

        let task_deadline = input
            .cfg
            .task_deadline_secs
            .map(|s| std::time::Instant::now() + std::time::Duration::from_secs(s));

        for retry_count in 0..=input.cfg.max_autonomic_retries {
            if let Some(dl) = task_deadline {
                if std::time::Instant::now() >= dl {
                    input.store.mark_failed(&task_id);
                    return Err(EngineError::DeadlineExceeded {
                        budget_secs: input.cfg.task_deadline_secs.unwrap_or(0),
                    });
                }
            }
            // ── Phase 2: Topology Provisioning ─────────────────────────────
            let role_specs: Vec<RoleSpec> = if input.manifest.explorers.roles.is_empty() {
                let count = current_params.n_agents.max(1);
                let tau_min_manifest = input.manifest.explorers.tau_min.unwrap_or(0.2);
                let tau_max_manifest = input.manifest.explorers.tau_max.unwrap_or(0.9);
                // Apply τ-spread expansion (Talagrand U-curve feedback) around the manifest centre.
                let tau_center = (tau_max_manifest + tau_min_manifest) / 2.0;
                let half_spread = (tau_max_manifest - tau_min_manifest) / 2.0;
                let max_half = tau_center.min(1.0 - tau_center); // can't exceed [0,1]
                let expanded_half = (half_spread * tau_spread_factor).min(max_half);
                let tau_min = tau_center - expanded_half;
                let tau_max = tau_center + expanded_half;
                let step = if count > 1 {
                    (tau_max - tau_min) / (count - 1) as f64
                } else {
                    0.0
                };
                (0..count)
                    .map(|i| RoleSpec {
                        agent_id: format!("exp_{}", (b'A' + (i % 26) as u8) as char),
                        role: AgentRole::Executor,
                        tau: Some(
                            TauValue::new(
                                ((tau_min + step * i as f64) * tau_reduction_factor)
                                    .clamp(0.05, 0.95),
                            )
                            .unwrap_or_else(|_| TauValue::new(0.05).unwrap()),
                        ),
                        role_error_cost: None,
                    })
                    .collect()
            } else {
                input.manifest.explorers.roles.clone()
            };

            input
                .store
                .set_phase(&task_id, TaskPhase::Provisioning, 0, retry_count);

            let (provisioned, _cg_collapse) = TopologyPlanner::provision(ProvisionInput {
                task_id: task_id.clone(),
                cc: &input.calibration.coefficients,
                pareto_weights: &input.manifest.pareto_weights,
                role_specs: &role_specs,
                review_gates: input.manifest.explorers.review_gates.clone(),
                auditor_config: input.auditor_config.clone(),
                explorer_adapter: explorer_adapter_kind.clone(),
                force_topology: force_topology.clone(),
                retry_count,
                cfg: input.cfg,
                eigen: input.calibration.eigen.as_ref(),
            });

            tried_topologies.push(provisioned.topology_kind.clone());
            let explorer_count = provisioned.explorer_configs.len() as u32;
            current_params.n_agents = explorer_count;

            // Guard: OutlierResistant requires n ≥ 2f+3. Fail early rather than silently falling back.
            if let MergeStrategy::OutlierResistant { f }
            | MergeStrategy::MultiOutlierResistant { f, .. } = &provisioned.merge_strategy
            {
                let f = *f;
                let n = provisioned.explorer_configs.len();
                let required = MergeStrategy::min_krum_quorum(f);
                if n < required {
                    return Err(EngineError::InsufficientQuorum { n, f, required });
                }
            }

            // ── Phase 2.5: Multiplication Condition Gate ───────────────────
            input.store.set_phase(
                &task_id,
                TaskPhase::MultiplicationCheck,
                explorer_count,
                retry_count,
            );

            // Derive p_mean, rho_mean, and prediction_basis from EnsembleCalibration when available.
            // Fallback proxies when calibration is absent (Heuristic basis):
            //   p = 0.5 + CG_mean / 2  (accuracy proxy from output similarity)
            //   ρ = 1 - CG_mean        (correlation proxy from output similarity)
            let (p_mean, rho_mean, attribution_basis) = match &input.calibration.ensemble {
                Some(ec) => (ec.p_mean, ec.rho_mean, ec.prediction_basis),
                None => (
                    0.5 + cg_mean / 2.0,
                    (1.0 - cg_mean).clamp(0.0, 1.0),
                    PredictionBasis::Heuristic,
                ),
            };
            let baseline_competence = p_mean;
            let error_correlation = rho_mean;

            if let Err(mc_event) = MultiplicationChecker::check(
                &task_id,
                &input.calibration.coefficients,
                &input.calibration.coordination_threshold,
                baseline_competence,
                error_correlation,
                retry_count,
                input.cfg,
            ) {
                last_multiplication_failure = Some(mc_event.failure.clone());
                let tau_values: Vec<f64> = provisioned
                    .explorer_configs
                    .iter()
                    .map(|ec| ec.tau.value())
                    .collect();
                tau_values_tried.push(tau_values);

                let zero_event = ZeroSurvivalEvent {
                    task_id: task_id.clone(),
                    retry_count,
                    timestamp: Utc::now(),
                };
                match RetryPolicy::decide(
                    &zero_event,
                    &tried_topologies,
                    all_pruned.clone(),
                    tau_values_tried.clone(),
                    last_multiplication_failure.clone(),
                ) {
                    RetryAction::Retry(next_topology) => {
                        force_topology = Some(next_topology);
                        Self::apply_optimizer(
                            &mut current_params,
                            &mut tao_config,
                            &mut verification_config,
                            &quality_history,
                            n_max_ceiling,
                            cg_mean,
                            1.0,
                            input.calibration.ensemble.as_ref(),
                            input.cfg,
                        );
                        continue;
                    }
                    RetryAction::RetryWithTauReduction {
                        topology,
                        tau_factor,
                    } => {
                        force_topology = Some(topology);
                        tau_reduction_factor *= tau_factor;
                        Self::apply_optimizer(
                            &mut current_params,
                            &mut tao_config,
                            &mut verification_config,
                            &quality_history,
                            n_max_ceiling,
                            cg_mean,
                            1.0,
                            input.calibration.ensemble.as_ref(),
                            input.cfg,
                        );
                        continue;
                    }
                    RetryAction::RetryWithHints { topology, .. } => {
                        force_topology = Some(topology);
                        Self::apply_optimizer(
                            &mut current_params,
                            &mut tao_config,
                            &mut verification_config,
                            &quality_history,
                            n_max_ceiling,
                            cg_mean,
                            1.0,
                            input.calibration.ensemble.as_ref(),
                            input.cfg,
                        );
                        continue;
                    }
                    RetryAction::Fail(_) => {
                        input.store.mark_failed(&task_id);
                        return Err(EngineError::MaxRetriesExhausted);
                    }
                }
            }

            // ── Phase 3: Parallel Generation ───────────────────────────────
            input.store.set_phase(
                &task_id,
                TaskPhase::ParallelGeneration,
                explorer_count,
                retry_count,
            );

            use crate::nats_dispatch_adapter::NatsDispatchAdapter;
            use std::future::Future;
            use std::pin::Pin;
            use std::sync::Arc;
            type ExplorerFuture<'f> = Pin<
                Box<
                    dyn Future<
                            Output = Result<
                                (ProposalEvent, u8, Option<String>),
                                ProposalFailedEvent,
                            >,
                        > + Send
                        + 'f,
                >,
            >;
            let futures_vec: Vec<ExplorerFuture<'_>> = provisioned
                .explorer_configs
                .iter()
                .enumerate()
                .map(|(idx, explorer_cfg)| {
                    let req = ComputeRequest {
                        system_context: system_context.clone(),
                        task: input.manifest.description.clone(),
                        tau: explorer_cfg.tau,
                        max_tokens: input.cfg.explorer_max_tokens,
                    };
                    let explorer_id = explorer_cfg.explorer_id.clone();
                    let task_id_clone = task_id.clone();
                    let tao_cfg = tao_config.clone();
                    if let Some(ref nd_cfg) = input.nats_dispatch {
                        let arc = Arc::new(NatsDispatchAdapter::new(NatsDispatchConfig {
                            nats: nd_cfg.nats.clone(),
                            provider: nd_cfg.provider.clone(),
                            agent_descriptor: nd_cfg.agent_descriptor.clone(),
                            task_requirements: nd_cfg.task_requirements.clone(),
                            task_timeout: nd_cfg.task_timeout,
                            payload_store: nd_cfg.payload_store.clone(),
                            offload_threshold_bytes: nd_cfg.offload_threshold_bytes,
                        }));
                        let generation = retry_count as u64;
                        let fut: ExplorerFuture<'_> = Box::pin(async move {
                            use crate::tao_loop::{TaoInput, TaoLoop};
                            match TaoLoop::run(TaoInput {
                                task_id: task_id_clone.clone(),
                                explorer_id: explorer_id.clone(),
                                adapter: arc.as_ref(),
                                initial_request: req,
                                config: tao_cfg,
                                schema_config: None,
                                generation,
                            })
                            .await
                            {
                                Ok(tao_proposal) => Ok((
                                    tao_proposal.event,
                                    tao_proposal.tao_turns,
                                    tao_proposal.turn1_output,
                                )),
                                Err(e) => Err(ProposalFailedEvent {
                                    task_id: task_id_clone,
                                    explorer_id,
                                    reason: ProposalFailureReason::AdapterError(e.to_string()),
                                    timestamp: Utc::now(),
                                }),
                            }
                        });
                        fut
                    } else {
                        let adapter_idx = idx % input.explorer_adapters.len();
                        let adapter = input.explorer_adapters[adapter_idx];
                        let generation = retry_count as u64;
                        let fut: ExplorerFuture<'_> = Box::pin(async move {
                            use crate::tao_loop::{TaoInput, TaoLoop};
                            match TaoLoop::run(TaoInput {
                                task_id: task_id_clone.clone(),
                                explorer_id: explorer_id.clone(),
                                adapter,
                                initial_request: req,
                                config: tao_cfg,
                                schema_config: None,
                                generation,
                            })
                            .await
                            {
                                Ok(tao_proposal) => Ok((
                                    tao_proposal.event,
                                    tao_proposal.tao_turns,
                                    tao_proposal.turn1_output,
                                )),
                                Err(e) => Err(ProposalFailedEvent {
                                    task_id: task_id_clone,
                                    explorer_id,
                                    reason: ProposalFailureReason::AdapterError(e.to_string()),
                                    timestamp: Utc::now(),
                                }),
                            }
                        });
                        fut
                    }
                })
                .collect();

            let results = join_all(futures_vec).await;

            let mut proposals: Vec<ProposalEvent> = Vec::new();
            let mut tao_turns_collected: Vec<u8> = Vec::new();
            let mut failed_proposals: Vec<ProposalFailedEvent> = Vec::new();
            let mut turn1_map: std::collections::HashMap<h2ai_types::identity::ExplorerId, String> =
                std::collections::HashMap::new();

            for result in results {
                match result {
                    Ok((proposal, turns, turn1_output)) => {
                        input.store.increment_completed(&task_id);
                        tao_turns_collected.push(turns);
                        if let Some(t1) = turn1_output {
                            turn1_map.insert(proposal.explorer_id.clone(), t1);
                        }
                        proposals.push(proposal);
                    }
                    Err(failed) => {
                        input.store.increment_completed(&task_id);
                        failed_proposals.push(failed);
                    }
                }
            }

            let tao_turns_mean = if tao_turns_collected.is_empty() {
                1.0
            } else {
                tao_turns_collected.iter().map(|&t| t as f64).sum::<f64>()
                    / tao_turns_collected.len() as f64
            };

            let _gen_completed = GenerationPhaseCompletedEvent {
                task_id: task_id.clone(),
                total_explorers: explorer_count,
                successful: proposals.len() as u32,
                failed: failed_proposals.len() as u32,
                timestamp: Utc::now(),
            };

            // Collect tau values for this batch before verification
            let tau_values: Vec<f64> = provisioned
                .explorer_configs
                .iter()
                .map(|ec| ec.tau.value())
                .collect();

            // Diversity gate: all pairwise proposal outputs too similar → collective hallucination.
            if crate::diversity::is_uniform(&proposals, input.cfg.diversity_threshold) {
                tau_values_tried.push(tau_values);
                let zero_event = ZeroSurvivalEvent {
                    task_id: task_id.clone(),
                    retry_count,
                    timestamp: Utc::now(),
                };
                match RetryPolicy::decide(
                    &zero_event,
                    &tried_topologies,
                    all_pruned.clone(),
                    tau_values_tried.clone(),
                    last_multiplication_failure.clone(),
                ) {
                    RetryAction::Retry(next_topology) => {
                        force_topology = Some(next_topology);
                        Self::apply_optimizer(
                            &mut current_params,
                            &mut tao_config,
                            &mut verification_config,
                            &quality_history,
                            n_max_ceiling,
                            cg_mean,
                            1.0,
                            input.calibration.ensemble.as_ref(),
                            input.cfg,
                        );
                        continue;
                    }
                    RetryAction::RetryWithTauReduction {
                        topology,
                        tau_factor,
                    } => {
                        force_topology = Some(topology);
                        tau_reduction_factor *= tau_factor;
                        Self::apply_optimizer(
                            &mut current_params,
                            &mut tao_config,
                            &mut verification_config,
                            &quality_history,
                            n_max_ceiling,
                            cg_mean,
                            1.0,
                            input.calibration.ensemble.as_ref(),
                            input.cfg,
                        );
                        continue;
                    }
                    RetryAction::RetryWithHints { topology, .. } => {
                        force_topology = Some(topology);
                        Self::apply_optimizer(
                            &mut current_params,
                            &mut tao_config,
                            &mut verification_config,
                            &quality_history,
                            n_max_ceiling,
                            cg_mean,
                            1.0,
                            input.calibration.ensemble.as_ref(),
                            input.cfg,
                        );
                        continue;
                    }
                    RetryAction::Fail(_) => {
                        input.store.mark_failed(&task_id);
                        return Err(EngineError::MaxRetriesExhausted);
                    }
                }
            }

            // ── Phase 3.5: Verification Loop (LLM-as-Judge) ──────────────
            use crate::verification::{VerificationInput, VerificationPhase};
            let mut pruned: Vec<BranchPrunedEvent> = Vec::new();
            let mut iteration_verification_events: Vec<VerificationScoredEvent> = Vec::new();
            let ver_out = VerificationPhase::run(VerificationInput {
                proposals,
                constraint_corpus: &input.constraint_corpus,
                evaluator: input.verification_adapter,
                config: verification_config.clone(),
            })
            .await;

            let mut proposals: Vec<ProposalEvent> = Vec::new();
            for (prop, results) in ver_out.passed {
                let score = h2ai_constraints::types::aggregate_compliance_score(&results);
                iteration_verification_events.push(VerificationScoredEvent {
                    task_id: task_id.clone(),
                    explorer_id: prop.explorer_id.clone(),
                    score,
                    reason: String::new(),
                    passed: true,
                    timestamp: Utc::now(),
                });
                input.store.record_validation(&task_id, true);
                proposals.push(prop);
            }
            for (prop, results, violations) in ver_out.failed {
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
                pruned.push(BranchPrunedEvent {
                    task_id: task_id.clone(),
                    explorer_id: prop.explorer_id,
                    reason: format!("verification compliance {compliance:.2}"),
                    constraint_error_cost: cost,
                    violated_constraints: violations,
                    timestamp: Utc::now(),
                });
                input.store.record_validation(&task_id, false);
            }

            // Build turn-1 proposals for Option B estimator feed.
            // Only accepted (passed) proposals that ran multiple TAO turns.
            let turn1_proposals_for_scoring: Vec<ProposalEvent> = proposals
                .iter()
                .filter_map(|prop| {
                    turn1_map
                        .get(&prop.explorer_id)
                        .map(|t1_output| ProposalEvent {
                            raw_output: t1_output.clone(),
                            ..prop.clone()
                        })
                })
                .collect();

            // ── Phase 4: Auditor Gate ──────────────────────────────────────
            input.store.set_phase(
                &task_id,
                TaskPhase::AuditorGate,
                explorer_count,
                retry_count,
            );

            let mut proposal_set = ProposalSet::new();

            for proposal in proposals {
                let audit_prompt = input
                    .auditor_config
                    .prompt_template
                    .replace("{constraints}", &input.manifest.constraints.join(", "))
                    .replace("{proposal}", &proposal.raw_output);
                let audit_req = ComputeRequest {
                    system_context: system_context.clone(),
                    task: audit_prompt,
                    tau: input.auditor_config.tau,
                    max_tokens: input.auditor_config.max_tokens,
                };
                let audit_result = input
                    .auditor_adapter
                    .execute(audit_req)
                    .await
                    .map_err(|e| EngineError::Adapter(e.to_string()))?;

                let (rejected, audit_reason) =
                    match serde_json::from_str::<AuditResponse>(&audit_result.output) {
                        Ok(r) => (!r.approved, r.reason),
                        Err(_) => {
                            tracing::warn!(
                                task_id = %task_id,
                                output = %audit_result.output,
                                "auditor returned non-JSON; failing safe (treating as rejected)"
                            );
                            (true, "auditor parse failure".to_string())
                        }
                    };

                if rejected {
                    let explorer_id = proposal.explorer_id.clone();
                    let cost = provisioned
                        .explorer_configs
                        .iter()
                        .position(|ec| ec.explorer_id == explorer_id)
                        .and_then(|idx| provisioned.role_error_costs.get(idx))
                        .cloned()
                        .unwrap_or_else(|| RoleErrorCost::new(0.5).unwrap());
                    pruned.push(BranchPrunedEvent {
                        task_id: task_id.clone(),
                        explorer_id,
                        reason: audit_reason,
                        constraint_error_cost: cost,
                        violated_constraints: vec![],
                        timestamp: Utc::now(),
                    });
                    input.store.record_validation(&task_id, false);
                } else {
                    input.store.record_validation(&task_id, true);
                    let ver_score = iteration_verification_events
                        .iter()
                        .find(|e| e.explorer_id == proposal.explorer_id)
                        .map(|e| e.score)
                        .unwrap_or(0.0);
                    proposal_set.insert_scored(proposal, ver_score);
                }
            }

            // ── Phase 5: Merge ──────────────────────────────────────────────
            input
                .store
                .set_phase(&task_id, TaskPhase::Merging, explorer_count, retry_count);

            let total_evaluated = proposal_set.len() + pruned.len();
            let filter_ratio = if total_evaluated > 0 {
                proposal_set.len() as f64 / total_evaluated as f64
            } else {
                1.0
            };

            let (attribution, attribution_interval) = {
                use crate::attribution::{
                    bootstrap_interval, AttributionInput, HarnessAttribution,
                };
                // Compute per-iteration Talagrand from current verification scores for S7 ρ correction.
                let iter_talagrand_state = {
                    let scores: Vec<f64> = iteration_verification_events
                        .iter()
                        .map(|e| e.score)
                        .collect();
                    TalagrandDiagnostic::from_verification_scores(&[scores])
                        .map(|d| d.calibration_state)
                };
                let attr_input = AttributionInput {
                    p_mean,
                    rho_mean,
                    n_agents: explorer_count,
                    verification_filter_ratio: filter_ratio,
                    tao_turns_mean,
                    tao_per_turn_factor: input.tao_multiplier,
                    prediction_basis: attribution_basis,
                    talagrand_state: iter_talagrand_state,
                    eigen_calibration: input.calibration.eigen.clone(),
                };
                let attr = HarnessAttribution::compute(&attr_input);
                let interval = {
                    let cg_samples = &input.calibration.coefficients.cg_samples;
                    if cg_samples.len() >= 2 {
                        Some(bootstrap_interval(&attr_input, cg_samples, 1000))
                    } else {
                        None
                    }
                };
                (attr, interval)
            };

            // ── Option B: feed TaoMultiplierEstimator with turn-1 vs final scores ──────
            if !turn1_proposals_for_scoring.is_empty() {
                use crate::verification::VerificationPhase;
                let turn1_scores = VerificationPhase::score_proposals(
                    turn1_proposals_for_scoring,
                    input.verification_adapter,
                    &verification_config,
                    &input.constraint_corpus,
                )
                .await;

                let mut est = input.tao_estimator.write().await;
                for (t1_prop, t1_score) in &turn1_scores {
                    if let Some(final_ev) = iteration_verification_events
                        .iter()
                        .find(|e| e.explorer_id == t1_prop.explorer_id && e.passed)
                    {
                        est.update(*t1_score, final_ev.score);
                    }
                }
            }
            // ─────────────────────────────────────────────────────────────────────────────

            // Accumulate all_pruned before moving pruned into resolve
            all_pruned.extend(pruned.iter().cloned());
            tau_values_tried.push(tau_values);

            let outcome = MergeEngine::resolve(
                task_id.clone(),
                proposal_set,
                pruned,
                provisioned.merge_strategy.clone(),
                retry_count,
                input.embedding_model,
            )
            .await;

            all_verification_events.extend(iteration_verification_events.clone());

            // Talagrand τ feedback: after each iteration, update τ-spread expansion factor.
            // Using the current iteration's scores — typically Insufficient (< 20 runs) for a
            // single task iteration, but may trigger on longer sessions with accumulated data.
            {
                let iter_scores: Vec<f64> = iteration_verification_events
                    .iter()
                    .map(|e| e.score)
                    .collect();
                if let Some(diag) = TalagrandDiagnostic::from_verification_scores(&[iter_scores]) {
                    tau_spread_factor = diag
                        .tau_expansion_factor(tau_spread_factor, input.cfg.tau_spread_max_factor);
                }
            }

            match outcome {
                MergeOutcome::Resolved {
                    compiled: semilattice,
                    resolved,
                } => {
                    quality_history.push(QualityMeasurement {
                        params: current_params.clone(),
                        q_total: attribution.total_quality,
                    });
                    let suggested_next = SelfOptimizer::suggest(SuggestInput {
                        current: &current_params,
                        history: &quality_history,
                        n_max_ceiling,
                        n_optimal: input
                            .calibration
                            .ensemble
                            .as_ref()
                            .map(|ec| ec.n_optimal as u32),
                        p_mean,
                        rho_mean,
                        filter_ratio,
                        cfg: input.cfg,
                    });

                    let waste_ratio = filter_ratio;
                    let applied_optimizations = if waste_ratio < input.cfg.optimizer_waste_threshold
                    {
                        // N changes are bandit-owned; only record threshold suggestions.
                        let mut opts = Vec::new();
                        if (suggested_next.verify_threshold - current_params.verify_threshold).abs()
                            > 1e-9
                        {
                            opts.push(h2ai_types::events::AppliedOptimization {
                                kind: h2ai_types::events::OptimizationKind::TauSpreadAdjusted,
                                reason: format!(
                                    "waste_ratio={:.2} < threshold {:.2}; \
                                         tighten verify_threshold to reduce pruned proposals",
                                    waste_ratio, input.cfg.optimizer_waste_threshold
                                ),
                                before: format!("{:.3}", current_params.verify_threshold),
                                after: format!("{:.3}", suggested_next.verify_threshold),
                            });
                        }
                        opts
                    } else {
                        vec![]
                    };

                    input.store.mark_resolved(&task_id);
                    let run_scores: Vec<f64> =
                        all_verification_events.iter().map(|e| e.score).collect();
                    let talagrand = TalagrandDiagnostic::from_verification_scores(&[run_scores]);
                    return Ok(EngineOutput {
                        task_id,
                        resolved_output: resolved.resolved_output,
                        semilattice,
                        attribution,
                        attribution_interval,
                        verification_events: all_verification_events,
                        talagrand,
                        suggested_next_params: Some(suggested_next),
                        waste_ratio,
                        applied_optimizations,
                    });
                }
                MergeOutcome::ZeroSurvival(zero_event) => {
                    match RetryPolicy::decide(
                        &zero_event,
                        &tried_topologies,
                        all_pruned.clone(),
                        tau_values_tried.clone(),
                        last_multiplication_failure.clone(),
                    ) {
                        RetryAction::Retry(next_topology) => {
                            force_topology = Some(next_topology);
                            Self::apply_optimizer(
                                &mut current_params,
                                &mut tao_config,
                                &mut verification_config,
                                &quality_history,
                                n_max_ceiling,
                                cg_mean,
                                filter_ratio,
                                input.calibration.ensemble.as_ref(),
                                input.cfg,
                            );
                        }
                        RetryAction::RetryWithTauReduction {
                            topology,
                            tau_factor,
                        } => {
                            force_topology = Some(topology);
                            tau_reduction_factor *= tau_factor;
                            Self::apply_optimizer(
                                &mut current_params,
                                &mut tao_config,
                                &mut verification_config,
                                &quality_history,
                                n_max_ceiling,
                                cg_mean,
                                filter_ratio,
                                input.calibration.ensemble.as_ref(),
                                input.cfg,
                            );
                        }
                        RetryAction::RetryWithHints { topology, .. } => {
                            force_topology = Some(topology);
                            Self::apply_optimizer(
                                &mut current_params,
                                &mut tao_config,
                                &mut verification_config,
                                &quality_history,
                                n_max_ceiling,
                                cg_mean,
                                filter_ratio,
                                input.calibration.ensemble.as_ref(),
                                input.cfg,
                            );
                        }
                        RetryAction::Fail(_) => {
                            input.store.mark_failed(&task_id);
                            return Err(EngineError::MaxRetriesExhausted);
                        }
                    }
                }
            }
        }

        input.store.mark_failed(&task_id);
        Err(EngineError::MaxRetriesExhausted)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_optimizer(
        current_params: &mut OptimizerParams,
        tao_config: &mut TaoConfig,
        verification_config: &mut VerificationConfig,
        history: &[QualityMeasurement],
        n_max_ceiling: u32,
        cg_mean: f64,
        filter_ratio: f64,
        ensemble: Option<&h2ai_types::physics::EnsembleCalibration>,
        cfg: &H2AIConfig,
    ) {
        let (p_mean, rho_mean) = match ensemble {
            Some(ec) => (ec.p_mean, ec.rho_mean),
            None => (0.5 + cg_mean / 2.0, (1.0 - cg_mean).clamp(0.0, 1.0)),
        };
        let n_optimal = ensemble.map(|ec| ec.n_optimal as u32);
        let suggested = SelfOptimizer::suggest(SuggestInput {
            current: current_params,
            history,
            n_max_ceiling,
            n_optimal,
            p_mean,
            rho_mean,
            filter_ratio,
            cfg,
        });
        if suggested.max_turns != current_params.max_turns {
            tao_config.max_turns = suggested.max_turns as u8;
        }
        if (suggested.verify_threshold - current_params.verify_threshold).abs() > 1e-9 {
            verification_config.threshold = suggested.verify_threshold;
        }
        *current_params = suggested;
    }
}
