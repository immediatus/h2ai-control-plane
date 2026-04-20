use crate::task_store::{TaskPhase, TaskState, TaskStore};
use chrono::Utc;
use futures::future::join_all;
use h2ai_autonomic::checker::MultiplicationChecker;
use h2ai_autonomic::merger::{MergeEngine, MergeOutcome};
use h2ai_autonomic::planner::{ProvisionInput, TopologyPlanner};
use h2ai_config::H2AIConfig;
use h2ai_context::adr::AdrConstraints;
use h2ai_context::compaction::{compact, CompactionConfig};
use h2ai_context::compiler;
use h2ai_state::semilattice::ProposalSet;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::config::{AuditorConfig, RoleSpec, TaoConfig, VerificationConfig};
use h2ai_types::events::{
    BranchPrunedEvent, CalibrationCompletedEvent, GenerationPhaseCompletedEvent, ProposalEvent,
    ProposalFailedEvent, ProposalFailureReason, SemilatticeCompiledEvent, TaskBootstrappedEvent,
    VerificationScoredEvent,
};
use h2ai_types::identity::TaskId;
use h2ai_types::manifest::TaskManifest;
use h2ai_types::physics::{RoleErrorCost, TauValue};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("context underflow: J_eff={j_eff:.3} < {threshold:.1}")]
    ContextUnderflow { j_eff: f64, threshold: f64 },
    #[error("multiplication condition failed: {0}")]
    MultiplicationConditionFailed(String),
    #[error("max retries exhausted")]
    MaxRetriesExhausted,
    #[error("adapter error: {0}")]
    Adapter(String),
    #[error("parse error: {0}")]
    Parse(String),
}

pub struct EngineInput<'a> {
    pub manifest: TaskManifest,
    pub calibration: CalibrationCompletedEvent,
    pub explorer_adapters: Vec<&'a dyn IComputeAdapter>,
    pub auditor_adapter: &'a dyn IComputeAdapter,
    pub auditor_config: AuditorConfig,
    pub tao_config: TaoConfig,
    pub verification_config: VerificationConfig,
    pub adr_corpus: Vec<AdrConstraints>,
    pub cfg: &'a H2AIConfig,
    pub store: TaskStore,
}

#[derive(Debug)]
pub struct EngineOutput {
    pub task_id: TaskId,
    pub semilattice: SemilatticeCompiledEvent,
    pub attribution: crate::attribution::HarnessAttribution,
    pub verification_events: Vec<VerificationScoredEvent>,
}

pub struct ExecutionEngine;

impl ExecutionEngine {
    /// Run all 6 phases without NATS publishing (unit-testable).
    pub async fn run_offline(input: EngineInput<'_>) -> Result<EngineOutput, EngineError> {
        let task_id = TaskId::new();
        input
            .store
            .insert(task_id.clone(), TaskState::new(task_id.clone()));

        // ── Phase 1: Bootstrap ──────────────────────────────────────────────
        let description = &input.manifest.description;
        let required_kw = input
            .adr_corpus
            .iter()
            .flat_map(|a| a.keywords.iter().cloned())
            .chain(input.manifest.constraints.iter().cloned())
            .collect::<Vec<_>>()
            .join(" ");
        let required_kw = if required_kw.is_empty() {
            description.clone()
        } else {
            required_kw
        };

        let compiled = compiler::compile(description, &input.adr_corpus, &required_kw, input.cfg)
            .map_err(|e| {
            let h2ai_context::compiler::ContextError::ContextUnderflow { j_eff, threshold } = e;
            EngineError::ContextUnderflow { j_eff, threshold }
        })?;

        let adr_keywords: Vec<String> = input
            .adr_corpus
            .iter()
            .flat_map(|a| a.keywords.iter().cloned())
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

        input
            .store
            .set_phase(&task_id, TaskPhase::Provisioning, 0, 0);

        let role_specs: Vec<RoleSpec> = if input.manifest.explorers.roles.is_empty() {
            let count = input.manifest.explorers.count.max(1);
            let tau_min = input.manifest.explorers.tau_min.unwrap_or(0.2);
            let tau_max = input.manifest.explorers.tau_max.unwrap_or(0.9);
            let step = if count > 1 {
                (tau_max - tau_min) / (count - 1) as f64
            } else {
                0.0
            };
            (0..count)
                .map(|i| RoleSpec {
                    agent_id: format!("exp_{}", (b'A' + i as u8) as char),
                    role: h2ai_types::config::AgentRole::Executor,
                    tau: Some(TauValue::new(tau_min + step * i as f64).expect("tau in (0,1]")),
                    role_error_cost: None,
                })
                .collect()
        } else {
            input.manifest.explorers.roles.clone()
        };

        let explorer_adapter_kind = input
            .explorer_adapters
            .first()
            .map(|a| a.kind().clone())
            .unwrap_or_else(|| input.auditor_config.adapter.clone());

        // ── Phase 2: Topology Provisioning ─────────────────────────────────
        let provisioned = TopologyPlanner::provision(ProvisionInput {
            task_id: task_id.clone(),
            cc: &input.calibration.coefficients,
            pareto_weights: &input.manifest.pareto_weights,
            role_specs: &role_specs,
            review_gates: input.manifest.explorers.review_gates.clone(),
            auditor_config: input.auditor_config.clone(),
            explorer_adapter: explorer_adapter_kind,
            retry_count: 0,
            cfg: input.cfg,
        });

        let explorer_count = provisioned.explorer_configs.len() as u32;
        input
            .store
            .set_phase(&task_id, TaskPhase::MultiplicationCheck, explorer_count, 0);

        // ── Phase 2.5: Multiplication Condition Gate ────────────────────────
        let cg_mean = input.calibration.coefficients.cg_mean();
        let baseline_competence = cg_mean;
        let error_correlation = 1.0 - cg_mean;

        if let Err(mc_failed) = MultiplicationChecker::check(
            &task_id,
            &input.calibration.coefficients,
            &input.calibration.coordination_threshold,
            baseline_competence,
            error_correlation,
            0,
            input.cfg,
        ) {
            return Err(EngineError::MultiplicationConditionFailed(
                mc_failed.failure.to_string(),
            ));
        }

        // ── Phase 3: Parallel Generation ────────────────────────────────────
        input
            .store
            .set_phase(&task_id, TaskPhase::ParallelGeneration, explorer_count, 0);

        let futures_vec: Vec<_> = provisioned
            .explorer_configs
            .iter()
            .enumerate()
            .map(|(idx, explorer_cfg)| {
                let adapter_idx = idx % input.explorer_adapters.len();
                let adapter = input.explorer_adapters[adapter_idx];
                let req = ComputeRequest {
                    system_context: system_context.clone(),
                    task: input.manifest.description.clone(),
                    tau: explorer_cfg.tau,
                    max_tokens: input.cfg.explorer_max_tokens,
                };
                let explorer_id = explorer_cfg.explorer_id.clone();
                let task_id_clone = task_id.clone();
                let tao_cfg = input.tao_config.clone();
                async move {
                    use crate::tao_loop::{TaoInput, TaoLoop};
                    match TaoLoop::run(TaoInput {
                        task_id: task_id_clone.clone(),
                        explorer_id: explorer_id.clone(),
                        adapter,
                        initial_request: req,
                        config: tao_cfg,
                        schema_config: None,
                    })
                    .await
                    {
                        Ok(tao_proposal) => Ok((tao_proposal.event, tao_proposal.tao_turns)),
                        Err(e) => Err(ProposalFailedEvent {
                            task_id: task_id_clone,
                            explorer_id,
                            reason: ProposalFailureReason::AdapterError(e.to_string()),
                            timestamp: Utc::now(),
                        }),
                    }
                }
            })
            .collect();

        let results = join_all(futures_vec).await;

        let mut proposals: Vec<ProposalEvent> = Vec::new();
        let mut tao_turns_collected: Vec<u8> = Vec::new();
        let mut failed_proposals: Vec<ProposalFailedEvent> = Vec::new();

        for result in results {
            match result {
                Ok((proposal, turns)) => {
                    input.store.increment_completed(&task_id);
                    tao_turns_collected.push(turns);
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

        // ── Phase 3.5: Verification Loop (LLM-as-Judge) ──────────────────────
        use crate::verification::{VerificationInput, VerificationPhase};
        let mut pruned: Vec<BranchPrunedEvent> = Vec::new();
        let mut verification_events: Vec<VerificationScoredEvent> = Vec::new();
        let ver_out = VerificationPhase::run(VerificationInput {
            proposals,
            constraints: &input.manifest.constraints,
            evaluator: input.auditor_adapter,
            config: input.verification_config.clone(),
        })
        .await;

        let mut proposals: Vec<ProposalEvent> = Vec::new();
        for (prop, score) in ver_out.passed {
            verification_events.push(VerificationScoredEvent {
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
        for (prop, score, reason) in ver_out.failed {
            verification_events.push(VerificationScoredEvent {
                task_id: task_id.clone(),
                explorer_id: prop.explorer_id.clone(),
                score,
                reason: reason.clone(),
                passed: false,
                timestamp: Utc::now(),
            });
            let cost = provisioned
                .role_error_costs
                .first()
                .cloned()
                .unwrap_or_else(|| RoleErrorCost::new(0.5).unwrap());
            pruned.push(BranchPrunedEvent {
                task_id: task_id.clone(),
                explorer_id: prop.explorer_id,
                reason: format!("verification score {score:.2}: {reason}"),
                constraint_error_cost: cost,
                timestamp: Utc::now(),
            });
            input.store.record_validation(&task_id, false);
        }

        // ── Phase 4: Auditor Gate ────────────────────────────────────────────
        input
            .store
            .set_phase(&task_id, TaskPhase::AuditorGate, explorer_count, 0);

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

            let rejected = audit_result.output.to_lowercase().contains("reject")
                || audit_result.output.to_lowercase().contains("violation");

            if rejected {
                let explorer_id = proposal.explorer_id.clone();
                let cost = provisioned
                    .role_error_costs
                    .first()
                    .cloned()
                    .unwrap_or_else(|| RoleErrorCost::new(0.5).unwrap());
                pruned.push(BranchPrunedEvent {
                    task_id: task_id.clone(),
                    explorer_id,
                    reason: audit_result.output.clone(),
                    constraint_error_cost: cost,
                    timestamp: Utc::now(),
                });
                input.store.record_validation(&task_id, false);
            } else {
                input.store.record_validation(&task_id, true);
                proposal_set.insert(proposal);
            }
        }

        // ── Phase 5: Merge ───────────────────────────────────────────────────
        input
            .store
            .set_phase(&task_id, TaskPhase::Merging, explorer_count, 0);

        // Compute harness attribution before resolve() moves `pruned`.
        let attribution = {
            use crate::attribution::{AttributionInput, HarnessAttribution};
            let total_evaluated = proposal_set.len() + pruned.len();
            let filter_ratio = if total_evaluated > 0 {
                proposal_set.len() as f64 / total_evaluated as f64
            } else {
                1.0
            };
            HarnessAttribution::compute(&AttributionInput {
                baseline_c_i: 1.0 - cg_mean,
                n_agents: explorer_count,
                alpha: input.calibration.coefficients.alpha,
                kappa_eff: input.calibration.coefficients.kappa_eff(),
                verification_filter_ratio: filter_ratio,
                tao_turns_mean,
            })
        };

        let outcome = MergeEngine::resolve(
            task_id.clone(),
            proposal_set,
            pruned,
            provisioned.merge_strategy.clone(),
            0,
        );

        match outcome {
            MergeOutcome::Resolved {
                compiled: semilattice,
                ..
            } => Ok(EngineOutput {
                task_id,
                semilattice,
                attribution,
                verification_events,
            }),
            MergeOutcome::ZeroSurvival(_) => {
                input.store.mark_failed(&task_id);
                Err(EngineError::MaxRetriesExhausted)
            }
        }
    }
}
