use crate::engine::{EngineError, EngineInput};
use crate::phases::StepResult;
use crate::task_store::TaskPhase;
use crate::verification::extract_json_object;
use chrono::Utc;
use h2ai_state::semilattice::ProposalSet;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::events::{
    BranchPrunedEvent, ProposalEvent, ShadowAuditorResultEvent, TopologyProvisionedEvent,
};
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::RoleErrorCost;

use super::verify::Output as VerifyOutput;

#[derive(serde::Deserialize)]
struct AuditResponse {
    approved: bool,
    reason: String,
    #[serde(default)]
    violated: Vec<String>,
}

pub struct Input<'a> {
    pub engine_input: &'a EngineInput<'a>,
    pub task_id: &'a TaskId,
    pub retry_count: u32,
    pub explorer_count: u32,
    pub provisioned: &'a TopologyProvisionedEvent,
}

pub struct Output {
    pub proposal_set: ProposalSet,
    pub synthesis_candidates: Vec<ProposalEvent>,
    /// Proposals pruned in the audit gate (appended to the verify-phase pruned list).
    pub pruned: Vec<BranchPrunedEvent>,
    pub shadow_audit_events: Vec<ShadowAuditorResultEvent>,
    /// Verification events passed through (needed downstream for score lookup).
    pub iteration_verification_events: Vec<h2ai_types::events::VerificationScoredEvent>,
}

/// Run Phase 4: Auditor Gate.
///
/// For each surviving proposal from verification, runs the primary auditor and
/// optionally a shadow auditor concurrently.  Builds `ProposalSet` and
/// `synthesis_candidates` for the merge / synthesis phases.
///
/// Always returns `StepResult::Done` — ZeroSurvival detection after audit is
/// handled by the merge phase (Task 8).
pub async fn run(verify_out: VerifyOutput, input: Input<'_>) -> StepResult<Output> {
    let engine_input = input.engine_input;
    let task_id = input.task_id;
    let retry_count = input.retry_count;
    let explorer_count = input.explorer_count;
    let provisioned = input.provisioned;

    engine_input
        .store
        .set_phase(task_id, TaskPhase::AuditorGate, explorer_count, retry_count);

    let mut proposal_set = ProposalSet::new();
    let mut synthesis_candidates: Vec<ProposalEvent> = Vec::new();
    let mut pruned: Vec<BranchPrunedEvent> = verify_out.pruned;

    // Shadow auditor setup — domain and vote mode for this task.
    let task_domain = engine_input
        .manifest
        .constraint_tags
        .first()
        .cloned()
        .unwrap_or_else(|| "default".to_string());
    let majority_vote_active = engine_input
        .shadow_audit_ctx
        .as_ref()
        .map(|ctx| ctx.promoted_domains.contains(&task_domain))
        .unwrap_or(false);
    let mut shadow_events_this_wave: Vec<ShadowAuditorResultEvent> = Vec::new();

    // In single-family mode the auditor is the same model as the explorer.
    // Adversarial self-evaluation produces systematic rejection bias — skip the
    // audit phase entirely and let all proposals through to the verifier.
    let auditor_addr =
        engine_input.auditor_adapter as *const dyn IComputeAdapter as *const () as usize;
    let skip_audit = !engine_input.explorer_adapters.is_empty()
        && engine_input
            .explorer_adapters
            .iter()
            .all(|a| *a as *const dyn IComputeAdapter as *const () as usize == auditor_addr);
    if skip_audit {
        tracing::info!(
            target: "h2ai.engine",
            task_id = %task_id,
            adapter = ?engine_input.auditor_adapter.kind(),
            "single-adapter mode: skipping auditor (same adapter instance as explorer)"
        );
    }

    for proposal in verify_out.proposals {
        let (primary_approved, audit_reason, audit_violated, shadow_result_opt) = if skip_audit {
            (true, String::new(), vec![], None)
        } else {
            let audit_prompt = engine_input
                .auditor_config
                .prompt_template
                .replace(
                    "{constraints}",
                    &engine_input.manifest.constraints.join(", "),
                )
                .replace("{proposal}", &proposal.raw_output);

            let audit_prompt_str = audit_prompt;
            let make_req = || ComputeRequest {
                system_context: engine_input.auditor_config.system_prompt.clone(),
                task: audit_prompt_str.clone(),
                tau: engine_input.auditor_config.tau,
                max_tokens: engine_input.auditor_config.max_tokens,
            };

            // Run shadow concurrently with primary when shadow ctx is present.
            let (primary_result, shadow_opt) = match engine_input.shadow_audit_ctx.as_ref() {
                Some(ctx) => {
                    let (p, s) = tokio::join!(
                        engine_input.auditor_adapter.execute(make_req()),
                        ctx.adapter.execute(make_req())
                    );
                    (p, Some(s))
                }
                None => (engine_input.auditor_adapter.execute(make_req()).await, None),
            };

            let audit_result = match primary_result {
                Ok(r) => r,
                Err(e) => {
                    return StepResult::Fatal(EngineError::Adapter(e.to_string()));
                }
            };

            let (approved, reason, violated) =
                match extract_json_object::<AuditResponse>(&audit_result.output) {
                    Some(r) => (r.approved, r.reason, r.violated),
                    None => {
                        tracing::warn!(
                            task_id = %task_id,
                            output = %audit_result.output,
                            "auditor returned non-JSON; failing safe (treating as rejected)"
                        );
                        (false, "auditor parse failure".to_string(), vec![])
                    }
                };
            (approved, reason, violated, shadow_opt)
        };

        // Extract shadow decision (None if shadow errored or absent).
        let shadow_approved_opt: Option<bool> = shadow_result_opt.and_then(|sr| {
            sr.ok()
                .and_then(|r| extract_json_object::<AuditResponse>(&r.output))
                .map(|a| a.approved)
        });

        // Pruning decision.
        let rejected = if majority_vote_active {
            // Promoted domain: both must approve (AND vote).
            // If shadow errored (None), fall back to primary-only — shadow error ≠ rejection.
            let shadow_vote = shadow_approved_opt.unwrap_or(primary_approved);
            !(primary_approved && shadow_vote)
        } else {
            // Shadow mode or no shadow: primary always decides.
            !primary_approved
        };

        // Collect shadow event when shadow ran successfully.
        if let (Some(shadow_approved), Some(ctx)) =
            (shadow_approved_opt, engine_input.shadow_audit_ctx.as_ref())
        {
            shadow_events_this_wave.push(ShadowAuditorResultEvent {
                task_id: task_id.clone(),
                explorer_id: proposal.explorer_id.clone(),
                primary_approved,
                shadow_approved,
                disagreement: primary_approved != shadow_approved,
                domain: task_domain.clone(),
                primary_family: format!("{:?}", engine_input.auditor_adapter.kind()),
                shadow_family: format!("{:?}", ctx.adapter.kind()),
                timestamp_ms: chrono::Utc::now().timestamp_millis() as u64,
            });
        }

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
                violated_constraints: audit_violated
                    .iter()
                    .map(|id| h2ai_types::events::ConstraintViolation {
                        constraint_id: id.clone(),
                        score: 0.0,
                        severity_label: "Hard".to_string(),
                        remediation_hint: None,
                    })
                    .collect(),
                timestamp: Utc::now(),
            });
            engine_input.store.record_validation(task_id, false);
        } else {
            engine_input.store.record_validation(task_id, true);
            let ver_score = verify_out
                .iteration_verification_events
                .iter()
                .find(|e| e.explorer_id == proposal.explorer_id)
                .map(|e| e.score)
                .unwrap_or(0.0);
            synthesis_candidates.push(proposal.clone());
            proposal_set.insert_scored(proposal, ver_score);
        }
    }

    StepResult::Done(Output {
        proposal_set,
        synthesis_candidates,
        pruned,
        shadow_audit_events: shadow_events_this_wave,
        iteration_verification_events: verify_out.iteration_verification_events,
    })
}
