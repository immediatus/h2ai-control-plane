use crate::engine::EngineInput;
use crate::phases::StepResult;
use crate::task_store::TaskPhase;
use chrono::Utc;
use futures::future::join_all;
use h2ai_types::adapter::ComputeRequest;
use h2ai_types::events::{
    GenerationPhaseCompletedEvent, ProposalEvent, ProposalFailedEvent, ProposalFailureReason,
    ResearcherGroundingEvent, TopologyProvisionedEvent,
};
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::TauValue;
use std::collections::HashMap;

pub struct Input<'a> {
    pub engine_input: &'a EngineInput<'a>,
    pub task_id: &'a TaskId,
    pub retry_count: u32,
    /// The active context string (retry_context if Some, else system_context).
    pub active_ctx: String,
    /// The base system context without retry feedback.
    pub system_context: String,
    pub system_context_with_rubric: String,
    pub explorer_count: u32,
    pub provisioned: &'a TopologyProvisionedEvent,
    /// Tombstone text appended to every slot's effective context.
    pub pending_tombstone: Option<String>,
    /// Adapter pool rotation offset (index modulo).
    pub adapter_rotation_offset: usize,
}

pub struct Output {
    pub proposals: Vec<ProposalEvent>,
    pub failed_proposals: Vec<ProposalFailedEvent>,
    pub all_raw_texts: Vec<String>,
    pub tao_turns_mean: f64,
    pub tau_values: Vec<f64>,
    pub turn1_map: HashMap<h2ai_types::identity::ExplorerId, String>,
    pub researcher_grounding_events: Vec<ResearcherGroundingEvent>,
    /// Captured for epistemic yield — count of proposals dispatched.
    pub failed_count: u32,
}

/// Run Phase 3: Parallel Generation.
///
/// Dispatches explorer slots in parallel via TAO loops, collects proposals and failures.
/// Also runs the proactive researcher pre-pass for search-enabled slots (GAP-C1 proactive).
///
/// Always returns `StepResult::Done(Output)`. Failed individual explorer TAO loops are
/// captured in `Output::failed_proposals` rather than triggering a fatal error.
/// Never returns `StepResult::EarlyExit` or `StepResult::Fatal`.
pub async fn run(input: Input<'_>) -> StepResult<Output> {
    let engine_input = input.engine_input;
    let task_id = input.task_id;
    let retry_count = input.retry_count;
    let provisioned = input.provisioned;
    let explorer_count = input.explorer_count;
    let active_ctx = &input.active_ctx;

    engine_input.store.set_phase(
        task_id,
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
            dyn Future<Output = Result<(ProposalEvent, u8, Option<String>), ProposalFailedEvent>>
                + Send
                + 'f,
        >,
    >;

    let effective_slot_configs: &[h2ai_types::manifest::ExplorerSlotConfig] =
        &engine_input.manifest.explorers.slot_configs;

    // ── Proactive Researcher Pre-pass (GAP-C1 proactive path) ──────────
    // For slots with search_enabled=true, call the researcher adapter to fetch
    // current state-of-the-art grounding before generating proposals.
    let mut slot_groundings: Vec<Option<String>> = vec![None; provisioned.explorer_configs.len()];
    let mut researcher_grounding_events: Vec<ResearcherGroundingEvent> = Vec::new();
    if let Some(ref researcher) = engine_input.researcher_adapter {
        for idx in 0..provisioned.explorer_configs.len() {
            let sc_opt = if effective_slot_configs.is_empty() {
                None
            } else {
                Some(&effective_slot_configs[idx % effective_slot_configs.len()])
            };
            if sc_opt.map(|sc| sc.search_enabled).unwrap_or(false) {
                let req = ComputeRequest {
                    system_context: active_ctx.clone(),
                    task: format!(
                        "Search for current state-of-the-art evidence relevant to: {}. \
                         Return a concise grounding statement in 2-3 sentences that \
                         the explorer should treat as established fact.",
                        engine_input.manifest.description
                    ),
                    tau: TauValue::new(0.2).unwrap(),
                    max_tokens: 512,
                };
                if let Ok(resp) = researcher.execute(req).await {
                    researcher_grounding_events.push(ResearcherGroundingEvent {
                        task_id: task_id.clone(),
                        shared_assumption: String::new(),
                        literature_summary: resp.output.clone(),
                        slot: Some(format!("slot_{idx}")),
                        source: h2ai_types::events::GroundingSource::LlmResearcher,
                    });
                    slot_groundings[idx] = Some(format!("[STATE-OF-THE-ART]: {}", resp.output));
                }
            }
        }
    }

    let futures_vec: Vec<ExplorerFuture<'_>> = provisioned
        .explorer_configs
        .iter()
        .enumerate()
        .map(|(idx, explorer_cfg)| {
            let (slot_task, slot_system_ctx) = {
                let configs = effective_slot_configs;
                if configs.is_empty() {
                    (
                        engine_input.manifest.description.clone(),
                        active_ctx.clone(),
                    )
                } else {
                    let sc = &configs[idx % configs.len()];
                    let cot = sc.cot_style.instruction();
                    let task = if cot.is_empty() {
                        engine_input.manifest.description.clone()
                    } else {
                        format!("{}\n\n{}", cot, engine_input.manifest.description)
                    };
                    let mut preamble = String::new();
                    if !sc.role_frame.is_empty() {
                        preamble.push_str(&sc.role_frame);
                    }
                    if !sc.focus_mandate.is_empty() {
                        if !preamble.is_empty() {
                            preamble.push_str("\n\n");
                        }
                        preamble.push_str("[MANDATE]: ");
                        preamble.push_str(&sc.focus_mandate);
                    }
                    if !sc.rejection_criteria.is_empty() {
                        if !preamble.is_empty() {
                            preamble.push_str("\n\n");
                        }
                        preamble
                            .push_str("[AFTER WRITING YOUR PROPOSAL, IDENTIFY THE BIGGEST RISK]: ");
                        preamble.push_str(&sc.rejection_criteria);
                    }
                    let base_ctx = if preamble.is_empty() {
                        active_ctx.clone()
                    } else {
                        format!("{}\n\n{}", preamble, active_ctx)
                    };
                    let ctx = if let Some(grounding) =
                        slot_groundings.get(idx).and_then(|g| g.as_ref())
                    {
                        format!("{}\n\n{}", grounding, base_ctx)
                    } else {
                        base_ctx
                    };
                    (task, ctx)
                }
            };
            let effective_ctx = if let Some(ref tombstone) = input.pending_tombstone {
                format!("{}\n\n{}", slot_system_ctx, tombstone)
            } else {
                slot_system_ctx
            };
            let req = ComputeRequest {
                system_context: effective_ctx,
                task: slot_task,
                tau: explorer_cfg.tau,
                max_tokens: engine_input.cfg.explorer_max_tokens,
            };
            let explorer_id = explorer_cfg.explorer_id.clone();
            let task_id_clone = task_id.clone();
            let tao_cfg = engine_input.tao_config.clone();
            if let Some(ref nd_cfg) = engine_input.nats_dispatch {
                let arc = Arc::new(NatsDispatchAdapter::new(
                    crate::nats_dispatch_adapter::NatsDispatchConfig {
                        nats: nd_cfg.nats.clone(),
                        provider: nd_cfg.provider.clone(),
                        agent_descriptor: nd_cfg.agent_descriptor.clone(),
                        task_requirements: nd_cfg.task_requirements.clone(),
                        task_timeout: nd_cfg.task_timeout,
                        payload_store: nd_cfg.payload_store.clone(),
                        offload_threshold_bytes: nd_cfg.offload_threshold_bytes,
                    },
                ));
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
                        bypass_tao: false,
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
                let pool_len = engine_input.explorer_adapters.len();
                let adapter_idx = (idx + input.adapter_rotation_offset) % pool_len;
                let adapter = engine_input.explorer_adapters[adapter_idx];
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
                        bypass_tao: explorer_cfg.is_reasoning_model,
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
    let mut turn1_map: HashMap<h2ai_types::identity::ExplorerId, String> = HashMap::new();

    for result in results {
        match result {
            Ok((proposal, turns, turn1_output)) => {
                engine_input.store.increment_completed(task_id);
                tao_turns_collected.push(turns);
                if let Some(t1) = turn1_output {
                    turn1_map.insert(proposal.explorer_id.clone(), t1);
                }
                proposals.push(proposal);
            }
            Err(failed) => {
                engine_input.store.increment_completed(task_id);
                failed_proposals.push(failed);
            }
        }
    }
    let failed_count = failed_proposals.len() as u32;

    // Capture raw texts for epistemic yield / FailureMode classification.
    let all_raw_texts: Vec<String> = proposals.iter().map(|p| p.raw_output.clone()).collect();

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
        failed: failed_count,
        timestamp: Utc::now(),
    };

    // Collect tau values for this batch before verification
    let tau_values: Vec<f64> = provisioned
        .explorer_configs
        .iter()
        .map(|ec| ec.tau.value())
        .collect();

    StepResult::Done(Output {
        proposals,
        failed_proposals,
        all_raw_texts,
        tao_turns_mean,
        tau_values,
        turn1_map,
        researcher_grounding_events,
        failed_count,
    })
}
