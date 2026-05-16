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
    /// Leader context snapshot for per-slot prefix injection.
    pub leader_context: Option<crate::leader::LeaderContextSnapshot>,
    /// Assembled contexts from the previous wave for delta encoding.
    pub prev_assembled_contexts: Vec<Option<crate::context_assembler::AssembledContext>>,
    /// Compression adapter for LLM summarization pass.
    pub compression_adapter: Option<std::sync::Arc<dyn h2ai_types::adapter::IComputeAdapter>>,
    /// Cross-task stable context cache.
    pub stable_cache:
        Option<std::sync::Arc<crate::context_assembler::stable_cache::StableContextCache>>,
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
    /// Assembled contexts from this wave, for threading to next wave.
    pub assembled_contexts: Vec<Option<crate::context_assembler::AssembledContext>>,
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

    // ── Phase A: collect per-slot inputs (sync) ──────────────────────────────
    struct SlotData {
        slot_task: String,
        leader_prefix: Option<String>,
        role_frame: Option<String>,
        mandate: Option<String>,
        rejection_criteria: Option<String>,
        grounding: Option<String>,
        tombstone: Option<String>,
    }

    let slot_data: Vec<SlotData> = provisioned
        .explorer_configs
        .iter()
        .enumerate()
        .map(|(idx, explorer_cfg)| {
            let slot_task = {
                let configs = effective_slot_configs;
                if configs.is_empty() {
                    engine_input.manifest.description.clone()
                } else {
                    let sc = &configs[idx % configs.len()];
                    let cot = sc.cot_style.instruction();
                    if cot.is_empty() {
                        engine_input.manifest.description.clone()
                    } else {
                        format!("{}\n\n{}", cot, engine_input.manifest.description)
                    }
                }
            };

            let (role_frame, mandate, rejection_criteria) = {
                let configs = effective_slot_configs;
                if configs.is_empty() {
                    (None, None, None)
                } else {
                    let sc = &configs[idx % configs.len()];
                    (
                        if sc.role_frame.is_empty() {
                            None
                        } else {
                            Some(sc.role_frame.clone())
                        },
                        if sc.focus_mandate.is_empty() {
                            None
                        } else {
                            Some(sc.focus_mandate.clone())
                        },
                        if sc.rejection_criteria.is_empty() {
                            None
                        } else {
                            Some(sc.rejection_criteria.clone())
                        },
                    )
                }
            };

            let leader_prefix: Option<String> = input.leader_context.as_ref().and_then(|ls| {
                if ls.term == 0 {
                    return None;
                }
                if explorer_cfg.explorer_id == ls.leader_explorer_id {
                    Some(crate::leader::build_leader_prefix(
                        ls,
                        &explorer_cfg.explorer_id,
                    ))
                } else {
                    let follower_idx = {
                        let mut fi = 0usize;
                        let mut count = 0usize;
                        for (j, ec) in provisioned.explorer_configs.iter().enumerate() {
                            if ec.explorer_id != ls.leader_explorer_id {
                                if j == idx {
                                    fi = count;
                                }
                                count += 1;
                            }
                        }
                        fi
                    };
                    Some(crate::leader::build_follower_prefix_with_aspect(
                        ls,
                        follower_idx,
                        0.4,
                    ))
                }
            });

            let grounding = slot_groundings.get(idx).and_then(|g| g.as_ref()).cloned();
            let tombstone = input.pending_tombstone.clone();

            SlotData {
                slot_task,
                leader_prefix,
                role_frame,
                mandate,
                rejection_criteria,
                grounding,
                tombstone,
            }
        })
        .collect();

    // ── Phase B: async context assembly ──────────────────────────────────────
    use crate::context_assembler::{ContextAssembler, ContextAssemblerInput};

    let assembled_ctx_futs: Vec<_> = slot_data
        .iter()
        .enumerate()
        .map(|(idx, sd)| {
            let assembler_input = ContextAssemblerInput {
                active_ctx: active_ctx.as_str(),
                retry_context: None,
                leader_prefix: sd.leader_prefix.as_deref(),
                grounding: sd.grounding.as_deref(),
                tombstone: sd.tombstone.as_deref(),
                role_frame: sd.role_frame.as_deref(),
                mandate: sd.mandate.as_deref(),
                rejection_criteria: sd.rejection_criteria.as_deref(),
                prev_wave_blob: input
                    .prev_assembled_contexts
                    .get(idx)
                    .and_then(|x| x.as_ref()),
                budget: engine_input.cfg.context_budget_tokens,
                quality_guard_ratio: engine_input.cfg.context_quality_guard_ratio,
                compression_adapter: input.compression_adapter.as_deref(),
                stable_cache: input.stable_cache.as_deref(),
                global_knowledge: None,
                topic_knowledge: None,
            };
            ContextAssembler::build(assembler_input)
        })
        .collect();

    let assembled_contexts: Vec<crate::context_assembler::AssembledContext> =
        join_all(assembled_ctx_futs).await;

    // ── Phase C: build futures_vec using assembled contexts ───────────────────
    let futures_vec: Vec<ExplorerFuture<'_>> = provisioned
        .explorer_configs
        .iter()
        .enumerate()
        .map(|(idx, explorer_cfg)| {
            let req = ComputeRequest {
                system_context: assembled_contexts[idx].text.clone(),
                task: slot_data[idx].slot_task.clone(),
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
        assembled_contexts: assembled_contexts.into_iter().map(Some).collect(),
    })
}
