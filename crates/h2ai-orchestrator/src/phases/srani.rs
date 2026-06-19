use crate::engine::EngineInput;
use crate::phases::StepResult;
use chrono::Utc;
use h2ai_types::events::{CorrelatedFabricationEvent, ResearcherGroundingEvent};
use h2ai_types::identity::TaskId;

use super::generation::Output as GenerationOutput;

pub struct Input<'a> {
    pub engine_input: &'a EngineInput<'a>,
    pub task_id: &'a TaskId,
    /// Current tier for the grounding chain (escalates on consecutive fires).
    pub srani_tier: usize,
    /// Whether SRANI fired in the previous wave (used to escalate tier).
    pub srani_last_wave_fired: bool,
    /// The current MAPE-K `retry_context` (may have been set by hallucination or prior SRANI).
    pub retry_context: Option<String>,
}

pub struct Output {
    pub generation: GenerationOutput,
    /// Updated EMA midpoint after absorbing this wave's CFI observation.
    pub srani_ema_cfi_updated: f64,
    /// Updated count after absorbing this wave's CFI observation.
    pub srani_count_updated: usize,
    /// Updated tier (may increment when hint fires on consecutive waves).
    pub srani_tier_updated: usize,
    /// Updated last-wave-fired flag.
    pub srani_last_wave_fired_updated: bool,
    /// Fabrication events emitted this wave (extend `all_srani_events` in caller).
    pub srani_events: Vec<CorrelatedFabricationEvent>,
    /// Researcher grounding events emitted this wave (extend `all_researcher_grounding_events`).
    pub researcher_grounding_events: Vec<ResearcherGroundingEvent>,
    /// Updated `retry_context` — may have been extended with SRANI hint.
    pub retry_context: Option<String>,
}

/// Run the SRANI (Specification-Relative Architectural Noun Intersection) phase.
///
/// Fires when diverse proposals share a fabricated entity not grounded in the task spec.
/// Orthogonal to C1 hallucination detection: SRANI detects entity-level fabrication
/// while C1 detects semantic clustering.
///
/// Always returns `StepResult::Done(Output)`. SRANI is enrichment-only — it never
/// triggers a retry loop `continue` directly. Hint injection updates `retry_context`
/// in the output so the next wave incorporates the grounding.
///
/// The caller updates MAPE-K state from output fields after this call:
///   `srani_ema_updated = out.srani_ema_cfi_updated`
///   `srani_count_updated = out.srani_count_updated`
///   `srani_tier = out.srani_tier_updated`
///   `srani_last_wave_fired = out.srani_last_wave_fired_updated`
///   `all_srani_events.extend(out.srani_events)`
///   `all_researcher_grounding_events.extend(out.researcher_grounding_events)`
///   `retry_context = out.retry_context`
///
/// Never returns `StepResult::EarlyExit` or `StepResult::Fatal`.
pub async fn run(generation: GenerationOutput, input: Input<'_>) -> StepResult<Output> {
    let engine_input = input.engine_input;
    let task_id = input.task_id;
    let mut srani_tier = input.srani_tier;
    let mut srani_last_wave_fired = input.srani_last_wave_fired;
    let mut retry_context = input.retry_context;

    let proposals = &generation.proposals;

    // Carry-forward EMA/count defaults — only updated when grounding fires.
    let mut srani_ema_cfi_updated = engine_input.srani_ema_cfi;
    let mut srani_count_updated = engine_input.srani_count;
    let mut srani_events: Vec<CorrelatedFabricationEvent> = Vec::new();
    let mut researcher_grounding_events: Vec<ResearcherGroundingEvent> = Vec::new();

    // ── SRANI: Specification-Relative Architectural Noun Intersection ──
    // Orthogonal to C1: fires when diverse proposals share a fabricated entity.
    if engine_input.cfg.srani.enabled && proposals.len() >= 2 {
        let proposal_texts: Vec<&str> = proposals.iter().map(|p| p.raw_output.as_str()).collect();
        // Build effective spec: task description + constraint corpus text so that
        // constraint-mandated technologies (e.g. Redis, Kafka required by constraints) are
        // not flagged as ungrounded entities absent from the task description alone.
        let constraint_text: String = engine_input
            .constraint_corpus
            .iter()
            .flat_map(|doc| {
                std::iter::once(doc.description.as_str())
                    .chain(doc.binary_checks.iter().map(String::as_str))
                    .chain(doc.pass_criteria.as_deref())
            })
            .collect::<Vec<_>>()
            .join("\n");
        let effective_spec = format!(
            "{}\n{}\n{}",
            engine_input.manifest.description,
            engine_input.manifest.context.as_deref().unwrap_or(""),
            constraint_text
        );
        if let Some(grounding) = crate::specification_grounding::check_specification_grounding(
            &effective_spec,
            &proposal_texts,
        ) {
            // Build grounded parents list: keys in implied_by that appear in effective_spec
            let grounded_parents: Vec<String> = engine_input
                .cfg
                .srani
                .implied_by
                .keys()
                .filter(|k| effective_spec.contains(k.as_str()))
                .cloned()
                .collect();

            let filtered_ungrounded = if grounded_parents.is_empty() {
                grounding.shared_ungrounded.clone()
            } else {
                crate::specification_grounding::apply_implied_by_suppression(
                    &grounding.shared_ungrounded,
                    &engine_input.cfg.srani.implied_by,
                    &grounded_parents,
                )
            };

            // Always update EMA regardless of whether the gate fires.
            let new_ema = crate::srani_gate::update_ema(
                engine_input.srani_ema_cfi,
                grounding.cfi,
                engine_input.cfg.srani.ema_alpha,
            );
            srani_ema_cfi_updated = new_ema;
            srani_count_updated = engine_input.srani_count + 1;

            let (pressure, hint_injected) = if engine_input.cfg.srani.adaptive {
                let mu = if engine_input.srani_count < 5 {
                    engine_input.cfg.srani.cold_start_midpoint()
                } else {
                    engine_input.srani_ema_cfi
                };
                let p = crate::srani_gate::compute_injection_pressure(
                    grounding.cfi,
                    mu,
                    engine_input.cfg.srani.temperature,
                );
                (p, p >= engine_input.cfg.srani.gate_threshold)
            } else {
                // Legacy static-threshold path (adaptive=false).
                let injected = grounding.cfi > engine_input.cfg.srani.inject_threshold;
                let p = if grounding.cfi > engine_input.cfg.srani.warn_threshold {
                    1.0
                } else {
                    0.0
                };
                (p, injected)
            };

            // Warn floor: 0.20 for adaptive, warn_threshold crossing for static.
            let should_emit = if engine_input.cfg.srani.adaptive {
                pressure >= 0.20
            } else {
                grounding.cfi > engine_input.cfg.srani.warn_threshold
            };

            if should_emit {
                srani_events.push(CorrelatedFabricationEvent {
                    task_id: task_id.clone(),
                    cfi: grounding.cfi,
                    injection_pressure: pressure,
                    shared_ungrounded_entities: filtered_ungrounded.clone(),
                    proposal_count: grounding.proposal_count,
                    hint_injected,
                    timestamp: Utc::now(),
                });
                if hint_injected {
                    let grounding_ctx = crate::srani_grounding::GroundingContext {
                        fabricated_entities: filtered_ungrounded.clone(),
                        task_description: engine_input.manifest.description.clone(),
                    };
                    let chain_result = if let Some(ref chain) = engine_input.srani_grounding_chain {
                        chain.resolve(&grounding_ctx, srani_tier).await
                    } else {
                        use crate::srani_grounding::GroundingProvider;
                        crate::srani_grounding::SpecAnchorGrounder
                            .ground(&grounding_ctx)
                            .await
                    };
                    if let Some(ref result) = chain_result {
                        let hint = crate::srani_grounding::format_grounding_hint(
                            result,
                            &filtered_ungrounded,
                        );
                        retry_context = Some(retry_context.unwrap_or_default() + &hint);
                        let grounding_slot =
                            crate::srani_grounding::classify_grounding_slot(&filtered_ungrounded);
                        researcher_grounding_events.push(ResearcherGroundingEvent {
                            task_id: task_id.clone(),
                            shared_assumption: filtered_ungrounded.join(", "),
                            literature_summary: result.grounding_statement.clone(),
                            slot: Some(grounding_slot),
                            source: result.source.clone(),
                        });
                    } else {
                        let entities = filtered_ungrounded.join(", ");
                        retry_context = Some(
                            retry_context.unwrap_or_default()
                                + &format!(
                                    "\n\n--- GROUNDING CONTEXT ---\n\
                                     Avoid (not in spec): {entities}\n\
                                     Design using spec-defined components only.\n---"
                                ),
                        );
                    }
                    if srani_last_wave_fired {
                        srani_tier = srani_tier.saturating_add(1);
                    }
                    srani_last_wave_fired = true;
                } else {
                    srani_last_wave_fired = false;
                }
            }
        }
    }

    StepResult::Done(Output {
        generation,
        srani_ema_cfi_updated,
        srani_count_updated,
        srani_tier_updated: srani_tier,
        srani_last_wave_fired_updated: srani_last_wave_fired,
        srani_events,
        researcher_grounding_events,
        retry_context,
    })
}
