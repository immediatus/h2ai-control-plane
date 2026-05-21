use crate::engine::EngineInput;
use crate::phases::StepResult;
use chrono::Utc;
use h2ai_types::events::ProposalEvent;
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::{TaskQuadrant, TauValue};

pub struct Input<'a> {
    pub engine_input: &'a EngineInput<'a>,
    pub task_id: &'a TaskId,
    pub assessed_quadrant: TaskQuadrant,
    pub wave_coherence: &'a crate::coherence::CoherenceState,
    pub synthesis_candidates: &'a [ProposalEvent],
}

pub struct Output {
    /// Synthesised text, or `None` when synthesis was skipped / failed.
    pub resolved_text: Option<String>,
    /// Q(synthesis) − max(Q(individuals)), zero when synthesis did not run.
    pub synthesis_gain: f64,
    /// Comparison events collected during the re-verification step.
    pub comparison_events: Vec<h2ai_types::events::VerifierComparisonEvent>,
}

/// Run Phase 5a: Synthesis (optional).
///
/// Checks whether synthesis should run (enabled flag, coherence closure,
/// enough candidates). When applicable, runs `SynthesisPhase::run`, then
/// re-verifies the output through `VerificationPhase`. Returns `Done` in all
/// cases — synthesis never causes an `EarlyExit`.
pub async fn run(input: Input<'_>) -> StepResult<Output> {
    let engine_input = input.engine_input;
    let task_id = input.task_id;
    let assessed_quadrant = input.assessed_quadrant;
    let wave_coherence = input.wave_coherence;
    let synthesis_candidates = input.synthesis_candidates;

    // Complex quadrant forces synthesis regardless of synthesis_enabled flag.
    let synthesis_forced =
        !engine_input.cfg.task_complexity.shadow_mode && assessed_quadrant == TaskQuadrant::Complex;

    // When coherence is closed, all surviving proposals already agree on every
    // constraint domain — synthesis reconciles nothing.
    let synthesis_bypass = wave_coherence.is_closed();

    if synthesis_bypass && (engine_input.cfg.synthesis_enabled || synthesis_forced) {
        tracing::debug!(
            target: "h2ai.coherence",
            "synthesis bypassed: coherence closed, proposals already agree"
        );
    }

    let should_run = !synthesis_bypass
        && (engine_input.cfg.synthesis_enabled || synthesis_forced)
        && synthesis_candidates.len() >= engine_input.cfg.synthesis_min_proposals;

    if !should_run {
        return StepResult::Done(Output {
            resolved_text: None,
            synthesis_gain: 0.0,
            comparison_events: vec![],
        });
    }

    let synth_adapter = match engine_input.synthesis_adapter {
        Some(a) => a,
        None => {
            return StepResult::Done(Output {
                resolved_text: None,
                synthesis_gain: 0.0,
                comparison_events: vec![],
            });
        }
    };

    use crate::synthesis::{SynthesisInput as SynthInput, SynthesisPhase};

    let constraint_list = engine_input.manifest.constraints.join("\n");
    let synth_input = SynthInput {
        task_description: &engine_input.manifest.description,
        constraint_list: &constraint_list,
        proposals: synthesis_candidates,
        adapter: synth_adapter,
        cfg: engine_input.cfg,
    };

    let synth_out = match SynthesisPhase::run(synth_input).await {
        Ok(out) => out,
        Err(e) => {
            tracing::warn!(
                task_id = %task_id,
                error = %e,
                "synthesis phase error; falling back to selection chain"
            );
            return StepResult::Done(Output {
                resolved_text: None,
                synthesis_gain: 0.0,
                comparison_events: vec![],
            });
        }
    };

    // Re-verify the synthesis output through the full VerificationPhase.
    use crate::verification::{new_eval_cache, VerificationInput, VerificationPhase};

    let synth_proposal = ProposalEvent {
        task_id: task_id.clone(),
        explorer_id: h2ai_types::identity::ExplorerId::new(),
        tau: TauValue::new(engine_input.cfg.synthesis_tau)
            .unwrap_or_else(|_| TauValue::new(0.2).unwrap()),
        generation: 0,
        raw_output: synth_out.synthesis_text.clone(),
        token_cost: synth_out.synthesis_tokens,
        adapter_kind: h2ai_types::config::AdapterKind::CloudGeneric {
            endpoint: "synthesis".into(),
            api_key_env: "NONE".into(),
            model: None,
        },
        timestamp: Utc::now(),
    };

    // Use a fresh eval cache — synthesis output was never scored before.
    let eval_cache = new_eval_cache();
    let re_ver = VerificationPhase::run(VerificationInput {
        proposals: vec![synth_proposal],
        constraint_corpus: &engine_input.constraint_corpus,
        evaluator: engine_input.verification_adapter,
        config: engine_input.verification_config.clone(),
        eval_cache: std::sync::Arc::clone(&eval_cache),
        consensus_passes: engine_input.cfg.verifier_consensus_passes,
    })
    .await;

    let comparison_events = re_ver.comparison_events.clone();

    if re_ver.passed.is_empty() {
        tracing::warn!(
            task_id = %task_id,
            "synthesis re-verification failed; falling back to selection chain"
        );
        return StepResult::Done(Output {
            resolved_text: None,
            synthesis_gain: 0.0,
            comparison_events,
        });
    }

    // Compute synthesis_gain: Q(synthesis) − max(Q(individuals)).
    let indiv_scores = VerificationPhase::score_proposals(
        synthesis_candidates.to_vec(),
        engine_input.verification_adapter,
        &engine_input.verification_config,
        &engine_input.constraint_corpus,
    )
    .await;

    let max_indiv = indiv_scores
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max)
        .max(0.0);

    let synth_score = re_ver.passed.first().map_or(0.0, |(_, results, _)| {
        h2ai_constraints::types::aggregate_compliance_score(results)
    });

    let synthesis_gain = synth_score - max_indiv;

    tracing::debug!(
        task_id = %task_id,
        synthesis_gain,
        "synthesis re-verification passed"
    );

    StepResult::Done(Output {
        resolved_text: Some(synth_out.synthesis_text),
        synthesis_gain,
        comparison_events,
    })
}
