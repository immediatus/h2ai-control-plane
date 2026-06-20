//! Pre-execution thinking loop: multi-archetype brainstorm → synthesis → quality gate.
//!
//! The loop runs up to `ThinkingLoopConfig::max_iterations` rounds. Each round:
//!   1. Select `n_archetypes` expert archetypes via LLM (iteration-aware prompt).
//!   2. Run each archetype's brainstorm in parallel.
//!   3. Synthesize all outputs into a `ThinkingReport`.
//!   4. Check convergence/coverage thresholds and an LLM quality gate.
//!
//! When `ThinkingLoopConfig::enabled` is `false`, returns a default `ThinkingReport`
//! immediately without calling the adapter.

use futures::future::join_all;
use h2ai_adapters::chain::{execute_chain, tournament_merge};
use h2ai_config::prompts::{
    THINKING_ARCHETYPE_MD_ITER1, THINKING_ARCHETYPE_MD_ITERN, THINKING_ARCHETYPE_SYSTEM_MD,
    THINKING_BRAINSTORM_TASK, THINKING_QUALITY_GATE_SYSTEM, THINKING_QUALITY_GATE_TASK,
    THINKING_SYNTHESIS_MD_PAIRWISE, THINKING_SYNTHESIS_MD_SYSTEM,
};
use h2ai_config::ThinkingLoopConfig;
use h2ai_context::embedding::EmbeddingModel;
use h2ai_knowledge::provider::KnowledgeProvider;
use h2ai_knowledge::types::{KnowledgeQuery, NodeDepth, NodeSource, SearchScope};
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::chain::{ChainStep, ChainedRequest};
use h2ai_types::config::AgentRole;
use h2ai_types::events::OracleGateResultEvent;
use h2ai_types::knowledge::{profile_for_role, RetrievalMode as TypesRetrievalMode};
use h2ai_types::manifest::CotStyle;
use h2ai_types::memory::RetryHintPattern;
use h2ai_types::sizing::TauValue;
use h2ai_types::thinking::{ArchetypeOutput, ArchetypeSpec, ModelTier, ThinkingReport};
use serde::Deserialize;
use std::sync::Arc;

use crate::llm_parse::{extract_first_json_array, strip_json_fences};
// ─── Public input struct ──────────────────────────────────────────────────────

pub struct ThinkingLoopInput<'a> {
    pub task_description: &'a str,
    pub constraint_ids: &'a [String],
    /// Domain tags used to focus knowledge retrieval (e.g. ["rtb", "latency"]).
    pub constraint_tags: &'a [String],
    /// Full constraint docs injected into the archetype selection prompt so the LLM
    /// can produce domain-scoped archetypes rather than generic personas.
    pub constraint_corpus: &'a [h2ai_constraints::types::ConstraintDoc],
    /// Static fallback context used when no knowledge provider is configured.
    pub research_context: &'a str,
    /// Knowledge provider queried at each iteration. When present, domain articles
    /// and constraint wiki content are fetched and injected into every brainstorm
    /// prompt — giving archetypes access to relevant knowledge without requiring
    /// manually-written hints.
    pub knowledge_provider: Option<Arc<dyn KnowledgeProvider>>,
    pub n_archetypes: usize,
    pub cfg: &'a ThinkingLoopConfig,
    pub adapter: &'a dyn IComputeAdapter,
    pub embedding_model: Option<&'a dyn EmbeddingModel>,
    /// Optional NATS client for inline oracle checks per archetype (Stage 2).
    /// Pass `None` to skip oracle checks.
    pub nats_client: Option<async_nats::Client>,
    /// Task ID used in oracle gate payloads. May be empty when `nats_client` is `None`.
    pub task_id: &'a str,
    /// Induction patterns from `InductionStore::load_patterns` for the current task's domain.
    /// Pass `&[]` when no store is available — `format_induction_priors` returns empty string.
    pub induction_patterns: &'a [h2ai_types::knowledge::KnowledgeNodePattern],
    /// Retry hint priors loaded by `load_priming_hints` before the task starts.
    /// Top patterns (by success_rate) are formatted into the archetype selection system prompt.
    /// Pass `&[]` when no scheduler is available — `format_retry_hint_priors` returns empty string.
    pub retry_hint_priors: &'a [RetryHintPattern],
}

// ─── Pure helpers (pub for unit tests) ───────────────────────────────────────

/// Adaptive archetype count: full exploration in iter 0; contracts by coverage deficit after.
/// Clamps result to [2, `max_n`].
#[must_use]
pub fn adaptive_n(iteration: usize, max_n: usize, coverage_score: f64) -> usize {
    if iteration == 0 {
        return max_n;
    }
    let deficit = 1.0 - coverage_score.clamp(0.0, 1.0);
    ((max_n as f64 * deficit).ceil() as usize).max(2)
}

/// Like `adaptive_n` but gated by quality floor: if `filter_ratio` < floor, return `max_n` unchanged.
#[must_use]
pub fn adaptive_n_guarded(
    iteration: usize,
    max_n: usize,
    coverage_score: f64,
    filter_ratio: f64,
    floor: f64,
) -> usize {
    if iteration == 0 || filter_ratio < floor {
        return max_n;
    }
    adaptive_n(iteration, max_n, coverage_score)
}

/// Linear temperature schedule: `tau_max` at iter 0, `tau_min` at iter (`max_iterations` - 1).
#[must_use]
pub fn scheduled_tau(iteration: usize, max_iterations: u32, tau_max: f64, tau_min: f64) -> f64 {
    if max_iterations <= 1 {
        return tau_max;
    }
    let progress = iteration as f64 / f64::from(max_iterations - 1);
    (tau_max - progress * (tau_max - tau_min)).clamp(tau_min, tau_max)
}

/// Extract the `candidate_solution` field value from structured LLM output text.
/// Searches for the last occurrence of `"candidate_solution"` and extracts the quoted string after the colon.
#[must_use]
pub fn extract_candidate_solution(text: &str) -> Option<String> {
    let marker = "\"candidate_solution\"";
    let pos = text.rfind(marker)?;
    let after = &text[pos + marker.len()..];
    // Find colon
    let colon = after.find(':')?;
    let after_colon = after[colon + 1..].trim_start();
    // Find opening quote
    if !after_colon.starts_with('"') {
        return None;
    }
    let inner = &after_colon[1..];
    // Find closing quote (handle escaped quotes)
    let end = inner.find('"')?;
    Some(inner[..end].to_string())
}

// ─── Entry point ─────────────────────────────────────────────────────────────

/// Run the thinking loop and return the final `ThinkingReport`.
///
/// Returns `ThinkingReport::default()` immediately when `input.cfg.enabled` is `false`.
pub async fn run(input: ThinkingLoopInput<'_>) -> ThinkingReport {
    if !input.cfg.enabled {
        return ThinkingReport::default();
    }

    let mut report = ThinkingReport::default();
    let mut all_retrieved: Vec<(String, NodeSource)> = vec![];

    for iteration in 0..input.cfg.max_iterations as usize {
        // Approximation: no per-run pass_rate tracked yet; coverage_score is directionally correlated.
        let current_filter_ratio = if iteration == 0 {
            1.0
        } else {
            report.coverage_score
        };
        let n_this_iter = adaptive_n_guarded(
            iteration,
            input.cfg.max_archetypes,
            report.coverage_score,
            current_filter_ratio,
            input.cfg.expansion_quality_floor,
        );

        // Query knowledge provider at each iteration. Later iterations refine the
        // query with unresolved tensions so retrieval focuses on current gaps.
        let (iteration_knowledge, retrieved) =
            fetch_iteration_knowledge(&input, &report, iteration).await;
        all_retrieved.extend(retrieved);
        let research_context = if iteration_knowledge.is_empty() {
            input.research_context
        } else {
            &iteration_knowledge
        };

        let raw_archetypes =
            select_archetypes(&input, research_context, &report, iteration, n_this_iter).await;
        let used_fallback = raw_archetypes.is_empty();
        let archetypes = if used_fallback {
            fallback_archetypes()
        } else {
            raw_archetypes
        };
        {
            let names: Vec<&str> = archetypes.iter().map(|a| a.name.as_str()).collect();
            let scopes: Vec<&str> = archetypes.iter().map(|a| a.scope.as_str()).collect();
            tracing::info!(
                target: "h2ai.thinking",
                iteration,
                fallback = used_fallback,
                ?names,
                ?scopes,
                "archetypes selected"
            );
        }

        let iteration_tau = scheduled_tau(
            iteration,
            input.cfg.max_iterations,
            input.cfg.tau_max,
            input.cfg.tau_min,
        );

        let outputs = brainstorm_all(&input, research_context, &archetypes, iteration_tau).await;
        let mut new_report = synthesize(&input, &outputs, &report).await;

        new_report.prev_similarity = compute_similarity(
            &report.shared_understanding,
            &new_report.shared_understanding,
            input.embedding_model,
        );
        new_report.iteration = iteration as u32 + 1;
        report = new_report;

        let is_last = iteration + 1 >= input.cfg.max_iterations as usize;
        if is_last {
            break;
        }
        if report.coverage_score < input.cfg.coverage_threshold {
            continue;
        }
        if report.prev_similarity < input.cfg.convergence_threshold && iteration > 0 {
            continue;
        }
        if llm_gate(&input, &report).await {
            break;
        }
    }

    // Populate ThinkingReport with retrieval tracking for post_run feedback.
    {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let deduplicated: Vec<(String, NodeSource)> = all_retrieved
            .into_iter()
            .filter(|(id, _)| seen.insert(id.clone()))
            .collect();
        report.skill_nodes_used = deduplicated
            .iter()
            .filter(|(_, src)| matches!(src, NodeSource::Synthetic))
            .count() as u32;
        report.retrieved_node_ids = deduplicated.into_iter().map(|(id, _)| id).collect();
    }

    report
}

// ─── Knowledge retrieval ─────────────────────────────────────────────────────

/// Fetch domain knowledge for this iteration from the knowledge provider.
/// Iteration 0 queries with the bare task description; later iterations
/// refine with unresolved tensions so retrieval focuses on current gaps.
/// Returns an empty string when no provider is configured.
async fn fetch_iteration_knowledge(
    input: &ThinkingLoopInput<'_>,
    report: &ThinkingReport,
    iteration: usize,
) -> (String, Vec<(String, NodeSource)>) {
    let provider = match &input.knowledge_provider {
        Some(p) => p,
        None => return (String::new(), vec![]),
    };

    let query_text = if iteration == 0 || report.tensions.is_empty() {
        input.task_description.to_string()
    } else {
        format!("{} {}", input.task_description, report.tensions.join(" "))
    };

    let profile = profile_for_role(&AgentRole::Synthesizer);
    let mode = match profile.mode {
        TypesRetrievalMode::TreeTraversal => h2ai_knowledge::types::RetrievalMode::TreeTraversal,
        TypesRetrievalMode::CollapsedTree => h2ai_knowledge::types::RetrievalMode::CollapsedTree,
    };
    let query = KnowledgeQuery {
        text: &query_text,
        tags: input.constraint_tags,
        explicit_ids: input.constraint_ids,
        top_k: profile.top_k,
        depths: &[NodeDepth::Topic, NodeDepth::Leaf],
        mode,
        scope: SearchScope::Auto,
        expand_hops: profile.expand_hops,
    };

    let result = provider.query(&query).await;
    let retrieved: Vec<(String, NodeSource)> = result
        .nodes
        .iter()
        .map(|(n, _)| (n.id.clone(), n.source.clone()))
        .collect();
    let synthesis: Vec<&str> = result
        .nodes
        .iter()
        .map(|(n, _)| n.synthesis.as_str())
        .collect();
    (synthesis.join("\n\n"), retrieved)
}

// ─── Archetype selection ──────────────────────────────────────────────────────

/// Build a compact constraint spec block injected into the archetype selection prompt.
/// Returns empty string when the corpus is empty (graceful no-op).
/// Format per constraint:
///   CONSTRAINT-004 — <description>
///     [1] <binary_check_text>
///     [2] …
pub fn format_constraint_context(corpus: &[h2ai_constraints::types::ConstraintDoc]) -> String {
    if corpus.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "CONSTRAINT SPECIFICATIONS (read these carefully to choose domain-matched archetypes):\n",
    );
    for doc in corpus {
        out.push_str(&format!("\n{}", doc.id));
        if !doc.description.is_empty() {
            out.push_str(&format!(" — {}", doc.description));
        }
        for (i, check) in doc.binary_checks.iter().enumerate() {
            out.push_str(&format!("\n  [{}] {}", i + 1, check));
        }
    }
    out
}

/// Single pragmatic archetype used when LLM archetype selection fails to parse.
/// Keeps the thinking loop alive for at least one brainstorm iteration.
pub fn fallback_archetypes() -> Vec<ArchetypeSpec> {
    vec![
        ArchetypeSpec {
            name: "Constraint Satisfier".to_string(),
            persona: "A systems engineer who reads each constraint literally and builds the minimal design that satisfies all of them simultaneously. Prefers proven patterns; checks each constraint explicitly before finalising.".to_string(),
            scope: "Complete solution satisfying all stated constraints — no constraint is optional.".to_string(),
            confidence: 0.8,
            tau: 0.3,
            model_tier: ModelTier::Standard,
            cot_style: CotStyle::StepByStep,
            focus_constraints: vec![],
        },
        ArchetypeSpec {
            name: "Failure Mode Analyst".to_string(),
            persona: "An adversarial reviewer who stress-tests each proposed design against concurrent retries, partial failures, and schema migrations. Identifies which constraint appears satisfied but breaks under an edge case.".to_string(),
            scope: "Critique of the design from the perspective of the most likely failure mode for each constraint.".to_string(),
            confidence: 0.7,
            tau: 0.5,
            model_tier: ModelTier::Standard,
            cot_style: CotStyle::DevilsAdvocate,
            focus_constraints: vec![],
        },
        ArchetypeSpec {
            name: "Migration Safety Specialist".to_string(),
            persona: "A database reliability engineer who has managed live schema migrations on 100M-row tables. Specifies every DDL step explicitly, provides rollback scripts, and identifies which operations are destructive versus additive.".to_string(),
            scope: "Concrete migration plan with explicit additive-only DDL for the schema migration constraint, plus rollback procedures.".to_string(),
            confidence: 0.7,
            tau: 0.4,
            model_tier: ModelTier::Standard,
            cot_style: CotStyle::BackwardChaining,
            focus_constraints: vec![],
        },
    ]
}

/// Returns constraint IDs from `constraint_ids` that have no dedicated archetype in `archetypes`.
/// An archetype "covers" a constraint if that constraint ID appears in its `focus_constraints`.
/// Archetypes with empty `focus_constraints` cover nothing specifically.
#[must_use]
pub fn find_uncovered_constraints(
    archetypes: &[ArchetypeSpec],
    constraint_ids: &[String],
) -> Vec<String> {
    constraint_ids
        .iter()
        .filter(|cid| {
            !archetypes
                .iter()
                .any(|a| a.focus_constraints.iter().any(|f| f == *cid))
        })
        .cloned()
        .collect()
}

/// Synthesize a minimal coverage archetype targeting `constraint_id`.
///
/// Used when no existing archetype declares this constraint in `focus_constraints`.
/// Looks up the constraint description from `corpus` for a richer persona; falls back
/// to a generic description when the constraint is absent from the corpus.
#[must_use]
pub fn synthesize_coverage_archetype(
    constraint_id: &str,
    corpus: &[h2ai_constraints::types::ConstraintDoc],
) -> ArchetypeSpec {
    let description = corpus
        .iter()
        .find(|d| d.id == constraint_id)
        .map(|d| d.description.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("the specified constraint");
    ArchetypeSpec {
        name: format!("{constraint_id}-specialist"),
        persona: format!(
            "You are a constraint compliance specialist who focuses exclusively on {constraint_id}: \
             {description}. You read each binary check literally and build the minimal design that \
             satisfies all sub-conditions simultaneously without regressing any other constraint."
        ),
        scope: format!(
            "Complete and explicit satisfaction of {constraint_id} — every binary check must pass."
        ),
        confidence: 0.8,
        tau: 0.3,
        model_tier: ModelTier::Standard,
        cot_style: CotStyle::StepByStep,
        focus_constraints: vec![constraint_id.to_string()],
    }
}

async fn select_archetypes(
    input: &ThinkingLoopInput<'_>,
    research_context: &str,
    report: &ThinkingReport,
    iteration: usize,
    n_this_iter: usize,
) -> Vec<ArchetypeSpec> {
    let n = n_this_iter.to_string();
    let constraints = input.constraint_ids.join(", ");

    // Build constraint-spec context from the corpus so the archetype selection LLM
    // knows what each constraint ID actually demands — enabling domain-specific
    // archetypes (e.g. "redis-atomicity-specialist") rather than generic personas.
    let corpus_context = format_constraint_context(input.constraint_corpus);
    let combined_context = match (corpus_context.is_empty(), research_context.is_empty()) {
        (true, _) => research_context.to_string(),
        (false, true) => corpus_context,
        (false, false) => format!("{}\n\n{}", corpus_context, research_context),
    };

    let task = if iteration == 0 {
        THINKING_ARCHETYPE_MD_ITER1.render(&[
            ("description", input.task_description),
            ("constraints", &constraints),
            ("research_context", &combined_context),
            ("n", &n),
        ])
    } else {
        let tensions_joined = report.tensions.join("; ");
        THINKING_ARCHETYPE_MD_ITERN.render(&[
            ("description", input.task_description),
            ("understanding", &report.shared_understanding),
            ("tensions", &tensions_joined),
            ("n", &n),
        ])
    };

    let priors = format_induction_priors(input.induction_patterns);
    let retry_priors = format_retry_hint_priors(input.retry_hint_priors);
    let system_context = match (priors.is_empty(), retry_priors.is_empty()) {
        (true, true) => THINKING_ARCHETYPE_SYSTEM_MD.to_string(),
        (false, true) => format!("{}\n\n{}", THINKING_ARCHETYPE_SYSTEM_MD, priors),
        (true, false) => format!("{}\n\n{}", THINKING_ARCHETYPE_SYSTEM_MD, retry_priors),
        (false, false) => format!(
            "{}\n\n{}\n\n{}",
            THINKING_ARCHETYPE_SYSTEM_MD, priors, retry_priors
        ),
    };

    let mut archetypes = match execute_chain(
        input.adapter,
        ChainedRequest {
            initial_system_context: system_context,
            steps: vec![ChainStep {
                template: task,
                tau: TauValue::new(0.2).unwrap(),
                max_tokens: input.cfg.archetype_select_max_tokens,
            }],
        },
    )
    .await
    {
        Ok(filled) => {
            tracing::debug!(
                target: "h2ai.thinking",
                iteration,
                raw = %filled,
                "select_archetypes: raw LLM output"
            );
            match parse_archetypes_from_markdown(&filled) {
                Ok(specs) if !specs.is_empty() => specs,
                Ok(_) => {
                    tracing::warn!(
                        target: "h2ai.thinking",
                        "select_archetypes: markdown parser returned empty vec (iteration {iteration})"
                    );
                    vec![]
                }
                Err(e) => {
                    tracing::warn!(
                        target: "h2ai.thinking",
                        "select_archetypes: failed to parse archetype markdown (iteration {iteration}): {e}"
                    );
                    vec![]
                }
            }
        }
        Err(e) => {
            tracing::warn!(
                target: "h2ai.thinking",
                "select_archetypes: adapter error at iteration {iteration}: {e}"
            );
            vec![]
        }
    };

    // Guarantee per-constraint coverage: synthesize a specialist archetype for any
    // constraint that has no dedicated archetype in focus_constraints.
    let uncovered = find_uncovered_constraints(&archetypes, input.constraint_ids);
    for cid in &uncovered {
        tracing::info!(
            target: "h2ai.thinking",
            constraint_id = %cid,
            iteration,
            "no archetype covers constraint — synthesizing coverage archetype"
        );
        archetypes.push(synthesize_coverage_archetype(cid, input.constraint_corpus));
    }

    archetypes
}

// ─── Brainstorm ───────────────────────────────────────────────────────────────

async fn brainstorm_all(
    input: &ThinkingLoopInput<'_>,
    research_context: &str,
    archetypes: &[ArchetypeSpec],
    iteration_tau: f64,
) -> Vec<ArchetypeOutput> {
    let futures: Vec<_> = archetypes
        .iter()
        .map(|a| {
            brainstorm_one(
                input,
                research_context,
                a,
                iteration_tau,
                input.nats_client.clone(),
            )
        })
        .collect();
    join_all(futures).await
}

async fn brainstorm_one(
    input: &ThinkingLoopInput<'_>,
    research_context: &str,
    archetype: &ArchetypeSpec,
    iteration_tau: f64,
    nats_client: Option<async_nats::Client>,
) -> ArchetypeOutput {
    let cot_instruction = archetype.cot_style.instruction();
    let task = THINKING_BRAINSTORM_TASK.render(&[
        ("cot_instruction", cot_instruction),
        ("description", input.task_description),
        ("research_context", research_context),
    ]);

    let tau = TauValue::new(archetype.tau.clamp(0.0, iteration_tau))
        .unwrap_or_else(|_| TauValue::new(0.5).unwrap());

    let req = ComputeRequest {
        system_context: archetype.persona.clone(),
        task,
        tau,
        max_tokens: input.cfg.brainstorm_max_tokens,
    };

    let (llm_response_text, problem_analysis, solution_sketch, confidence) = match input
        .adapter
        .execute(req)
        .await
    {
        Ok(resp) => {
            let (pa, ss, conf) =
                parse_brainstorm_output(&resp.output, archetype.confidence.clamp(0.0, 1.0));
            (resp.output, pa, ss, conf)
        }
        Err(e) => {
            tracing::warn!(target: "h2ai.thinking_loop", archetype = %archetype.name, error = %e, "brainstorm call failed");
            (
                String::new(),
                String::new(),
                String::new(),
                archetype.confidence.clamp(0.0, 1.0),
            )
        }
    };

    // Inline oracle check (Stage 2)
    let oracle_result = if input.cfg.oracle_timeout_secs > 0 {
        if let Some(candidate) = extract_candidate_solution(&llm_response_text) {
            if let Some(nats) = &nats_client {
                let payload = serde_json::json!({
                    "task_id": input.task_id,
                    "candidate_solution": candidate,
                    "stage": "thinking_loop",
                });
                let timeout = std::time::Duration::from_secs(input.cfg.oracle_timeout_secs);
                match tokio::time::timeout(
                    timeout,
                    nats.request(
                        "h2ai.oracle.gate",
                        serde_json::to_vec(&payload).unwrap_or_default().into(),
                    ),
                )
                .await
                {
                    Ok(Ok(response)) => {
                        serde_json::from_slice::<OracleGateResultEvent>(&response.payload).ok()
                    }
                    _ => None,
                }
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    ArchetypeOutput {
        archetype: archetype.clone(),
        problem_analysis,
        solution_sketch,
        confidence,
        oracle_result,
    }
}

/// Extract `problem_analysis`, `solution_sketch`, and self-reported confidence from brainstorm output.
/// Falls back gracefully: full text → `problem_analysis` when no structure found.
fn parse_brainstorm_output(text: &str, default_confidence: f64) -> (String, String, f64) {
    // Try to find the trailing {"confidence": N} pattern.
    let confidence = text
        .rfind(r#""confidence""#)
        .and_then(|pos| {
            let snippet = &text[pos..];
            // grab the number after the colon
            snippet
                .split(':')
                .nth(1)
                .and_then(|s| {
                    s.chars()
                        .skip_while(|c| c.is_whitespace())
                        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                        .collect::<String>()
                        .parse::<f64>()
                        .ok()
                })
                .map(|v| v.clamp(0.0, 1.0))
        })
        .unwrap_or(default_confidence);

    // Split on "SOLUTION SKETCH:" if present.
    if let Some(idx) = text.find("SOLUTION SKETCH:") {
        let problem = text[..idx].trim().to_string();
        let solution = text[idx + "SOLUTION SKETCH:".len()..].trim().to_string();
        return (problem, solution, confidence);
    }

    // Fallback: entire text as problem_analysis.
    (text.to_string(), String::new(), confidence)
}

// ─── Synthesis ────────────────────────────────────────────────────────────────

async fn synthesize(
    input: &ThinkingLoopInput<'_>,
    outputs: &[ArchetypeOutput],
    prior: &ThinkingReport,
) -> ThinkingReport {
    if outputs.is_empty() {
        return ThinkingReport::default();
    }

    // Sort by Krum-adjusted confidence (highest first) so proposals[0] is the Krum leader.
    let mut sorted_outputs = outputs.to_vec();
    sorted_outputs.sort_by(|a, b| {
        let ja = if a.oracle_result.as_ref().is_some_and(|r| r.gate_passed) {
            (a.confidence + input.cfg.oracle_confidence_bonus).min(1.0)
        } else {
            a.confidence
        };
        let jb = if b.oracle_result.as_ref().is_some_and(|r| r.gate_passed) {
            (b.confidence + input.cfg.oracle_confidence_bonus).min(1.0)
        } else {
            b.confidence
        };
        jb.partial_cmp(&ja).unwrap_or(std::cmp::Ordering::Equal)
    });

    let proposals: Vec<String> = sorted_outputs
        .iter()
        .map(|o| {
            let j_eff = if o.oracle_result.as_ref().is_some_and(|r| r.gate_passed) {
                (o.confidence + input.cfg.oracle_confidence_bonus).min(1.0)
            } else {
                o.confidence
            };
            format!(
                "[{} | confidence={:.2}]\nPROBLEM: {}\nSOLUTION: {}",
                o.archetype.name, j_eff, o.problem_analysis, o.solution_sketch
            )
        })
        .collect();

    let system_context = format!(
        "{}\n\nPRIOR UNDERSTANDING (from previous iteration — empty on first pass):\n{}",
        THINKING_SYNTHESIS_MD_SYSTEM, prior.shared_understanding,
    );

    match tournament_merge(
        input.adapter,
        &system_context,
        proposals,
        THINKING_SYNTHESIS_MD_PAIRWISE,
        TauValue::new(0.3).unwrap(),
        input.cfg.synthesis_tournament_max_round_tokens,
    )
    .await
    {
        Ok(merged) => parse_synthesis_from_markdown(&merged),
        Err(e) => {
            tracing::warn!(
                target: "h2ai.thinking",
                "synthesize: adapter error — returning default ThinkingReport: {e}"
            );
            ThinkingReport::default()
        }
    }
}

// ─── LLM quality gate ─────────────────────────────────────────────────────────

/// Returns `true` when the LLM says YES (ready to stop), `false` otherwise.
async fn llm_gate(input: &ThinkingLoopInput<'_>, report: &ThinkingReport) -> bool {
    let tensions = report.tensions.join("; ");
    let coverage = format!("{:.2}", report.coverage_score);
    let task = THINKING_QUALITY_GATE_TASK.render(&[
        ("understanding", &report.shared_understanding),
        ("tensions", &tensions),
        ("coverage", &coverage),
    ]);

    let req = ComputeRequest {
        system_context: THINKING_QUALITY_GATE_SYSTEM.into(),
        task,
        tau: TauValue::new(0.1).unwrap(),
        max_tokens: input.cfg.quality_gate_max_tokens,
    };

    match input.adapter.execute(req).await {
        Ok(resp) => resp.output.trim().to_uppercase().starts_with("YES"),
        Err(_) => false,
    }
}

// ─── Similarity ───────────────────────────────────────────────────────────────

/// Cosine similarity between two strings via the embedding model.
/// Falls back to exact-equality (1.0 / 0.0) when no model is provided or either string is empty.
fn compute_similarity(a: &str, b: &str, model: Option<&dyn EmbeddingModel>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    h2ai_context::embedding::semantic_jaccard(a, b, model)
}

// ─── Public parse helpers (also used in tests) ────────────────────────────────

/// Parse a JSON array of `ArchetypeSpec` from LLM output.
/// Returns `None` if the text is not a JSON array.
///
/// Handles two common LLM deviations from the "output only JSON" instruction:
/// 1. Markdown code fences wrapping the array.
/// 2. Preamble/postamble prose around the array (e.g. reasoning tokens, "Here are the
///    archetypes:"). In that case we locate the outermost `[…]` by bracket-depth scanning.
#[must_use]
pub fn parse_archetypes(text: &str) -> Option<Vec<ArchetypeSpec>> {
    let stripped = strip_json_fences(text);
    // Fast path: the stripped text IS the JSON array.
    let json_value = serde_json::from_str::<serde_json::Value>(stripped.trim())
        .ok()
        // Fallback: extract the outermost JSON array from mixed-content text.
        .or_else(|| {
            extract_first_json_array(stripped).and_then(|s| serde_json::from_str(s).ok())
        })?;
    let arr = json_value.as_array()?;
    let specs: Vec<ArchetypeSpec> = arr
        .iter()
        .filter_map(|item| serde_json::from_value(item.clone()).ok())
        .collect();
    if specs.is_empty() {
        None
    } else {
        Some(specs)
    }
}

/// Parse a markdown-filled archetype document into a `Vec<ArchetypeSpec>`.
///
/// Blocks split on `## Archetype` headers; each block extracts labelled fields.
///
/// # Errors
/// Returns `Err(String)` when no blocks are found or a block is missing `**Persona:**`.
pub fn parse_archetypes_from_markdown(text: &str) -> Result<Vec<ArchetypeSpec>, String> {
    let chunks: Vec<&str> = text.split("## Archetype").skip(1).collect();
    if chunks.is_empty() {
        return Err("no '## Archetype' headers found in model output".to_string());
    }
    chunks
        .iter()
        .map(|block| parse_archetype_block(block))
        .collect()
}

fn parse_archetype_block(block: &str) -> Result<ArchetypeSpec, String> {
    // First non-empty line: "N: name-in-kebab-case" or just "name-in-kebab-case"
    let name_line = block
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    let name = name_line
        .split_once(':')
        .map_or(name_line, |(_prefix, rest)| rest.trim())
        .to_string();

    let persona = extract_md_field(block, "Persona")
        .ok_or_else(|| format!("archetype '{name}': missing **Persona:** field"))?;
    let scope = extract_md_field(block, "Scope").unwrap_or_default();
    let confidence = extract_md_field(block, "Confidence")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.7);
    let tau = extract_md_field(block, "Tau")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.5);
    let model_tier = match extract_md_field(block, "Model tier")
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_lowercase()
        .as_str()
    {
        "fast" => ModelTier::Fast,
        "capable" => ModelTier::Capable,
        _ => ModelTier::Standard,
    };
    let cot_style = match extract_md_field(block, "CoT style")
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_lowercase()
        .as_str()
    {
        "step_by_step" => CotStyle::StepByStep,
        "backward_chaining" => CotStyle::BackwardChaining,
        "first_principles" => CotStyle::FirstPrinciples,
        "devil_s_advocate" | "devils_advocate" | "devil's_advocate" => CotStyle::DevilsAdvocate,
        _ => CotStyle::None,
    };

    let focus_constraints = extract_md_field(block, "Focus Constraints")
        .map(|s| {
            s.split(',')
                .map(|id| id.trim().to_string())
                .filter(|id| !id.is_empty() && !id.eq_ignore_ascii_case("all"))
                .collect()
        })
        .unwrap_or_default();

    Ok(ArchetypeSpec {
        name,
        persona,
        scope,
        confidence,
        tau,
        model_tier,
        cot_style,
        focus_constraints,
    })
}

/// Extract the value of a `**Field:**` line from a markdown block.
/// Returns the text after the marker, trimmed. Returns `None` if the field is absent.
pub(crate) fn extract_md_field(block: &str, field: &str) -> Option<String> {
    let marker = format!("**{field}:**");
    for line in block.lines() {
        if let Some(idx) = line.find(&marker) {
            let after = &line[idx + marker.len()..];
            return Some(after.trim().to_string());
        }
    }
    None
}

/// Format `InductionStore` node patterns as a markdown prior-context block.
///
/// Prepended to the archetype-selection `system_context` so the model can
/// bias archetype selection toward high-hit-rate knowledge patterns.
/// Returns an empty string when `patterns` is empty (cold start / store unavailable).
#[must_use]
pub fn format_induction_priors(patterns: &[h2ai_types::knowledge::KnowledgeNodePattern]) -> String {
    if patterns.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "## Prior Knowledge: High-Impact Retrieval Patterns\n\
         The following patterns were most effective for this domain in past tasks. \
         Prefer archetypes whose scope aligns with these signals.\n\n",
    );
    for p in patterns.iter().take(5) {
        let tags = p.domain_tags.join(", ");
        out.push_str(&format!(
            "- {} (hit_rate: {:.1}, domains: [{}])\n",
            p.node_id, p.hit_rate, tags
        ));
    }
    out
}

/// Format top-5 `RetryHintPattern` records as a markdown prior-context block.
///
/// Injected into the archetype selection system prompt so the LLM knows which
/// retry strategies have worked in the past for similar task domains.
/// Returns empty string when `patterns` is empty — callers treat this as a no-op.
pub fn format_retry_hint_priors(patterns: &[RetryHintPattern]) -> String {
    if patterns.is_empty() {
        return String::new();
    }
    let mut out =
        String::from("RETRY HISTORY (apply these learnings when selecting archetypes):\n");
    for p in patterns.iter().take(5) {
        let tags = p.trigger_tags.join(", ");
        out.push_str(&format!(
            "- [{}] {} → \"{}\" (rate={:.2}, n={})\n",
            tags,
            p.exit_reason_kind,
            p.hint_text,
            p.success_rate(),
            p.attempt_count,
        ));
    }
    out
}

/// Parse a `ThinkingReport` from LLM synthesis output.
/// Falls back to treating the entire text as `shared_understanding` with `coverage_score = 0.5`.
#[must_use]
pub fn parse_thinking_report(text: &str) -> ThinkingReport {
    let stripped = strip_json_fences(text);

    #[derive(Deserialize, Default)]
    struct SynthesisJson {
        #[serde(default)]
        shared_understanding: String,
        #[serde(default)]
        tensions: Vec<String>,
        #[serde(default)]
        coverage_score: f64,
    }

    if let Ok(parsed) = serde_json::from_str::<SynthesisJson>(stripped.trim()) {
        if !parsed.shared_understanding.is_empty() || parsed.coverage_score > 0.0 {
            return ThinkingReport {
                shared_understanding: parsed.shared_understanding,
                tensions: parsed.tensions,
                coverage_score: parsed.coverage_score,
                iteration: 0,
                prev_similarity: 0.0,
                retrieved_node_ids: vec![],
                skill_nodes_used: 0,
            };
        }
    }

    // Plain text fallback.
    ThinkingReport {
        shared_understanding: text.to_string(),
        tensions: vec![],
        coverage_score: 0.5,
        iteration: 0,
        prev_similarity: 0.0,
        retrieved_node_ids: vec![],
        skill_nodes_used: 0,
    }
}

/// Parse a markdown synthesis document produced by `tournament_merge` into a `ThinkingReport`.
///
/// Sections:
/// - `## Shared Understanding` — everything until the next `##` header.
/// - `## Unresolved Tensions` — each `- ` bullet as a tension string.
/// - `## Coverage Assessment` — the float on the `**Score:** ` line.
///
/// Falls back to treating the whole text as `shared_understanding` with `coverage_score = 0.5`
/// when no section headers are found.
#[must_use]
pub fn parse_synthesis_from_markdown(text: &str) -> ThinkingReport {
    let shared_understanding = extract_section(text, "## Shared Understanding");
    let tensions_section = extract_section(text, "## Unresolved Tensions");
    let coverage_section = extract_section(text, "## Coverage Assessment");

    // If no sections found at all, plain-text fallback
    if shared_understanding.is_empty() && tensions_section.is_empty() && coverage_section.is_empty()
    {
        return ThinkingReport {
            shared_understanding: text.to_string(),
            tensions: vec![],
            coverage_score: 0.5,
            iteration: 0,
            prev_similarity: 0.0,
            retrieved_node_ids: vec![],
            skill_nodes_used: 0,
        };
    }

    let tensions: Vec<String> = tensions_section
        .lines()
        .filter(|l| l.trim_start().starts_with("- "))
        .map(|l| l.trim_start_matches("- ").trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    let coverage_score = coverage_section
        .lines()
        .find(|l| l.contains("**Score:**"))
        .and_then(|l| {
            let marker = "**Score:**";
            l.find(marker).map(|idx| l[idx + marker.len()..].trim())
        })
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.5);

    ThinkingReport {
        shared_understanding,
        tensions,
        coverage_score,
        iteration: 0,
        prev_similarity: 0.0,
        retrieved_node_ids: vec![],
        skill_nodes_used: 0,
    }
}

/// Extract the text content of a markdown section up to the next `##` header.
fn extract_section(text: &str, header: &str) -> String {
    let Some(start) = text.find(header) else {
        return String::new();
    };
    let after = &text[start + header.len()..];
    let end = after.find("\n## ").unwrap_or(after.len());
    after[..end].trim().to_string()
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
