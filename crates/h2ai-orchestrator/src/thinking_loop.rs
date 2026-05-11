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
use h2ai_config::prompts::{
    THINKING_ARCHETYPE_SELECT_ITER1, THINKING_ARCHETYPE_SELECT_ITERN, THINKING_ARCHETYPE_SYSTEM,
    THINKING_BRAINSTORM_TASK, THINKING_QUALITY_GATE_SYSTEM, THINKING_QUALITY_GATE_TASK,
    THINKING_SYNTHESIS_SYSTEM, THINKING_SYNTHESIS_TASK,
};
use h2ai_config::ThinkingLoopConfig;
use h2ai_context::embedding::EmbeddingModel;
use h2ai_types::adapter::{ComputeRequest, IComputeAdapter};
use h2ai_types::sizing::TauValue;
use h2ai_types::thinking::{ArchetypeOutput, ArchetypeSpec, ThinkingReport};
use serde::Deserialize;

// ─── Public input struct ──────────────────────────────────────────────────────

pub struct ThinkingLoopInput<'a> {
    pub task_description: &'a str,
    pub constraint_ids: &'a [String],
    pub research_context: &'a str,
    pub n_archetypes: usize,
    pub cfg: &'a ThinkingLoopConfig,
    pub adapter: &'a dyn IComputeAdapter,
    pub embedding_model: Option<&'a dyn EmbeddingModel>,
}

// ─── Pure helpers (pub for unit tests) ───────────────────────────────────────

/// Adaptive archetype count: full exploration in iter 0; contracts by coverage deficit after.
/// Clamps result to [2, max_n].
pub fn adaptive_n(iteration: usize, max_n: usize, coverage_score: f64) -> usize {
    if iteration == 0 {
        return max_n;
    }
    let deficit = 1.0 - coverage_score.clamp(0.0, 1.0);
    ((max_n as f64 * deficit).ceil() as usize).max(2)
}

/// Like `adaptive_n` but gated by quality floor: if filter_ratio < floor, return max_n unchanged.
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

/// Linear temperature schedule: tau_max at iter 0, tau_min at iter (max_iterations - 1).
pub fn scheduled_tau(iteration: usize, max_iterations: u32, tau_max: f64, tau_min: f64) -> f64 {
    if max_iterations <= 1 {
        return tau_max;
    }
    let progress = iteration as f64 / (max_iterations - 1) as f64;
    (tau_max - progress * (tau_max - tau_min)).clamp(tau_min, tau_max)
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

        let archetypes = select_archetypes(&input, &report, iteration, n_this_iter).await;
        if archetypes.is_empty() {
            break;
        }

        let iteration_tau = scheduled_tau(
            iteration,
            input.cfg.max_iterations,
            input.cfg.tau_max,
            input.cfg.tau_min,
        );

        let outputs = brainstorm_all(&input, &archetypes, iteration_tau).await;
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

    report
}

// ─── Archetype selection ──────────────────────────────────────────────────────

async fn select_archetypes(
    input: &ThinkingLoopInput<'_>,
    report: &ThinkingReport,
    iteration: usize,
    n_this_iter: usize,
) -> Vec<ArchetypeSpec> {
    let n = n_this_iter.to_string();
    let constraints = input.constraint_ids.join(", ");

    let task = if iteration == 0 {
        THINKING_ARCHETYPE_SELECT_ITER1.render(&[
            ("description", input.task_description),
            ("constraints", &constraints),
            ("research_context", input.research_context),
            ("n", &n),
        ])
    } else {
        let tensions_joined = report.tensions.join("; ");
        let base_task = THINKING_ARCHETYPE_SELECT_ITERN.render(&[
            ("description", input.task_description),
            ("understanding", &report.shared_understanding),
            ("tensions", &tensions_joined),
            ("n", &n),
        ]);
        if report.tensions.is_empty() {
            base_task
        } else {
            let gaps = report
                .tensions
                .iter()
                .enumerate()
                .map(|(i, t)| format!("{}. {}", i + 1, t))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "{}\n\nUnresolved tensions from the previous pass:\n{}\n\
                 Generate archetypes that specifically address these gaps. \
                 Do not repeat perspectives already covered.",
                base_task, gaps
            )
        }
    };

    let req = ComputeRequest {
        system_context: THINKING_ARCHETYPE_SYSTEM.into(),
        task,
        tau: TauValue::new(0.2).unwrap(),
        max_tokens: 1024,
    };

    match input.adapter.execute(req).await {
        Ok(resp) => parse_archetypes(&resp.output).unwrap_or_default(),
        Err(_) => vec![],
    }
}

// ─── Brainstorm ───────────────────────────────────────────────────────────────

async fn brainstorm_all(
    input: &ThinkingLoopInput<'_>,
    archetypes: &[ArchetypeSpec],
    iteration_tau: f64,
) -> Vec<ArchetypeOutput> {
    let futures: Vec<_> = archetypes
        .iter()
        .map(|a| brainstorm_one(input, a, iteration_tau))
        .collect();
    join_all(futures).await
}

async fn brainstorm_one(
    input: &ThinkingLoopInput<'_>,
    archetype: &ArchetypeSpec,
    iteration_tau: f64,
) -> ArchetypeOutput {
    let cot_instruction = archetype.cot_style.instruction();
    let task = THINKING_BRAINSTORM_TASK.render(&[
        ("cot_instruction", cot_instruction),
        ("description", input.task_description),
        ("research_context", input.research_context),
    ]);

    let tau = TauValue::new(archetype.tau.clamp(0.0, iteration_tau))
        .unwrap_or_else(|_| TauValue::new(0.5).unwrap());

    let req = ComputeRequest {
        system_context: archetype.persona.clone(),
        task,
        tau,
        max_tokens: 1024,
    };

    let (problem_analysis, solution_sketch, confidence) = match input.adapter.execute(req).await {
        Ok(resp) => parse_brainstorm_output(&resp.output, archetype.confidence.clamp(0.0, 1.0)),
        Err(e) => {
            tracing::warn!(target: "h2ai.thinking_loop", archetype = %archetype.name, error = %e, "brainstorm call failed");
            (
                String::new(),
                String::new(),
                archetype.confidence.clamp(0.0, 1.0),
            )
        }
    };

    ArchetypeOutput {
        archetype: archetype.clone(),
        problem_analysis,
        solution_sketch,
        confidence,
    }
}

/// Extract problem_analysis, solution_sketch, and self-reported confidence from brainstorm output.
/// Falls back gracefully: full text → problem_analysis when no structure found.
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
    let perspectives = outputs
        .iter()
        .map(|o| {
            format!(
                "[{} | confidence={:.2}]\nPROBLEM: {}\nSOLUTION: {}",
                o.archetype.name, o.confidence, o.problem_analysis, o.solution_sketch
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");

    let task = THINKING_SYNTHESIS_TASK.render(&[
        ("perspectives", &perspectives),
        ("prior_understanding", &prior.shared_understanding),
    ]);

    let req = ComputeRequest {
        system_context: THINKING_SYNTHESIS_SYSTEM.into(),
        task,
        tau: TauValue::new(0.3).unwrap(),
        max_tokens: 512,
    };

    match input.adapter.execute(req).await {
        Ok(resp) => parse_thinking_report(&resp.output),
        Err(_) => ThinkingReport::default(),
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
        max_tokens: 64,
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
pub fn parse_archetypes(text: &str) -> Option<Vec<ArchetypeSpec>> {
    // Strip markdown fences if present.
    let stripped = strip_json_fences(text);
    let v: serde_json::Value = serde_json::from_str(stripped.trim()).ok()?;
    let arr = v.as_array()?;
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

/// Parse a `ThinkingReport` from LLM synthesis output.
/// Falls back to treating the entire text as `shared_understanding` with `coverage_score = 0.5`.
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
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Remove ```json ... ``` or ``` ... ``` fences from LLM output.
fn strip_json_fences(s: &str) -> &str {
    let s = s.trim();
    if s.starts_with("```") {
        let after_open = s.find('\n').map(|i| &s[i + 1..]).unwrap_or(s);
        if let Some(close) = after_open.rfind("```") {
            return after_open[..close].trim();
        }
    }
    s
}
