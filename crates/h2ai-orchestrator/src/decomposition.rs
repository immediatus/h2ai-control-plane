use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate};
use h2ai_context::embedding::{cosine_similarity, EmbeddingModel};
use h2ai_types::adapter::{AdapterError, ComputeRequest, IComputeAdapter};
use h2ai_types::config::ParetoWeights;
use h2ai_types::manifest::{CotStyle, ExplorerSlotConfig};
use h2ai_types::sizing::TauValue;
use serde::Deserialize;
use std::collections::BTreeMap;
use thiserror::Error;

// ── Step 1: Failure Mode Analysis ────────────────────────────────────────────
// The LLM reads actual constraint rubrics and surfaces counter-intuitive requirements.
const STEP1_SYSTEM: &str = "\
You are a failure mode analyst. Your job is to read constraint requirements and \
identify the specific requirements that general-purpose engineers miss on first pass — \
not the obvious ones, but the ones that cause production incidents.";

// ── Step 2: Role Frame Design ─────────────────────────────────────────────────
// The LLM designs personas anchored to the failure modes surfaced in Step 1.
const STEP2_SYSTEM: &str = "\
You are designing expert reviewer personas for a technical committee. Each persona \
must be defined by what they notice FIRST when reading a proposal — anchored to \
specific professional experience with a concrete failure type, not a generic title.";

// ── Step 3: JSON Assembly ─────────────────────────────────────────────────────
// The LLM formats the designed roles as a JSON array. Narrow formatting task only.
const STEP3_SYSTEM: &str = "\
You are a JSON formatter. Convert structured expert role descriptions into a precise \
JSON array. Output only valid JSON — no markdown fences, no explanation.";

#[derive(Debug, Error)]
pub enum DecompositionError {
    #[error("response contains no JSON array")]
    NoJsonArray,
    #[error("JSON parse failed: {0}")]
    ParseError(String),
    #[error("all slots had empty role_frame after filtering")]
    EmptyResult,
}

/// Coerce a JSON value to a String: strings pass through, arrays are joined with "; ".
fn coerce_to_string<'de, D: serde::Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(value_to_string(v))
}

fn coerce_to_opt_string<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<String>, D::Error> {
    let v = Option::<serde_json::Value>::deserialize(d)?;
    Ok(v.map(value_to_string))
}

fn value_to_string(v: serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s,
        serde_json::Value::Array(arr) => arr
            .into_iter()
            .map(|item| match item {
                serde_json::Value::String(s) => s,
                other => other.to_string(),
            })
            .collect::<Vec<_>>()
            .join("; "),
        other => other.to_string(),
    }
}

#[derive(Deserialize)]
struct RawSlot {
    #[serde(deserialize_with = "coerce_to_string")]
    role_frame: String,
    #[serde(deserialize_with = "coerce_to_string")]
    cot_style: String,
    #[serde(default, deserialize_with = "coerce_to_opt_string")]
    focus_mandate: Option<String>,
    #[serde(default, deserialize_with = "coerce_to_opt_string")]
    rejection_criteria: Option<String>,
}

fn raw_to_cot(s: &str) -> CotStyle {
    match s {
        "step_by_step" => CotStyle::StepByStep,
        "devil_s_advocate" | "devils_advocate" => CotStyle::DevilsAdvocate,
        "first_principles" => CotStyle::FirstPrinciples,
        "backward_chaining" => CotStyle::BackwardChaining,
        _ => CotStyle::None,
    }
}

/// Strip `<think>...</think>` blocks produced by reasoning models before parsing.
/// These blocks often contain brackets that confuse the JSON array scanner.
fn strip_thinking_tags(s: &str) -> std::borrow::Cow<'_, str> {
    let open = s.find("<think>");
    let close = s.rfind("</think>");
    match (open, close) {
        (Some(o), Some(c)) if c > o => {
            let after = c + "</think>".len();
            std::borrow::Cow::Owned(s[after..].to_owned())
        }
        _ => std::borrow::Cow::Borrowed(s),
    }
}

/// Extract and parse the first `[...]` JSON array from a free-form LLM response.
pub fn parse_decomposition_response(
    response: &str,
) -> Result<Vec<ExplorerSlotConfig>, DecompositionError> {
    let response = strip_thinking_tags(response);
    let response = response.as_ref();
    let start = response.find('[').ok_or(DecompositionError::NoJsonArray)?;
    let end = response.rfind(']').ok_or(DecompositionError::NoJsonArray)?;
    if end <= start {
        return Err(DecompositionError::NoJsonArray);
    }
    let json_str = &response[start..=end];
    let raw: Vec<RawSlot> = serde_json::from_str(json_str)
        .map_err(|e| DecompositionError::ParseError(e.to_string()))?;

    let slots: Vec<ExplorerSlotConfig> = raw
        .into_iter()
        .filter(|s| !s.role_frame.trim().is_empty())
        .map(|s| ExplorerSlotConfig {
            role_frame: s.role_frame,
            cot_style: raw_to_cot(&s.cot_style),
            focus_mandate: s.focus_mandate.unwrap_or_default(),
            rejection_criteria: s.rejection_criteria.unwrap_or_default(),
        })
        .collect();

    if slots.is_empty() {
        return Err(DecompositionError::EmptyResult);
    }
    Ok(slots)
}

/// Prune `slots` to at most `n_max` entries by dropping the least semantically
/// independent slot (highest mean cosine similarity to all retained peers) until
/// `slots.len() <= n_max`.
pub fn prune_by_orthogonality(
    mut slots: Vec<ExplorerSlotConfig>,
    n_max: usize,
    model: &dyn EmbeddingModel,
) -> Vec<ExplorerSlotConfig> {
    while slots.len() > n_max {
        let n = slots.len();
        let embeddings: Vec<Vec<f32>> = slots.iter().map(|s| model.embed(&s.role_frame)).collect();

        let mean_sim: Vec<f64> = (0..n)
            .map(|i| {
                let sum: f64 = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| cosine_similarity(&embeddings[i], &embeddings[j]).max(0.0))
                    .sum();
                if n > 1 {
                    sum / (n - 1) as f64
                } else {
                    0.0
                }
            })
            .collect();

        let drop_idx = mean_sim
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(n - 1);

        slots.remove(drop_idx);
    }
    slots
}

/// Compute `n_eff_cosine` for a set of `ExplorerSlotConfig` role_frames.
///
/// Returns 1.0 when fewer than 2 slots or no embedding model is available.
pub fn compute_role_diversity(
    slots: &[ExplorerSlotConfig],
    model: Option<&dyn EmbeddingModel>,
) -> f64 {
    match model {
        Some(m) => {
            let texts: Vec<String> = slots.iter().map(|s| s.role_frame.clone()).collect();
            h2ai_autonomic::epistemic::compute_n_eff_cosine(&texts, m, 0.05)
        }
        None => 1.0,
    }
}

/// Generate `ExplorerSlotConfig` entries driven by the constraint corpus domain tags.
///
/// Groups constraints by their `domains` field. One slot per distinct domain.
/// Prunes to `n_max` by truncating alphabetically-sorted domain list.
/// Returns a single default slot when the corpus is empty or all constraints
/// have no domain tags.
pub fn corpus_fallback(
    corpus: &[ConstraintDoc],
    _pareto_weights: &ParetoWeights,
    n_max: usize,
) -> Vec<ExplorerSlotConfig> {
    let mut domain_map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for doc in corpus {
        for domain in &doc.domains {
            domain_map
                .entry(domain.clone())
                .or_default()
                .push(doc.id.clone());
        }
    }

    if domain_map.is_empty() {
        return vec![default_slot()];
    }

    let mut slots: Vec<ExplorerSlotConfig> = domain_map
        .into_iter()
        .map(|(domain, ids)| domain_to_slot(&domain, &ids))
        .collect();

    slots.truncate(n_max.max(1));
    slots
}

fn domain_to_slot(domain: &str, constraint_ids: &[String]) -> ExplorerSlotConfig {
    let ids_str = constraint_ids.join(", ");
    match domain {
        "security" | "auth" | "authentication" | "authz" => ExplorerSlotConfig {
            role_frame: "You are a security engineer. Your first concern is what an attacker \
                         can do with this interface and where trust boundaries are violated."
                .into(),
            cot_style: CotStyle::DevilsAdvocate,
            focus_mandate: format!(
                "You are responsible for: {ids_str}. Ensure the proposal satisfies all \
                 security and authentication requirements."
            ),
            rejection_criteria: "The single most likely way an attacker exploits \
                                  or bypasses this proposal."
                .into(),
        },
        "performance" | "latency" | "throughput" | "scalability" => ExplorerSlotConfig {
            role_frame: "You are a systems performance engineer. Your first concern is \
                         latency under load and resource exhaustion under concurrency."
                .into(),
            cot_style: CotStyle::FirstPrinciples,
            focus_mandate: format!(
                "You are responsible for: {ids_str}. Ensure the proposal meets all \
                 performance and throughput requirements."
            ),
            rejection_criteria: "The single most likely way this proposal degrades \
                                  under high concurrency or a load spike."
                .into(),
        },
        "correctness" | "accuracy" | "validation" | "integrity" => ExplorerSlotConfig {
            role_frame: "You are a formal verification engineer. Your first concern is \
                         edge cases, state machine violations, and invariant breaks."
                .into(),
            cot_style: CotStyle::BackwardChaining,
            focus_mandate: format!(
                "You are responsible for: {ids_str}. Ensure the proposal is logically \
                 correct and handles all edge cases."
            ),
            rejection_criteria: "The single invariant violation or edge case that \
                                  produces incorrect output under valid inputs."
                .into(),
        },
        "consistency" | "distributed" | "concurrency" | "synchronization" => ExplorerSlotConfig {
            role_frame: "You are a distributed systems architect. Your first concern is \
                         split-brain scenarios, partial failures, and ordering violations."
                .into(),
            cot_style: CotStyle::FirstPrinciples,
            focus_mandate: format!(
                "You are responsible for: {ids_str}. Ensure the proposal handles \
                 distributed consistency requirements."
            ),
            rejection_criteria: "The single consistency violation under concurrent \
                                  operations or partial network failure."
                .into(),
        },
        "compliance" | "regulatory" | "legal" | "audit" => ExplorerSlotConfig {
            role_frame: "You are a regulatory compliance analyst. Your first concern is \
                         what regulators flag and what must be documented for audit."
                .into(),
            cot_style: CotStyle::StepByStep,
            focus_mandate: format!(
                "You are responsible for: {ids_str}. Ensure the proposal satisfies all \
                 compliance and regulatory requirements."
            ),
            rejection_criteria: "The single compliance gap that would fail a \
                                  regulatory audit."
                .into(),
        },
        _ => ExplorerSlotConfig {
            role_frame: "You are a senior software architect. Your first concern is \
                         what breaks under load and what is impossible to change later."
                .into(),
            cot_style: CotStyle::StepByStep,
            focus_mandate: format!(
                "You are responsible for: {ids_str}. Ensure the proposal addresses \
                 all specified constraints."
            ),
            rejection_criteria: "The single architectural decision that creates \
                                  irreversible technical debt."
                .into(),
        },
    }
}

fn default_slot() -> ExplorerSlotConfig {
    ExplorerSlotConfig {
        role_frame: "You are a senior software architect. Your first concern is what \
                     breaks under load and what is impossible to change later."
            .into(),
        cot_style: CotStyle::StepByStep,
        focus_mandate: String::new(),
        rejection_criteria: "The single architectural decision that creates \
                              irreversible technical debt."
            .into(),
    }
}

/// Extract the rubric text from a predicate tree (LlmJudge rubrics only).
fn extract_rubric(pred: &ConstraintPredicate) -> String {
    match pred {
        ConstraintPredicate::LlmJudge { rubric } => rubric.clone(),
        ConstraintPredicate::Composite { children, .. } => children
            .iter()
            .map(extract_rubric)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Step 1 task: give the LLM actual rubric text and ask it to surface what engineers miss.
fn step1_analyze_task(description: &str, corpus: &[ConstraintDoc]) -> String {
    let constraint_block = if corpus.is_empty() {
        "No constraints loaded. Analyze based on general engineering best practices.".to_string()
    } else {
        corpus
            .iter()
            .map(|doc| {
                let domains = if doc.domains.is_empty() {
                    "untagged".to_string()
                } else {
                    doc.domains.join(", ")
                };
                let rubric = extract_rubric(&doc.predicate);
                let hint = doc
                    .remediation_hint
                    .as_deref()
                    .unwrap_or("")
                    .to_string();
                format!(
                    "CONSTRAINT {id} [{domains}]\n\
                     Rubric: {rubric}\n\
                     Remediation hint: {hint}",
                    id = doc.id,
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    };

    format!(
        "TASK: {description}\n\n\
         ACTIVE CONSTRAINTS:\n{constraint_block}\n\n\
         For each constraint domain above, answer three questions:\n\
         1. What is the single most counter-intuitive requirement \
            (the one a general-purpose engineer misses on first pass)?\n\
         2. What is the typical violation pattern — how does the design usually fail this?\n\
         3. What epistemic blindspot causes engineers to miss it \
            (wrong mental model, missing context, false assumption)?"
    )
}

/// Step 2 task: given the failure analysis, design concrete expert personas.
fn step2_design_roles_task(step1_analysis: &str, n_target: usize) -> String {
    format!(
        "FAILURE MODE ANALYSIS:\n{step1_analysis}\n\n\
         Design exactly {n_target} expert reviewer personas for a proposal review committee.\n\n\
         For each failure mode identified above, create one expert whose role is defined \
         by direct professional experience with that specific failure:\n\
         - role_frame: Start with \"You are a [role] who has [specific experience with this failure].\"\n\
           The role must change what the expert notices FIRST in any proposal — \
           not a generic title, but an identity anchored to the failure mode.\n\
         - reasoning_style: Choose backward_chaining (trace from the failure backward), \
           devil_s_advocate (prove the design is wrong), \
           first_principles (derive from invariants, ignore precedent), or \
           step_by_step (enumerate every state transition).\n\
         - what_they_hunt: In one sentence, the specific failure this expert looks for \
           before anything else.\n\n\
         The final role ({n_target}) is an integration reviewer who detects cascade failures \
         between the domains above — what breaks when both failure modes occur simultaneously.\n\n\
         Describe each role in plain text. Be concrete about the failure experience."
    )
}

/// Step 3 task: format the designed roles as a JSON array.
fn step3_assemble_json_task(step2_roles: &str, n_max: usize) -> String {
    format!(
        "EXPERT ROLES:\n{step2_roles}\n\n\
         Convert these roles into a JSON array. Maximum {n_max} elements.\n\
         Each element must have exactly these fields:\n\
         - \"role_frame\": string. 1-2 sentences starting with \"You are a [specific role].\"\n\
         - \"cot_style\": exactly one of: \"step_by_step\", \"devil_s_advocate\", \
           \"first_principles\", \"backward_chaining\", \"none\"\n\
         - \"focus_mandate\": string. The constraint domain(s) this expert covers.\n\
         - \"rejection_criteria\": string. The specific failure mode this expert hunts.\n\n\
         Output ONLY the JSON array. No markdown, no explanation."
    )
}

/// Run one adapter call, returning the output text or a DecompositionError.
async fn call_step(
    adapter: &dyn IComputeAdapter,
    system: &str,
    task: String,
    max_tokens: u64,
) -> Result<String, DecompositionError> {
    let request = ComputeRequest {
        system_context: system.to_string(),
        task,
        tau: TauValue::new(0.3).unwrap(),
        max_tokens,
    };
    adapter
        .execute(request)
        .await
        .map(|r| r.output)
        .map_err(|e: AdapterError| DecompositionError::ParseError(e.to_string()))
}

/// Run the 3-step decomposition pipeline: analyze → design roles → assemble JSON.
///
/// Step 1 grounds the analysis in actual constraint rubrics.
/// Step 2 designs personas anchored to the surfaced failure modes.
/// Step 3 formats the roles as JSON without creative load.
///
/// Returns `Err` if any step fails or the final JSON cannot be parsed.
/// No silent fallback — the caller publishes `TaskFailed` on error.
pub async fn run_decomposition_agent(
    description: &str,
    corpus: &[ConstraintDoc],
    pareto_weights: &ParetoWeights,
    n_target: usize,
    n_max: usize,
    adapter: &dyn IComputeAdapter,
    embedding_model: Option<&dyn EmbeddingModel>,
) -> Result<Vec<ExplorerSlotConfig>, DecompositionError> {
    let _ = pareto_weights; // Pareto weights inform n_target at call site; not needed in prompt pipeline.

    // Step 1: Identify what engineers miss in each constraint domain.
    let analysis =
        call_step(adapter, STEP1_SYSTEM, step1_analyze_task(description, corpus), 1024).await?;

    // Step 2: Design expert personas anchored to those failure modes.
    let roles = call_step(
        adapter,
        STEP2_SYSTEM,
        step2_design_roles_task(&analysis, n_target),
        1024,
    )
    .await?;

    // Step 3: Format roles as JSON. Narrow formatting task — no creative reasoning.
    let json_output = call_step(
        adapter,
        STEP3_SYSTEM,
        step3_assemble_json_task(&roles, n_max),
        2048,
    )
    .await?;

    let slots = parse_decomposition_response(&json_output)?;

    let limit = n_max.max(1);
    let pruned = if let Some(model) = embedding_model {
        prune_by_orthogonality(slots, limit, model)
    } else {
        let mut s = slots;
        s.truncate(limit);
        s
    };
    Ok(pruned)
}
