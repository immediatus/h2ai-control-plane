use h2ai_config::prompts::{
    DECOMPOSITION_CONSTRAINT_ENTRY, DECOMPOSITION_STEP1_SYSTEM, DECOMPOSITION_STEP1_TASK,
    DECOMPOSITION_STEP2_SYSTEM, DECOMPOSITION_STEP2_TASK, DECOMPOSITION_STEP3_SYSTEM,
    DECOMPOSITION_STEP3_TASK,
};
use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate};
use h2ai_context::embedding::{cosine_similarity, EmbeddingModel};
use h2ai_types::adapter::{AdapterError, ComputeRequest, IComputeAdapter};
use h2ai_types::config::AgentRole;
use h2ai_types::config::ParetoWeights;
use h2ai_types::manifest::{CotStyle, ExplorerSlotConfig};
use h2ai_types::sizing::TauValue;
use serde::Deserialize;
use std::collections::BTreeMap;
use thiserror::Error;

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
    #[serde(default)]
    constraint_domains: Vec<String>,
    #[serde(default)]
    search_enabled: bool,
    #[serde(default)]
    agent_role: AgentRole,
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
    let tail = &response[start..];
    let mut stream = serde_json::Deserializer::from_str(tail).into_iter::<serde_json::Value>();
    let json_str = match stream.next() {
        Some(Ok(_)) => &tail[..stream.byte_offset()],
        _ => return Err(DecompositionError::NoJsonArray),
    };
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
            constraint_domains: s.constraint_domains,
            search_enabled: s.search_enabled,
            agent_role: s.agent_role,
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
            .map_or(n - 1, |(i, _)| i);

        slots.remove(drop_idx);
    }
    slots
}

/// Compute `n_eff_cosine` for a set of `ExplorerSlotConfig` `role_frames`.
///
/// Returns 1.0 when fewer than 2 slots or no embedding model is available.
#[must_use]
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
#[must_use]
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
            constraint_domains: vec![domain.to_string()],
            ..Default::default()
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
            constraint_domains: vec![domain.to_string()],
            ..Default::default()
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
            constraint_domains: vec![domain.to_string()],
            ..Default::default()
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
            constraint_domains: vec![domain.to_string()],
            ..Default::default()
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
            constraint_domains: vec![domain.to_string()],
            ..Default::default()
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
            constraint_domains: vec![domain.to_string()],
            ..Default::default()
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
        constraint_domains: vec![],
        search_enabled: false,
        agent_role: AgentRole::default(),
    }
}

/// Extract the rubric text from a predicate tree (`LlmJudge` rubrics only).
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

/// Build the Step 1 task by rendering the template with the constraint corpus content.
fn step1_analyze_task(
    description: &str,
    corpus: &[ConstraintDoc],
    thinking_context: &str,
) -> String {
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
                let hint = doc.remediation_hint.as_deref().unwrap_or("");
                DECOMPOSITION_CONSTRAINT_ENTRY.render(&[
                    ("id", &doc.id),
                    ("domains", &domains),
                    ("rubric", &rubric),
                    ("hint", hint),
                ])
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    };
    let context_block = if thinking_context.is_empty() {
        String::new()
    } else {
        format!("PRIOR THINKING CONTEXT:\n{thinking_context}\n\n")
    };
    DECOMPOSITION_STEP1_TASK.render(&[
        ("thinking_context", &context_block),
        ("description", description),
        ("constraints", &constraint_block),
    ])
}

/// Build the Step 2 task with explicit domain-to-slot assignments.
///
/// Each domain gets exactly one role; an integration slot is appended.
/// Passing domain names explicitly prevents the LLM from over-expanding one domain.
fn step2_design_roles_task(step1_analysis: &str, constraint_domains: &[String]) -> String {
    let domain_assignments: String =
        constraint_domains
            .iter()
            .enumerate()
            .fold(String::new(), |mut acc, (i, d)| {
                use std::fmt::Write as _;
                writeln!(acc, "  Role {}: covers the \"{}\" domain.", i + 1, d).unwrap();
                acc
            });
    let integration_idx = (constraint_domains.len() + 1).to_string();
    let n_total = integration_idx.clone();
    DECOMPOSITION_STEP2_TASK.render(&[
        ("analysis", step1_analysis),
        ("n_total", &n_total),
        ("domain_assignments", &domain_assignments),
        ("integration_idx", &integration_idx),
    ])
}

/// Build the Step 3 task for JSON assembly from the designed roles.
///
/// `corpus_domains` is the authoritative vocabulary for `constraint_domains` fields.
/// Passing it here ensures the LLM emits only verbatim corpus strings, preventing
/// vocabulary mismatch between generated tags and the coverage validator.
fn step3_assemble_json_task(step2_roles: &str, n_max: usize, corpus_domains: &[String]) -> String {
    let domains_str = if corpus_domains.is_empty() {
        "[] — no corpus domains defined, always use empty array".to_string()
    } else {
        let quoted: Vec<String> = corpus_domains.iter().map(|d| format!("\"{d}\"")).collect();
        format!("[{}]", quoted.join(", "))
    };
    DECOMPOSITION_STEP3_TASK.render(&[
        ("roles", step2_roles),
        ("n_max", &n_max.to_string()),
        ("corpus_domains", &domains_str),
    ])
}

/// Run one adapter call, returning the output text or a `DecompositionError`.
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
#[allow(clippy::too_many_arguments)]
pub async fn run_decomposition_agent(
    description: &str,
    corpus: &[ConstraintDoc],
    pareto_weights: &ParetoWeights,
    n_target: usize,
    n_max: usize,
    adapter: &dyn IComputeAdapter,
    embedding_model: Option<&dyn EmbeddingModel>,
    step_max_tokens: u64,
    json_max_tokens: u64,
    thinking_context: &str,
) -> Result<Vec<ExplorerSlotConfig>, DecompositionError> {
    let _ = (pareto_weights, n_target); // n_target is recomputed from domains here; pareto weights unused.

    // Collect the unique constraint domains from the corpus — Step 2 anchors one role per domain.
    let constraint_domains: Vec<String> = {
        let mut seen = std::collections::HashSet::new();
        let mut ordered = Vec::new();
        for doc in corpus {
            for d in &doc.domains {
                if seen.insert(d.clone()) {
                    ordered.push(d.clone());
                }
            }
        }
        ordered.sort();
        ordered
    };

    // Step 1: Identify what engineers miss in each constraint domain.
    let analysis = call_step(
        adapter,
        DECOMPOSITION_STEP1_SYSTEM.as_str(),
        step1_analyze_task(description, corpus, thinking_context),
        step_max_tokens,
    )
    .await?;

    // Step 2: Design one expert persona per domain (explicit domain list prevents over-expansion).
    let roles = call_step(
        adapter,
        DECOMPOSITION_STEP2_SYSTEM.as_str(),
        step2_design_roles_task(&analysis, &constraint_domains),
        step_max_tokens,
    )
    .await?;

    // Step 3: Format roles as JSON. Narrow formatting task — no creative reasoning.
    // Pass corpus_domains so the LLM emits verbatim vocabulary strings, preventing
    // vocabulary mismatch between generated constraint_domains and the coverage validator.
    let json_output = call_step(
        adapter,
        DECOMPOSITION_STEP3_SYSTEM.as_str(),
        step3_assemble_json_task(&roles, n_max, &constraint_domains),
        json_max_tokens,
    )
    .await?;

    let mut slots = parse_decomposition_response(&json_output)?;

    // Coverage check: ensure every constraint domain has at least one slot covering it.
    // If the LLM dropped a domain, add a corpus_fallback slot for the missing ones.
    for domain in &constraint_domains {
        let covered = slots.iter().any(|s| {
            s.focus_mandate
                .to_lowercase()
                .contains(&domain.to_lowercase())
                || s.role_frame.to_lowercase().contains(&domain.to_lowercase())
        });
        if !covered {
            // Find constraints for this domain and inject a fallback slot.
            let domain_ids: Vec<String> = corpus
                .iter()
                .filter(|doc| doc.domains.iter().any(|d| d == domain))
                .map(|doc| doc.id.clone())
                .collect();
            if !domain_ids.is_empty() {
                slots.push(domain_to_slot(domain, &domain_ids));
            }
        }
    }

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
