use std::collections::{HashMap, HashSet};

use h2ai_constraints::types::ConstraintDoc;
use h2ai_knowledge::types::{KnowledgeNode, NodeDepth, NodeSource, TensionRef};
use h2ai_types::events::{
    SocraticDiagnosisEvent, TopologyProvisionedEvent, VerificationScoredEvent,
};
use h2ai_types::identity::TaskId;

use crate::engine::EngineOutput;

// ── Private helpers ───────────────────────────────────────────────────────────

/// Trim `s` to the last whitespace at or before `limit` chars.
/// Falls back to a hard cut at `limit` when no whitespace is found.
/// Safely converts char-count limit to a byte boundary to avoid panicking on UTF-8.
pub fn trim_at_word_boundary(s: &str, limit: usize) -> String {
    // Convert char-count limit to a safe byte boundary.
    let byte_limit = s
        .char_indices()
        .nth(limit)
        .map(|(b, _)| b)
        .unwrap_or(s.len());
    if byte_limit >= s.len() {
        return s.to_string();
    }
    match s[..byte_limit].rfind(char::is_whitespace) {
        Some(pos) => s[..pos].to_string(),
        None => s[..byte_limit].to_string(),
    }
}

/// Word-bag Jaccard similarity: |A ∩ B| / |A ∪ B|. Returns 1.0 when both bags are empty.
fn jaccard_similarity(a: &str, b: &str) -> f64 {
    let bag_a: HashSet<&str> = a.split_whitespace().collect();
    let bag_b: HashSet<&str> = b.split_whitespace().collect();
    let union = bag_a.union(&bag_b).count();
    if union == 0 {
        return 1.0;
    }
    bag_a.intersection(&bag_b).count() as f64 / union as f64
}

/// FNV-32a hash — stable identifier for reason-keyed leaf node IDs.
pub fn fnv32a(s: &str) -> u32 {
    const OFFSET: u32 = 2_166_136_261;
    const PRIME: u32 = 16_777_619;
    let mut hash = OFFSET;
    for byte in s.bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// Extract the first `[A-Z]+-\d+` token from `text` (e.g. "C-007" from "violated C-007 constraint").
pub fn parse_constraint_id(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        if bytes[i].is_ascii_uppercase() {
            let start = i;
            while i < len && bytes[i].is_ascii_uppercase() {
                i += 1;
            }
            if i < len && bytes[i] == b'-' {
                i += 1;
                let digit_start = i;
                while i < len && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                if i > digit_start {
                    return Some(text[start..i].to_string());
                }
            }
        } else {
            i += 1;
        }
    }
    None
}

/// Return deduplicated `(score, reason)` pairs: drop a candidate when its Jaccard similarity
/// with any already-selected reason exceeds `threshold`.
fn jaccard_dedup(mut pairs: Vec<(f32, String)>, threshold: f64) -> Vec<(f32, String)> {
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)); // ascending score
    let mut result: Vec<(f32, String)> = Vec::new();
    for (score, reason) in pairs {
        let is_dup = result
            .iter()
            .any(|(_, r)| jaccard_similarity(r, &reason) >= threshold);
        if !is_dup {
            result.push((score, reason));
        }
    }
    result
}

// ── Core extractor ────────────────────────────────────────────────────────────

/// All skill node logic lives here. Both public entry points delegate to this function.
#[allow(clippy::too_many_arguments)]
fn extract_skill_nodes(
    n_valid: usize,
    topology_retry_events: &[TopologyProvisionedEvent],
    uncovered_domains: &[String],
    verification_events: &[VerificationScoredEvent],
    resolved_output: &str,
    socratic_diagnosis_events: &[SocraticDiagnosisEvent],
    corpus: &[ConstraintDoc],
    task_id: &TaskId,
) -> Vec<KnowledgeNode> {
    let n_retries = topology_retry_events.len();
    if n_valid == 0 && n_retries == 0 && verification_events.is_empty() {
        return vec![];
    }

    // Build corpus indexes.
    let mut domain_constraints: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut constraint_domain: HashMap<&str, &str> = HashMap::new();
    for doc in corpus {
        for domain in &doc.domains {
            domain_constraints
                .entry(domain.as_str())
                .or_default()
                .push(doc.id.as_str());
        }
        if let Some(primary) = doc.domains.first() {
            constraint_domain.insert(doc.id.as_str(), primary.as_str());
        }
    }

    let repair = match (n_valid, n_retries) {
        (_, 0) => format!("resolved with {n_valid} valid proposals on first topology"),
        (0, r) => format!("failed after {r} topology retries — no proposals survived"),
        (_, r) => format!("resolved with {n_valid} valid proposals after {r} topology retries"),
    };
    let importance = (0.5_f32 + 0.5 * (n_retries as f32 / 5.0).min(1.0)).min(1.0);

    // Collect per-domain failure signals (for Topic node failure_modes).
    let mut domain_failures: HashMap<&str, Vec<String>> = HashMap::new();

    for ev in topology_retry_events {
        if ev.retry_count > 0 {
            let msg = ev
                .constraint_tombstone
                .clone()
                .unwrap_or_else(|| format!("topology retry #{}", ev.retry_count));
            for domain in domain_constraints.keys() {
                domain_failures.entry(domain).or_default().push(msg.clone());
            }
        }
    }

    for domain in uncovered_domains {
        if domain_constraints.contains_key(domain.as_str()) {
            domain_failures
                .entry(domain.as_str())
                .or_default()
                .push(format!(
                    "domain '{}' remained uncovered after {} topology waves",
                    domain, n_retries
                ));
        }
    }

    let has_verifier_failures = verification_events.iter().any(|ev| ev.score < 0.5);
    if domain_failures.is_empty() && !has_verifier_failures {
        return vec![];
    }

    // Socratic questions — exact-dedup, preserve insertion order.
    let mut socratic_qs: Vec<String> = Vec::new();
    for ev in socratic_diagnosis_events {
        if !socratic_qs.contains(&ev.question) {
            socratic_qs.push(ev.question.clone());
        }
    }
    let socratic_summary = if socratic_qs.is_empty() {
        "no diagnostic questions".to_string()
    } else {
        socratic_qs.join(" | ")
    };

    // Resolved output excerpt for entry_points.
    let excerpt = trim_at_word_boundary(resolved_output, 300);
    let entry_point = format!("Resolution pattern: {excerpt}");

    // Emit one Topic node per domain with failure signals.
    let mut nodes: Vec<KnowledgeNode> = domain_failures
        .into_iter()
        .map(|(domain, failure_modes)| {
            let related = domain_constraints
                .get(domain)
                .map(|ids| ids.iter().map(|s| s.to_string()).collect())
                .unwrap_or_default();
            let tensions: Vec<TensionRef> = socratic_qs
                .iter()
                .map(|q| TensionRef {
                    domain: domain.to_string(),
                    reason: q.clone(),
                })
                .collect();
            KnowledgeNode {
                id: format!("skill:{task_id}:{domain}:topic"),
                depth: NodeDepth::Topic,
                source: NodeSource::Synthetic,
                domains: vec![domain.to_string()],
                related,
                synthesis: format!(
                    "Domain '{domain}' [{n_retries} retries]: {socratic_summary}. Resolution: {repair}."
                ),
                failure_modes,
                invariants: vec![repair.clone()],
                importance,
                entry_points: vec![entry_point.clone()],
                tensions,
                cross_references: vec![],
            }
        })
        .collect();

    // ── Constraint-keyed Leaf nodes ───────────────────────────────────────────

    let mut tombstone_map: HashMap<String, u32> = HashMap::new();
    for ev in topology_retry_events {
        if ev.retry_count > 0 {
            if let Some(ref t) = ev.constraint_tombstone {
                *tombstone_map.entry(t.clone()).or_insert(0) += 1;
            }
        }
    }

    let raw_reasons: Vec<(f32, String)> = verification_events
        .iter()
        .filter(|ev| ev.score < 0.5)
        .map(|ev| (ev.score as f32, ev.reason.clone()))
        .collect();
    let verifier_reasons = jaccard_dedup(raw_reasons, 0.85);

    let mut covered_constraint_ids: HashSet<String> = HashSet::new();

    for (tombstone, count) in &tombstone_map {
        if let Some(constraint_id) = parse_constraint_id(tombstone) {
            if let Some(&domain) = constraint_domain.get(constraint_id.as_str()) {
                let best_reason = verifier_reasons
                    .first()
                    .map(|(_, r)| r.clone())
                    .unwrap_or_default();
                let leaf_importance = if *count >= 2 { 1.0_f32 } else { 0.6_f32 };
                let failure_modes = if best_reason.is_empty() {
                    vec![tombstone.clone()]
                } else {
                    vec![tombstone.clone(), best_reason.clone()]
                };
                nodes.push(KnowledgeNode {
                    id: format!("skill:{task_id}:{constraint_id}"),
                    depth: NodeDepth::Leaf,
                    source: NodeSource::Synthetic,
                    domains: vec![domain.to_string()],
                    related: vec![constraint_id.clone()],
                    synthesis: format!(
                        "Constraint {constraint_id} [{domain}]: {tombstone}. Verifier: {best_reason}."
                    ),
                    failure_modes,
                    invariants: vec![format!("passed after {n_retries} retries")],
                    importance: leaf_importance,
                    entry_points: vec![],
                    tensions: vec![],
                    cross_references: vec![],
                });
                covered_constraint_ids.insert(constraint_id);
            }
        }
    }

    // ── Reason-keyed Leaf nodes (fallback) ────────────────────────────────────

    let all_corpus_domains: Vec<String> =
        domain_constraints.keys().map(|s| s.to_string()).collect();

    for (_, reason) in &verifier_reasons {
        let already_covered = parse_constraint_id(reason)
            .map(|id| covered_constraint_ids.contains(&id))
            .unwrap_or(false);
        if already_covered {
            continue;
        }
        let hash = fnv32a(reason);
        nodes.push(KnowledgeNode {
            id: format!("skill:{task_id}:reason:{hash}"),
            depth: NodeDepth::Leaf,
            source: NodeSource::Synthetic,
            domains: all_corpus_domains.clone(),
            related: vec![],
            synthesis: format!("Verifier rejection: {reason}"),
            failure_modes: vec![reason.clone()],
            invariants: vec![],
            importance: 0.6_f32,
            entry_points: vec![],
            tensions: vec![],
            cross_references: vec![],
        });
    }

    nodes
}

// ── Public API ────────────────────────────────────────────────────────────────

pub fn skill_from_output(
    output: &EngineOutput,
    corpus: &[ConstraintDoc],
    task_id: &TaskId,
) -> Vec<KnowledgeNode> {
    extract_skill_nodes(
        output.selection_resolved.valid_proposals.len(),
        &output.topology_retry_events,
        &output.coherence_state.uncovered_domains,
        &output.verification_events,
        &output.resolved_output,
        &output.socratic_diagnosis_events,
        corpus,
        task_id,
    )
}

/// Entry point for the `TaskFailed` path where no `EngineOutput` is available.
/// Takes topology retry events and partial verification events that exist on the failure path.
pub fn skill_from_retry_events(
    topology_retry_events: Vec<TopologyProvisionedEvent>,
    partial_verification_events: &[VerificationScoredEvent],
    corpus: &[ConstraintDoc],
    task_id: &TaskId,
) -> Vec<KnowledgeNode> {
    extract_skill_nodes(
        0,
        &topology_retry_events,
        &[],
        partial_verification_events,
        "",
        &[],
        corpus,
        task_id,
    )
}
