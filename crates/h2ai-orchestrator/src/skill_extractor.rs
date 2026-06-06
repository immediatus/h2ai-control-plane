use std::collections::{HashMap, HashSet};

use h2ai_constraints::types::ConstraintDoc;
use h2ai_knowledge::types::{KnowledgeNode, NodeDepth, NodeSource, TensionRef};
use h2ai_types::identity::TaskId;

use crate::engine::EngineOutput;

// ── Private helpers ───────────────────────────────────────────────────────────

/// Trim `s` to the last whitespace at or before `limit` chars.
/// Falls back to a hard cut at `limit` when no whitespace is found.
/// Safely converts char-count limit to a byte boundary to avoid panicking on UTF-8.
fn trim_at_word_boundary(s: &str, limit: usize) -> String {
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
fn fnv32a(s: &str) -> u32 {
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
fn parse_constraint_id(text: &str) -> Option<String> {
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

// ── Public API ────────────────────────────────────────────────────────────────

pub fn skill_from_output(
    output: &EngineOutput,
    corpus: &[ConstraintDoc],
    task_id: &TaskId,
) -> Vec<KnowledgeNode> {
    // Guard: must have at least one resolved proposal.
    let n_valid = output.selection_resolved.valid_proposals.len();
    if n_valid == 0 {
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
    // constraint_domain is used in Leaf node emission below.

    let n_retries = output.topology_retry_events.len();
    let repair = match n_retries {
        0 => format!("resolved with {n_valid} valid proposals on first topology"),
        r => format!("resolved with {n_valid} valid proposals after {r} topology retries"),
    };
    let importance = (0.5_f32 + 0.5 * (n_retries as f32 / 5.0).min(1.0)).min(1.0);

    // Collect per-domain failure signals (for Topic node failure_modes).
    let mut domain_failures: HashMap<&str, Vec<String>> = HashMap::new();

    for ev in &output.topology_retry_events {
        if ev.retry_count > 0 {
            let msg = ev.constraint_tombstone.clone().unwrap_or_else(|| {
                format!("topology retry #{}", ev.retry_count)
            });
            for domain in domain_constraints.keys() {
                domain_failures.entry(domain).or_default().push(msg.clone());
            }
        }
    }

    for domain in &output.coherence_state.uncovered_domains {
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

    for ev in &output.srani_events {
        if ev.hint_injected && !ev.shared_ungrounded_entities.is_empty() {
            let msg = format!(
                "ungrounded entities: {}",
                ev.shared_ungrounded_entities.join(", ")
            );
            for domain in domain_constraints.keys() {
                domain_failures.entry(domain).or_default().push(msg.clone());
            }
        }
    }

    if domain_failures.is_empty() {
        return vec![];
    }

    // Socratic questions — exact-dedup, preserve insertion order.
    let mut socratic_qs: Vec<String> = Vec::new();
    for ev in &output.socratic_diagnosis_events {
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
    let excerpt = trim_at_word_boundary(&output.resolved_output, 300);
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

    // Count tombstone appearances across all retry events.
    let mut tombstone_map: HashMap<String, u32> = HashMap::new();
    for ev in &output.topology_retry_events {
        if ev.retry_count > 0 {
            if let Some(ref t) = ev.constraint_tombstone {
                *tombstone_map.entry(t.clone()).or_insert(0) += 1;
            }
        }
    }

    // Collect low-scoring verifier reasons, sorted by score ascending, Jaccard-deduped.
    let raw_reasons: Vec<(f32, String)> = output
        .verification_events
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
        // Skip if this reason mentions a constraint_id already covered by a Constraint-keyed Leaf.
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coherence::CoherenceState;
    use h2ai_types::events::{SelectionResolvedEvent, TaskComplexityAssessedEvent};
    use h2ai_types::identity::ExplorerId;
    use h2ai_types::sizing::{MergeStrategy, ProbeSkipReason, TaskQuadrant};

    fn stub_corpus(domains: &[&str]) -> Vec<ConstraintDoc> {
        use h2ai_constraints::types::{ConstraintPredicate, ConstraintSeverity};
        domains
            .iter()
            .enumerate()
            .map(|(i, d)| ConstraintDoc {
                id: format!("C-{i:03}"),
                source_file: format!("{d}.yaml"),
                description: format!("constraint in {d}"),
                severity: ConstraintSeverity::Advisory,
                predicate: ConstraintPredicate::LlmJudge {
                    rubric: "stub rubric".into(),
                },
                remediation_hint: None,
                domains: vec![d.to_string()],
                mandatory_for_tags: vec![],
                related_to: vec![],
                binary_checks: vec![],
                version: 1,
                repair_provenance: None,
                pass_criteria: None,
            })
            .collect()
    }

    fn stub_selection(valid: usize) -> SelectionResolvedEvent {
        SelectionResolvedEvent {
            task_id: TaskId::new(),
            valid_proposals: (0..valid).map(|_| ExplorerId::new()).collect(),
            pruned_proposals: vec![],
            merge_strategy: MergeStrategy::ScoreOrdered,
            timestamp: chrono::Utc::now(),
            merge_elapsed_secs: None,
            n_input_proposals: valid,
            n_failed_proposals: 0,
        }
    }

    fn stub_complexity(task_id: TaskId) -> TaskComplexityAssessedEvent {
        TaskComplexityAssessedEvent {
            task_id,
            tcc_structural: 0.5,
            tcc_empirical: None,
            tcc_effective: 0.5,
            n_eff_pool: None,
            task_quadrant: TaskQuadrant::Precision,
            probe_skipped: true,
            probe_skip_reason: ProbeSkipReason::None,
            heavy_fraction: 0.0,
            tcc_mismatch: false,
            probe_cost_tokens: 0,
            n_informative_static: 0,
            timestamp: chrono::Utc::now(),
        }
    }

    fn stub_attribution() -> crate::attribution::HarnessAttribution {
        crate::attribution::HarnessAttribution {
            baseline_quality: 0.7,
            topology_gain: 0.1,
            verification_gain: 0.0,
            tao_gain: 0.0,
            q_confidence: 0.8,
            prediction_basis: h2ai_types::sizing::PredictionBasis::Heuristic,
            q_measured: None,
            rho_adjusted: 0.7,
            case_b_flag: false,
            synthesis_gain: 0.0,
        }
    }

    fn make_output(
        task_id: TaskId,
        valid_proposals: usize,
        topology_retry_events: Vec<h2ai_types::events::TopologyProvisionedEvent>,
        coherence_state: CoherenceState,
        srani_events: Vec<h2ai_types::events::CorrelatedFabricationEvent>,
        verification_events: Vec<h2ai_types::events::VerificationScoredEvent>,
    ) -> EngineOutput {
        EngineOutput {
            task_id: task_id.clone(),
            resolved_output: "stub resolution".into(),
            selection_resolved: stub_selection(valid_proposals),
            attribution: stub_attribution(),
            attribution_interval: None,
            verification_events,
            failed_proposals: vec![],
            talagrand: None,
            suggested_next_params: None,
            waste_ratio: 0.0,
            applied_optimizations: vec![],
            topology_retry_events,
            mode_collapse_count: 0,
            epistemic_yield: None,
            task_quadrant: None,
            complexity_event: stub_complexity(task_id),
            frontier_event: None,
            adapter_correctness: vec![],
            coherence_state,
            comparison_events: vec![],
            shadow_audit_events: vec![],
            correlated_warnings: vec![],
            researcher_grounding_events: vec![],
            diversity_degraded_event: None,
            srani_events,
            srani_ema_cfi_updated: 0.0,
            srani_count_updated: 0,
            oracle_gate_passed: None,
            leader_elected_events: vec![],
            socratic_diagnosis_events: vec![],
            consensus_agreement_rate: None,
            tokens_used: 0,
        }
    }

    fn closed_coherence() -> CoherenceState {
        CoherenceState {
            uncovered_domains: vec![],
            active_contradictions: vec![],
        }
    }

    fn retry_event(
        task_id: TaskId,
        retry_count: u32,
        tombstone: Option<String>,
    ) -> h2ai_types::events::TopologyProvisionedEvent {
        use h2ai_types::config::{AuditorConfig, TopologyKind};
        use h2ai_types::sizing::{CoherencyCoefficients, CoordinationThreshold, MergeStrategy};
        let cc = CoherencyCoefficients {
            alpha: 0.1,
            beta_base: 0.01,
            beta_quality: None,
            cg_samples: vec![0.5],
            sample_timestamps: vec![],
        };
        h2ai_types::events::TopologyProvisionedEvent {
            task_id,
            topology_kind: TopologyKind::Ensemble,
            explorer_configs: vec![],
            auditor_config: AuditorConfig::default(),
            n_max: 2.0,
            interface_n_max: None,
            beta_eff: 0.03,
            role_error_costs: vec![],
            merge_strategy: MergeStrategy::ScoreOrdered,
            coordination_threshold: CoordinationThreshold::from_calibration(&cc, 1.0),
            review_gates: vec![],
            retry_count,
            timestamp: chrono::Utc::now(),
            constraint_tombstone: tombstone,
        }
    }

    // ── Regression tests (preserved from old implementation) ─────────────────

    #[test]
    fn clean_run_produces_no_skills() {
        let task_id = TaskId::new();
        let output = make_output(task_id.clone(), 2, vec![], closed_coherence(), vec![], vec![]);
        let corpus = stub_corpus(&["auth"]);
        let nodes = skill_from_output(&output, &corpus, &task_id);
        assert!(nodes.is_empty(), "clean run must produce no skills");
    }

    #[test]
    fn zero_valid_proposals_returns_empty() {
        let task_id = TaskId::new();
        let output = make_output(
            task_id.clone(),
            0,
            vec![retry_event(task_id.clone(), 2, Some("violated C-001".into()))],
            closed_coherence(),
            vec![],
            vec![],
        );
        let corpus = stub_corpus(&["auth"]);
        let nodes = skill_from_output(&output, &corpus, &task_id);
        assert!(nodes.is_empty(), "unresolved task must produce no skills");
    }

    #[test]
    fn topology_retry_produces_topic_per_domain() {
        let task_id = TaskId::new();
        let output = make_output(
            task_id.clone(),
            1,
            vec![retry_event(task_id.clone(), 1, Some("violated auth constraint".into()))],
            closed_coherence(),
            vec![],
            vec![],
        );
        let corpus = stub_corpus(&["auth", "billing"]);
        let nodes = skill_from_output(&output, &corpus, &task_id);
        // 2 Topic nodes (one per corpus domain) — no Leaf nodes from this tombstone
        let topic_nodes: Vec<_> = nodes.iter().filter(|n| n.depth == NodeDepth::Topic).collect();
        assert_eq!(topic_nodes.len(), 2, "one Topic node per corpus domain");
        for node in &topic_nodes {
            assert!(
                node.failure_modes.iter().any(|f| f.contains("violated auth constraint")),
                "tombstone text must appear in failure_modes"
            );
            assert!(!node.invariants.is_empty(), "invariants must contain repair summary");
            assert!(node.importance > 0.5, "retried task must have importance > 0.5");
        }
    }

    #[test]
    fn uncovered_domain_produces_targeted_topic_node() {
        let task_id = TaskId::new();
        let output = make_output(
            task_id.clone(),
            1,
            vec![],
            CoherenceState {
                uncovered_domains: vec!["security".into()],
                active_contradictions: vec![],
            },
            vec![],
            vec![],
        );
        let corpus = stub_corpus(&["auth", "security"]);
        let nodes = skill_from_output(&output, &corpus, &task_id);
        assert_eq!(nodes.len(), 1, "only the uncovered domain produces a node");
        assert_eq!(nodes[0].domains, vec!["security"]);
        assert_eq!(nodes[0].depth, NodeDepth::Topic);
    }

    #[test]
    fn srani_produces_fabrication_failure_mode() {
        use h2ai_types::events::CorrelatedFabricationEvent;
        let task_id = TaskId::new();
        let srani_ev = CorrelatedFabricationEvent {
            task_id: task_id.clone(),
            cfi: 0.6,
            injection_pressure: 0.55,
            shared_ungrounded_entities: vec!["AuthService".into(), "TokenVault".into()],
            proposal_count: 2,
            hint_injected: true,
            timestamp: chrono::Utc::now(),
        };
        let output =
            make_output(task_id.clone(), 1, vec![], closed_coherence(), vec![srani_ev], vec![]);
        let corpus = stub_corpus(&["auth"]);
        let nodes = skill_from_output(&output, &corpus, &task_id);
        assert_eq!(nodes.len(), 1);
        let failure_text = nodes[0].failure_modes.join(" ");
        assert!(
            failure_text.contains("AuthService") && failure_text.contains("TokenVault"),
            "ungrounded entities must appear in failure_modes"
        );
    }

    #[test]
    fn importance_scales_with_retry_count() {
        let task_id = TaskId::new();
        let corpus = stub_corpus(&["auth"]);
        let output_low = make_output(
            task_id.clone(),
            1,
            vec![],
            CoherenceState {
                uncovered_domains: vec!["auth".into()],
                active_contradictions: vec![],
            },
            vec![],
            vec![],
        );
        let output_high = make_output(
            task_id.clone(),
            1,
            vec![retry_event(task_id.clone(), 5, Some("heavy failure".into()))],
            closed_coherence(),
            vec![],
            vec![],
        );
        let low_nodes = skill_from_output(&output_low, &corpus, &task_id);
        let high_nodes = skill_from_output(&output_high, &corpus, &task_id);
        assert!(!low_nodes.is_empty() && !high_nodes.is_empty());
        assert!(
            high_nodes[0].importance > low_nodes[0].importance,
            "more retries → higher importance"
        );
    }

    #[test]
    fn topic_node_id_is_deterministic() {
        let task_id = TaskId::new();
        let output = make_output(
            task_id.clone(),
            1,
            vec![],
            CoherenceState {
                uncovered_domains: vec!["auth".into()],
                active_contradictions: vec![],
            },
            vec![],
            vec![],
        );
        let corpus = stub_corpus(&["auth"]);
        let nodes1 = skill_from_output(&output, &corpus, &task_id);
        let nodes2 = skill_from_output(&output, &corpus, &task_id);
        assert_eq!(nodes1[0].id, nodes2[0].id, "same inputs → same node id");
        assert_eq!(
            nodes1[0].id,
            format!("skill:{}:auth:topic", task_id),
            "Topic id must be skill:{{task_id}}:{{domain}}:topic"
        );
    }

    // ── New tests ─────────────────────────────────────────────────────────────

    #[test]
    fn parse_constraint_id_extracts_c007_from_tombstone() {
        assert_eq!(
            parse_constraint_id("violated C-007 auth constraint"),
            Some("C-007".to_string())
        );
        assert_eq!(parse_constraint_id("no constraint here"), None);
        assert_eq!(parse_constraint_id("AUTH-123 failed"), Some("AUTH-123".to_string()));
    }

    #[test]
    fn fnv32a_is_deterministic_and_nonzero() {
        let h1 = fnv32a("auth token missing");
        let h2 = fnv32a("auth token missing");
        assert_eq!(h1, h2);
        assert_ne!(h1, 0);
    }

    fn verification_event(
        task_id: TaskId,
        score: f64,
        reason: &str,
    ) -> h2ai_types::events::VerificationScoredEvent {
        h2ai_types::events::VerificationScoredEvent {
            task_id,
            explorer_id: ExplorerId::new(),
            score,
            reason: reason.to_string(),
            passed: false,
            cache_hit: false,
            timestamp: chrono::Utc::now(),
        }
    }

    #[test]
    fn constraint_leaf_emitted_when_tombstone_has_parseable_id() {
        let task_id = TaskId::new();
        // Tombstone "violated C-000 auth quota" → regex matches "C-000"
        // C-000 maps to "auth" domain (stub_corpus index 0)
        let output = make_output(
            task_id.clone(), 1,
            vec![retry_event(task_id.clone(), 1, Some("violated C-000 auth quota".into()))],
            closed_coherence(), vec![], vec![],
        );
        let corpus = stub_corpus(&["auth"]);
        let nodes = skill_from_output(&output, &corpus, &task_id);
        let leaf = nodes.iter().find(|n| n.depth == NodeDepth::Leaf);
        assert!(leaf.is_some(), "must emit a Constraint-keyed Leaf for parseable constraint ID");
        let leaf = leaf.unwrap();
        assert_eq!(leaf.id, format!("skill:{task_id}:C-000"),
            "Constraint-keyed Leaf id must be skill:{{task_id}}:{{constraint_id}}");
        assert!(leaf.synthesis.contains("C-000"), "synthesis must contain the constraint ID");
        assert!(leaf.synthesis.contains("violated C-000 auth quota"),
            "synthesis must contain the tombstone text");
    }

    #[test]
    fn constraint_leaf_importance_1_when_tombstone_appears_twice() {
        let task_id = TaskId::new();
        // Same tombstone in two retry events → count = 2 → importance = 1.0
        let output = make_output(
            task_id.clone(), 1,
            vec![
                retry_event(task_id.clone(), 1, Some("violated C-000 quota".into())),
                retry_event(task_id.clone(), 2, Some("violated C-000 quota".into())),
            ],
            closed_coherence(), vec![], vec![],
        );
        let corpus = stub_corpus(&["auth"]);
        let nodes = skill_from_output(&output, &corpus, &task_id);
        let leaf = nodes.iter().find(|n| n.depth == NodeDepth::Leaf && n.id.contains("C-000"));
        assert!(leaf.is_some());
        assert!(
            (leaf.unwrap().importance - 1.0).abs() < 1e-5,
            "tombstone appearing ≥2 times → importance must be 1.0"
        );
    }

    #[test]
    fn reason_leaf_emitted_when_no_constraint_id_in_tombstone() {
        let task_id = TaskId::new();
        // Tombstone without [A-Z]+-\d+ pattern → no Constraint-keyed Leaf
        // Verification event with score < 0.5 → Reason-keyed Leaf
        let output = make_output(
            task_id.clone(), 1,
            vec![retry_event(task_id.clone(), 1, Some("auth quota exceeded".into()))],
            closed_coherence(), vec![],
            vec![verification_event(task_id.clone(), 0.3, "auth token was missing from header")],
        );
        let corpus = stub_corpus(&["auth"]);
        let nodes = skill_from_output(&output, &corpus, &task_id);
        let reason_leaf = nodes.iter().find(|n| n.depth == NodeDepth::Leaf && n.id.contains(":reason:"));
        assert!(reason_leaf.is_some(), "must emit Reason-keyed Leaf when tombstone has no constraint ID");
        assert!(
            reason_leaf.unwrap().synthesis.contains("auth token was missing from header"),
            "Reason-keyed Leaf synthesis must contain the verifier reason"
        );
    }

    #[test]
    fn jaccard_dedup_prevents_near_duplicate_reason_leaves() {
        let task_id = TaskId::new();
        // Two near-identical verifier reasons (high Jaccard) → only one Reason-keyed Leaf
        let output = make_output(
            task_id.clone(), 1,
            vec![retry_event(task_id.clone(), 1, Some("generic failure".into()))],
            closed_coherence(), vec![],
            vec![
                verification_event(task_id.clone(), 0.2, "auth token header missing from request"),
                verification_event(task_id.clone(), 0.3, "auth token header missing from request field"), // Jaccard = 6/7 ≈ 0.857
            ],
        );
        let corpus = stub_corpus(&["auth"]);
        let nodes = skill_from_output(&output, &corpus, &task_id);
        let reason_leaves: Vec<_> = nodes.iter()
            .filter(|n| n.depth == NodeDepth::Leaf && n.id.contains(":reason:"))
            .collect();
        assert_eq!(reason_leaves.len(), 1,
            "near-duplicate reasons (Jaccard ≥ 0.85) must collapse to one Reason-keyed Leaf");
    }

    #[test]
    fn topic_node_id_has_topic_suffix() {
        let task_id = TaskId::new();
        let output = make_output(
            task_id.clone(), 1,
            vec![retry_event(task_id.clone(), 1, Some("violated C-001 constraint".into()))],
            closed_coherence(), vec![], vec![],
        );
        let corpus = stub_corpus(&["auth"]);
        let nodes = skill_from_output(&output, &corpus, &task_id);
        let topic = nodes.iter().find(|n| n.depth == NodeDepth::Topic);
        assert!(topic.is_some(), "must emit at least one Topic node");
        assert_eq!(
            topic.unwrap().id,
            format!("skill:{task_id}:auth:topic"),
            "Topic node id must be skill:{{task_id}}:{{domain}}:topic"
        );
    }

    #[test]
    fn topic_node_tensions_contain_socratic_questions() {
        use h2ai_types::events::SocraticDiagnosisEvent;
        let task_id = TaskId::new();
        let mut output = make_output(
            task_id.clone(), 1,
            vec![retry_event(task_id.clone(), 1, Some("tombstone".into()))],
            closed_coherence(), vec![], vec![],
        );
        output.socratic_diagnosis_events = vec![
            SocraticDiagnosisEvent {
                task_id: task_id.clone(),
                term: 0,
                question: "Why did auth quota fail?".to_string(),
                violated_constraints: vec![],
                eig_rank: 1,
                dedup_candidates_tried: 0,
                timestamp: chrono::Utc::now(),
            },
        ];
        let corpus = stub_corpus(&["auth"]);
        let nodes = skill_from_output(&output, &corpus, &task_id);
        let topic = nodes.iter().find(|n| n.depth == NodeDepth::Topic).unwrap();
        assert_eq!(topic.tensions.len(), 1);
        assert_eq!(topic.tensions[0].reason, "Why did auth quota fail?");
        assert!(
            topic.synthesis.contains("Why did auth quota fail?"),
            "synthesis must include the Socratic question"
        );
    }

    #[test]
    fn topic_node_entry_points_contain_resolution_excerpt() {
        let task_id = TaskId::new();
        let mut output = make_output(
            task_id.clone(), 1,
            vec![retry_event(task_id.clone(), 1, Some("tombstone".into()))],
            closed_coherence(), vec![], vec![],
        );
        output.resolved_output = "word ".repeat(100); // 500 chars
        let corpus = stub_corpus(&["auth"]);
        let nodes = skill_from_output(&output, &corpus, &task_id);
        let topic = nodes.iter().find(|n| n.depth == NodeDepth::Topic).unwrap();
        assert!(!topic.entry_points.is_empty(), "Topic node must have entry_points");
        assert!(
            topic.entry_points[0].starts_with("Resolution pattern: "),
            "entry_points[0] must start with 'Resolution pattern: '"
        );
        assert!(
            topic.entry_points[0].len() <= 320,
            "entry_points[0] must not exceed 320 chars (prefix + 300-char excerpt)"
        );
    }

    #[test]
    fn trim_at_word_boundary_short_string_unchanged() {
        assert_eq!(trim_at_word_boundary("hello world", 300), "hello world");
    }

    #[test]
    fn trim_at_word_boundary_cuts_at_last_space() {
        let s = "hello world foo"; // 15 chars
        // limit=12 → s[..12] = "hello world " → rfind(' ') at 11 → s[..11] = "hello world"
        assert_eq!(trim_at_word_boundary(s, 12), "hello world");
    }

    #[test]
    fn trim_at_word_boundary_no_whitespace_falls_back_to_hard_cut() {
        assert_eq!(trim_at_word_boundary("abcdefghijklmnop", 5), "abcde");
    }

    #[test]
    fn trim_at_word_boundary_multibyte_does_not_panic() {
        // "café" is 5 chars, 6 bytes (é is 2 bytes). limit=3 → "caf"
        let s = "café world";
        let result = trim_at_word_boundary(s, 3);
        assert_eq!(result, "caf");
    }

    #[test]
    fn reason_leaf_not_emitted_when_constraint_already_covered_by_leaf() {
        let task_id = TaskId::new();
        // Tombstone with parseable C-000 → emits Constraint-keyed Leaf for C-000
        // Verification event reason also mentions C-000 → must NOT emit Reason-keyed Leaf for it
        let output = make_output(
            task_id.clone(), 1,
            vec![retry_event(task_id.clone(), 1, Some("violated C-000 auth quota".into()))],
            closed_coherence(), vec![],
            vec![verification_event(task_id.clone(), 0.3, "C-000 constraint auth quota exceeded")],
        );
        let corpus = stub_corpus(&["auth"]);
        let nodes = skill_from_output(&output, &corpus, &task_id);
        // Should have: 1 Topic node + 1 Constraint-keyed Leaf (C-000)
        // Should NOT have: a Reason-keyed Leaf (reason mentions C-000 which is already covered)
        let reason_leaves: Vec<_> = nodes.iter()
            .filter(|n| n.depth == NodeDepth::Leaf && n.id.contains(":reason:"))
            .collect();
        assert!(
            reason_leaves.is_empty(),
            "must not emit Reason-keyed Leaf when the reason's constraint ID is already covered by a Constraint Leaf"
        );
        // Also confirm the Constraint Leaf IS present
        let constraint_leaf = nodes.iter().find(|n| n.depth == NodeDepth::Leaf && n.id.contains("C-000"));
        assert!(constraint_leaf.is_some(), "Constraint Leaf for C-000 must still be present");
    }
}
