use h2ai_orchestrator::coherence::CoherenceState;
use h2ai_orchestrator::engine::EngineOutput;
use h2ai_orchestrator::skill_extractor::{
    fnv32a, parse_constraint_id, skill_from_output, skill_from_retry_events, trim_at_word_boundary,
};
use h2ai_types::events::{SelectionResolvedEvent, TaskComplexityAssessedEvent};
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::{MergeStrategy, ProbeSkipReason, TaskQuadrant};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn stub_corpus(domains: &[&str]) -> Vec<h2ai_constraints::types::ConstraintDoc> {
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};
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

fn stub_attribution() -> h2ai_orchestrator::attribution::HarnessAttribution {
    h2ai_orchestrator::attribution::HarnessAttribution {
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
        passed_checks: None,
        total_checks: None,
        score_lower: None,
        score_upper: None,
        timestamp: chrono::Utc::now(),
    }
}

// ── Regression tests (preserved from old implementation) ─────────────────────

#[test]
fn clean_run_produces_no_skills() {
    let task_id = TaskId::new();
    let output = make_output(
        task_id.clone(),
        2,
        vec![],
        closed_coherence(),
        vec![],
        vec![],
    );
    let corpus = stub_corpus(&["auth"]);
    let nodes = skill_from_output(&output, &corpus, &task_id);
    assert!(nodes.is_empty(), "clean run must produce no skills");
}

#[test]
fn zero_valid_proposals_with_retries_produces_skills() {
    // TaskFailed path: 0 valid proposals but retries occurred → skills SHOULD be emitted.
    let task_id = TaskId::new();
    let output = make_output(
        task_id.clone(),
        0,
        vec![retry_event(
            task_id.clone(),
            2,
            Some("violated C-001".into()),
        )],
        closed_coherence(),
        vec![],
        vec![],
    );
    let corpus = stub_corpus(&["auth"]);
    let nodes = skill_from_output(&output, &corpus, &task_id);
    assert!(
        !nodes.is_empty(),
        "failed task with retries must produce skill nodes"
    );
}

#[test]
fn zero_valid_proposals_and_zero_retries_returns_empty() {
    // No signal at all → no skill nodes.
    let task_id = TaskId::new();
    let output = make_output(
        task_id.clone(),
        0,
        vec![],
        closed_coherence(),
        vec![],
        vec![],
    );
    let corpus = stub_corpus(&["auth"]);
    let nodes = skill_from_output(&output, &corpus, &task_id);
    assert!(
        nodes.is_empty(),
        "no retries and no valid proposals → no skills"
    );
}

#[test]
fn topology_retry_produces_topic_per_domain() {
    use h2ai_knowledge::types::NodeDepth;
    let task_id = TaskId::new();
    let output = make_output(
        task_id.clone(),
        1,
        vec![retry_event(
            task_id.clone(),
            1,
            Some("violated auth constraint".into()),
        )],
        closed_coherence(),
        vec![],
        vec![],
    );
    let corpus = stub_corpus(&["auth", "billing"]);
    let nodes = skill_from_output(&output, &corpus, &task_id);
    // 2 Topic nodes (one per corpus domain) — no Leaf nodes from this tombstone
    let topic_nodes: Vec<_> = nodes
        .iter()
        .filter(|n| n.depth == NodeDepth::Topic)
        .collect();
    assert_eq!(topic_nodes.len(), 2, "one Topic node per corpus domain");
    for node in &topic_nodes {
        assert!(
            node.failure_modes
                .iter()
                .any(|f| f.contains("violated auth constraint")),
            "tombstone text must appear in failure_modes"
        );
        assert!(
            !node.invariants.is_empty(),
            "invariants must contain repair summary"
        );
        assert!(
            node.importance > 0.5,
            "retried task must have importance > 0.5"
        );
    }
}

#[test]
fn uncovered_domain_produces_targeted_topic_node() {
    use h2ai_knowledge::types::NodeDepth;
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
    let output = make_output(
        task_id.clone(),
        1,
        vec![],
        closed_coherence(),
        vec![srani_ev],
        vec![],
    );
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
        vec![retry_event(
            task_id.clone(),
            5,
            Some("heavy failure".into()),
        )],
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

// ── New tests ─────────────────────────────────────────────────────────────────

#[test]
fn parse_constraint_id_extracts_c007_from_tombstone() {
    assert_eq!(
        parse_constraint_id("violated C-007 auth constraint"),
        Some("C-007".to_string())
    );
    assert_eq!(parse_constraint_id("no constraint here"), None);
    assert_eq!(
        parse_constraint_id("AUTH-123 failed"),
        Some("AUTH-123".to_string())
    );
}

#[test]
fn fnv32a_is_deterministic_and_nonzero() {
    let h1 = fnv32a("auth token missing");
    let h2 = fnv32a("auth token missing");
    assert_eq!(h1, h2);
    assert_ne!(h1, 0);
}

#[test]
fn constraint_leaf_emitted_when_tombstone_has_parseable_id() {
    use h2ai_knowledge::types::NodeDepth;
    let task_id = TaskId::new();
    // Tombstone "violated C-000 auth quota" → regex matches "C-000"
    // C-000 maps to "auth" domain (stub_corpus index 0)
    let output = make_output(
        task_id.clone(),
        1,
        vec![retry_event(
            task_id.clone(),
            1,
            Some("violated C-000 auth quota".into()),
        )],
        closed_coherence(),
        vec![],
        vec![],
    );
    let corpus = stub_corpus(&["auth"]);
    let nodes = skill_from_output(&output, &corpus, &task_id);
    let leaf = nodes.iter().find(|n| n.depth == NodeDepth::Leaf);
    assert!(
        leaf.is_some(),
        "must emit a Constraint-keyed Leaf for parseable constraint ID"
    );
    let leaf = leaf.unwrap();
    assert_eq!(
        leaf.id,
        format!("skill:{task_id}:C-000"),
        "Constraint-keyed Leaf id must be skill:{{task_id}}:{{constraint_id}}"
    );
    assert!(
        leaf.synthesis.contains("C-000"),
        "synthesis must contain the constraint ID"
    );
    assert!(
        leaf.synthesis.contains("violated C-000 auth quota"),
        "synthesis must contain the tombstone text"
    );
}

#[test]
fn constraint_leaf_importance_1_when_tombstone_appears_twice() {
    use h2ai_knowledge::types::NodeDepth;
    let task_id = TaskId::new();
    // Same tombstone in two retry events → count = 2 → importance = 1.0
    let output = make_output(
        task_id.clone(),
        1,
        vec![
            retry_event(task_id.clone(), 1, Some("violated C-000 quota".into())),
            retry_event(task_id.clone(), 2, Some("violated C-000 quota".into())),
        ],
        closed_coherence(),
        vec![],
        vec![],
    );
    let corpus = stub_corpus(&["auth"]);
    let nodes = skill_from_output(&output, &corpus, &task_id);
    let leaf = nodes
        .iter()
        .find(|n| n.depth == NodeDepth::Leaf && n.id.contains("C-000"));
    assert!(leaf.is_some());
    assert!(
        (leaf.unwrap().importance - 1.0).abs() < 1e-5,
        "tombstone appearing ≥2 times → importance must be 1.0"
    );
}

#[test]
fn reason_leaf_emitted_when_no_constraint_id_in_tombstone() {
    use h2ai_knowledge::types::NodeDepth;
    let task_id = TaskId::new();
    // Tombstone without [A-Z]+-\d+ pattern → no Constraint-keyed Leaf
    // Verification event with score < 0.5 → Reason-keyed Leaf
    let output = make_output(
        task_id.clone(),
        1,
        vec![retry_event(
            task_id.clone(),
            1,
            Some("auth quota exceeded".into()),
        )],
        closed_coherence(),
        vec![],
        vec![verification_event(
            task_id.clone(),
            0.3,
            "auth token was missing from header",
        )],
    );
    let corpus = stub_corpus(&["auth"]);
    let nodes = skill_from_output(&output, &corpus, &task_id);
    let reason_leaf = nodes
        .iter()
        .find(|n| n.depth == NodeDepth::Leaf && n.id.contains(":reason:"));
    assert!(
        reason_leaf.is_some(),
        "must emit Reason-keyed Leaf when tombstone has no constraint ID"
    );
    assert!(
        reason_leaf
            .unwrap()
            .synthesis
            .contains("auth token was missing from header"),
        "Reason-keyed Leaf synthesis must contain the verifier reason"
    );
}

#[test]
fn jaccard_dedup_prevents_near_duplicate_reason_leaves() {
    use h2ai_knowledge::types::NodeDepth;
    let task_id = TaskId::new();
    // Two near-identical verifier reasons (high Jaccard) → only one Reason-keyed Leaf
    let output = make_output(
        task_id.clone(),
        1,
        vec![retry_event(
            task_id.clone(),
            1,
            Some("generic failure".into()),
        )],
        closed_coherence(),
        vec![],
        vec![
            verification_event(
                task_id.clone(),
                0.2,
                "auth token header missing from request",
            ),
            verification_event(
                task_id.clone(),
                0.3,
                "auth token header missing from request field",
            ), // Jaccard = 6/7 ≈ 0.857
        ],
    );
    let corpus = stub_corpus(&["auth"]);
    let nodes = skill_from_output(&output, &corpus, &task_id);
    let reason_leaves: Vec<_> = nodes
        .iter()
        .filter(|n| n.depth == NodeDepth::Leaf && n.id.contains(":reason:"))
        .collect();
    assert_eq!(
        reason_leaves.len(),
        1,
        "near-duplicate reasons (Jaccard ≥ 0.85) must collapse to one Reason-keyed Leaf"
    );
}

#[test]
fn topic_node_id_has_topic_suffix() {
    use h2ai_knowledge::types::NodeDepth;
    let task_id = TaskId::new();
    let output = make_output(
        task_id.clone(),
        1,
        vec![retry_event(
            task_id.clone(),
            1,
            Some("violated C-001 constraint".into()),
        )],
        closed_coherence(),
        vec![],
        vec![],
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
    use h2ai_knowledge::types::NodeDepth;
    use h2ai_types::events::SocraticDiagnosisEvent;
    let task_id = TaskId::new();
    let mut output = make_output(
        task_id.clone(),
        1,
        vec![retry_event(task_id.clone(), 1, Some("tombstone".into()))],
        closed_coherence(),
        vec![],
        vec![],
    );
    output.socratic_diagnosis_events = vec![SocraticDiagnosisEvent {
        task_id: task_id.clone(),
        term: 0,
        question: "Why did auth quota fail?".to_string(),
        violated_constraints: vec![],
        eig_rank: 1,
        dedup_candidates_tried: 0,
        timestamp: chrono::Utc::now(),
    }];
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
    use h2ai_knowledge::types::NodeDepth;
    let task_id = TaskId::new();
    let mut output = make_output(
        task_id.clone(),
        1,
        vec![retry_event(task_id.clone(), 1, Some("tombstone".into()))],
        closed_coherence(),
        vec![],
        vec![],
    );
    output.resolved_output = "word ".repeat(100); // 500 chars
    let corpus = stub_corpus(&["auth"]);
    let nodes = skill_from_output(&output, &corpus, &task_id);
    let topic = nodes.iter().find(|n| n.depth == NodeDepth::Topic).unwrap();
    assert!(
        !topic.entry_points.is_empty(),
        "Topic node must have entry_points"
    );
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
    use h2ai_knowledge::types::NodeDepth;
    let task_id = TaskId::new();
    // Tombstone with parseable C-000 → emits Constraint-keyed Leaf for C-000
    // Verification event reason also mentions C-000 → must NOT emit Reason-keyed Leaf for it
    let output = make_output(
        task_id.clone(),
        1,
        vec![retry_event(
            task_id.clone(),
            1,
            Some("violated C-000 auth quota".into()),
        )],
        closed_coherence(),
        vec![],
        vec![verification_event(
            task_id.clone(),
            0.3,
            "C-000 constraint auth quota exceeded",
        )],
    );
    let corpus = stub_corpus(&["auth"]);
    let nodes = skill_from_output(&output, &corpus, &task_id);
    // Should have: 1 Topic node + 1 Constraint-keyed Leaf (C-000)
    // Should NOT have: a Reason-keyed Leaf (reason mentions C-000 which is already covered)
    let reason_leaves: Vec<_> = nodes
        .iter()
        .filter(|n| n.depth == NodeDepth::Leaf && n.id.contains(":reason:"))
        .collect();
    assert!(
        reason_leaves.is_empty(),
        "must not emit Reason-keyed Leaf when the reason's constraint ID is already covered by a Constraint Leaf"
    );
    // Also confirm the Constraint Leaf IS present
    let constraint_leaf = nodes
        .iter()
        .find(|n| n.depth == NodeDepth::Leaf && n.id.contains("C-000"));
    assert!(
        constraint_leaf.is_some(),
        "Constraint Leaf for C-000 must still be present"
    );
}

// ── skill_from_retry_events (failure path) ───────────────────────────────────

#[test]
fn failure_path_with_verification_events_but_no_topology_retries_produces_reason_leaf_nodes() {
    use h2ai_knowledge::types::NodeDepth;
    // Regression: TaskFailed with partial_verification_events (score < 0.5) but empty
    // topology_retry_events. Old code: guard at n_valid==0 && n_retries==0 → vec![],
    // then domain_failures empty → vec![]. Fix: bypass both guards when verification
    // events carry failure signal, produce Reason-keyed Leaf nodes.
    let task_id = TaskId::new();
    let corpus = stub_corpus(&["billing"]);
    let nodes = skill_from_retry_events(
        vec![],
        &[verification_event(
            task_id.clone(),
            0.35,
            "billing quota constraint violated",
        )],
        &corpus,
        &task_id,
    );
    assert!(
        !nodes.is_empty(),
        "partial_verification_events on TaskFailed path must produce skill nodes"
    );
    let reason_leaf = nodes
        .iter()
        .find(|n| n.depth == NodeDepth::Leaf && n.id.contains(":reason:"));
    assert!(
        reason_leaf.is_some(),
        "must emit at least one Reason-keyed Leaf from partial verification failures"
    );
    assert!(
        reason_leaf
            .unwrap()
            .failure_modes
            .iter()
            .any(|f| f.contains("billing quota")),
        "Reason-keyed Leaf failure_modes must contain the verifier reason text"
    );
}

#[test]
fn failure_path_with_no_signals_returns_empty() {
    let task_id = TaskId::new();
    let corpus = stub_corpus(&["billing"]);
    let nodes = skill_from_retry_events(vec![], &[], &corpus, &task_id);
    assert!(
        nodes.is_empty(),
        "no retries and no verification events must produce no skill nodes"
    );
}

#[test]
fn failure_path_high_scoring_events_do_not_produce_reason_leaves() {
    // Verification events with score >= 0.5 are not failures → no Reason-keyed Leaf nodes.
    let task_id = TaskId::new();
    let corpus = stub_corpus(&["billing"]);
    let nodes = skill_from_retry_events(
        vec![],
        &[
            verification_event(task_id.clone(), 0.5, "barely passing"),
            verification_event(task_id.clone(), 0.9, "well above threshold"),
        ],
        &corpus,
        &task_id,
    );
    assert!(
        nodes.is_empty(),
        "events with score >= 0.5 must not produce skill nodes"
    );
}

// ── Branch-coverage completeness ─────────────────────────────────────────────

#[test]
fn parse_constraint_id_uppercase_block_without_dash_returns_none() {
    // "HELLO" is all uppercase but not followed by '-' → inner `if bytes[i] == b'-'` is
    // false, falling through to the outer `} else {` arm without returning anything.
    assert_eq!(parse_constraint_id("HELLO world"), None);
}

#[test]
fn parse_constraint_id_dash_followed_by_no_digits_returns_none() {
    // "C- x" matches uppercase then '-' but has no digit after the dash → digit_start == i
    // → `if i > digit_start` is false → continues without returning.
    assert_eq!(parse_constraint_id("C- x"), None);
}

#[test]
fn jaccard_dedup_treats_two_empty_reasons_as_duplicates() {
    use h2ai_knowledge::types::NodeDepth;
    let task_id = TaskId::new();
    // Two failing events both with empty reason strings:
    // jaccard_similarity("", "") → union=0 → returns 1.0 (the ≥0.85 guard fires) → second is deduped.
    // We need domain_failures or has_verifier_failures true so the early-return guard at line 183
    // doesn't fire. One topology retry (retry_count=1) creates domain_failures.
    let output = make_output(
        task_id.clone(),
        1,
        vec![retry_event(
            task_id.clone(),
            1,
            Some("generic tombstone".into()),
        )],
        closed_coherence(),
        vec![],
        vec![
            verification_event(task_id.clone(), 0.2, ""),
            verification_event(task_id.clone(), 0.3, ""),
        ],
    );
    let corpus = stub_corpus(&["auth"]);
    let nodes = skill_from_output(&output, &corpus, &task_id);
    let reason_leaves: Vec<_> = nodes
        .iter()
        .filter(|n| n.depth == NodeDepth::Leaf && n.id.contains(":reason:"))
        .collect();
    // Both reasons are empty; Jaccard(∅, ∅) = 1.0 ≥ 0.85 → second is a dup → only 1 reason leaf.
    assert_eq!(
        reason_leaves.len(),
        1,
        "two empty reasons must collapse to one Reason-keyed Leaf (Jaccard=1.0)"
    );
}

#[test]
fn topology_retry_event_with_retry_count_zero_does_not_add_failure_mode() {
    use h2ai_knowledge::types::NodeDepth;
    let task_id = TaskId::new();
    // retry_count=0 → `if ev.retry_count > 0` is false → tombstone text is NOT added to
    // domain_failures. We pair it with an uncovered domain to keep the early-return guard
    // from firing (domain_failures gets an entry from the uncovered_domain loop).
    let output = make_output(
        task_id.clone(),
        1,
        vec![retry_event(
            task_id.clone(),
            0,
            Some("should not appear".into()),
        )],
        CoherenceState {
            uncovered_domains: vec!["auth".into()],
            active_contradictions: vec![],
        },
        vec![],
        vec![],
    );
    let corpus = stub_corpus(&["auth"]);
    let nodes = skill_from_output(&output, &corpus, &task_id);
    let topic = nodes.iter().find(|n| n.depth == NodeDepth::Topic);
    assert!(
        topic.is_some(),
        "must still emit a Topic node for the uncovered domain"
    );
    let failure_modes = &topic.unwrap().failure_modes;
    assert!(
        failure_modes
            .iter()
            .all(|f| !f.contains("should not appear")),
        "retry_count=0 tombstone must NOT appear in failure_modes"
    );
}

#[test]
fn srani_event_with_hint_not_injected_does_not_add_failure_mode() {
    use h2ai_types::events::CorrelatedFabricationEvent;
    let task_id = TaskId::new();
    // hint_injected=false → inner if-body is skipped; entities present but not injected.
    // Pair with an uncovered domain so domain_failures is non-empty (avoids early return).
    let srani_ev = CorrelatedFabricationEvent {
        task_id: task_id.clone(),
        cfi: 0.6,
        injection_pressure: 0.55,
        shared_ungrounded_entities: vec!["SomeEntity".into()],
        proposal_count: 2,
        hint_injected: false,
        timestamp: chrono::Utc::now(),
    };
    let output = make_output(
        task_id.clone(),
        1,
        vec![],
        CoherenceState {
            uncovered_domains: vec!["auth".into()],
            active_contradictions: vec![],
        },
        vec![srani_ev],
        vec![],
    );
    let corpus = stub_corpus(&["auth"]);
    let nodes = skill_from_output(&output, &corpus, &task_id);
    assert!(
        !nodes.is_empty(),
        "uncovered domain must still produce a Topic node"
    );
    let all_failures: Vec<String> = nodes
        .iter()
        .flat_map(|n| n.failure_modes.iter().cloned())
        .collect();
    assert!(
        all_failures.iter().all(|f| !f.contains("SomeEntity")),
        "hint_injected=false must not add ungrounded entities to failure_modes"
    );
}
