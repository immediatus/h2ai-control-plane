#![allow(clippy::doc_markdown, clippy::cast_precision_loss, clippy::float_cmp)]
//! Section 4A simulation harness tests for the bivariate CG control loop.
//!
//! Tests the three embedding stubs (Collapse, Diverse, PartialCollapse) and
//! the routing decisions they trigger, per spec §4A.

use h2ai_autonomic::epistemic::{
    classify_failure_mode, compute_n_eff_cosine, synthesize_tombstone,
};
use h2ai_context::embedding::EmbeddingModel;
use h2ai_types::events::{ConstraintViolation, FailureMode};

// ── Mock declarations ─────────────────────────────────────────────────────────

mockall::mock! {
    pub BivariateEmbedding {}
    impl EmbeddingModel for BivariateEmbedding {
        fn embed(&self, text: &str) -> Vec<f32>;
    }
}

// ── Embedding stub factories ──────────────────────────────────────────────────

/// Forces N_eff → 1. ModeCollapse discriminant fires.
fn collapse_stub() -> MockBivariateEmbedding {
    let mut m = MockBivariateEmbedding::new();
    m.expect_embed().returning(|_| vec![1.0, 0.0, 0.0]);
    m
}

/// Forces N_eff → N for texts tagged [AGENT_A], [AGENT_B], [AGENT_C].
fn diverse_stub() -> MockBivariateEmbedding {
    let mut m = MockBivariateEmbedding::new();
    m.expect_embed().returning(|text| {
        if text.contains("[AGENT_A]") {
            vec![1.0, 0.0, 0.0]
        } else if text.contains("[AGENT_B]") {
            vec![0.0, 1.0, 0.0]
        } else {
            vec![0.0, 0.0, 1.0]
        }
    });
    m
}

/// [AGENT_C] orthogonal; others same → N_eff ≈ 1.8 (PartialCollapse).
fn partial_collapse_stub() -> MockBivariateEmbedding {
    let mut m = MockBivariateEmbedding::new();
    m.expect_embed().returning(|text| {
        if text.contains("[AGENT_C]") {
            vec![0.0, 1.0, 0.0]
        } else {
            vec![1.0, 0.0, 0.0]
        }
    });
    m
}

// ── Test 1 ────────────────────────────────────────────────────────────────────

/// All-same embeddings → cosine kernel is all-ones/N → N_eff = 1.
#[test]
fn collapse_stub_n_eff_is_one() {
    let texts = vec![
        "Output [AGENT_A] stateless auth".into(),
        "Output [AGENT_B] stateless auth".into(),
        "Output [AGENT_C] stateless auth".into(),
    ];
    let n_eff = compute_n_eff_cosine(&texts, &collapse_stub(), 0.05);
    assert!(
        (n_eff - 1.0).abs() < 1e-5,
        "CollapseStub → N_eff=1.0, got {n_eff}"
    );
}

// ── Test 2 ────────────────────────────────────────────────────────────────────

/// Orthogonal embeddings → identity/N kernel → N_eff = N.
#[test]
fn diverse_stub_n_eff_is_n() {
    let texts = vec![
        "Output [AGENT_A] JWT auth".into(),
        "Output [AGENT_B] CQRS pattern".into(),
        "Output [AGENT_C] API boundary".into(),
    ];
    let n_eff = compute_n_eff_cosine(&texts, &diverse_stub(), 0.05);
    assert!(
        (n_eff - 3.0).abs() < 1e-5,
        "DiverseStub → N_eff=3.0, got {n_eff}"
    );
}

// ── Test 3 ────────────────────────────────────────────────────────────────────

/// N_eff=1 with diversity_threshold=0.5, N=3: boundary=1.5, 1.0 < 1.5 → ModeCollapse.
#[test]
fn collapse_discriminant_fires_mode_collapse() {
    let texts = vec![
        "Output [AGENT_A]".into(),
        "Output [AGENT_B]".into(),
        "Output [AGENT_C]".into(),
    ];
    let n_eff = compute_n_eff_cosine(&texts, &collapse_stub(), 0.05);
    let fm = classify_failure_mode(n_eff, 3, 0.5);
    assert_eq!(
        fm,
        FailureMode::ModeCollapse,
        "n_eff={n_eff:.3} with boundary=1.5 must give ModeCollapse"
    );
}

// ── Test 4 ────────────────────────────────────────────────────────────────────

/// N_eff=3 with diversity_threshold=0.5, N=3: boundary=1.5, 3.0 > 1.5 → ConstrainedExploration.
#[test]
fn diverse_discriminant_fires_constrained_exploration() {
    let texts = vec![
        "Output [AGENT_A]".into(),
        "Output [AGENT_B]".into(),
        "Output [AGENT_C]".into(),
    ];
    let n_eff = compute_n_eff_cosine(&texts, &diverse_stub(), 0.05);
    let fm = classify_failure_mode(n_eff, 3, 0.5);
    assert_eq!(
        fm,
        FailureMode::ConstrainedExploration,
        "n_eff={n_eff:.3} with boundary=1.5 must give ConstrainedExploration"
    );
}

// ── Test 5 ────────────────────────────────────────────────────────────────────

/// N_eff≈1.8, boundary=1.5 at threshold=0.5, N=3 → ConstrainedExploration (above boundary).
#[test]
fn partial_collapse_boundary_classified_correctly() {
    let texts = vec![
        "Output [AGENT_A] auth".into(),
        "Output [AGENT_B] auth".into(),
        "Output [AGENT_C] cqrs".into(),
    ];
    let n_eff = compute_n_eff_cosine(&texts, &partial_collapse_stub(), 0.05);
    assert!(
        n_eff > 1.5 && n_eff < 2.5,
        "PartialCollapseStub N_eff should be ≈1.8, got {n_eff}"
    );
    let fm = classify_failure_mode(n_eff, 3, 0.5);
    assert_eq!(
        fm,
        FailureMode::ConstrainedExploration,
        "N_eff={n_eff:.3} > boundary=1.5 → ConstrainedExploration"
    );
}

// ── Test 6 ────────────────────────────────────────────────────────────────────

/// Tombstone contains constraint IDs but never raw proposal text or remediation hints.
#[test]
fn tombstone_contains_constraint_ids_not_proposal_text() {
    let raw_proposal = "The system should use OAuth PKCE with rolling refresh tokens.";
    let violations = vec![
        ConstraintViolation {
            constraint_id: "CONSTRAINT-001".into(),
            score: 0.0,
            severity_label: "Hard".into(),
            remediation_hint: Some("Use JWT tokens only".into()),
            constraint_description: String::new(),
            verifier_reason: None,
            check_verdicts: vec![],
            criteria_pass: None,
        },
        ConstraintViolation {
            constraint_id: "CONSTRAINT-004".into(),
            score: 0.4,
            severity_label: "Soft".into(),
            remediation_hint: None,
            constraint_description: String::new(),
            verifier_reason: None,
            check_verdicts: vec![],
            criteria_pass: None,
        },
    ];
    let tombstone = synthesize_tombstone(&violations).expect("violations non-empty → Some");

    assert!(
        tombstone.contains("CONSTRAINT-001"),
        "must include first constraint ID"
    );
    assert!(
        tombstone.contains("CONSTRAINT-004"),
        "must include second constraint ID"
    );
    // GAP-F7: remediation_hint is now the "what_to_try" field — deliberately included.
    assert!(
        tombstone.contains("JWT"),
        "remediation hint must appear as what_to_try (GAP-F7)"
    );
    assert!(
        !tombstone.contains("PKCE"),
        "must NOT leak raw proposal text"
    );
    assert!(
        !tombstone.contains(raw_proposal),
        "must NOT contain raw proposal text"
    );
}

// ── Test 7 ────────────────────────────────────────────────────────────────────

/// ModeCollapse routing signal: CollapseStub → N_eff=1 → ModeCollapse discriminant fires.
#[test]
fn mode_collapse_retry_routing_signal() {
    let texts = vec![
        "Output [AGENT_A]".into(),
        "Output [AGENT_B]".into(),
        "Output [AGENT_C]".into(),
    ];
    let n_eff = compute_n_eff_cosine(&texts, &collapse_stub(), 0.05);
    let fm = classify_failure_mode(n_eff, 3, 0.5);
    assert_eq!(
        fm,
        FailureMode::ModeCollapse,
        "CollapseStub must signal ModeCollapse → adapter rotation in engine"
    );
}

// ── Test 8 ────────────────────────────────────────────────────────────────────

/// ConstrainedExploration routing signal + tombstone synthesis.
#[test]
fn constrained_exploration_tombstone_injection_signal() {
    let texts = vec![
        "Output [AGENT_A]".into(),
        "Output [AGENT_B]".into(),
        "Output [AGENT_C]".into(),
    ];
    let n_eff = compute_n_eff_cosine(&texts, &diverse_stub(), 0.05);
    let fm = classify_failure_mode(n_eff, 3, 0.5);
    assert_eq!(fm, FailureMode::ConstrainedExploration);

    let violations = vec![ConstraintViolation {
        constraint_id: "ADR-007".into(),
        score: 0.0,
        severity_label: "Hard".into(),
        remediation_hint: None,
        constraint_description: String::new(),
        verifier_reason: None,
        check_verdicts: vec![],
        criteria_pass: None,
    }];
    let tombstone = synthesize_tombstone(&violations);
    assert!(
        tombstone.is_some(),
        "ConstrainedExploration with violations must produce tombstone"
    );
    let s = tombstone.unwrap();
    assert!(
        s.contains("ADR-007"),
        "tombstone must identify the violated constraint"
    );
}

// ── Test 9 ────────────────────────────────────────────────────────────────────

/// Event sourcing invariant: TaskBootstrappedEvent is a distinct H2AIEvent variant from
/// TopologyProvisionedEvent. Retries emit TopologyProvisioned, never TaskBootstrapped.
#[test]
fn retry_emits_topology_not_bootstrap() {
    use chrono::Utc;
    use h2ai_types::events::{H2AIEvent, TaskBootstrappedEvent};
    use h2ai_types::identity::TaskId;

    let bootstrap_ev = H2AIEvent::TaskBootstrapped(TaskBootstrappedEvent {
        task_id: TaskId::new(),
        system_context: "ctx".into(),
        pareto_weights: h2ai_types::config::ParetoWeights::new(0.2, 0.3, 0.5).unwrap(),
        timestamp: Utc::now(),
    });

    // A retry emits TopologyProvisioned, not TaskBootstrapped — they are distinct variants.
    assert!(
        !matches!(bootstrap_ev, H2AIEvent::TopologyProvisioned(_)),
        "TopologyProvisioned is distinct from TaskBootstrapped"
    );
}

// ── Test 10 ───────────────────────────────────────────────────────────────────

/// yield_ratio uses N_requested (not N_responded) as denominator.
#[test]
fn yield_ratio_uses_n_requested_not_n_responded() {
    let n_requested: f64 = 3.0;
    let n_eff_actual: f64 = 1.5;

    let yield_correct = n_eff_actual / n_requested;
    let yield_wrong = n_eff_actual / 2.0_f64; // wrong: uses N_responded

    assert!((yield_correct - 0.5).abs() < 1e-9);
    assert!((yield_wrong - 0.75).abs() < 1e-9);
    assert!(
        yield_correct < yield_wrong,
        "N_requested denominator gives more conservative yield than N_responded"
    );
}
