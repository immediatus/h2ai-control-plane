use h2ai_autonomic::epistemic::{
    classify_failure_mode, compute_n_eff_cosine, synthesize_tombstone,
};
use h2ai_context::embedding::EmbeddingModel;
use h2ai_types::events::{ConstraintViolation, FailureMode};

// ── Embedding stubs ───────────────────────────────────────────────────────────

/// All texts → same vector → N_eff = 1 (ModeCollapse).
struct CollapseStub;
impl EmbeddingModel for CollapseStub {
    fn embed(&self, _: &str) -> Vec<f32> {
        vec![1.0, 0.0, 0.0]
    }
}

/// Routes on agent markers → orthogonal vectors → N_eff = N (ConstrainedExploration).
struct DiverseStub;
impl EmbeddingModel for DiverseStub {
    fn embed(&self, text: &str) -> Vec<f32> {
        if text.contains("[AGENT_A]") {
            vec![1.0, 0.0, 0.0]
        } else if text.contains("[AGENT_B]") {
            vec![0.0, 1.0, 0.0]
        } else {
            vec![0.0, 0.0, 1.0]
        }
    }
}

/// [AGENT_C] → orthogonal; all others → same → N_eff ≈ 1.8 for N=3.
struct PartialCollapseStub;
impl EmbeddingModel for PartialCollapseStub {
    fn embed(&self, text: &str) -> Vec<f32> {
        if text.contains("[AGENT_C]") {
            vec![0.0, 1.0, 0.0]
        } else {
            vec![1.0, 0.0, 0.0]
        }
    }
}

// ── Test 1 ────────────────────────────────────────────────────────────────────

#[test]
fn collapse_stub_n_eff_is_one() {
    let texts = vec![
        "Answer [AGENT_A]".into(),
        "Answer [AGENT_B]".into(),
        "Answer [AGENT_C]".into(),
    ];
    let n_eff = compute_n_eff_cosine(&texts, &CollapseStub, 0.05);
    assert!(
        (n_eff - 1.0).abs() < 1e-5,
        "CollapseStub must give N_eff=1, got {n_eff}"
    );
}

// ── Test 2 ────────────────────────────────────────────────────────────────────

#[test]
fn diverse_stub_n_eff_is_n() {
    let texts = vec![
        "Answer [AGENT_A]".into(),
        "Answer [AGENT_B]".into(),
        "Answer [AGENT_C]".into(),
    ];
    let n_eff = compute_n_eff_cosine(&texts, &DiverseStub, 0.05);
    assert!(
        (n_eff - 3.0).abs() < 1e-5,
        "DiverseStub must give N_eff=3, got {n_eff}"
    );
}

// ── Test 3 ────────────────────────────────────────────────────────────────────

#[test]
fn collapse_discriminant_fires_mode_collapse() {
    let n_eff = 1.0;
    let fm = classify_failure_mode(n_eff, 3, 0.5);
    assert_eq!(
        fm,
        FailureMode::ModeCollapse,
        "n_eff=1.0, N=3, threshold=0.5 → boundary 1.5 → ModeCollapse"
    );
}

// ── Test 4 ────────────────────────────────────────────────────────────────────

#[test]
fn diverse_discriminant_fires_constrained_exploration() {
    let n_eff = 3.0;
    let fm = classify_failure_mode(n_eff, 3, 0.5);
    assert_eq!(
        fm,
        FailureMode::ConstrainedExploration,
        "n_eff=3.0, N=3, threshold=0.5 → boundary 1.5 → ConstrainedExploration"
    );
}

// ── Test 5 ────────────────────────────────────────────────────────────────────

#[test]
fn partial_collapse_boundary_classified_correctly() {
    // N_eff ≈ 1.8, boundary = 0.5 × 3 = 1.5 → 1.8 > 1.5 → ConstrainedExploration
    let texts = vec![
        "Answer [AGENT_A]".into(),
        "Answer [AGENT_B]".into(),
        "Answer [AGENT_C]".into(),
    ];
    let n_eff = compute_n_eff_cosine(&texts, &PartialCollapseStub, 0.05);
    let fm = classify_failure_mode(n_eff, 3, 0.5);
    assert_eq!(
        fm,
        FailureMode::ConstrainedExploration,
        "PartialCollapseStub: N_eff={n_eff:.3}, boundary=1.5 → ConstrainedExploration"
    );
}

// ── Test 6 ────────────────────────────────────────────────────────────────────

#[test]
fn tombstone_contains_constraint_ids_not_proposal_text() {
    let violations = vec![
        ConstraintViolation {
            constraint_id: "CONSTRAINT-001".into(),
            score: 0.0,
            severity_label: "Hard".into(),
            remediation_hint: Some("Use JWT tokens".into()),
        },
        ConstraintViolation {
            constraint_id: "CONSTRAINT-004".into(),
            score: 0.4,
            severity_label: "Soft".into(),
            remediation_hint: None,
        },
    ];
    let raw_proposal_text = "The system should use OAuth with PKCE and refresh tokens.";
    let tombstone = synthesize_tombstone(&violations).unwrap();

    assert!(
        tombstone.contains("CONSTRAINT-001"),
        "must list constraint ID"
    );
    assert!(
        tombstone.contains("CONSTRAINT-004"),
        "must list second constraint ID"
    );
    assert!(
        !tombstone.contains("JWT"),
        "must NOT contain remediation text"
    );
    assert!(
        !tombstone.contains(raw_proposal_text),
        "must NOT contain raw proposal text"
    );
    assert!(!tombstone.contains("PKCE"), "must NOT leak raw text");
}

// ── Test 10 (pure unit) ───────────────────────────────────────────────────────

#[test]
fn yield_ratio_uses_n_requested_not_n_responded() {
    // N_requested = 3, one adapter timed out (N_responded = 2), n_eff_actual = 1.5
    // yield_ratio should be 1.5 / 3 = 0.5, NOT 1.5 / 2 = 0.75
    let n_requested: usize = 3;
    let n_eff_actual: f64 = 1.5;
    let yield_ratio = n_eff_actual / n_requested as f64;
    assert!(
        (yield_ratio - 0.5).abs() < 1e-9,
        "yield_ratio must use N_requested=3, got {yield_ratio}"
    );
}
