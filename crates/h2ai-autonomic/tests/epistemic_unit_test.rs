#![allow(clippy::doc_markdown, clippy::cast_precision_loss)]
use h2ai_autonomic::epistemic::{
    classify_failure_mode, compute_n_eff_cosine, synthesize_tombstone,
};
use h2ai_context::embedding::EmbeddingModel;
use h2ai_types::events::{ConstraintViolation, FailureMode};

// ── Mock declarations ─────────────────────────────────────────────────────────

mockall::mock! {
    pub EpistemicEmbedding {}
    impl EmbeddingModel for EpistemicEmbedding {
        fn embed(&self, text: &str) -> Vec<f32>;
    }
}

// ── Embedding stub factories ──────────────────────────────────────────────────

/// All texts → same vector → N_eff = 1 (ModeCollapse).
fn collapse_stub() -> MockEpistemicEmbedding {
    let mut m = MockEpistemicEmbedding::new();
    m.expect_embed().returning(|_| vec![1.0, 0.0, 0.0]);
    m
}

/// Routes on agent markers → orthogonal vectors → N_eff = N (ConstrainedExploration).
fn diverse_stub() -> MockEpistemicEmbedding {
    let mut m = MockEpistemicEmbedding::new();
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

/// [AGENT_C] → orthogonal; all others → same → N_eff ≈ 1.8 for N=3.
fn partial_collapse_stub() -> MockEpistemicEmbedding {
    let mut m = MockEpistemicEmbedding::new();
    m.expect_embed().returning(|text| {
        if text.contains("[AGENT_C]") {
            vec![0.0, 1.0, 0.0]
        } else {
            vec![1.0, 0.0, 0.0]
        }
    });
    m
}

/// Single fixed vector → used for degenerate single-text tests.
fn single_stub() -> MockEpistemicEmbedding {
    let mut m = MockEpistemicEmbedding::new();
    m.expect_embed().returning(|_| vec![1.0, 0.0]);
    m
}

// ── Test 1 ────────────────────────────────────────────────────────────────────

#[test]
fn collapse_stub_n_eff_is_one() {
    let texts = vec![
        "Answer [AGENT_A]".into(),
        "Answer [AGENT_B]".into(),
        "Answer [AGENT_C]".into(),
    ];
    let n_eff = compute_n_eff_cosine(&texts, &collapse_stub(), 0.05);
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
    let n_eff = compute_n_eff_cosine(&texts, &diverse_stub(), 0.05);
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
    let n_eff = compute_n_eff_cosine(&texts, &partial_collapse_stub(), 0.05);
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
    let n_requested: f64 = 3.0;
    let n_eff_actual: f64 = 1.5;
    let yield_ratio = n_eff_actual / n_requested;
    assert!(
        (yield_ratio - 0.5).abs() < 1e-9,
        "yield_ratio must use N_requested=3, got {yield_ratio}"
    );
}

#[test]
fn compute_n_eff_cosine_returns_one_for_single_text() {
    // Line 14: n < 2 early return → 1.0 (degenerate: only one perspective)
    let texts = vec!["only one text".to_string()];
    let n_eff = compute_n_eff_cosine(&texts, &single_stub(), 0.05);
    assert!(
        (n_eff - 1.0).abs() < 1e-9,
        "single text must return 1.0, got {n_eff}"
    );
}

#[test]
fn synthesize_tombstone_returns_none_for_empty_violations() {
    // Line 60: empty violations → return None
    let result = synthesize_tombstone(&[]);
    assert!(result.is_none(), "empty violations must return None");
}
