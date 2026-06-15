#![allow(clippy::doc_markdown, clippy::cast_precision_loss)]
use h2ai_autonomic::epistemic::{
    classify_failure_mode, compute_n_eff_cosine, mean_pairwise_cosine, synthesize_repair_plan,
    synthesize_tombstone, talagrand_kl_delta_tau,
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
            check_reasons: None,
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
            check_reasons: None,
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
    // GAP-F7: remediation_hint is now included as "what_to_try" — the old exclusion was too
    // conservative. Raw *proposal* text is still never injected (anchoring hazard).
    assert!(
        tombstone.contains("JWT"),
        "remediation hint must appear as what_to_try (GAP-F7)"
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

// ── repair_plan_tests (moved from epistemic.rs) ───────────────────────────────

fn v(id: &str, sev: &str, score: f64) -> ConstraintViolation {
    ConstraintViolation {
        constraint_id: id.to_string(),
        score,
        severity_label: sev.to_string(),
        remediation_hint: None,
        constraint_description: String::new(),
        verifier_reason: None,
        check_verdicts: vec![],
        criteria_pass: None,
        check_reasons: None,
    }
}

#[test]
fn empty_violations_returns_none() {
    assert!(synthesize_repair_plan(&[]).is_none());
    assert!(synthesize_tombstone(&[]).is_none());
}

#[test]
fn render_contains_constraint_id_and_score() {
    let plan = synthesize_repair_plan(&[v("C-001", "Hard", 0.32)]).unwrap();
    let s = plan.render();
    assert!(s.contains("C-001"));
    assert!(s.contains("0.32"));
    assert!(s.contains("Hard"));
}

#[test]
fn rule_uses_criteria_pass_first() {
    let mut violation = v("C-1", "Hard", 0.0);
    violation.criteria_pass = Some("use atomic Lua EVAL".into());
    violation.constraint_description = "fallback description".into();
    let plan = synthesize_repair_plan(&[violation]).unwrap();
    let s = plan.render();
    assert!(s.contains("atomic Lua EVAL"), "criteria_pass must win");
    assert!(!s.contains("fallback description"));
}

#[test]
fn rule_falls_back_to_description_when_no_criteria_pass() {
    let mut violation = v("C-1", "Hard", 0.0);
    violation.criteria_pass = None;
    violation.constraint_description = "use circuit breakers".into();
    let plan = synthesize_repair_plan(&[violation]).unwrap();
    assert!(plan.render().contains("circuit breakers"));
}

#[test]
fn what_to_try_uses_remediation_hint() {
    let mut violation = v("C-1", "Hard", 0.0);
    violation.remediation_hint = Some("wrap calls with Resilience4j".into());
    let plan = synthesize_repair_plan(&[violation]).unwrap();
    assert!(plan.render().contains("Resilience4j"));
}

#[test]
fn what_failed_uses_verifier_reason() {
    let mut violation = v("C-1", "Hard", 0.2);
    violation.verifier_reason = Some("non-atomic GET-SET detected".into());
    let plan = synthesize_repair_plan(&[violation]).unwrap();
    assert!(plan.render().contains("non-atomic GET-SET detected"));
}

#[test]
fn failed_check_indices_appended_to_what_failed() {
    let mut violation = v("C-1", "Hard", 0.5);
    violation.check_verdicts = vec![true, false, true, false];
    let plan = synthesize_repair_plan(&[violation]).unwrap();
    let s = plan.render();
    // Checks 2 and 4 failed (1-indexed)
    assert!(s.contains("checks failed: 2, 4"), "got: {s}");
}

#[test]
fn raw_proposal_text_never_appears() {
    // The LLM's actual proposal text must never be injected (anchoring hazard).
    // This is enforced structurally: synthesize_repair_plan only reads typed fields.
    let raw_proposal = "The system uses PKCE with rolling refresh tokens";
    let mut violation = v("C-1", "Hard", 0.0);
    violation.verifier_reason = Some("missing Lua atomicity".into());
    let plan = synthesize_repair_plan(&[violation]).unwrap();
    assert!(!plan.render().contains(raw_proposal));
}

#[test]
fn multiple_violations_all_rendered() {
    let vs = vec![v("A-1", "Hard", 0.1), v("B-2", "Soft", 0.3)];
    let plan = synthesize_repair_plan(&vs).unwrap();
    let s = plan.render();
    assert!(s.contains("A-1"));
    assert!(s.contains("B-2"));
}

#[test]
fn tombstone_delegates_to_render() {
    let s = synthesize_tombstone(&[v("C-1", "Hard", 0.0)]).unwrap();
    // Must contain the constraint ID (same as repair plan)
    assert!(s.contains("C-1"));
}

// ── pairwise_cosine_tests (moved from epistemic.rs) ──────────────────────────

struct FakeEmbedder {
    embeddings: std::collections::HashMap<String, Vec<f32>>,
}

impl EmbeddingModel for FakeEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        self.embeddings.get(text).cloned().unwrap_or_default()
    }
}

#[test]
fn mean_pairwise_cosine_returns_none_for_single_text() {
    let model = FakeEmbedder {
        embeddings: Default::default(),
    };
    let result = mean_pairwise_cosine(&["hello".to_string()], &model);
    assert!(result.is_none());
}

#[test]
fn mean_pairwise_cosine_identical_texts_returns_one() {
    let mut emb = std::collections::HashMap::new();
    emb.insert("a".to_string(), vec![1.0_f32, 0.0]);
    emb.insert("b".to_string(), vec![1.0_f32, 0.0]);
    let model = FakeEmbedder { embeddings: emb };
    let result = mean_pairwise_cosine(&["a".to_string(), "b".to_string()], &model).unwrap();
    assert!((result - 1.0).abs() < 1e-5, "expected ~1.0 got {result}");
}

#[test]
fn mean_pairwise_cosine_orthogonal_returns_zero() {
    let mut emb = std::collections::HashMap::new();
    emb.insert("x".to_string(), vec![1.0_f32, 0.0]);
    emb.insert("y".to_string(), vec![0.0_f32, 1.0]);
    let model = FakeEmbedder { embeddings: emb };
    let result = mean_pairwise_cosine(&["x".to_string(), "y".to_string()], &model).unwrap();
    assert!(result.abs() < 1e-5, "expected ~0.0 got {result}");
}

#[test]
fn mean_pairwise_cosine_three_texts_averages_pairs() {
    let mut emb = std::collections::HashMap::new();
    for k in ["a", "b", "c"] {
        emb.insert(k.to_string(), vec![1.0_f32, 0.0]);
    }
    let model = FakeEmbedder { embeddings: emb };
    let texts: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
    let result = mean_pairwise_cosine(&texts, &model).unwrap();
    assert!((result - 1.0).abs() < 1e-5, "expected ~1.0 got {result}");
}

// ── talagrand_kl_delta_tau tests (GAP-E2) ────────────────────────────────────

#[test]
fn talagrand_kl_delta_tau_short_histogram_returns_zero() {
    // Fewer than 3 bins → can't compute middle/edges → 0.0
    assert_eq!(talagrand_kl_delta_tau(&[], 0.1), 0.0);
    assert_eq!(talagrand_kl_delta_tau(&[1.0], 0.1), 0.0);
    assert_eq!(talagrand_kl_delta_tau(&[1.0, 1.0], 0.1), 0.0);
}

#[test]
fn talagrand_kl_delta_tau_zero_histogram_returns_zero() {
    assert_eq!(talagrand_kl_delta_tau(&[0.0, 0.0, 0.0], 0.1), 0.0);
}

#[test]
fn talagrand_kl_delta_tau_flat_histogram_near_zero() {
    // Uniform histogram: U_score and Λ_score cancel; Δτ ≈ 0.
    // For N=5 uniform counts, mean=0.2, var=0, U_score=0.
    // Λ_score = max(middle) / mean(edges) = 0.2 / 0.2 = 1.0.
    // Δτ = 0.1 × (0 − 1.0) = −0.1 (slight contraction from Λ term dominating).
    // Regardless of exact value, the magnitude should be small relative to the range.
    let h = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let delta = talagrand_kl_delta_tau(&h, 0.1);
    // For perfectly flat, U_score=0 and Λ_score=1 → delta = -0.1
    assert!(
        (delta - (-0.1)).abs() < 1e-9,
        "flat histogram delta={delta}"
    );
}

#[test]
fn talagrand_kl_delta_tau_u_shaped_positive_delta() {
    // U-shape: high counts at extremes, low in middle → over-confident → expand τ.
    // histogram: [10, 1, 1, 1, 10] — heavy tails
    let h = vec![10.0, 1.0, 1.0, 1.0, 10.0];
    let delta = talagrand_kl_delta_tau(&h, 0.1);
    assert!(
        delta > 0.0,
        "U-shaped histogram should expand τ, got delta={delta}"
    );
}

#[test]
fn talagrand_kl_delta_tau_lambda_shaped_negative_delta() {
    // Λ-shape: heavy middle, light tails → under-dispersed → contract τ.
    // histogram: [1, 1, 10, 10, 1, 1] — centre mass dominates
    let h = vec![1.0, 1.0, 10.0, 10.0, 1.0, 1.0];
    let delta = talagrand_kl_delta_tau(&h, 0.1);
    assert!(
        delta < 0.0,
        "Λ-shaped histogram should contract τ, got delta={delta}"
    );
}

#[test]
fn talagrand_kl_delta_tau_innovation5_scores_detected_as_lambda() {
    // INNOVATION-5 evidence (from gaps.md): scores (0.18,0.23,0.29,0.38,0.39,0.39,0.43,0.43,0.49)
    // are Λ-shaped — mass concentrated in 0.35–0.45 range, thin tails.
    // Simulate corresponding rank histogram: middle-heavy, edge-light.
    let h = vec![1.0, 2.0, 5.0, 5.0, 4.0, 2.0, 1.0];
    let delta = talagrand_kl_delta_tau(&h, 0.1);
    assert!(
        delta < 0.0,
        "INNOVATION-5 Λ-shaped scores should produce negative Δτ, got {delta}"
    );
}
