#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::unused_async,
    clippy::default_trait_access,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::similar_names,
    clippy::match_same_arms,
    clippy::literal_string_with_formatting_args,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::useless_format,
    clippy::option_if_let_else,
    clippy::map_unwrap_or,
    clippy::cloned_instead_of_copied,
    clippy::trivially_copy_pass_by_ref,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::needless_pass_by_value,
    clippy::explicit_iter_loop,
    clippy::needless_borrow,
    clippy::large_futures,
    clippy::manual_string_new,
    clippy::needless_lifetimes,
    clippy::elidable_lifetime_names,
    clippy::redundant_else,
    clippy::stable_sort_primitive,
    clippy::type_complexity,
    clippy::wildcard_imports,
    clippy::single_match_else,
    clippy::missing_fields_in_debug,
    clippy::doc_link_with_quotes,
    clippy::implicit_hasher,
    clippy::needless_collect,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::needless_type_cast,
    clippy::unreadable_literal,
    clippy::no_effect_underscore_binding
)]
use async_trait::async_trait;
use chrono::Utc;
use h2ai_config::H2AIConfig;
use h2ai_constraints::types::{
    ConstraintDoc, ConstraintPredicate, ConstraintSeverity, NumericOp, VocabularyMode,
};
use h2ai_orchestrator::complexity::{
    assess_task_complexity, classify_quadrant, participation_ratio, run_probe, ProbeInput,
};
use h2ai_types::adapter::{AdapterError, ComputeRequest, ComputeResponse, IComputeAdapter};
use h2ai_types::config::AdapterKind;
use h2ai_types::events::{CalibrationCompletedEvent, CalibrationQuality};
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::{
    CoherencyCoefficients, CoordinationThreshold, ProbeSkipReason, TaskQuadrant,
};

/// Adapter that always returns Err — used to test run_probe fallback.
#[derive(Debug)]
struct FailAdapter {
    kind: AdapterKind,
}

impl FailAdapter {
    fn new() -> Self {
        Self {
            kind: AdapterKind::CloudGeneric {
                endpoint: "fail://mock".into(),
                api_key_env: "MOCK".into(),
                model: None,
            },
        }
    }
}

#[async_trait]
impl IComputeAdapter for FailAdapter {
    async fn execute(&self, _req: ComputeRequest) -> Result<ComputeResponse, AdapterError> {
        Err(AdapterError::Timeout)
    }

    fn kind(&self) -> &AdapterKind {
        &self.kind
    }
}

fn dummy_calibration() -> CalibrationCompletedEvent {
    use h2ai_types::events::CgMode;
    let cc = CoherencyCoefficients::new(0.12, 0.039, vec![0.7]).unwrap();
    let thresh = CoordinationThreshold::from_calibration(&cc, 0.3);
    CalibrationCompletedEvent {
        calibration_id: TaskId::new(),
        coefficients: cc,
        coordination_threshold: thresh,
        ensemble: None,
        eigen: None,
        timestamp: Utc::now(),
        pairwise_beta: None,
        cg_mode: CgMode::default(),
        adapter_families: vec![],
        explorer_verification_family_match: false,
        single_family_warning: false,
        n_max_lo: 0.0,
        n_max_hi: 0.0,
        n_eff_cosine_prior: 0.0,
        calibration_quality: CalibrationQuality::Domain,
        calibration_source: Default::default(),
        beta_quality: None,
    }
}

fn vocab_constraint(id: &str) -> ConstraintDoc {
    ConstraintDoc {
        id: id.into(),
        source_file: "test".into(),
        description: "vocab".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.9 },
        predicate: ConstraintPredicate::VocabularyPresence {
            mode: VocabularyMode::AllOf,
            terms: vec!["stateless".into()],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    }
}

#[tokio::test]
async fn bootstrap_calibration_routes_coverage() {
    let cfg = H2AIConfig::default().task_complexity;
    let mut cal = dummy_calibration();
    cal.calibration_quality = CalibrationQuality::Bootstrap;
    let result =
        assess_task_complexity(&[vocab_constraint("c1")], &cal, &cfg, TaskId::new(), None).await;
    assert_eq!(result.task_quadrant, TaskQuadrant::Coverage);
    assert_eq!(
        result.probe_skip_reason,
        ProbeSkipReason::BootstrapCalibration
    );
}

#[tokio::test]
async fn empty_corpus_routes_precision() {
    let cfg = H2AIConfig::default().task_complexity;
    let cal = dummy_calibration();
    let result = assess_task_complexity(&[], &cal, &cfg, TaskId::new(), None).await;
    assert_eq!(result.task_quadrant, TaskQuadrant::Precision);
    assert!((result.tcc_effective - 1.0).abs() < 1e-9);
}

#[tokio::test]
async fn single_hard_static_constraint_routes_precision() {
    let cfg = H2AIConfig::default().task_complexity;
    let cal = dummy_calibration();
    let corpus = vec![vocab_constraint("c1")];
    let result = assess_task_complexity(&corpus, &cal, &cfg, TaskId::new(), None).await;
    assert_eq!(result.task_quadrant, TaskQuadrant::Precision);
    assert_eq!(
        result.probe_skip_reason,
        ProbeSkipReason::UnambiguousPrecision
    );
}

#[tokio::test]
async fn ambiguous_band_tcc_defers_probe_and_routes_coverage() {
    let mut cfg = H2AIConfig::default().task_complexity;
    cfg.tcc_precision_threshold = 1.2;
    cfg.tcc_coverage_threshold = 1.8;
    let cal = dummy_calibration();

    let make_hard = |id: &str, pred: ConstraintPredicate| ConstraintDoc {
        id: id.into(),
        source_file: "test".into(),
        description: "hard static".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.9 },
        predicate: pred,
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    };
    let corpus = vec![
        make_hard(
            "c1",
            ConstraintPredicate::VocabularyPresence {
                mode: VocabularyMode::AllOf,
                terms: vec!["foo".into()],
            },
        ),
        make_hard(
            "c2",
            ConstraintPredicate::NegativeKeyword {
                terms: vec!["bad".into()],
            },
        ),
        make_hard(
            "c3",
            ConstraintPredicate::RegexMatch {
                pattern: "ok".into(),
                must_match: true,
            },
        ),
        make_hard(
            "c4",
            ConstraintPredicate::LengthRange {
                min_chars: Some(10),
                max_chars: Some(500),
            },
        ),
        make_hard(
            "c5",
            ConstraintPredicate::NumericThreshold {
                field_pattern: r"(\d+)".into(),
                op: NumericOp::Ge,
                value: 1.0,
            },
        ),
    ];
    let result = assess_task_complexity(&corpus, &cal, &cfg, TaskId::new(), None).await;

    assert!(
        result.tcc_effective > cfg.tcc_precision_threshold
            && result.tcc_effective < cfg.tcc_coverage_threshold,
        "expected TCC in ambiguous band [{:.2}, {:.2}], got {:.3}",
        cfg.tcc_precision_threshold,
        cfg.tcc_coverage_threshold,
        result.tcc_effective
    );
    assert_eq!(
        result.probe_skip_reason,
        ProbeSkipReason::AmbiguousBandProbeDeferred
    );
    assert!(result.probe_skipped);
    assert_eq!(result.task_quadrant, TaskQuadrant::Coverage);
}

// ── classify_quadrant ──────────────────────────────────────────────────────────

#[test]
fn classify_quadrant_high_tcc_good_pool_is_coverage() {
    let cfg = H2AIConfig::default().task_complexity;
    let q = classify_quadrant(3.0, Some(2.0), &cfg);
    assert_eq!(q, TaskQuadrant::Coverage);
}

#[test]
fn classify_quadrant_high_tcc_poor_pool_is_complex() {
    let cfg = H2AIConfig::default().task_complexity;
    let q = classify_quadrant(3.0, Some(1.0), &cfg);
    assert_eq!(q, TaskQuadrant::Complex);
}

#[test]
fn classify_quadrant_low_tcc_good_pool_is_precision() {
    let cfg = H2AIConfig::default().task_complexity;
    let q = classify_quadrant(1.1, Some(2.0), &cfg);
    assert_eq!(q, TaskQuadrant::Precision);
}

#[test]
fn classify_quadrant_low_tcc_poor_pool_is_degenerate() {
    let cfg = H2AIConfig::default().task_complexity;
    let q = classify_quadrant(1.1, Some(1.0), &cfg);
    assert_eq!(q, TaskQuadrant::Degenerate);
}

// ── participation_ratio ────────────────────────────────────────────────────────

#[test]
fn participation_ratio_empty_returns_one() {
    assert!((participation_ratio(&[]) - 1.0).abs() < 1e-9);
}

#[test]
fn participation_ratio_single_row_returns_one() {
    let m = vec![vec![1.0, 0.0, 1.0]];
    assert!((participation_ratio(&m) - 1.0).abs() < 1e-9);
}

#[test]
fn participation_ratio_all_identical_rows_returns_one() {
    let row = vec![1.0, 0.0, 1.0, 1.0];
    let m = vec![row.clone(), row.clone(), row.clone()];
    assert!((participation_ratio(&m) - 1.0).abs() < 1e-9);
}

#[test]
fn participation_ratio_perfect_diversity_is_above_one() {
    let m = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let pr = participation_ratio(&m);
    assert!((pr - 2.0).abs() < 1e-6, "expected PR=2.0, got {pr:.6}");
}

#[test]
fn participation_ratio_rank_one_returns_one() {
    let m = vec![
        vec![1.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0],
    ];
    let pr = participation_ratio(&m);
    assert!((pr - 1.0).abs() < 1e-6, "expected PR=1.0, got {pr:.6}");
}

#[test]
fn participation_ratio_empty_columns_returns_one() {
    // Row exists but has no columns → n_cols = 0 → 1.0.
    let m = vec![vec![]];
    assert!((participation_ratio(&m) - 1.0).abs() < 1e-9);
}

// ── Path B: unambiguously Coverage ────────────────────────────────────────────

#[tokio::test]
async fn unambiguous_coverage_path_b_routes_coverage() {
    let mut cfg = H2AIConfig::default().task_complexity;
    // Force all TCC values above coverage threshold so Path B fires.
    cfg.tcc_precision_threshold = -1.0;
    cfg.tcc_coverage_threshold = 0.5;
    let cal = dummy_calibration();
    // Empty corpus → TCC_structural = 1.0 ≥ 0.5 → Path B.
    let result = assess_task_complexity(&[], &cal, &cfg, TaskId::new(), None).await;
    assert_eq!(result.task_quadrant, TaskQuadrant::Coverage);
    assert_eq!(
        result.probe_skip_reason,
        ProbeSkipReason::UnambiguousCoverage
    );
    assert!(result.probe_skipped);
}

// ── classify_quadrant — ambiguous band with poor pool ─────────────────────────

#[test]
fn classify_quadrant_ambiguous_band_poor_pool_is_complex() {
    let cfg = H2AIConfig::default().task_complexity;
    // TCC in ambiguous band (2.0 < 2.2 < 2.5), n_eff below n_eff_complex_threshold (1.3).
    let q = classify_quadrant(2.2, Some(1.0), &cfg);
    assert_eq!(q, TaskQuadrant::Complex);
}

// ── run_probe: all probes fail → fallback to structural TCC ───────────────────

#[tokio::test]
async fn run_probe_all_probes_fail_returns_structural_tcc() {
    let mut cfg = H2AIConfig::default().task_complexity;
    cfg.n_probe = 2;
    cfg.tcc_precision_threshold = -1.0;
    cfg.tcc_coverage_threshold = 100.0;
    cfg.min_static_coverage_for_probe = 0.0;

    let adapter = FailAdapter::new();
    let cal = dummy_calibration();
    let corpus = vec![vocab_constraint("c1")];

    let result = assess_task_complexity(
        &corpus,
        &cal,
        &cfg,
        TaskId::new(),
        Some((&adapter, "system")),
    )
    .await;

    // All probes failed → probe_outputs is empty → fallback to structural TCC.
    assert!(!result.probe_skipped);
    assert_eq!(result.probe_skip_reason, ProbeSkipReason::None);
    assert!(result.tcc_empirical.is_none());
}

// ── run_probe: probes agree on all constraints → n_informative too low ────────

#[tokio::test]
async fn run_probe_all_probes_agree_uses_amplified_tcc() {
    use h2ai_adapters::MockAdapter;

    let mut cfg = H2AIConfig::default().task_complexity;
    cfg.n_probe = 3;
    cfg.tcc_precision_threshold = -1.0;
    cfg.tcc_coverage_threshold = 100.0;
    cfg.min_static_coverage_for_probe = 0.0;
    cfg.tcc_min_informative_constraints = 2;
    cfg.k_heavy = 0.5;

    // All 3 probes return "present stateless" → all VocabularyPresence("stateless") pass.
    let adapter = MockAdapter::new("present stateless".into());
    let cal = dummy_calibration();
    // Single Static constraint: all probes agree it passes → n_informative = 0 < 2.
    let corpus = vec![vocab_constraint("c1")];

    let result = assess_task_complexity(
        &corpus,
        &cal,
        &cfg,
        TaskId::new(),
        Some((&adapter, "system")),
    )
    .await;

    assert!(!result.probe_skipped);
    assert!(
        result.tcc_empirical.is_none(),
        "degenerate probe → no empirical TCC"
    );
    assert!(result.tcc_effective > 0.0);
}

// ── run_probe: mixed results → success path with tcc_empirical ────────────────

#[tokio::test]
async fn run_probe_mixed_results_computes_tcc_empirical() {
    use h2ai_adapters::SequencedMockAdapter;

    let mut cfg = H2AIConfig::default().task_complexity;
    cfg.n_probe = 3;
    cfg.tcc_precision_threshold = -1.0;
    cfg.tcc_coverage_threshold = 100.0;
    cfg.min_static_coverage_for_probe = 0.0;
    cfg.tcc_min_informative_constraints = 2;

    // Probe 1 satisfies "alpha" but not "beta".
    // Probe 2 satisfies "beta" but not "alpha".
    // Probe 3 satisfies neither.
    // Both constraints are informative (split across probes) → n_informative = 2 ≥ 2.
    let adapter = SequencedMockAdapter::new(vec!["alpha".into(), "beta".into(), "neither".into()]);

    let cal = dummy_calibration();
    let make_vocab = |id: &str, term: &str| ConstraintDoc {
        id: id.into(),
        source_file: "test".into(),
        description: "vocab".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.9 },
        predicate: ConstraintPredicate::VocabularyPresence {
            mode: VocabularyMode::AllOf,
            terms: vec![term.into()],
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    };
    let corpus = vec![make_vocab("ca", "alpha"), make_vocab("cb", "beta")];

    let result = assess_task_complexity(
        &corpus,
        &cal,
        &cfg,
        TaskId::new(),
        Some((&adapter, "system")),
    )
    .await;

    assert!(!result.probe_skipped);
    assert!(
        result.tcc_empirical.is_some(),
        "mixed probes must yield empirical TCC"
    );
    assert!(result.probe_cost_tokens > 0);
}

// ── run_probe direct invocation: ProbeInput ───────────────────────────────────

#[tokio::test]
async fn run_probe_direct_empty_static_corpus_degenerate() {
    use h2ai_adapters::MockAdapter;

    let mut cfg = H2AIConfig::default().task_complexity;
    cfg.n_probe = 2;
    cfg.tcc_min_informative_constraints = 1;
    cfg.k_heavy = 0.5;

    let adapter = MockAdapter::new("output".into());
    let result = run_probe(ProbeInput {
        meta_tcc_structural: 1.5,
        meta_heavy_fraction: 0.2,
        static_corpus: &[],
        n_eff_pool: None,
        cfg: &cfg,
        task_id: TaskId::new(),
        adapter: &adapter,
        system_context: "system",
    })
    .await;

    // Empty static corpus → n_informative = 0 < tcc_min_informative_constraints (1).
    assert!(result.tcc_empirical.is_none());
    assert!(!result.probe_skipped);
}
