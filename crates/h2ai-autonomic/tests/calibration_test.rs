use h2ai_adapters::mock::MockAdapter;
use h2ai_autonomic::calibration::{beta_from_merge_spans, CalibrationHarness, CalibrationInput};
use h2ai_config::H2AIConfig;
use h2ai_types::identity::TaskId;

fn mock_adapter() -> MockAdapter {
    MockAdapter::new("The proposed solution uses stateless JWT auth.".into())
}

#[tokio::test]
async fn calibration_produces_valid_coefficients() {
    let adapter = mock_adapter();
    let cfg = H2AIConfig::default();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec![
            "Describe a stateless auth strategy".into(),
            "What makes a good API design?".into(),
            "Explain event sourcing".into(),
        ],
        adapters: vec![&adapter as &dyn h2ai_types::adapter::IComputeAdapter],
        cfg: &cfg,
        constraint_corpus: &[],
    };
    let event = CalibrationHarness::run(input).await.unwrap();
    assert!(event.coefficients.alpha >= 0.0 && event.coefficients.alpha < 1.0);
    assert!(event.coefficients.beta_base > 0.0);
    assert!(!event.coefficients.cg_samples.is_empty());
    assert!(event.coordination_threshold.value() <= 0.3);
}

#[tokio::test]
async fn calibration_single_adapter_uses_default_cg() {
    let adapter = mock_adapter();
    let cfg = H2AIConfig::default();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["Single prompt".into()],
        adapters: vec![&adapter as &dyn h2ai_types::adapter::IComputeAdapter],
        cfg: &cfg,
        constraint_corpus: &[],
    };
    let event = CalibrationHarness::run(input).await.unwrap();
    assert_eq!(event.coefficients.cg_samples, vec![0.7]);
}

#[tokio::test]
async fn calibration_two_adapters_empty_corpus_uses_fallback_cg() {
    // Two adapters with empty corpus → single pair → fallback CG = cfg.calibration_cg_fallback
    let a = mock_adapter();
    let b = mock_adapter(); // same output
    let cfg = H2AIConfig::default();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["Prompt one".into(), "Prompt two".into()],
        adapters: vec![
            &a as &dyn h2ai_types::adapter::IComputeAdapter,
            &b as &dyn h2ai_types::adapter::IComputeAdapter,
        ],
        cfg: &cfg,
        constraint_corpus: &[],
    };
    let event = CalibrationHarness::run(input).await.unwrap();
    // Two adapters with empty corpus → single pair → fallback CG = cfg.calibration_cg_fallback
    assert_eq!(event.coefficients.cg_samples.len(), 1);
    let expected_fallback = cfg.calibration_cg_fallback; // default: 0.7
    assert!(
        (event.coefficients.cg_samples[0] - expected_fallback).abs() < 1e-9,
        "empty-corpus calibration must fall back to cg_fallback={expected_fallback}, got: {}",
        event.coefficients.cg_samples[0]
    );
}

#[tokio::test]
async fn calibration_empty_task_prompts_with_two_adapters_produces_cg_zero() {
    // No prompts → outputs_i.is_empty() → triggers fallback path.
    let a = mock_adapter();
    let b = mock_adapter();
    let cfg = H2AIConfig::default();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec![], // no prompts
        adapters: vec![
            &a as &dyn h2ai_types::adapter::IComputeAdapter,
            &b as &dyn h2ai_types::adapter::IComputeAdapter,
        ],
        cfg: &cfg,
        constraint_corpus: &[],
    };
    // Should not panic — produces CG from fallback path.
    let event = CalibrationHarness::run(input).await.unwrap();
    assert_eq!(event.coefficients.cg_samples.len(), 1);
    assert!(
        event.coefficients.cg_samples[0] >= 0.0,
        "CG must be non-negative"
    );
}

#[tokio::test]
async fn calibration_zero_adapters_returns_no_adapters_error() {
    use h2ai_autonomic::calibration::CalibrationError;
    let cfg = H2AIConfig::default();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["any prompt".into()],
        adapters: vec![],
        cfg: &cfg,
        constraint_corpus: &[],
    };
    let result = CalibrationHarness::run(input).await;
    assert!(
        matches!(result, Err(CalibrationError::NoAdapters)),
        "zero adapters must return NoAdapters error, got {:?}",
        result.err()
    );
}

#[tokio::test]
async fn calibration_two_adapters_populates_ensemble() {
    let cfg = H2AIConfig::default();
    let a1 = MockAdapter::new("alpha beta gamma delta".into());
    let a2 = MockAdapter::new("delta epsilon zeta omega".into());
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["test prompt".into()],
        adapters: vec![
            &a1 as &dyn h2ai_types::adapter::IComputeAdapter,
            &a2 as &dyn h2ai_types::adapter::IComputeAdapter,
        ],
        cfg: &cfg,
        constraint_corpus: &[],
    };
    let event = CalibrationHarness::run(input).await.unwrap();
    let ec = event
        .ensemble
        .expect("ensemble should be Some with 2 adapters");
    assert!(
        ec.p_mean > 0.5 && ec.p_mean <= 1.0,
        "p_mean out of range: {}",
        ec.p_mean
    );
    assert!(
        ec.rho_mean >= 0.0 && ec.rho_mean <= 1.0,
        "rho_mean out of range: {}",
        ec.rho_mean
    );
    assert!(
        ec.n_optimal >= 1 && ec.n_optimal <= 9,
        "n_optimal out of range: {}",
        ec.n_optimal
    );
    assert!(
        ec.q_optimal >= ec.p_mean,
        "q_optimal should be >= p_mean: {} < {}",
        ec.q_optimal,
        ec.p_mean
    );
}

#[test]
fn beta_from_merge_spans_derives_correct_value() {
    // 5 proposals → n*(n-1)/2 = 10 modelled pairs; elapsed 0.001s; T₁ = 1.0s
    // β₀ = 0.001 / 10 / 1.0 = 0.0001
    let spans = vec![(0.001f64, 5usize)];
    let result = beta_from_merge_spans(&spans, 1.0);
    let expected = 0.001 / 10.0 / 1.0;
    assert!(result.is_some());
    let b = result.unwrap();
    assert!((b - expected).abs() < 1e-9, "expected {expected}, got {b}");
}

#[test]
fn beta_from_merge_spans_returns_none_for_empty_input() {
    assert!(beta_from_merge_spans(&[], 1.0).is_none());
}

#[test]
fn beta_from_merge_spans_returns_none_when_t1_zero() {
    let spans = vec![(0.1f64, 4usize)];
    assert!(beta_from_merge_spans(&spans, 0.0).is_none());
}

#[test]
fn beta_from_merge_spans_n_one_uses_pairs_guard() {
    // n=1 → n*(n-1)/2 = 0 → max(1, 0) = 1 to avoid division by zero
    let spans = vec![(0.002f64, 1usize)];
    let result = beta_from_merge_spans(&spans, 1.0);
    assert!(result.is_some(), "n=1 must not return None");
    let b = result.unwrap();
    // elapsed / 1 / T1 = 0.002
    assert!((b - 0.002).abs() < 1e-9, "n=1: expected 0.002, got {b}");
}

#[test]
fn beta_from_merge_spans_clamps_to_max() {
    // Extremely slow merge → clamped at 0.1
    // 4 proposals → 6 pairs; 1000s elapsed; T1 = 0.001s → β_raw = 1000/6/0.001 >> 0.1
    let spans = vec![(1000.0f64, 4usize)];
    let result = beta_from_merge_spans(&spans, 0.001);
    assert!(result.is_some());
    assert!((result.unwrap() - 0.1).abs() < 1e-9, "should clamp at 0.1");
}

#[tokio::test]
async fn calibration_harness_m3_populates_ensemble_and_eigen() {
    // Use 3 adapters with distinct outputs. With M=3, the harness executes Phase A
    // (2 adapters) and Phase B (3 adapters) in parallel. MockAdapter timing is
    // sub-nanosecond, so USL fit always falls back to config default α = 0.12.
    // We verify the multi-adapter code path ran by checking ensemble and eigen.
    let a1 = MockAdapter::new("stateless JWT authentication using signed tokens".into());
    let a2 = MockAdapter::new("event sourcing with CQRS separating reads from writes".into());
    let a3 = MockAdapter::new(
        "clean API boundary defined by domain interfaces not implementation".into(),
    );
    let cfg = H2AIConfig::default();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec![
            "Describe a stateless auth approach".into(),
            "Explain CQRS and event sourcing".into(),
            "What is a good API boundary?".into(),
        ],
        adapters: vec![
            &a1 as &dyn h2ai_types::adapter::IComputeAdapter,
            &a2 as &dyn h2ai_types::adapter::IComputeAdapter,
            &a3 as &dyn h2ai_types::adapter::IComputeAdapter,
        ],
        cfg: &cfg,
        constraint_corpus: &[],
    };
    let event = CalibrationHarness::run(input).await.unwrap();

    // Structural assertions: verify multi-adapter code path ran via ensemble/eigen.
    assert!(
        event.ensemble.is_some(),
        "ensemble must be Some with 3 adapters"
    );
    assert!(event.eigen.is_some(), "eigen must be Some with 3 adapters");
    // 3 adapters → C(3,2) = 3 pairwise CG samples.
    assert_eq!(
        event.coefficients.cg_samples.len(),
        3,
        "3 adapters must produce 3 pairwise CG samples"
    );
}

#[tokio::test]
async fn calibration_accepts_empty_corpus() {
    let a = mock_adapter();
    let cfg = H2AIConfig::default();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["Prompt".into()],
        adapters: vec![&a as &dyn h2ai_types::adapter::IComputeAdapter],
        cfg: &cfg,
        constraint_corpus: &[],
    };
    assert!(CalibrationHarness::run(input).await.is_ok());
}

#[tokio::test]
async fn calibration_non_empty_corpus_computes_hamming() {
    // Adapter A always returns "jwt auth" → VocabularyPresence("jwt") → true (score=1.0 >= 0.5)
    // Adapter B always returns "session cookie" → VocabularyPresence("jwt") → false (score=0.0 < 0.5)
    // One Hard constraint → Hamming distance = 1.0 → CG = 1.0 * align
    use h2ai_constraints::types::{
        ConstraintDoc, ConstraintPredicate, ConstraintSeverity, VocabularyMode,
    };
    let corpus = vec![ConstraintDoc {
        id: "auth-jwt".into(),
        source_file: "test".into(),
        description: "must use jwt".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::VocabularyPresence {
            mode: VocabularyMode::AnyOf,
            terms: vec!["jwt".into()],
        },
        remediation_hint: None,
    }];
    let a = MockAdapter::new("jwt authentication token".into());
    let b = MockAdapter::new("session cookie storage".into());
    let cfg = H2AIConfig::default();
    let input = CalibrationInput {
        calibration_id: TaskId::new(),
        task_prompts: vec!["auth strategy".into()],
        adapters: vec![
            &a as &dyn h2ai_types::adapter::IComputeAdapter,
            &b as &dyn h2ai_types::adapter::IComputeAdapter,
        ],
        cfg: &cfg,
        constraint_corpus: &corpus,
    };
    let event = CalibrationHarness::run(input).await.unwrap();
    assert_eq!(event.coefficients.cg_samples.len(), 1);
    // Opposite fingerprints → Hamming = 1.0 → CG ≈ 1.0 (tau_alignment at equal taus = 1.0)
    assert!(
        (event.coefficients.cg_samples[0] - 1.0).abs() < 1e-9,
        "opposite constraint profiles must produce CG=1.0, got: {}",
        event.coefficients.cg_samples[0]
    );
}

#[tokio::test]
async fn cg_mode_is_constraint_profile() {
    use h2ai_types::events::CgMode;
    let mode = CgMode::default();
    assert!(matches!(mode, CgMode::ConstraintProfile));
    let json = serde_json::to_string(&mode).unwrap();
    let back: CgMode = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, CgMode::ConstraintProfile));
}
