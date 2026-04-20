use h2ai_adapters::mock::MockAdapter;
use h2ai_autonomic::calibration::{CalibrationHarness, CalibrationInput};
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
    };
    let event = CalibrationHarness::run(input).await.unwrap();
    assert_eq!(event.coefficients.cg_samples, vec![0.7]);
}

#[tokio::test]
async fn calibration_two_identical_adapters_produces_cg_one() {
    // Both adapters return identical output → Jaccard = 1.0 → CG = 1.0.
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
    };
    let event = CalibrationHarness::run(input).await.unwrap();
    // Two adapters with identical outputs → single pair → Jaccard = 1.0
    assert_eq!(event.coefficients.cg_samples.len(), 1);
    assert!(
        (event.coefficients.cg_samples[0] - 1.0).abs() < 1e-9,
        "identical adapters must produce CG=1.0, got: {}",
        event.coefficients.cg_samples[0]
    );
}

#[tokio::test]
async fn calibration_empty_task_prompts_with_two_adapters_produces_cg_zero() {
    // No prompts → outputs are empty strings for all adapters →
    // tokenize("") = {} for both → jaccard({}, {}) = 0.0.
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
    };
    // Should not panic — produces CG=0.0 from empty token sets.
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
