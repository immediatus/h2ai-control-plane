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
    assert!(event.coefficients.kappa_base > 0.0);
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
