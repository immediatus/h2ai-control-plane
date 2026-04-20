use h2ai_config::{ConfigLoadError, H2AIConfig};
use std::io::Write;

// ── defaults ─────────────────────────────────────────────────────────────────

#[test]
fn default_j_eff_gate_is_0_4() {
    assert_eq!(H2AIConfig::default().j_eff_gate, 0.4);
}

#[test]
fn default_bft_threshold_is_0_85() {
    assert_eq!(H2AIConfig::default().bft_threshold, 0.85);
}

#[test]
fn default_coordination_threshold_max_is_0_3() {
    assert_eq!(H2AIConfig::default().coordination_threshold_max, 0.3);
}

#[test]
fn default_min_baseline_competence_is_0_5() {
    assert_eq!(H2AIConfig::default().min_baseline_competence, 0.5);
}

#[test]
fn default_max_error_correlation_is_0_9() {
    assert_eq!(H2AIConfig::default().max_error_correlation, 0.9);
}

#[test]
fn default_role_tau_values() {
    let c = H2AIConfig::default();
    assert_eq!(c.tau_coordinator, 0.05);
    assert_eq!(c.tau_executor, 0.40);
    assert_eq!(c.tau_evaluator, 0.10);
    assert_eq!(c.tau_synthesizer, 0.80);
}

#[test]
fn default_role_error_cost_values() {
    let c = H2AIConfig::default();
    assert_eq!(c.cost_coordinator, 0.1);
    assert_eq!(c.cost_executor, 0.5);
    assert_eq!(c.cost_evaluator, 0.9);
    assert_eq!(c.cost_synthesizer, 0.1);
}

// ── JSON serialisation ────────────────────────────────────────────────────────

#[test]
fn config_serializes_to_json_with_expected_field_names() {
    let json = serde_json::to_string(&H2AIConfig::default()).unwrap();
    assert!(json.contains("j_eff_gate"));
    assert!(json.contains("bft_threshold"));
    assert!(json.contains("tau_coordinator"));
}

#[test]
fn config_round_trips_through_json() {
    let original = H2AIConfig::default();
    let json = serde_json::to_string(&original).unwrap();
    let restored: H2AIConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.j_eff_gate, original.j_eff_gate);
    assert_eq!(restored.bft_threshold, original.bft_threshold);
    assert_eq!(restored.tau_synthesizer, original.tau_synthesizer);
    assert_eq!(restored.cost_evaluator, original.cost_evaluator);
}

#[test]
fn config_override_single_field_from_json() {
    let json = r#"{"j_eff_gate":0.5,"bft_threshold":0.85,"coordination_threshold_max":0.3,
        "min_baseline_competence":0.5,"max_error_correlation":0.9,
        "tau_coordinator":0.05,"tau_executor":0.40,"tau_evaluator":0.10,"tau_synthesizer":0.80,
        "cost_coordinator":0.1,"cost_executor":0.5,"cost_evaluator":0.9,"cost_synthesizer":0.1}"#;
    let cfg: H2AIConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.j_eff_gate, 0.5);
    assert_eq!(cfg.bft_threshold, 0.85); // unchanged
}

// ── file load ─────────────────────────────────────────────────────────────────

#[test]
fn load_from_file_returns_config_with_overridden_values() {
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    let json = serde_json::to_string(&H2AIConfig {
        j_eff_gate: 0.6,
        ..H2AIConfig::default()
    })
    .unwrap();
    write!(tmp, "{json}").unwrap();

    let cfg = H2AIConfig::load_from_file(tmp.path()).unwrap();
    assert_eq!(cfg.j_eff_gate, 0.6);
    assert_eq!(cfg.bft_threshold, 0.85);
}

#[test]
fn load_from_file_returns_error_for_missing_file() {
    let result = H2AIConfig::load_from_file("/nonexistent/path/config.json".as_ref());
    assert!(matches!(result, Err(ConfigLoadError::Io(_))));
}

#[test]
fn load_from_file_returns_error_for_invalid_json() {
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    write!(tmp, "not valid json").unwrap();
    let result = H2AIConfig::load_from_file(tmp.path());
    assert!(matches!(result, Err(ConfigLoadError::Parse(_))));
}
