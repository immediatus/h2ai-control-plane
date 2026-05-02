use h2ai_config::{ConfigLoadError, H2AIConfig, SchedulerPolicy};
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
fn default_min_baseline_competence_is_0_3() {
    assert_eq!(H2AIConfig::default().min_baseline_competence, 0.3);
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

// ── serde alias ───────────────────────────────────────────────────────────────

#[test]
fn beta_base_default_loads_from_kappa_eff_factor_alias() {
    // Serialize a complete config, swap the field name to the legacy alias, then round-trip.
    let mut v: serde_json::Value = serde_json::to_value(&H2AIConfig::default()).unwrap();
    let obj = v.as_object_mut().unwrap();
    obj.remove("beta_base_default");
    obj.insert("kappa_eff_factor".into(), serde_json::json!(0.019));
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert!(
        (cfg.beta_base_default - 0.019).abs() < 1e-10,
        "kappa_eff_factor alias must deserialize into beta_base_default, got {}",
        cfg.beta_base_default
    );
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

// ── load_layered ──────────────────────────────────────────────────────────────

#[test]
fn load_layered_no_override_matches_default() {
    let from_layered = H2AIConfig::load_layered(None).unwrap();
    let from_default = H2AIConfig::default();
    assert_eq!(from_layered.j_eff_gate, from_default.j_eff_gate);
    assert_eq!(from_layered.bft_threshold, from_default.bft_threshold);
    assert_eq!(from_layered.tau_synthesizer, from_default.tau_synthesizer);
    assert_eq!(
        from_layered.beta_base_default,
        from_default.beta_base_default
    );
    assert_eq!(from_layered.scheduler_policy, from_default.scheduler_policy);
}

#[test]
fn load_layered_override_changes_only_specified_field() {
    let mut tmp = tempfile::NamedTempFile::with_suffix(".toml").unwrap();
    write!(tmp, "j_eff_gate = 0.7\n").unwrap();

    let cfg = H2AIConfig::load_layered(Some(tmp.path())).unwrap();
    assert!(
        (cfg.j_eff_gate - 0.7).abs() < 1e-10,
        "override must apply, got {}",
        cfg.j_eff_gate
    );
    assert_eq!(
        cfg.bft_threshold,
        H2AIConfig::default().bft_threshold,
        "non-overridden field must fall through to reference"
    );
    assert_eq!(
        cfg.scheduler_policy,
        SchedulerPolicy::CostAwareSpillover,
        "scheduler_policy must fall through to reference default"
    );
}

/// RAII guard: sets an env var on construction, removes it on drop (even on panic).
struct EnvGuard(&'static str);
impl EnvGuard {
    fn set(key: &'static str, val: &str) -> Self {
        std::env::set_var(key, val);
        Self(key)
    }
}
impl Drop for EnvGuard {
    fn drop(&mut self) {
        std::env::remove_var(self.0);
    }
}

#[test]
fn load_layered_env_var_wins_over_file() {
    // Use max_autonomic_retries — not asserted by any other test, so parallel races are harmless.
    let mut tmp = tempfile::NamedTempFile::with_suffix(".toml").unwrap();
    write!(tmp, "max_autonomic_retries = 5\n").unwrap();

    let _guard = EnvGuard::set("H2AI__MAX_AUTONOMIC_RETRIES", "99");
    let cfg = H2AIConfig::load_layered(Some(tmp.path())).unwrap();

    assert_eq!(
        cfg.max_autonomic_retries, 99,
        "env var must beat the override file, got {}",
        cfg.max_autonomic_retries
    );
}

#[test]
fn load_layered_missing_override_path_returns_error() {
    let result = H2AIConfig::load_layered(Some("/nonexistent/override.toml".as_ref()));
    assert!(
        matches!(result, Err(ConfigLoadError::Config(_))),
        "missing override path must return ConfigLoadError::Config"
    );
}
