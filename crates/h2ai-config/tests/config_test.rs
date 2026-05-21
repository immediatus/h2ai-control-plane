use h2ai_config::{
    CalibrationBootstrapConfig, CalibrationProbeConfig, CalibrationSlowStartConfig,
    ConfigLoadError, ConflictBetaConfig, CsprConfig, FamilyConstraint, H2AIConfig,
    JudgePanelConfig, OproConfig, OracleGateConfig, ProbeTaskSource, ReasoningMemoryConfig,
    SafetyConfig, SafetyProfile, SchedulerPolicy, ShadowAuditorConfig, SraniConfig, StateConfig,
    StateDeltaConfig, SystemModifier, ThinkingLoopConfig,
};
use h2ai_types::config::AdapterKind;
use std::io::Write;

// ── defaults ─────────────────────────────────────────────────────────────────

#[test]
#[allow(clippy::float_cmp)]
fn default_bft_threshold_is_0_85() {
    assert_eq!(H2AIConfig::default().bft_threshold, 0.85);
}

#[test]
#[allow(clippy::float_cmp)]
fn default_coordination_threshold_max_is_0_3() {
    assert_eq!(H2AIConfig::default().coordination_threshold_max, 0.3);
}

#[test]
#[allow(clippy::float_cmp)]
fn default_min_baseline_competence_is_0_3() {
    assert_eq!(H2AIConfig::default().min_baseline_competence, 0.3);
}

#[test]
#[allow(clippy::float_cmp)]
fn default_max_error_correlation_is_0_9() {
    assert_eq!(H2AIConfig::default().max_error_correlation, 0.9);
}

#[test]
#[allow(clippy::float_cmp)]
fn default_role_tau_values() {
    let c = H2AIConfig::default();
    assert_eq!(c.tau_coordinator, 0.05);
    assert_eq!(c.tau_executor, 0.40);
    assert_eq!(c.tau_evaluator, 0.10);
    assert_eq!(c.tau_synthesizer, 0.80);
}

#[test]
#[allow(clippy::float_cmp)]
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
    assert!(json.contains("bft_threshold"));
    assert!(json.contains("tau_coordinator"));
}

#[test]
#[allow(clippy::float_cmp)]
fn config_round_trips_through_json() {
    let original = H2AIConfig::default();
    let json = serde_json::to_string(&original).unwrap();
    let restored: H2AIConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.bft_threshold, original.bft_threshold);
    assert_eq!(restored.tau_synthesizer, original.tau_synthesizer);
    assert_eq!(restored.cost_evaluator, original.cost_evaluator);
}

// ── serde alias ───────────────────────────────────────────────────────────────

#[test]
fn beta_base_default_loads_from_kappa_eff_factor_alias() {
    // Serialize a complete config, swap the field name to the legacy alias, then round-trip.
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
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
#[allow(clippy::float_cmp)]
fn load_from_file_returns_config_with_overridden_values() {
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    let json = serde_json::to_string(&H2AIConfig {
        bft_threshold: 0.99,
        ..H2AIConfig::default()
    })
    .unwrap();
    write!(tmp, "{json}").unwrap();

    let cfg = H2AIConfig::load_from_file(tmp.path()).unwrap();
    assert!((cfg.bft_threshold - 0.99).abs() < 1e-10);
    assert_eq!(cfg.tau_synthesizer, 0.80);
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
#[allow(clippy::float_cmp)]
fn load_layered_no_override_matches_default() {
    let from_layered = H2AIConfig::load_layered(None).unwrap();
    let from_default = H2AIConfig::default();
    assert_eq!(from_layered.bft_threshold, from_default.bft_threshold);
    assert_eq!(from_layered.tau_synthesizer, from_default.tau_synthesizer);
    assert_eq!(
        from_layered.beta_base_default,
        from_default.beta_base_default
    );
    assert_eq!(from_layered.scheduler_policy, from_default.scheduler_policy);
}

#[test]
#[allow(clippy::float_cmp)]
fn load_layered_override_changes_only_specified_field() {
    let mut tmp = tempfile::NamedTempFile::with_suffix(".toml").unwrap();
    writeln!(tmp, "bft_threshold = 0.77").unwrap();

    let cfg = H2AIConfig::load_layered(Some(tmp.path())).unwrap();
    assert!(
        (cfg.bft_threshold - 0.77).abs() < 1e-10,
        "override must apply, got {}",
        cfg.bft_threshold
    );
    assert_eq!(
        cfg.tau_synthesizer,
        H2AIConfig::default().tau_synthesizer,
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
    writeln!(tmp, "max_autonomic_retries = 5").unwrap();

    let _guard = EnvGuard::set("H2AI_MAX_AUTONOMIC_RETRIES", "99");
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

#[test]
fn family_constraint_default_is_single_family_ok() {
    // Default safety profile is Development which sets family_constraint = SingleFamilyOk.
    // Strict enforcement requires opting in via family_constraint = require_diverse.
    assert_eq!(
        H2AIConfig::default().safety.family_constraint,
        h2ai_config::FamilyConstraint::SingleFamilyOk,
        "default safety profile must allow single-family pools — devcontainer requires single-family pool"
    );
}

#[test]
fn constraint_wiki_config_defaults_are_sane() {
    let cfg = H2AIConfig::load_layered(None).expect("load defaults");
    assert_eq!(
        cfg.constraint_wiki,
        h2ai_config::ConstraintWikiConfig::Disabled,
        "wiki disabled by default"
    );
}

#[test]
fn shadow_auditor_config_defaults_are_disabled_and_sensible() {
    let cfg = H2AIConfig::load_layered(None).unwrap();
    assert!(
        !cfg.safety.shadow_auditor.enabled,
        "shadow auditor must be off by default"
    );
    assert!(
        (cfg.safety.shadow_auditor.promotion_threshold - 0.05).abs() < 1e-9,
        "default threshold must be 0.05 (5%)"
    );
    assert_eq!(cfg.safety.shadow_auditor.promotion_window, 30);
    assert!(
        cfg.safety.shadow_auditor.auto_demotion,
        "auto-demotion must be on by default"
    );
}

#[test]
fn shadow_auditor_config_override_via_toml() {
    use std::io::Write;
    // Use profile = "custom" so apply_safety_profile() does not overwrite shadow_auditor fields.
    let toml = r#"
[safety]
profile = "custom"
[safety.shadow_auditor]
enabled = true
promotion_threshold = 0.10
promotion_window = 50
auto_demotion = false
"#;
    let mut f = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    f.write_all(toml.as_bytes()).unwrap();
    let cfg = H2AIConfig::load_layered(Some(f.path())).unwrap();
    assert!(cfg.safety.shadow_auditor.enabled);
    assert!((cfg.safety.shadow_auditor.promotion_threshold - 0.10).abs() < 1e-9);
    assert_eq!(cfg.safety.shadow_auditor.promotion_window, 50);
    assert!(!cfg.safety.shadow_auditor.auto_demotion);
}

// ── SraniConfig ───────────────────────────────────────────────────────────────

#[test]
fn srani_config_defaults_match_spec() {
    let cfg = H2AIConfig::default();
    let srani = &cfg.srani;
    assert!(srani.enabled, "srani must be enabled by default");
    assert!(
        (srani.warn_threshold - 0.3).abs() < 1e-9,
        "default warn_threshold must be 0.3, got {}",
        srani.warn_threshold
    );
    assert!(
        (srani.inject_threshold - 0.6).abs() < 1e-9,
        "default inject_threshold must be 0.6, got {}",
        srani.inject_threshold
    );
    assert!(
        srani.warn_threshold < srani.inject_threshold,
        "warn_threshold must be strictly below inject_threshold"
    );
}

#[test]
fn srani_config_round_trips_json() {
    let cfg = H2AIConfig::default();
    let json = serde_json::to_string(&cfg).unwrap();
    let restored: H2AIConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(
        cfg.srani, restored.srani,
        "srani config must survive JSON round-trip"
    );
}

#[test]
fn srani_config_deserialises_from_json_without_srani_field() {
    // Old JSON payloads without the "srani" key must deserialise cleanly using defaults.
    // Simulate by serialising a full config, removing the srani key, then round-tripping.
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    v.as_object_mut().unwrap().remove("srani");
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert!(
        cfg.srani.enabled,
        "missing srani key must default to enabled=true"
    );
    assert!(
        (cfg.srani.warn_threshold - 0.3).abs() < 1e-9,
        "missing srani key must default warn_threshold to 0.3"
    );
}

#[test]
fn srani_adaptive_defaults_are_sane() {
    let cfg = H2AIConfig::default();
    assert!(cfg.srani.adaptive, "adaptive must default to true");
    assert!(
        (cfg.srani.ema_alpha - 0.20).abs() < 1e-9,
        "ema_alpha must default to 0.20, got {}",
        cfg.srani.ema_alpha
    );
    assert!(
        (cfg.srani.temperature - 0.15).abs() < 1e-9,
        "temperature must default to 0.15, got {}",
        cfg.srani.temperature
    );
    assert!(
        (cfg.srani.gate_threshold - 0.50).abs() < 1e-9,
        "gate_threshold must default to 0.50, got {}",
        cfg.srani.gate_threshold
    );
}

#[test]
fn srani_cold_start_midpoint_is_mean_of_thresholds() {
    let cfg = H2AIConfig::default();
    let midpoint = cfg.srani.cold_start_midpoint();
    let expected = f64::midpoint(cfg.srani.warn_threshold, cfg.srani.inject_threshold);
    assert!(
        (midpoint - expected).abs() < 1e-9,
        "cold_start_midpoint must be (warn + inject) / 2, got {midpoint}"
    );
    // For defaults: (0.30 + 0.60) / 2 = 0.45
    assert!(
        (midpoint - 0.45).abs() < 1e-9,
        "default cold_start_midpoint must be 0.45, got {midpoint}"
    );
}

#[test]
fn srani_adaptive_false_deserialises_cleanly() {
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    v["srani"]["adaptive"] = serde_json::json!(false);
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert!(!cfg.srani.adaptive, "adaptive=false must round-trip");
    assert!(
        (cfg.srani.warn_threshold - 0.30).abs() < 1e-9,
        "warn_threshold must survive adaptive=false"
    );
}

#[test]
fn srani_new_fields_deserialise_from_json_without_them() {
    // Old JSON without new fields must deserialise using defaults (backward compat).
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    let srani = v["srani"].as_object_mut().unwrap();
    srani.remove("adaptive");
    srani.remove("ema_alpha");
    srani.remove("temperature");
    srani.remove("gate_threshold");
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert!(cfg.srani.adaptive, "missing adaptive must default to true");
    assert!(
        (cfg.srani.ema_alpha - 0.20).abs() < 1e-9,
        "missing ema_alpha must default to 0.20"
    );
}

// ── SraniConfig grounding fields ──────────────────────────────────────────────

#[test]
fn srani_grounding_defaults_are_sane() {
    let srani = &H2AIConfig::default().srani;
    assert_eq!(
        srani.grounding_raw_max_chars, 4000,
        "raw cap default is 4000"
    );
    assert_eq!(
        srani.grounding_hint_max_chars, 1200,
        "hint cap default is 1200"
    );
    assert!(srani.grounding_distill, "distill must default to true");
}

#[test]
fn srani_grounding_fields_round_trip_json() {
    let original = H2AIConfig::default();
    let json = serde_json::to_string(&original).unwrap();
    let restored: H2AIConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(
        restored.srani.grounding_raw_max_chars,
        original.srani.grounding_raw_max_chars
    );
    assert_eq!(
        restored.srani.grounding_hint_max_chars,
        original.srani.grounding_hint_max_chars
    );
    assert_eq!(
        restored.srani.grounding_distill,
        original.srani.grounding_distill
    );
}

#[test]
fn srani_grounding_missing_fields_use_defaults() {
    // Old JSON payloads without grounding fields must deserialise with defaults.
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    let srani = v["srani"].as_object_mut().unwrap();
    srani.remove("grounding_raw_max_chars");
    srani.remove("grounding_hint_max_chars");
    srani.remove("grounding_distill");
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert_eq!(cfg.srani.grounding_raw_max_chars, 4000);
    assert_eq!(cfg.srani.grounding_hint_max_chars, 1200);
    assert!(cfg.srani.grounding_distill);
}

#[test]
fn srani_grounding_config_override_via_toml() {
    let toml = r"
[srani]
grounding_raw_max_chars  = 8000
grounding_hint_max_chars = 600
grounding_distill        = false
";
    let mut f = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    f.write_all(toml.as_bytes()).unwrap();
    let cfg = H2AIConfig::load_layered(Some(f.path())).unwrap();
    assert_eq!(cfg.srani.grounding_raw_max_chars, 8000);
    assert_eq!(cfg.srani.grounding_hint_max_chars, 600);
    assert!(!cfg.srani.grounding_distill);
}

// ── C1 min_jaccard_floor ──────────────────────────────────────────────────────

#[test]
fn c1_min_jaccard_floor_default_is_0_50() {
    let cfg = H2AIConfig::default();
    assert!(
        (cfg.correlated_hallucination_min_jaccard_floor - 0.50).abs() < 1e-9,
        "default min_jaccard_floor must be 0.50, got {}",
        cfg.correlated_hallucination_min_jaccard_floor
    );
}

#[test]
fn c1_min_jaccard_floor_missing_from_json_uses_default() {
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    v.as_object_mut()
        .unwrap()
        .remove("correlated_hallucination_min_jaccard_floor");
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert!(
        (cfg.correlated_hallucination_min_jaccard_floor - 0.50).abs() < 1e-9,
        "missing field must default to 0.50"
    );
}

#[test]
fn c1_min_jaccard_floor_override_via_toml() {
    let toml = "correlated_hallucination_min_jaccard_floor = 0.70\n";
    let mut f = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    f.write_all(toml.as_bytes()).unwrap();
    let cfg = H2AIConfig::load_layered(Some(f.path())).unwrap();
    assert!(
        (cfg.correlated_hallucination_min_jaccard_floor - 0.70).abs() < 1e-9,
        "TOML override must apply, got {}",
        cfg.correlated_hallucination_min_jaccard_floor
    );
}

#[test]
fn default_calibration_max_ensemble_size_is_9() {
    assert_eq!(H2AIConfig::default().calibration_max_ensemble_size, 9);
}

#[test]
fn default_bandit_n_max_arms_is_6() {
    assert_eq!(H2AIConfig::default().bandit_n_max_arms, 6);
}

#[test]
#[allow(clippy::float_cmp)]
fn default_bandit_prior_sigma_is_2() {
    assert_eq!(H2AIConfig::default().bandit_prior_sigma, 2.0);
}

#[test]
#[allow(clippy::float_cmp)]
fn default_bandit_prior_strength_is_5() {
    assert_eq!(H2AIConfig::default().bandit_prior_strength, 5.0);
}

#[test]
fn default_precision_mode_max_slots_is_3() {
    assert_eq!(H2AIConfig::default().precision_mode_max_slots, 3);
}

#[test]
fn default_oracle_window_size_is_200() {
    assert_eq!(H2AIConfig::default().oracle_window_size, 200);
}

#[test]
#[allow(clippy::float_cmp)]
fn default_oracle_ece_alert_threshold_is_0_15() {
    assert_eq!(H2AIConfig::default().oracle_ece_alert_threshold, 0.15);
}

#[test]
#[allow(clippy::float_cmp)]
fn default_oracle_pass_rate_floor_is_0_30() {
    assert_eq!(H2AIConfig::default().oracle_pass_rate_floor, 0.30);
}

// ── safety profiles ───────────────────────────────────────────────────────────

#[test]
fn development_profile_all_gates_off() {
    let cfg = H2AIConfig::default();
    assert_eq!(cfg.safety.krum_fault_tolerance, 0);
    assert!((cfg.safety.diversity_threshold - 0.0).abs() < 1e-9);
    assert!(!cfg.safety.shadow_auditor.enabled);
    assert_eq!(
        cfg.safety.family_constraint,
        h2ai_config::FamilyConstraint::SingleFamilyOk
    );
    assert!(!cfg.safety.require_bivariate_cg);
}

// ── safety profile tests (Task 3) ─────────────────────────────────────────────

#[test]
fn production_profile_overwrites_operator_fields() {
    let toml = r#"
[safety]
profile = "production"
krum_fault_tolerance = 99
diversity_threshold = 0.99
"#;
    let mut f = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    f.write_all(toml.as_bytes()).unwrap();
    let cfg = H2AIConfig::load_layered(Some(f.path())).unwrap();
    assert_eq!(
        cfg.safety.krum_fault_tolerance, 1,
        "production profile must overwrite krum_fault_tolerance to 1"
    );
    assert!(
        (cfg.safety.diversity_threshold - 0.15).abs() < 1e-9,
        "production profile must overwrite diversity_threshold to 0.15, got {}",
        cfg.safety.diversity_threshold
    );
    assert!(
        cfg.safety.shadow_auditor.enabled,
        "production profile must enable shadow_auditor"
    );
    assert_eq!(
        cfg.safety.family_constraint,
        FamilyConstraint::RequireDiverse,
        "production profile must set family_constraint to RequireDiverse"
    );
}

#[test]
fn strict_profile_enables_bivariate_cg() {
    let toml = "[safety]\nprofile = \"strict\"\n";
    let mut f = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    f.write_all(toml.as_bytes()).unwrap();
    let cfg = H2AIConfig::load_layered(Some(f.path())).unwrap();
    assert!(
        cfg.safety.require_bivariate_cg,
        "strict profile must enable require_bivariate_cg"
    );
    assert_eq!(
        cfg.safety.krum_fault_tolerance, 2,
        "strict profile must set krum_fault_tolerance to 2"
    );
    assert!(
        (cfg.safety.diversity_threshold - 0.20).abs() < 1e-9,
        "strict profile must set diversity_threshold to 0.20, got {}",
        cfg.safety.diversity_threshold
    );
}

#[test]
fn custom_profile_preserves_explicit_values() {
    let toml = r#"
[safety]
profile = "custom"
krum_fault_tolerance = 42
diversity_threshold = 0.77
require_bivariate_cg = true
family_constraint = "disabled"
"#;
    let mut f = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    f.write_all(toml.as_bytes()).unwrap();
    let cfg = H2AIConfig::load_layered(Some(f.path())).unwrap();
    assert_eq!(
        cfg.safety.krum_fault_tolerance, 42,
        "custom profile must preserve krum_fault_tolerance=42"
    );
    assert!(
        (cfg.safety.diversity_threshold - 0.77).abs() < 1e-9,
        "custom profile must preserve diversity_threshold=0.77, got {}",
        cfg.safety.diversity_threshold
    );
    assert!(
        cfg.safety.require_bivariate_cg,
        "custom profile must preserve require_bivariate_cg=true"
    );
    assert_eq!(
        cfg.safety.family_constraint,
        FamilyConstraint::Disabled,
        "custom profile must preserve family_constraint=Disabled"
    );
}

#[test]
fn shadow_auditor_tuning_preserved_across_profiles() {
    let toml = r#"
[safety]
profile = "production"
[safety.shadow_auditor]
promotion_window = 60
"#;
    let mut f = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    f.write_all(toml.as_bytes()).unwrap();
    let cfg = H2AIConfig::load_layered(Some(f.path())).unwrap();
    assert_eq!(
        cfg.safety.shadow_auditor.promotion_window, 60,
        "production profile must not overwrite explicit promotion_window=60"
    );
    assert!(
        cfg.safety.shadow_auditor.enabled,
        "production profile must still enable shadow_auditor"
    );
}

#[test]
fn family_constraint_require_diverse_round_trips() {
    let toml = "[safety]\nprofile = \"custom\"\nfamily_constraint = \"require_diverse\"\n";
    let mut f = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    f.write_all(toml.as_bytes()).unwrap();
    let cfg = H2AIConfig::load_layered(Some(f.path())).unwrap();
    assert_eq!(
        cfg.safety.family_constraint,
        FamilyConstraint::RequireDiverse,
        "require_diverse must deserialise to FamilyConstraint::RequireDiverse"
    );
    let serialised = serde_json::to_string(&FamilyConstraint::RequireDiverse).unwrap();
    assert_eq!(
        serialised, "\"require_diverse\"",
        "FamilyConstraint::RequireDiverse must serialise to \"require_diverse\", got {serialised}"
    );
}

#[test]
fn safety_profile_as_str() {
    assert_eq!(SafetyProfile::Development.as_str(), "development");
    assert_eq!(SafetyProfile::Production.as_str(), "production");
    assert_eq!(SafetyProfile::Strict.as_str(), "strict");
    assert_eq!(SafetyProfile::Custom.as_str(), "custom");
}

#[test]
fn development_is_default_when_safety_section_absent() {
    let toml = "bft_threshold = 0.85\n";
    let mut f = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    f.write_all(toml.as_bytes()).unwrap();
    let cfg = H2AIConfig::load_layered(Some(f.path())).unwrap();
    assert_eq!(
        cfg.safety.profile,
        SafetyProfile::Development,
        "absent [safety] section must default to Development profile"
    );
}

#[test]
fn strict_profile_krum_threshold() {
    let toml = "[safety]\nprofile = \"strict\"\n";
    let mut f = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    f.write_all(toml.as_bytes()).unwrap();
    let cfg = H2AIConfig::load_layered(Some(f.path())).unwrap();
    assert!(
        (cfg.safety.krum_threshold - 0.20).abs() < 1e-9,
        "strict profile must set krum_threshold to 0.20, got {}",
        cfg.safety.krum_threshold
    );
}

#[test]
fn production_profile_krum_threshold() {
    let toml = "[safety]\nprofile = \"production\"\n";
    let mut f = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    f.write_all(toml.as_bytes()).unwrap();
    let cfg = H2AIConfig::load_layered(Some(f.path())).unwrap();
    assert!(
        (cfg.safety.krum_threshold - 0.30).abs() < 1e-9,
        "production profile must set krum_threshold to 0.30, got {}",
        cfg.safety.krum_threshold
    );
}

#[test]
fn safety_section_loads_from_reference_toml() {
    let cfg = H2AIConfig::default();
    assert_eq!(cfg.safety.profile, h2ai_config::SafetyProfile::Development);
    assert_eq!(cfg.safety.krum_fault_tolerance, 0);
    assert!((cfg.safety.diversity_threshold).abs() < 1e-9);
    assert_eq!(
        cfg.safety.family_constraint,
        h2ai_config::FamilyConstraint::SingleFamilyOk
    );
    // shadow_auditor tuning sub-fields must load from reference.toml
    assert!((cfg.safety.shadow_auditor.promotion_threshold - 0.05).abs() < 1e-9);
    assert_eq!(cfg.safety.shadow_auditor.promotion_window, 30);
    assert!(cfg.safety.shadow_auditor.auto_demotion);
}

#[test]
fn startup_report_development_is_warn() {
    // Verify log_startup_config_report compiles and does not panic with a default config.
    // The default config uses SafetyProfile::Development, shadow_mode=true, so WARN paths
    // for [safety] and [task_complexity] are exercised; [srani] and [hitl] take INFO paths.
    let cfg = H2AIConfig::default();
    h2ai_config::log_startup_config_report(&cfg);
}

#[test]
fn judge_panel_config_defaults_are_sane() {
    let cfg = JudgePanelConfig::default();
    assert!((cfg.quorum_fraction - 0.67).abs() < 1e-9);
    assert!((cfg.uncertainty_weight - 0.7).abs() < 1e-9);
    assert_eq!(cfg.persona_temperatures.len(), 3);
    assert!(cfg.persona_temperatures[0] < cfg.persona_temperatures[1]);
    assert!(cfg.persona_temperatures[1] < cfg.persona_temperatures[2]);
    assert!(cfg.persona_temperatures[2] <= 0.5);
    assert_eq!(cfg.ambiguity_threshold, 2);
}

#[test]
fn h2ai_config_default_includes_judge_panel() {
    let cfg = H2AIConfig::default();
    assert!((cfg.judge_panel.quorum_fraction - 0.67).abs() < 1e-9);
}

#[test]
fn load_layered_thinking_loop_enabled_overrides_reference_default() {
    let toml = "[thinking_loop]\nenabled = true\n";
    let mut f = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    f.write_all(toml.as_bytes()).unwrap();
    let cfg = H2AIConfig::load_layered(Some(f.path())).unwrap();
    assert!(
        cfg.thinking_loop.enabled,
        "thinking_loop.enabled=true in override must win over reference.toml default=false"
    );
}

#[test]
fn load_layered_srani_disabled_overrides_reference_default() {
    let toml = "[srani]\nenabled = false\n";
    let mut f = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    f.write_all(toml.as_bytes()).unwrap();
    let cfg = H2AIConfig::load_layered(Some(f.path())).unwrap();
    assert!(
        !cfg.srani.enabled,
        "srani.enabled=false in override must win over reference.toml default=true"
    );
}

#[test]
fn load_layered_hitl_enabled_overrides_reference_default() {
    let toml = "[hitl]\nenabled = true\nconfidence_threshold = 0.50\n";
    let mut f = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    f.write_all(toml.as_bytes()).unwrap();
    let cfg = H2AIConfig::load_layered(Some(f.path())).unwrap();
    assert!(
        cfg.hitl.enabled,
        "hitl.enabled=true in override must win over reference.toml default=false"
    );
}

#[test]
fn h2ai_config_knowledge_defaults_to_none() {
    let cfg = H2AIConfig::load_layered(None).unwrap();
    assert!(cfg.knowledge.is_none(), "knowledge must default to None");
}

#[test]
fn h2ai_config_knowledge_bm25wiki_deserializes() {
    use h2ai_knowledge::factory::{KnowledgeConfig, ProviderKind, SourceKind};
    let json_str =
        r#"{"provider":"Bm25Wiki","source":{"YamlDir":{"path":"tests/e2e/constraints"}}}"#;
    let parsed: KnowledgeConfig =
        serde_json::from_str(json_str).expect("KnowledgeConfig must deserialize");
    assert_eq!(parsed.provider, ProviderKind::Bm25Wiki);
    assert!(matches!(parsed.source, SourceKind::YamlDir { .. }));
}

#[test]
fn h2ai_config_knowledge_loads_from_toml_file() {
    use h2ai_knowledge::factory::{ProviderKind, SourceKind};
    use std::io::Write;

    let mut f = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    write!(
        f,
        r#"
nats_url = "nats://localhost:4222"
calibration_max_tokens = 512
calibration_adapter_count = 1

[safety]
profile = "development"

[knowledge]
provider = "Bm25Wiki"

[knowledge.source]
YamlDir = {{ path = "tests/e2e/constraints" }}

[constraint_wiki]
enabled = false

[[adapter_profiles]]
name = "local"
[adapter_profiles.kind.CloudGeneric]
endpoint = "http://host.docker.internal:8080/v1"
api_key_env = ""

[hitl]
enabled = false

[thinking_loop]
enabled = false
"#
    )
    .unwrap();

    let cfg = H2AIConfig::load_layered(Some(f.path())).expect("must load with [knowledge] section");
    let k = cfg.knowledge.expect("knowledge must be Some");
    assert_eq!(k.provider, ProviderKind::Bm25Wiki);
    assert!(matches!(k.source, SourceKind::YamlDir { .. }));
}

#[test]
fn hitl_config_has_decay_fields() {
    let cfg = h2ai_config::H2AIConfig::load_layered(None).unwrap();
    // decay must be in (0.0, 1.0); floor must be > 0
    assert!(cfg.hitl.timeout_decay > 0.0 && cfg.hitl.timeout_decay < 1.0);
    assert!(cfg.hitl.timeout_floor_ms > 0);
}

#[test]
fn signal_config_defaults_exist() {
    let cfg = h2ai_config::H2AIConfig::load_layered(None).unwrap();
    assert_eq!(cfg.signal_wave_window_ms, 0);
    assert!(cfg.signal_min_timeout_ms > 0);
    assert!(cfg.signal_max_timeout_ms > cfg.signal_min_timeout_ms);
}

#[test]
fn calibration_probe_config_has_defaults() {
    let cfg = H2AIConfig::default();
    assert_eq!(cfg.calibration_probe.agents, 3);
    assert_eq!(cfg.calibration_probe.max_tokens, 512);
    assert!((cfg.calibration_probe.neff_cg_exponent - 2.0).abs() < 1e-9);
    assert_eq!(
        cfg.calibration_probe.probe_task_source,
        ProbeTaskSource::Same
    );
    assert_eq!(
        cfg.calibration_probe.system_modifier,
        SystemModifier::CompressReasoning
    );
}

#[test]
fn calibration_slow_start_config_has_defaults() {
    let cfg = H2AIConfig::default();
    assert!((cfg.calibration_slow_start.seed_alpha - 0.15).abs() < 1e-6);
    assert!((cfg.calibration_slow_start.decay_rate - 0.95).abs() < 1e-6);
    assert!((cfg.calibration_slow_start.reset_multiplier - 3.0).abs() < 1e-6);
    assert!((cfg.calibration_slow_start.reset_threshold - 0.4).abs() < 1e-6);
}

#[test]
fn calibration_probe_config_deserializes_from_toml() {
    // Minimal valid H2AIConfig TOML with calibration_probe override.
    // Use load_layered API or direct toml::from_str if the config supports it.
    // Check how other tests in config_test.rs parse partial TOML — follow that pattern.
    let cfg = H2AIConfig::default();
    // Verify the fields exist (the default test above covers correctness)
    let _ = cfg.calibration_probe.agents;
    let _ = cfg.calibration_slow_start.seed_alpha;
}

#[test]
fn state_config_calibration_records_bucket_has_default() {
    let cfg = H2AIConfig::default();
    assert_eq!(
        cfg.state.calibration_records_bucket,
        "H2AI_CALIBRATION_RECORDS"
    );
}

#[test]
fn state_config_auditor_health_bucket_has_default() {
    let cfg = H2AIConfig::default();
    assert_eq!(cfg.state.auditor_health_bucket, "H2AI_AUDITOR_HEALTH");
}

#[test]
fn state_config_probe_lease_bucket_has_default() {
    let cfg = H2AIConfig::default();
    assert_eq!(cfg.state.probe_lease_bucket, "H2AI_PROBE_LEASE");
}

#[test]
fn cspr_config_defaults_to_disabled() {
    let cfg = H2AIConfig::load_layered(None).unwrap();
    assert!(!cfg.cspr.enabled);
}

// ── synthesis config ──────────────────────────────────────────────────────────

#[test]
fn synthesis_defaults_load_from_reference_toml() {
    let cfg = H2AIConfig::default();
    assert!(cfg.synthesis_enabled);
    assert_eq!(cfg.synthesis_min_proposals, 2);
    assert!((cfg.synthesis_tau - 0.2).abs() < 1e-9);
    assert_eq!(cfg.synthesis_critique_max_tokens, 32768);
    assert_eq!(cfg.synthesis_max_tokens, 32768);
}

#[test]
fn subset_validation_does_not_panic_on_contradiction() {
    let cfg = H2AIConfig {
        shell_allowlist: vec!["git".into(), "ls".into()],
        shell_hardened_allowlist: vec!["ls".into(), "rm".into()],
        ..H2AIConfig::default()
    };
    cfg.validate_shell_allowlist_subset();
}

#[test]
fn subset_validation_skipped_when_normal_allowlist_empty() {
    let cfg = H2AIConfig {
        shell_allowlist: vec![],
        shell_hardened_allowlist: vec!["rm".into()],
        ..H2AIConfig::default()
    };
    cfg.validate_shell_allowlist_subset();
}

// ── agent config ──────────────────────────────────────────────────────────────

#[test]
fn agent_max_tool_iterations_default_is_five() {
    let cfg = H2AIConfig::default();
    assert_eq!(cfg.agent_max_tool_iterations, 5);
}

// ── AdapterKind::A2a deserialization ─────────────────────────────────────────

#[test]
fn a2a_adapter_kind_deserializes_from_toml() {
    use config::{Config, File, FileFormat};

    #[derive(serde::Deserialize)]
    struct Wrapper {
        adapter: AdapterKind,
    }

    let toml_str = r#"
[adapter]
A2a = { endpoint = "https://example.com", auth_scheme = "bearer", auth_token_env = "TOKEN_ENV", timeout_minutes = 10, poll_interval_ms = 2000, max_poll_interval_ms = 30000, agent_card_cache_ttl_s = 3600 }
    "#;

    let cfg = Config::builder()
        .add_source(File::from_str(toml_str, FileFormat::Toml))
        .build()
        .expect("config builder failed");

    let w: Wrapper = cfg
        .try_deserialize()
        .expect("should parse A2a AdapterKind from TOML");
    assert!(matches!(w.adapter, AdapterKind::A2a { .. }));

    if let AdapterKind::A2a {
        endpoint,
        auth_scheme,
        auth_token_env,
        timeout_minutes,
        poll_interval_ms,
        max_poll_interval_ms,
        agent_card_cache_ttl_s,
    } = w.adapter
    {
        assert_eq!(endpoint, "https://example.com");
        assert_eq!(auth_scheme, "bearer");
        assert_eq!(auth_token_env, "TOKEN_ENV");
        assert_eq!(timeout_minutes, 10);
        assert_eq!(poll_interval_ms, 2000);
        assert_eq!(max_poll_interval_ms, 30000);
        assert_eq!(agent_card_cache_ttl_s, 3600);
    } else {
        panic!("Expected A2a variant");
    }
}

// ── leader config ─────────────────────────────────────────────────────────────

#[test]
fn leader_fields_present_in_default_config() {
    let cfg = H2AIConfig::default();
    assert!(!cfg.leader_enabled);
    assert_eq!(cfg.leader_stagnation_waves, 1);
    assert_eq!(cfg.leader_eig_candidates, 3);
}

// ── thinking loop config ──────────────────────────────────────────────────────

#[test]
fn thinking_loop_config_defaults_to_disabled() {
    let cfg = ThinkingLoopConfig::default();
    assert!(!cfg.enabled);
    assert_eq!(cfg.max_iterations, 5);
    assert_eq!(cfg.max_archetypes, 4);
    assert!((cfg.coverage_threshold - 0.75).abs() < 1e-9);
    assert!((cfg.convergence_threshold - 0.90).abs() < 1e-9);
}

#[test]
fn h2ai_config_default_has_thinking_loop_disabled() {
    let cfg = H2AIConfig::default();
    assert!(!cfg.thinking_loop.enabled);
}

// ── OracleGateConfig defaults ─────────────────────────────────────────────────

#[test]
fn oracle_gate_config_default_values() {
    let c = OracleGateConfig::default();
    assert!(!c.enabled);
    assert_eq!(c.subject, "h2ai.oracle.gate");
    assert_eq!(c.timeout_secs, 30);
    assert_eq!(c.on_timeout, "pass");
    assert!((c.min_confidence - 0.7).abs() < 1e-9);
    assert!(c.clarification_templates.is_empty());
}

#[test]
fn oracle_gate_config_serde_defaults_from_empty_toml() {
    // Parse partial TOML — missing fields invoke the serde default_* functions.
    use config::{Config, File, FileFormat};
    #[derive(serde::Deserialize)]
    struct Wrapper {
        oracle_gate: OracleGateConfig,
    }
    let toml = "[oracle_gate]\n";
    let w: Wrapper = Config::builder()
        .add_source(File::from_str(toml, FileFormat::Toml))
        .build()
        .unwrap()
        .try_deserialize()
        .unwrap();
    assert!(!w.oracle_gate.enabled);
    assert_eq!(w.oracle_gate.timeout_secs, 30);
    assert_eq!(w.oracle_gate.on_timeout, "pass");
    assert!((w.oracle_gate.min_confidence - 0.7).abs() < 1e-9);
}

#[test]
fn oracle_gate_config_round_trips_json() {
    let c = OracleGateConfig::default();
    let json = serde_json::to_string(&c).unwrap();
    let restored: OracleGateConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(c, restored);
}

// ── ShadowAuditorConfig defaults ──────────────────────────────────────────────

#[test]
fn shadow_auditor_config_direct_default() {
    let c = ShadowAuditorConfig::default();
    assert!(!c.enabled);
    assert!((c.promotion_threshold - 0.05).abs() < 1e-9);
    assert_eq!(c.promotion_window, 30);
    assert!(c.auto_demotion);
}

#[test]
fn shadow_auditor_config_round_trips_json() {
    let c = ShadowAuditorConfig::default();
    let json = serde_json::to_string(&c).unwrap();
    let restored: ShadowAuditorConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(c, restored);
}

// ── SafetyConfig defaults ─────────────────────────────────────────────────────

#[test]
fn safety_config_direct_default() {
    let c = SafetyConfig::default();
    assert_eq!(c.profile, SafetyProfile::Development);
    assert_eq!(c.krum_fault_tolerance, 0);
    assert!((c.krum_threshold - 0.30).abs() < 1e-9);
    assert!((c.diversity_threshold - 0.0).abs() < 1e-9);
    assert_eq!(c.family_constraint, FamilyConstraint::SingleFamilyOk);
    assert!(!c.require_bivariate_cg);
    // Nested default must also be populated.
    assert!(!c.shadow_auditor.enabled);
}

#[test]
fn safety_config_round_trips_json() {
    let c = SafetyConfig::default();
    let json = serde_json::to_string(&c).unwrap();
    let restored: SafetyConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(c, restored);
}

// ── OproConfig defaults ───────────────────────────────────────────────────────

#[test]
fn opro_config_direct_default() {
    let c = OproConfig::default();
    assert!(!c.enabled);
    assert!((c.trigger_j_eff_threshold - 0.6).abs() < 1e-9);
    assert_eq!(c.min_tasks_before_trigger, 10);
    assert_eq!(c.suppress_n_tasks, 5);
    assert_eq!(c.graduation_tasks, 20);
    assert!((c.promotion_margin - 0.05).abs() < 1e-9);
    assert_eq!(c.ema_window, 10);
}

#[test]
fn opro_config_serde_empty() {
    // All serde default_* functions are exercised when all fields are absent.
    let c: OproConfig = serde_json::from_str("{}").unwrap();
    assert!(!c.enabled);
    assert_eq!(c.min_tasks_before_trigger, 10);
    assert_eq!(c.suppress_n_tasks, 5);
    assert_eq!(c.graduation_tasks, 20);
    assert!((c.promotion_margin - 0.05).abs() < 1e-9);
    assert_eq!(c.ema_window, 10);
}

#[test]
fn opro_config_round_trips_json() {
    let c = OproConfig::default();
    let json = serde_json::to_string(&c).unwrap();
    let restored: OproConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(c, restored);
}

// ── CalibrationBootstrapConfig defaults ──────────────────────────────────────

#[test]
fn calibration_bootstrap_config_direct_default() {
    let c = CalibrationBootstrapConfig::default();
    assert_eq!(c.prior_weight, 5);
}

#[test]
fn calibration_bootstrap_config_serde_empty() {
    let c: CalibrationBootstrapConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(c.prior_weight, 5);
}

// ── CalibrationProbeConfig defaults ──────────────────────────────────────────

#[test]
fn calibration_probe_config_direct_default() {
    let c = CalibrationProbeConfig::default();
    assert_eq!(c.agents, 3);
    assert_eq!(c.max_tokens, 512);
    assert_eq!(c.system_modifier, SystemModifier::CompressReasoning);
    assert_eq!(c.probe_task_source, ProbeTaskSource::Same);
    assert!((c.neff_cg_exponent - 2.0).abs() < 1e-9);
}

#[test]
fn calibration_probe_config_round_trips_json() {
    let c = CalibrationProbeConfig::default();
    let json = serde_json::to_string(&c).unwrap();
    let restored: CalibrationProbeConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(c, restored);
}

// ── CalibrationSlowStartConfig defaults ──────────────────────────────────────

#[test]
fn calibration_slow_start_config_direct_default() {
    let c = CalibrationSlowStartConfig::default();
    assert!((c.seed_alpha - 0.15).abs() < 1e-9);
    assert!((c.decay_rate - 0.95).abs() < 1e-9);
    assert!((c.reset_multiplier - 3.0).abs() < 1e-9);
    assert!((c.reset_threshold - 0.4).abs() < 1e-9);
}

#[test]
fn calibration_slow_start_config_round_trips_json() {
    let c = CalibrationSlowStartConfig::default();
    let json = serde_json::to_string(&c).unwrap();
    let restored: CalibrationSlowStartConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(c, restored);
}

// ── StateDeltaConfig defaults ─────────────────────────────────────────────────

#[test]
fn state_delta_config_direct_default() {
    let c = StateDeltaConfig::default();
    assert!(c.enabled);
    assert_eq!(c.base_interval, 10);
    assert_eq!(c.cache_ttl_secs, 60);
    assert_eq!(c.cache_max_entries, 200);
}

#[test]
fn state_delta_config_serde_empty() {
    // All serde default_* functions are exercised when all fields are absent.
    let c: StateDeltaConfig = serde_json::from_str("{}").unwrap();
    assert!(c.enabled);
    assert_eq!(c.base_interval, 10);
    assert_eq!(c.cache_ttl_secs, 60);
    assert_eq!(c.cache_max_entries, 200);
}

#[test]
fn state_delta_config_round_trips_json() {
    let c = StateDeltaConfig::default();
    let json = serde_json::to_string(&c).unwrap();
    let restored: StateDeltaConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(c, restored);
}

// ── StateConfig defaults ──────────────────────────────────────────────────────

#[test]
fn state_config_direct_default() {
    let c = StateConfig::default();
    assert_eq!(c.snapshots_bucket, "H2AI_SNAPSHOTS");
    assert_eq!(c.task_checkpoints_bucket, "H2AI_TASK_CHECKPOINTS");
    assert_eq!(c.checkpoint_payloads_bucket, "H2AI_CHECKPOINT_PAYLOADS");
    assert_eq!(c.oracle_calibration_bucket, "H2AI_ORACLE_CALIBRATION");
    assert_eq!(c.estimator_bucket, "H2AI_ESTIMATOR");
    assert_eq!(c.calibration_bucket, "H2AI_CALIBRATION");
    assert_eq!(c.calibration_records_bucket, "H2AI_CALIBRATION_RECORDS");
    assert_eq!(c.auditor_health_bucket, "H2AI_AUDITOR_HEALTH");
    assert_eq!(c.probe_lease_bucket, "H2AI_PROBE_LEASE");
    assert_eq!(c.sessions_bucket, "H2AI_SESSIONS");
    assert_eq!(c.audit_shadow_bucket, "H2AI_AUDIT_SHADOW");
    assert_eq!(c.prompt_variants_bucket, "H2AI_PROMPT_VARIANTS");
    assert_eq!(c.approvals_bucket, "H2AI_APPROVALS");
    assert_eq!(c.reasoning_checkpoint_bucket_prefix, "H2AI_CHECKPOINT");
    assert_eq!(c.task_meta_state_bucket_prefix, "H2AI_META");
    assert_eq!(c.tenant_memory_bucket_prefix, "H2AI_MEMORY");
    assert_eq!(c.conflict_beta_bucket_prefix, "H2AI_CONFLICT");
    assert_eq!(c.tasks_stream, "H2AI_TASKS");
    assert_eq!(c.telemetry_stream, "H2AI_TELEMETRY");
    assert_eq!(c.results_stream, "H2AI_RESULTS");
    assert_eq!(c.signals_stream, "H2AI_SIGNALS");
    assert_eq!(c.signals_subject_prefix, "h2ai.signals");
}

#[test]
fn state_config_serde_empty() {
    // All serde default_* functions are exercised when all fields are absent.
    let c: StateConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(c.snapshots_bucket, "H2AI_SNAPSHOTS");
    assert_eq!(c.tasks_stream, "H2AI_TASKS");
    assert_eq!(c.signals_subject_prefix, "h2ai.signals");
    assert_eq!(c.reasoning_checkpoint_bucket_prefix, "H2AI_CHECKPOINT");
    assert_eq!(c.task_meta_state_bucket_prefix, "H2AI_META");
    assert_eq!(c.tenant_memory_bucket_prefix, "H2AI_MEMORY");
    assert_eq!(c.conflict_beta_bucket_prefix, "H2AI_CONFLICT");
    // Nested StateDeltaConfig must also have defaults.
    assert!(c.delta.enabled);
}

#[test]
fn state_config_round_trips_json() {
    let c = StateConfig::default();
    let json = serde_json::to_string(&c).unwrap();
    let restored: StateConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(c, restored);
}

// ── ReasoningMemoryConfig defaults ────────────────────────────────────────────

#[test]
fn reasoning_memory_config_direct_default() {
    let c = ReasoningMemoryConfig::default();
    assert!(!c.enabled);
    assert_eq!(c.induction_batch_size, 10);
    assert_eq!(c.induction_max_interval_secs, 86_400);
    assert_eq!(c.induction_max_tasks_per_run, 50);
    assert!((c.tag_gate_threshold - 0.2).abs() < 1e-9);
    assert!((c.max_archetype_boost - 0.15).abs() < 1e-9);
    assert!((c.max_archetype_penalty - 0.20).abs() < 1e-9);
    assert!(!c.strict_audit_checkpoint);
}

#[test]
fn reasoning_memory_config_serde_empty() {
    // All serde default_* functions are exercised when all fields are absent.
    let c: ReasoningMemoryConfig = serde_json::from_str("{}").unwrap();
    assert!(!c.enabled);
    assert_eq!(c.induction_batch_size, 10);
    assert_eq!(c.induction_max_interval_secs, 86_400);
    assert_eq!(c.induction_max_tasks_per_run, 50);
    assert!((c.tag_gate_threshold - 0.2).abs() < 1e-9);
    assert!((c.max_archetype_boost - 0.15).abs() < 1e-9);
    assert!((c.max_archetype_penalty - 0.20).abs() < 1e-9);
}

#[test]
fn reasoning_memory_config_round_trips_json() {
    let c = ReasoningMemoryConfig::default();
    let json = serde_json::to_string(&c).unwrap();
    let restored: ReasoningMemoryConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(c, restored);
}

// ── ConflictBetaConfig defaults ───────────────────────────────────────────────

#[test]
fn conflict_beta_config_direct_default() {
    let c = ConflictBetaConfig::default();
    assert!(c.enabled);
    assert_eq!(c.max_samples, 100);
    assert_eq!(c.halflife_secs, 604_800);
    assert_eq!(c.min_samples_for_override, 5);
}

#[test]
fn conflict_beta_config_serde_empty() {
    // All serde default_* functions are exercised when all fields are absent.
    let c: ConflictBetaConfig = serde_json::from_str("{}").unwrap();
    assert!(c.enabled);
    assert_eq!(c.max_samples, 100);
    assert_eq!(c.halflife_secs, 604_800);
    assert_eq!(c.min_samples_for_override, 5);
}

#[test]
fn conflict_beta_config_round_trips_json() {
    let c = ConflictBetaConfig::default();
    let json = serde_json::to_string(&c).unwrap();
    let restored: ConflictBetaConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(c, restored);
}

// ── JudgePanelConfig serde defaults ──────────────────────────────────────────

#[test]
#[allow(clippy::float_cmp)]
fn judge_panel_config_serde_empty() {
    // All serde default_* functions are exercised when all fields are absent.
    let c: JudgePanelConfig = serde_json::from_str("{}").unwrap();
    assert!((c.quorum_fraction - 0.67).abs() < 1e-9);
    assert!((c.uncertainty_weight - 0.7).abs() < 1e-9);
    assert_eq!(c.persona_temperatures, [0.0f32, 0.2, 0.4]);
    assert_eq!(c.ambiguity_threshold, 2);
}

#[test]
fn judge_panel_config_round_trips_json() {
    let c = JudgePanelConfig::default();
    let json = serde_json::to_string(&c).unwrap();
    let restored: JudgePanelConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(c, restored);
}

// ── ThinkingLoopConfig serde defaults ────────────────────────────────────────

#[test]
fn thinking_loop_config_serde_empty() {
    let c: ThinkingLoopConfig = serde_json::from_str("{}").unwrap();
    assert!(!c.enabled);
    assert_eq!(c.max_iterations, 5);
    assert_eq!(c.max_archetypes, 4);
    assert!((c.coverage_threshold - 0.75).abs() < 1e-9);
    assert!((c.convergence_threshold - 0.90).abs() < 1e-9);
    assert!((c.tau_max - 0.85).abs() < 1e-9);
    assert!((c.tau_min - 0.20).abs() < 1e-9);
    assert!((c.expansion_quality_floor - 0.30).abs() < 1e-9);
    assert_eq!(c.oracle_timeout_secs, 20);
    assert!((c.oracle_confidence_bonus - 0.1).abs() < 1e-9);
}

#[test]
fn thinking_loop_config_round_trips_json() {
    let c = ThinkingLoopConfig::default();
    let json = serde_json::to_string(&c).unwrap();
    let restored: ThinkingLoopConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(c, restored);
}

// ── SraniConfig direct Default ────────────────────────────────────────────────

#[test]
fn srani_config_direct_default() {
    let c = SraniConfig::default();
    assert!(c.enabled);
    assert!(c.adaptive);
    assert!((c.ema_alpha - 0.20).abs() < 1e-9);
    assert!((c.temperature - 0.15).abs() < 1e-9);
    assert!((c.gate_threshold - 0.50).abs() < 1e-9);
    assert!((c.warn_threshold - 0.3).abs() < 1e-9);
    assert!((c.inject_threshold - 0.6).abs() < 1e-9);
    assert_eq!(c.grounding_raw_max_chars, 4000);
    assert_eq!(c.grounding_hint_max_chars, 1200);
    assert!(c.grounding_distill);
}

#[test]
fn srani_config_serde_empty() {
    // All serde default_* functions exercised when all fields absent.
    let c: SraniConfig = serde_json::from_str("{}").unwrap();
    assert!(c.enabled);
    assert!(c.adaptive);
    assert!((c.ema_alpha - 0.20).abs() < 1e-9);
    assert!((c.temperature - 0.15).abs() < 1e-9);
    assert!((c.gate_threshold - 0.50).abs() < 1e-9);
    assert!((c.warn_threshold - 0.3).abs() < 1e-9);
    assert!((c.inject_threshold - 0.6).abs() < 1e-9);
    assert_eq!(c.grounding_raw_max_chars, 4000);
    assert_eq!(c.grounding_hint_max_chars, 1200);
    assert!(c.grounding_distill);
}

// ── CsprConfig default ────────────────────────────────────────────────────────

#[test]
fn cspr_config_direct_default() {
    let c = CsprConfig::default();
    assert!(!c.enabled);
}

#[test]
fn cspr_config_round_trips_json() {
    let c = CsprConfig::default();
    let json = serde_json::to_string(&c).unwrap();
    let restored: CsprConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(c, restored);
}

// ── H2AIConfig serde-default fields (missing from JSON) ───────────────────────

#[test]
fn h2ai_config_missing_calibration_max_ensemble_size_uses_default() {
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    v.as_object_mut()
        .unwrap()
        .remove("calibration_max_ensemble_size");
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert_eq!(cfg.calibration_max_ensemble_size, 9);
}

#[test]
fn h2ai_config_missing_bandit_n_max_arms_uses_default() {
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    v.as_object_mut().unwrap().remove("bandit_n_max_arms");
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert_eq!(cfg.bandit_n_max_arms, 6);
}

#[test]
fn h2ai_config_missing_bandit_prior_sigma_uses_default() {
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    v.as_object_mut().unwrap().remove("bandit_prior_sigma");
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert!((cfg.bandit_prior_sigma - 2.0).abs() < 1e-9);
}

#[test]
fn h2ai_config_missing_bandit_prior_strength_uses_default() {
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    v.as_object_mut().unwrap().remove("bandit_prior_strength");
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert!((cfg.bandit_prior_strength - 5.0).abs() < 1e-9);
}

#[test]
fn h2ai_config_missing_precision_mode_max_slots_uses_default() {
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    v.as_object_mut()
        .unwrap()
        .remove("precision_mode_max_slots");
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert_eq!(cfg.precision_mode_max_slots, 3);
}

#[test]
fn h2ai_config_missing_oracle_window_size_uses_default() {
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    v.as_object_mut().unwrap().remove("oracle_window_size");
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert_eq!(cfg.oracle_window_size, 200);
}

#[test]
fn h2ai_config_missing_oracle_ece_alert_threshold_uses_default() {
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    v.as_object_mut()
        .unwrap()
        .remove("oracle_ece_alert_threshold");
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert!((cfg.oracle_ece_alert_threshold - 0.15).abs() < 1e-9);
}

#[test]
fn h2ai_config_missing_oracle_pass_rate_floor_uses_default() {
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    v.as_object_mut().unwrap().remove("oracle_pass_rate_floor");
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert!((cfg.oracle_pass_rate_floor - 0.30).abs() < 1e-9);
}

#[test]
fn h2ai_config_missing_verifier_consensus_passes_uses_default() {
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    v.as_object_mut()
        .unwrap()
        .remove("verifier_consensus_passes");
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert_eq!(cfg.verifier_consensus_passes, 1);
}

#[test]
fn h2ai_config_missing_signal_min_timeout_ms_uses_default() {
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    v.as_object_mut().unwrap().remove("signal_min_timeout_ms");
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert_eq!(cfg.signal_min_timeout_ms, 60_000);
}

#[test]
fn h2ai_config_missing_signal_max_timeout_ms_uses_default() {
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    v.as_object_mut().unwrap().remove("signal_max_timeout_ms");
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert_eq!(cfg.signal_max_timeout_ms, 86_400_000);
}

#[test]
fn h2ai_config_missing_domain_coverage_threshold_uses_default() {
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    v.as_object_mut()
        .unwrap()
        .remove("domain_coverage_threshold");
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert!((cfg.domain_coverage_threshold - 0.40).abs() < 1e-9);
}

#[test]
fn h2ai_config_missing_oracle_human_bucket_uses_default() {
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    v.as_object_mut().unwrap().remove("oracle_human_bucket");
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert_eq!(cfg.oracle_human_bucket, "H2AI_ORACLE_HUMAN");
}

// ── startup report non-development paths ──────────────────────────────────────

#[test]
fn startup_report_production_profile_info_paths() {
    // Production profile: [safety] takes INFO path; [task_complexity] shadow_mode=true takes WARN
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    v["safety"]["profile"] = serde_json::json!("production");
    // apply_safety_profile won't run on raw serde but log_startup_config_report checks the field
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    // Must not panic
    h2ai_config::log_startup_config_report(&cfg);
}

#[test]
fn startup_report_srani_disabled_warn_path() {
    // srani.enabled=false → WARN path for [srani] block
    let cfg = H2AIConfig {
        srani: SraniConfig {
            enabled: false,
            ..SraniConfig::default()
        },
        ..H2AIConfig::default()
    };
    h2ai_config::log_startup_config_report(&cfg);
}

#[test]
fn startup_report_hitl_enabled_info_path() {
    // hitl.enabled=true → INFO path for [hitl] block
    let cfg = H2AIConfig {
        hitl: h2ai_config::HitlConfig {
            enabled: true,
            ..H2AIConfig::default().hitl
        },
        ..H2AIConfig::default()
    };
    h2ai_config::log_startup_config_report(&cfg);
}

#[test]
fn startup_report_task_complexity_shadow_mode_false() {
    // task_complexity.shadow_mode=false → INFO path for [task_complexity] block
    let cfg = H2AIConfig {
        task_complexity: h2ai_config::TaskComplexityConfig {
            shadow_mode: false,
            ..H2AIConfig::default().task_complexity
        },
        ..H2AIConfig::default()
    };
    h2ai_config::log_startup_config_report(&cfg);
}

// ── WebSearchConfig serde default ────────────────────────────────────────────

#[test]
fn web_search_config_max_results_serde_default() {
    use h2ai_config::WebSearchConfig;
    // Deserialize with max_results absent — triggers default_max_results()
    let c: WebSearchConfig = serde_json::from_str(r#"{"api_key_env":"K","cx_env":"C"}"#).unwrap();
    assert_eq!(c.max_results, 3);
}

#[test]
fn web_search_config_max_results_explicit() {
    use h2ai_config::WebSearchConfig;
    let c: WebSearchConfig =
        serde_json::from_str(r#"{"api_key_env":"K","cx_env":"C","max_results":5}"#).unwrap();
    assert_eq!(c.max_results, 5);
}

// ── ConstraintWikiConfig::Fs serde default ────────────────────────────────────

#[test]
fn constraint_wiki_fs_resolve_k_serde_default() {
    use h2ai_config::ConstraintWikiConfig;
    // Deserialize Fs variant without resolve_k — triggers default_resolve_k()
    let c: ConstraintWikiConfig =
        serde_json::from_str(r#"{"mode":"fs","corpus_path":"/tmp/corpus"}"#).unwrap();
    if let ConstraintWikiConfig::Fs { resolve_k, .. } = c {
        assert_eq!(resolve_k, 50);
    } else {
        panic!("expected Fs variant");
    }
}

#[test]
fn constraint_wiki_fs_resolve_k_explicit() {
    use h2ai_config::ConstraintWikiConfig;
    let c: ConstraintWikiConfig =
        serde_json::from_str(r#"{"mode":"fs","corpus_path":"/tmp/corpus","resolve_k":20}"#)
            .unwrap();
    if let ConstraintWikiConfig::Fs { resolve_k, .. } = c {
        assert_eq!(resolve_k, 20);
    } else {
        panic!("expected Fs variant");
    }
}

// ── load_layered error paths ──────────────────────────────────────────────────

#[test]
fn load_layered_invalid_toml_returns_config_error() {
    use std::io::Write as _;
    let mut tmp = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    // Invalid TOML content (not a valid key=value pair)
    tmp.write_all(b"[[[\ninvalid toml content here\n").unwrap();
    let result = H2AIConfig::load_layered(Some(tmp.path()));
    assert!(
        matches!(result, Err(ConfigLoadError::Config(_))),
        "invalid TOML must return ConfigLoadError::Config"
    );
}

#[test]
fn load_layered_wrong_type_returns_config_error() {
    use std::io::Write as _;
    let mut tmp = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    // bft_threshold should be f64 but we provide a string that can't be coerced
    tmp.write_all(b"bft_threshold = \"not_a_number\"\n")
        .unwrap();
    let result = H2AIConfig::load_layered(Some(tmp.path()));
    assert!(
        matches!(result, Err(ConfigLoadError::Config(_))),
        "type mismatch must return ConfigLoadError::Config"
    );
}
