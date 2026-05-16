use h2ai_config::{
    ConfigLoadError, FamilyConstraint, H2AIConfig, JudgePanelConfig, SafetyProfile, SchedulerPolicy,
};
use std::io::Write;

// ── defaults ─────────────────────────────────────────────────────────────────

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
    assert!(json.contains("bft_threshold"));
    assert!(json.contains("tau_coordinator"));
}

#[test]
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
    assert!(!cfg.constraint_wiki.enabled, "wiki disabled by default");
    assert_eq!(cfg.constraint_wiki.resolve_k, 50);
    assert_eq!(
        cfg.constraint_wiki.corpus_path.as_deref(),
        Some("/constraints")
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
    let expected = (cfg.srani.warn_threshold + cfg.srani.inject_threshold) / 2.0;
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
    let toml = r#"
[srani]
grounding_raw_max_chars  = 8000
grounding_hint_max_chars = 600
grounding_distill        = false
"#;
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
fn default_bandit_prior_sigma_is_2() {
    assert_eq!(H2AIConfig::default().bandit_prior_sigma, 2.0);
}

#[test]
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
fn default_oracle_ece_alert_threshold_is_0_15() {
    assert_eq!(H2AIConfig::default().oracle_ece_alert_threshold, 0.15);
}

#[test]
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
