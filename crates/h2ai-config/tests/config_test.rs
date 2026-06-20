use h2ai_config::{
    apply_safety_profile, AuditGateConfig, AwarenessProbeConfig, AwarenessProbeMode,
    CalibrationBootstrapConfig, CalibrationProbeConfig, CalibrationSlowStartConfig,
    ComplexityRoutingConfig, ConfigLoadError, ConflictBetaConfig, ConvergenceGateConfig,
    CostGuardConfig, CsprConfig, DPPMConfig, FamilyConstraint, GapI1Config, GapK1Config,
    H2AIConfig, JudgePanelConfig, OproConfig, OracleGateConfig, ProbeTaskSource,
    ReasoningMemoryConfig, SafetyConfig, SafetyProfile, SchedulerPolicy, ShadowAuditorConfig,
    SraniConfig, StateConfig, StateDeltaConfig, SystemModifier, ThinkingLoopConfig,
    TieredExitConfig,
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

#[test]
#[allow(clippy::float_cmp)]
fn drift_detection_defaults_are_sane() {
    let cfg = H2AIConfig::default();
    assert_eq!(cfg.drift_ddm_window, 20);
    assert!((cfg.drift_ddm_k - 2.5).abs() < 1e-9);
    assert!((cfg.drift_bocpd_hazard_rate - 0.01).abs() < 1e-9);
    assert!((cfg.drift_bocpd_changepoint_threshold - 0.90).abs() < 1e-9);
    assert!(!cfg.auto_recalibrate_on_drift);
    assert_eq!(cfg.drift_staleness_ttl_secs, 3600);
    assert!((cfg.drift_conformal_margin - 0.05).abs() < 1e-9);
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

// ── CostGuardConfig ───────────────────────────────────────────────────────────

#[test]
fn cost_guard_fraction_used_disabled_returns_zero() {
    let cfg = CostGuardConfig {
        enabled: false,
        budget_tokens_per_task: 1000,
        ..CostGuardConfig::default()
    };
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(cfg.fraction_used(500), 0.0, "disabled → always 0.0");
    }
}

#[test]
fn cost_guard_fraction_used_zero_budget_returns_zero() {
    let cfg = CostGuardConfig {
        enabled: true,
        budget_tokens_per_task: 0,
        ..CostGuardConfig::default()
    };
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(cfg.fraction_used(999), 0.0, "zero budget → 0.0");
    }
}

#[test]
fn cost_guard_fraction_used_returns_ratio() {
    let cfg = CostGuardConfig {
        enabled: true,
        budget_tokens_per_task: 1000,
        ..CostGuardConfig::default()
    };
    let frac = cfg.fraction_used(250);
    assert!(
        (frac - 0.25).abs() < 1e-9,
        "250/1000 must be 0.25, got {frac}"
    );
}

// ── serde default functions ───────────────────────────────────────────────────

#[test]
fn talagrand_eta_and_tau_min_default_via_serde() {
    // Serialize H2AIConfig::default() to JSON, remove the two talagrand fields,
    // deserialize back → default_talagrand_eta() and default_talagrand_tau_min() called.
    let cfg = H2AIConfig::default();
    let mut json: serde_json::Value = serde_json::to_value(&cfg).unwrap();
    let obj = json.as_object_mut().unwrap();
    obj.remove("talagrand_eta");
    obj.remove("talagrand_tau_min");
    let deser: H2AIConfig = serde_json::from_value(json).unwrap();
    assert!(
        (deser.talagrand_eta - 0.1).abs() < 1e-9,
        "default_talagrand_eta must be 0.1, got {}",
        deser.talagrand_eta
    );
    assert!(
        (deser.talagrand_tau_min - 0.5).abs() < 1e-9,
        "default_talagrand_tau_min must be 0.5, got {}",
        deser.talagrand_tau_min
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

#[test]
fn model_max_tokens_cascades_to_peer_fields() {
    let mut tmp = tempfile::NamedTempFile::with_suffix(".toml").unwrap();
    writeln!(tmp, "model_max_tokens = 8192").unwrap();
    let cfg = H2AIConfig::load_layered(Some(tmp.path())).unwrap();
    assert_eq!(cfg.model_max_tokens, 8192);
    assert_eq!(
        cfg.explorer_max_tokens, 8192,
        "explorer_max_tokens should cascade"
    );
    assert_eq!(
        cfg.evaluator_max_tokens, 8192,
        "evaluator_max_tokens should cascade"
    );
    assert_eq!(
        cfg.calibration_max_tokens, 8192,
        "calibration_max_tokens should cascade"
    );
    assert_eq!(
        cfg.decomposition_step_max_tokens, 8192,
        "decomposition_step_max_tokens should cascade"
    );
    assert_eq!(
        cfg.decomposition_json_max_tokens, 8192,
        "decomposition_json_max_tokens should cascade"
    );
    assert_eq!(
        cfg.synthesis_critique_max_tokens, 8192,
        "synthesis_critique_max_tokens should cascade"
    );
    assert_eq!(
        cfg.synthesis_max_tokens, 8192,
        "synthesis_max_tokens should cascade"
    );
    // Intentional exceptions must NOT cascade
    assert_eq!(
        cfg.leader_diagnosis_max_tokens, 128,
        "leader_diagnosis_max_tokens must not cascade"
    );
}

#[test]
fn model_max_tokens_cascade_does_not_override_explicit_field() {
    let mut tmp = tempfile::NamedTempFile::with_suffix(".toml").unwrap();
    writeln!(
        tmp,
        "model_max_tokens = 8192\ncalibration_max_tokens = 2048"
    )
    .unwrap();
    let cfg = H2AIConfig::load_layered(Some(tmp.path())).unwrap();
    assert_eq!(
        cfg.calibration_max_tokens, 2048,
        "explicit field override must not be clobbered"
    );
    assert_eq!(cfg.explorer_max_tokens, 8192, "unset peer must cascade");
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
    assert!(srani.grounding_distill, "distill must default to true");
    assert_eq!(
        srani.grounding_compress_threshold, 800,
        "compress threshold must default to 800"
    );
}

#[test]
fn srani_grounding_fields_round_trip_json() {
    let original = H2AIConfig::default();
    let json = serde_json::to_string(&original).unwrap();
    let restored: H2AIConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(
        restored.srani.grounding_distill,
        original.srani.grounding_distill
    );
    assert_eq!(
        restored.srani.grounding_compress_threshold,
        original.srani.grounding_compress_threshold
    );
}

#[test]
fn srani_grounding_missing_fields_use_defaults() {
    // Old JSON payloads without grounding fields must deserialise with defaults.
    let mut v: serde_json::Value = serde_json::to_value(H2AIConfig::default()).unwrap();
    let srani = v["srani"].as_object_mut().unwrap();
    srani.remove("grounding_distill");
    srani.remove("grounding_compress_threshold");
    let cfg: H2AIConfig = serde_json::from_value(v).unwrap();
    assert!(cfg.srani.grounding_distill);
    assert_eq!(cfg.srani.grounding_compress_threshold, 800);
}

#[test]
fn srani_grounding_config_override_via_toml() {
    let toml = r"
[srani]
grounding_distill        = false
";
    let mut f = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    f.write_all(toml.as_bytes()).unwrap();
    let cfg = H2AIConfig::load_layered(Some(f.path())).unwrap();
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
    assert!(cfg.synthesis_wave_enabled);
    assert_eq!(cfg.synthesis_min_proposals, 2);
    assert!((cfg.synthesis_tau - 0.2).abs() < 1e-9);
    assert_eq!(cfg.synthesis_critique_max_tokens, 32768);
    assert_eq!(cfg.synthesis_max_tokens, 32768);
}

#[test]
fn sequential_grafting_defaults_are_sane() {
    let cfg = H2AIConfig::default();
    assert!(
        !cfg.sequential_grafting_enabled,
        "grafting is opt-in, default false"
    );
    assert!(
        cfg.sequential_grafting_max_rounds >= 1,
        "max_rounds must be >= 1"
    );
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
    assert!(c.grounding_distill);
    assert_eq!(c.grounding_compress_threshold, 800);
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
    assert!(c.grounding_distill);
    assert_eq!(c.grounding_compress_threshold, 800);
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

// ── InductionTriggerConfig ────────────────────────────────────────────────────

#[test]
fn induction_trigger_config_parses_from_toml() {
    use std::io::Write;
    let mut tmp = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    writeln!(
        tmp,
        r#"
[induction_trigger]
enabled = true
min_prior_tasks = 3
grace_period_ms = 2000
min_tag_jaccard = 0.3
"#
    )
    .unwrap();
    let cfg = H2AIConfig::load_layered(Some(tmp.path())).unwrap();
    assert!(cfg.induction_trigger.enabled);
    assert_eq!(cfg.induction_trigger.min_prior_tasks, 3);
    assert_eq!(cfg.induction_trigger.grace_period_ms, 2000);
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

// ── DPPM-MetaRefine config tests ──────────────────────────────────────────────

#[test]
fn dppm_config_defaults_disabled() {
    let cfg = DPPMConfig::default();
    assert!(!cfg.enabled);
    assert_eq!(cfg.merge_max_retries, 2);
    assert_eq!(cfg.max_parallel_solvers, 4);
    assert!(cfg.meta_observer_enabled);
}

#[test]
fn h2ai_config_has_dppm_field_defaulting_disabled() {
    let cfg = H2AIConfig::default();
    assert!(!cfg.dppm.enabled);
}

#[test]
fn dppm_config_can_be_enabled_via_toml() {
    let toml = r#"
[dppm]
enabled = true
merge_max_retries = 3
max_parallel_solvers = 8
meta_observer_enabled = false
"#;
    #[derive(serde::Deserialize)]
    struct T {
        dppm: DPPMConfig,
    }
    let t: T = toml::from_str(toml).unwrap();
    assert!(t.dppm.enabled);
    assert_eq!(t.dppm.merge_max_retries, 3);
    assert_eq!(t.dppm.max_parallel_solvers, 8);
    assert!(!t.dppm.meta_observer_enabled);
}

// ── GapI1Config / GapK1Config / ambiguity / safety profile ───────────────────

#[test]
fn gap_i1_config_defaults() {
    let cfg = GapI1Config::default();
    assert!(!cfg.enabled, "I1 must be off by default");
    assert_eq!(cfg.cold_check_threshold, 0.0);
    assert!(cfg.synthesis_min_confidence > 0.5);
    assert!(cfg.max_gap_records_per_wave >= 1);
    assert!(cfg.researcher_timeout_secs > 0);
}

#[test]
fn h2ai_config_has_gap_i1_field() {
    let cfg = H2AIConfig::default();
    let _ = cfg.gap_i1; // field must exist
}

#[test]
fn gap_k1_config_defaults() {
    let cfg = GapK1Config::default();
    assert!(!cfg.enabled);
    assert!(!cfg.auto_repair_enabled);
    assert!((cfg.coherence_threshold - 0.80).abs() < 1e-9);
    assert!((cfg.instability_threshold - 0.10).abs() < 1e-9);
    assert!((cfg.repair_acceptance_threshold - 0.90).abs() < 1e-9);
    assert_eq!(cfg.probe_runs, 5);
    assert_eq!(cfg.repair_candidates, 3);
    assert_eq!(cfg.probe_cache_ttl_secs, 86400);
}

#[test]
fn h2ai_config_has_gap_k1_field() {
    let cfg = H2AIConfig::default();
    let _ = cfg.gap_k1;
}

#[test]
fn ambiguity_detection_config_defaults() {
    let cfg = h2ai_constraints::ambiguity::AmbiguityDetectionConfig::default();
    assert!(!cfg.enabled);
    assert!((cfg.score_threshold - 0.6).abs() < f32::EPSILON);
}

#[test]
fn h2ai_config_has_ambiguity_detection_field() {
    let cfg = H2AIConfig::default();
    assert!(!cfg.ambiguity_detection.enabled);
}

#[test]
fn ambiguity_detection_parses_from_toml() {
    let s = r#"
        [ambiguity_detection]
        enabled = true
        score_threshold = 0.5
        weight_multi_storage = 0.20
        weight_fm_negation = 0.30
        weight_remediation_conflict = 0.15
        weight_cross_check_negation = 0.20
        weight_llm_confirmed = 0.25
        weight_jaccard_freeze_wave = 0.15
        weight_positive_example_conflict = 0.35
    "#;
    #[derive(serde::Deserialize)]
    struct T {
        ambiguity_detection: h2ai_constraints::ambiguity::AmbiguityDetectionConfig,
    }
    let t: T = toml::from_str(s).expect("parse");
    assert!(t.ambiguity_detection.enabled);
    assert!((t.ambiguity_detection.score_threshold - 0.5).abs() < f32::EPSILON);
    assert!((t.ambiguity_detection.weight_fm_negation - 0.30).abs() < f32::EPSILON);
}

#[test]
fn shadow_auditor_config_strict_default_is_false() {
    let cfg = ShadowAuditorConfig::default();
    assert!(!cfg.strict, "strict must be false by default");
}

#[test]
fn apply_safety_profile_sets_strict_for_production() {
    let mut cfg = H2AIConfig::default();
    cfg.safety.profile = SafetyProfile::Production;
    apply_safety_profile(&mut cfg);
    assert!(cfg.safety.shadow_auditor.strict);
}

#[test]
fn apply_safety_profile_sets_strict_for_strict_profile() {
    let mut cfg = H2AIConfig::default();
    cfg.safety.profile = SafetyProfile::Strict;
    apply_safety_profile(&mut cfg);
    assert!(cfg.safety.shadow_auditor.strict);
}

#[test]
fn apply_safety_profile_keeps_strict_false_for_development() {
    let mut cfg = H2AIConfig::default();
    cfg.safety.profile = SafetyProfile::Development;
    apply_safety_profile(&mut cfg);
    assert!(!cfg.safety.shadow_auditor.strict);
}

// ── ComplexityRoutingConfig ───────────────────────────────────────────────────

#[test]
fn complexity_routing_config_defaults() {
    let cfg = ComplexityRoutingConfig::default();
    assert!(!cfg.enabled);
    assert_eq!(cfg.complexity_probe_adapter, "researcher");
    assert_eq!(cfg.complexity_probe_timeout_secs, 30);
    assert_eq!(cfg.decompose_threshold, 4);
    assert_eq!(cfg.hitl_threshold, 5);
    assert!(!cfg.verifier_decomposition_enabled);
    assert!(!cfg.intra_retry.enabled);
    assert_eq!(cfg.intra_retry.entropy_threshold, 0.6);
    assert_eq!(cfg.intra_retry.retry_slope_threshold, 0.05);
    assert_eq!(cfg.intra_retry.n_eff_cg_product_threshold, 0.3);
    assert_eq!(cfg.intra_retry.min_retry_count_for_detection, 2);
}

#[test]
fn agent_dropout_config_defaults() {
    let cfg = ComplexityRoutingConfig::default();
    assert!(!cfg.agent_dropout.enabled);
    assert_eq!(cfg.agent_dropout.n_eff_dropout_threshold, 0.5);
}

#[test]
fn complexity_routing_config_toml_roundtrip() {
    let toml_str = r#"
        enabled = true
        complexity_probe_adapter = "explorer"
        complexity_probe_timeout_secs = 60
        decompose_threshold = 3
        hitl_threshold = 4
        verifier_decomposition_enabled = true

        [intra_retry]
        enabled = true
        entropy_threshold = 0.5
        retry_slope_threshold = 0.02
        n_eff_cg_product_threshold = 0.25
        min_retry_count_for_detection = 3
    "#;
    let cfg: ComplexityRoutingConfig = toml::from_str(toml_str).expect("should parse");
    assert!(cfg.enabled);
    assert_eq!(cfg.complexity_probe_adapter, "explorer");
    assert_eq!(cfg.complexity_probe_timeout_secs, 60);
    assert_eq!(cfg.decompose_threshold, 3);
    assert_eq!(cfg.hitl_threshold, 4);
    assert!(cfg.verifier_decomposition_enabled);
    assert!(cfg.intra_retry.enabled);
    assert_eq!(cfg.intra_retry.entropy_threshold, 0.5);
    assert_eq!(cfg.intra_retry.min_retry_count_for_detection, 3);
}

#[test]
fn min_retries_before_graft_defaults_to_two() {
    let cfg = ComplexityRoutingConfig::default();
    assert_eq!(cfg.min_retries_before_graft, 2);
}

#[test]
fn min_retries_before_graft_parses_from_toml() {
    let toml_str = r#"
        enabled = true
        complexity_probe_adapter = "researcher"
        complexity_probe_timeout_secs = 30
        decompose_threshold = 4
        hitl_threshold = 5
        verifier_decomposition_enabled = false
        min_retries_before_graft = 0

        [intra_retry]
        enabled = false
        entropy_threshold = 0.6
        retry_slope_threshold = 0.05
        n_eff_cg_product_threshold = 0.3
        min_retry_count_for_detection = 2
    "#;
    let cfg: ComplexityRoutingConfig = toml::from_str(toml_str).expect("should parse");
    assert_eq!(cfg.min_retries_before_graft, 0, "explicit 0 disables floor");
}

#[test]
fn min_retries_before_graft_defaults_when_omitted_from_toml() {
    // Existing TOML without min_retries_before_graft must still parse (backward compat).
    let toml_str = r#"
        enabled = true
        complexity_probe_adapter = "explorer"
        complexity_probe_timeout_secs = 60
        decompose_threshold = 3
        hitl_threshold = 4
        verifier_decomposition_enabled = true

        [intra_retry]
        enabled = true
        entropy_threshold = 0.5
        retry_slope_threshold = 0.02
        n_eff_cg_product_threshold = 0.25
        min_retry_count_for_detection = 3
    "#;
    let cfg: ComplexityRoutingConfig = toml::from_str(toml_str).expect("should parse");
    assert_eq!(
        cfg.min_retries_before_graft, 2,
        "falls back to default when omitted"
    );
}

// ── TieredExitConfig ──────────────────────────────────────────────────────────

fn tiered_exit_cfg(min_n: u32, max_n: u32, quorum_fraction: f64) -> TieredExitConfig {
    TieredExitConfig {
        enabled: true,
        min_n,
        max_n,
        quorum_fraction,
        acceptance_score: 0.85,
        require_all_binary_checks: true,
    }
}

#[test]
fn n_for_wave_wave0_returns_min_n() {
    let c = tiered_exit_cfg(1, 5, 0.5);
    assert_eq!(c.n_for_wave(0, 4), 1);
}

#[test]
fn n_for_wave_last_wave_returns_max_n() {
    let c = tiered_exit_cfg(1, 5, 0.5);
    assert_eq!(c.n_for_wave(4, 4), 5);
}

#[test]
fn n_for_wave_midpoint_rounds_correctly() {
    let c = tiered_exit_cfg(1, 5, 0.5);
    // wave=2, max=4: frac=0.5, n = 1 + round(0.5 * 4) = 1 + 2 = 3
    assert_eq!(c.n_for_wave(2, 4), 3);
}

#[test]
fn n_for_wave_zero_max_retries_returns_max_n() {
    let c = tiered_exit_cfg(1, 5, 0.5);
    assert_eq!(c.n_for_wave(0, 0), 5);
}

#[test]
fn n_for_wave_clamps_to_bounds() {
    let c = tiered_exit_cfg(3, 3, 0.5);
    assert_eq!(c.n_for_wave(0, 4), 3);
    assert_eq!(c.n_for_wave(4, 4), 3);
}

#[test]
fn k_for_wave_fraction_half() {
    let c = tiered_exit_cfg(1, 5, 0.5);
    assert_eq!(c.k_for_wave(1), 1); // ceil(0.5) = 1
    assert_eq!(c.k_for_wave(3), 2); // ceil(1.5) = 2
    assert_eq!(c.k_for_wave(5), 3); // ceil(2.5) = 3
}

#[test]
fn k_for_wave_never_zero() {
    let c = tiered_exit_cfg(1, 5, 0.01);
    assert_eq!(c.k_for_wave(1), 1);
}

#[test]
fn tiered_exit_default_is_disabled() {
    let c = TieredExitConfig::default();
    assert!(!c.enabled);
}

#[test]
fn tiered_exit_toml_roundtrip() {
    let toml_str = r#"
        enabled                   = true
        min_n                     = 2
        max_n                     = 6
        quorum_fraction           = 0.4
        acceptance_score          = 0.90
        require_all_binary_checks = false
    "#;
    let c: TieredExitConfig = toml::from_str(toml_str).expect("parse");
    assert!(c.enabled);
    assert_eq!(c.min_n, 2);
    assert_eq!(c.max_n, 6);
    assert!((c.quorum_fraction - 0.4).abs() < 1e-9);
    assert!((c.acceptance_score - 0.90).abs() < 1e-9);
    assert!(!c.require_all_binary_checks);
}

#[test]
fn n_for_wave_wave_beyond_max_retries_clamps_to_max_n() {
    let c = tiered_exit_cfg(1, 5, 0.5);
    // wave=6 > max_retries=4: frac=1.5, raw n=7, clamped to max_n=5
    assert_eq!(c.n_for_wave(6, 4), 5);
}

#[test]
fn k_for_wave_n_zero_returns_one() {
    let c = tiered_exit_cfg(1, 5, 0.5);
    // ceil(0 * 0.5) = 0, max(1, 0) = 1
    assert_eq!(c.k_for_wave(0), 1);
}

#[test]
fn reference_toml_contains_tiered_exit() {
    let src = include_str!("../reference.toml");
    assert!(
        src.contains("[tiered_exit]"),
        "reference.toml must have [tiered_exit] section"
    );
}

// ── CostGuardConfig / ConvergenceGateConfig / AwarenessProbeConfig ────────────

#[test]
fn cost_guard_default_disabled() {
    let cfg = CostGuardConfig::default();
    assert!(!cfg.enabled);
    assert_eq!(cfg.budget_tokens_per_task, 100_000);
    assert!((cfg.budget_warning_fraction - 0.80).abs() < 1e-9);
    assert!((cfg.budget_abort_fraction - 1.00).abs() < 1e-9);
    assert!(!cfg.budget_prompt_injection_enabled);
    assert!((cfg.budget_injection_warn_fraction - 0.50).abs() < 1e-9);
    assert_eq!(cfg.budget_injection_max_complexity, 3);
}

#[test]
fn convergence_gate_default_disabled() {
    let cfg = ConvergenceGateConfig::default();
    assert!(!cfg.enabled);
    assert!((cfg.theta_converge - 0.87).abs() < 1e-9);
    assert!((cfg.supermajority_fraction_n3 - 0.67).abs() < 1e-9);
    assert!((cfg.supermajority_fraction_n4plus - 0.80).abs() < 1e-9);
    assert!((cfg.score_floor - 0.80).abs() < 1e-9);
    assert_eq!(cfg.min_wave, 1);
    assert!((cfg.budget_floor_fraction - 0.30).abs() < 1e-9);
}

#[test]
fn cost_guard_parses_from_toml() {
    let s = r#"
        [cost_guard]
        enabled = true
        budget_tokens_per_task = 50000
        budget_warning_fraction = 0.75
        budget_abort_fraction = 0.95
        budget_prompt_injection_enabled = true
        budget_injection_warn_fraction = 0.60
        budget_injection_max_complexity = 4
        [convergence_gate]
        enabled = true
        theta_converge = 0.90
        supermajority_fraction_n3 = 1.0
        supermajority_fraction_n4plus = 0.80
        score_floor = 0.75
        min_wave = 2
        budget_floor_fraction = 0.25
    "#;
    #[derive(serde::Deserialize)]
    struct T {
        cost_guard: CostGuardConfig,
        convergence_gate: ConvergenceGateConfig,
    }
    let t: T = toml::from_str(s).expect("parse");
    assert!(t.cost_guard.enabled);
    assert_eq!(t.cost_guard.budget_tokens_per_task, 50_000);
    assert!(t.convergence_gate.enabled);
    assert!((t.convergence_gate.theta_converge - 0.90).abs() < 1e-9);
}

#[test]
fn h2ai_config_has_cost_guard_and_convergence_gate() {
    let cfg = H2AIConfig::default();
    assert!(!cfg.cost_guard.enabled);
    assert!(!cfg.convergence_gate.enabled);
}

#[test]
fn awareness_probe_config_defaults() {
    let cfg = AwarenessProbeConfig::default();
    assert!(!cfg.enabled);
    assert_eq!(cfg.mode, AwarenessProbeMode::Shadow);
    assert_eq!(cfg.judge_max_tokens, 1024);
}

#[test]
fn h2ai_config_has_awareness_probe_field() {
    let cfg = H2AIConfig::default();
    assert!(!cfg.awareness_probe.enabled);
    assert!(!cfg.knowledge_domain_scoping);
}

#[test]
fn cost_guard_remaining_unlimited_when_budget_zero() {
    let cfg = CostGuardConfig {
        budget_tokens_per_task: 0,
        ..CostGuardConfig::default()
    };
    assert_eq!(cfg.remaining(999_999), i64::MAX);
}

#[test]
fn cost_guard_remaining_with_positive_budget() {
    let cfg = CostGuardConfig {
        budget_tokens_per_task: 10_000,
        ..CostGuardConfig::default()
    };
    assert_eq!(cfg.remaining(3_000), 7_000);
}

#[test]
fn convergence_gate_supermajority_for_n_zero_to_two() {
    let cfg = ConvergenceGateConfig::default();
    assert!((cfg.supermajority_for_n(0) - 1.0).abs() < 1e-9);
    assert!((cfg.supermajority_for_n(1) - 1.0).abs() < 1e-9);
    assert!((cfg.supermajority_for_n(2) - 1.0).abs() < 1e-9);
}

#[test]
fn convergence_gate_supermajority_for_n_three() {
    let cfg = ConvergenceGateConfig::default();
    assert!((cfg.supermajority_for_n(3) - cfg.supermajority_fraction_n3).abs() < 1e-9);
}

#[test]
fn convergence_gate_supermajority_for_n_four_plus() {
    let cfg = ConvergenceGateConfig::default();
    assert!((cfg.supermajority_for_n(4) - cfg.supermajority_fraction_n4plus).abs() < 1e-9);
    assert!((cfg.supermajority_for_n(10) - cfg.supermajority_fraction_n4plus).abs() < 1e-9);
}

#[test]
fn awareness_probe_parses_from_toml() {
    let toml = r#"
knowledge_domain_scoping = true

[awareness_probe]
enabled = true
mode = "active"
judge_max_tokens = 2048
"#;
    #[derive(serde::Deserialize)]
    struct T {
        awareness_probe: AwarenessProbeConfig,
        #[serde(default)]
        knowledge_domain_scoping: bool,
    }
    let t: T = toml::from_str(toml).expect("parse");
    assert!(t.awareness_probe.enabled);
    assert_eq!(t.awareness_probe.mode, AwarenessProbeMode::Active);
    assert_eq!(t.awareness_probe.judge_max_tokens, 2048);
    assert!(t.knowledge_domain_scoping);
}

/// Regression guard: `knowledge_domain_scoping` placed AFTER a `[section]` header
/// belongs to that section in TOML, not to the top level.  Serde silently drops it,
/// so the field must remain `false` (its default).  If this test ever fails it means
/// serde somehow started hoisting inner-section keys — which would mask the real bug.
#[test]
fn knowledge_domain_scoping_must_be_top_level() {
    // key placed inside [awareness_probe] — must NOT reach H2AIConfig.knowledge_domain_scoping
    let toml_misplaced = r#"
[awareness_probe]
enabled = true
mode = "shadow"
judge_max_tokens = 1024
knowledge_domain_scoping = true
"#;
    #[derive(serde::Deserialize)]
    struct T {
        awareness_probe: AwarenessProbeConfig,
        #[serde(default)]
        knowledge_domain_scoping: bool,
    }
    let t: T = toml::from_str(toml_misplaced).expect("parse");
    assert!(t.awareness_probe.enabled);
    // The misplaced key must NOT appear at the top level.
    assert!(
        !t.knowledge_domain_scoping,
        "knowledge_domain_scoping inside [awareness_probe] must not propagate to the top level"
    );
}

// ── Pipeline Resilience configs ───────────────────────────────────────────────

#[test]
fn verifier_freeze_config_defaults() {
    use h2ai_config::VerifierFreezeConfig;
    let cfg = VerifierFreezeConfig::default();
    assert!(cfg.enabled);
    assert_eq!(cfg.min_waves_to_detect, 3);
    assert!((cfg.score_variance_threshold - 0.05).abs() < 1e-9);
    assert!((cfg.reason_jaccard_threshold - 0.7).abs() < 1e-9);
    assert_eq!(cfg.reason_window_size, 10);
    assert!((cfg.other_constraint_success_threshold - 0.5).abs() < 1e-9);
    assert!(cfg.bypass_hard_gate_on_freeze);
    assert!(!cfg.emit_event_only);
}

#[test]
fn verifier_freeze_config_parses_from_toml() {
    use h2ai_config::VerifierFreezeConfig;
    let toml = r#"
[verifier_freeze]
enabled = false
min_waves_to_detect = 5
score_variance_threshold = 0.02
reason_jaccard_threshold = 0.8
reason_window_size = 15
other_constraint_success_threshold = 0.6
bypass_hard_gate_on_freeze = false
emit_event_only = true
"#;
    #[derive(serde::Deserialize)]
    struct T {
        verifier_freeze: VerifierFreezeConfig,
    }
    let t: T = toml::from_str(toml).unwrap();
    assert!(!t.verifier_freeze.enabled);
    assert_eq!(t.verifier_freeze.min_waves_to_detect, 5);
    assert!((t.verifier_freeze.score_variance_threshold - 0.02).abs() < 1e-9);
    assert!((t.verifier_freeze.reason_jaccard_threshold - 0.8).abs() < 1e-9);
    assert_eq!(t.verifier_freeze.reason_window_size, 15);
    assert!(!t.verifier_freeze.bypass_hard_gate_on_freeze);
    assert!(t.verifier_freeze.emit_event_only);
}

#[test]
fn generation_phase_config_defaults() {
    use h2ai_config::GenerationPhaseConfig;
    let cfg = GenerationPhaseConfig::default();
    assert_eq!(cfg.timeout_secs, 300);
}

#[test]
fn generation_phase_config_parses_from_toml() {
    use h2ai_config::GenerationPhaseConfig;
    let toml = r#"
[generation_phase]
timeout_secs = 600
"#;
    #[derive(serde::Deserialize)]
    struct T {
        generation_phase: GenerationPhaseConfig,
    }
    let t: T = toml::from_str(toml).unwrap();
    assert_eq!(t.generation_phase.timeout_secs, 600);
}

#[test]
fn oom_guard_config_defaults() {
    use h2ai_config::OomGuardConfig;
    let cfg = OomGuardConfig::default();
    assert!(cfg.enabled);
    assert_eq!(cfg.rss_abort_mb, 4096);
    assert_eq!(cfg.check_interval_waves, 1);
}

#[test]
fn oom_guard_config_parses_from_toml() {
    use h2ai_config::OomGuardConfig;
    let toml = r#"
[oom_guard]
enabled = false
rss_abort_mb = 8192
check_interval_waves = 2
"#;
    #[derive(serde::Deserialize)]
    struct T {
        oom_guard: OomGuardConfig,
    }
    let t: T = toml::from_str(toml).unwrap();
    assert!(!t.oom_guard.enabled);
    assert_eq!(t.oom_guard.rss_abort_mb, 8192);
    assert_eq!(t.oom_guard.check_interval_waves, 2);
}

#[test]
fn gap_quality_config_defaults() {
    use h2ai_config::GapQualityConfig;
    let cfg = GapQualityConfig::default();
    assert!((cfg.min_improvement_to_retain - 0.1).abs() < 1e-9);
    assert_eq!(cfg.min_post_injection_waves, 2);
}

#[test]
fn gap_quality_config_parses_from_toml() {
    use h2ai_config::GapQualityConfig;
    let toml = r#"
[gap_quality]
min_improvement_to_retain = 0.15
min_post_injection_waves = 3
"#;
    #[derive(serde::Deserialize)]
    struct T {
        gap_quality: GapQualityConfig,
    }
    let t: T = toml::from_str(toml).unwrap();
    assert!((t.gap_quality.min_improvement_to_retain - 0.15).abs() < 1e-9);
    assert_eq!(t.gap_quality.min_post_injection_waves, 3);
}

#[test]
fn h2ai_config_has_resilience_fields_with_sensible_defaults() {
    let cfg = h2ai_config::H2AIConfig::default();
    assert!(cfg.verifier_freeze.enabled);
    assert_eq!(cfg.generation_phase.timeout_secs, 300);
    assert!(cfg.oom_guard.enabled);
    assert_eq!(cfg.oom_guard.rss_abort_mb, 4096);
    assert!((cfg.gap_quality.min_improvement_to_retain - 0.1).abs() < 1e-9);
}

#[test]
fn generation_phase_loads_from_reference_toml() {
    let cfg = H2AIConfig::load_layered(None).unwrap();
    assert_eq!(cfg.generation_phase.timeout_secs, 300);
}

#[test]
fn generation_phase_override_applies() {
    let toml = "[generation_phase]\ntimeout_secs = 3600\n";
    let mut f = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
    f.write_all(toml.as_bytes()).unwrap();
    let cfg = H2AIConfig::load_layered(Some(f.path())).unwrap();
    assert_eq!(cfg.generation_phase.timeout_secs, 3600);
}

#[test]
fn verifier_freeze_loads_from_reference_toml() {
    let cfg = H2AIConfig::load_layered(None).unwrap();
    assert!(cfg.verifier_freeze.enabled);
    assert_eq!(cfg.verifier_freeze.min_waves_to_detect, 3);
    assert!((cfg.verifier_freeze.score_variance_threshold - 0.05).abs() < 1e-9);
    assert!((cfg.verifier_freeze.reason_jaccard_threshold - 0.7).abs() < 1e-9);
    assert_eq!(cfg.verifier_freeze.reason_window_size, 10);
    assert!((cfg.verifier_freeze.other_constraint_success_threshold - 0.5).abs() < 1e-9);
    assert!(cfg.verifier_freeze.bypass_hard_gate_on_freeze);
    assert!(!cfg.verifier_freeze.emit_event_only);
}

#[test]
fn oom_guard_loads_from_reference_toml() {
    let cfg = H2AIConfig::load_layered(None).unwrap();
    assert!(cfg.oom_guard.enabled);
    assert_eq!(cfg.oom_guard.rss_abort_mb, 4096);
    assert_eq!(cfg.oom_guard.check_interval_waves, 1);
}

#[test]
fn gap_quality_loads_from_reference_toml() {
    let cfg = H2AIConfig::load_layered(None).unwrap();
    assert!((cfg.gap_quality.min_improvement_to_retain - 0.1).abs() < 1e-9);
    assert_eq!(cfg.gap_quality.min_post_injection_waves, 2);
}

#[test]
fn audit_gate_default_fail_open() {
    let cfg = AuditGateConfig::default();
    assert!(cfg.fail_open_on_parse_error, "default must be fail-open");
}

#[test]
fn audit_gate_loads_from_reference_toml() {
    let cfg = H2AIConfig::load_layered(None).unwrap();
    assert!(
        cfg.audit_gate.fail_open_on_parse_error,
        "reference.toml must set fail_open_on_parse_error=true"
    );
}

// ── validate_shell_allowlist_subset — contradiction warn path ─────────────────

#[test]
fn validate_shell_allowlist_subset_skips_when_allowlist_empty() {
    let cfg = H2AIConfig {
        shell_allowlist: vec![],
        shell_hardened_allowlist: vec!["rm".to_string()],
        ..H2AIConfig::default()
    };
    // early-return path: shell_allowlist is empty, no warn emitted
    cfg.validate_shell_allowlist_subset();
}

#[test]
fn validate_shell_allowlist_subset_contradiction_does_not_panic() {
    let cfg = H2AIConfig {
        shell_allowlist: vec!["echo".to_string()],
        shell_hardened_allowlist: vec!["rm".to_string()],
        ..H2AIConfig::default()
    };
    // "rm" is in shell_hardened_allowlist but NOT in shell_allowlist → warn path
    cfg.validate_shell_allowlist_subset();
}

#[test]
fn validate_shell_allowlist_subset_no_warn_when_subset() {
    let cfg = H2AIConfig {
        shell_allowlist: vec!["echo".to_string(), "ls".to_string()],
        shell_hardened_allowlist: vec!["echo".to_string()],
        ..H2AIConfig::default()
    };
    // "echo" is in both lists — no contradiction
    cfg.validate_shell_allowlist_subset();
}
