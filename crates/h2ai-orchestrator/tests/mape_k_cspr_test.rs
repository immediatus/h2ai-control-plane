// Validates CSPR-v2 config toggle is accessible and disabled by default.
// Full MapeKController integration is tested in cspr_integration_test.rs (Task 7).

#[test]
fn cspr_config_disabled_by_default_in_h2ai_config() {
    let cfg = h2ai_config::H2AIConfig::load_layered(None).unwrap();
    assert!(
        !cfg.cspr.enabled,
        "cspr must be disabled by default to avoid regressions"
    );
}

#[test]
fn cspr_config_is_accessible_from_h2ai_config() {
    let cfg = h2ai_config::H2AIConfig::load_layered(None).unwrap();
    // Verify the field is structurally accessible (type-level check)
    let _enabled: bool = cfg.cspr.enabled;
}
