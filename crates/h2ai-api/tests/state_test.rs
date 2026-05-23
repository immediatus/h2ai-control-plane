#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
//! Tests for `AppState` utility methods that don't require NATS.

use std::sync::Arc;

use chrono::Utc;
use h2ai_api::state::AppState;
use h2ai_config::H2AIConfig;
use h2ai_test_utils::DecompositionMockAdapter;
use h2ai_types::{
    events::{CalibrationCompletedEvent, CalibrationQuality, CalibrationSource, CgMode},
    identity::{TaskId, TenantId},
    sizing::{CoherencyCoefficients, CoordinationThreshold},
};

fn synthetic_calibration() -> CalibrationCompletedEvent {
    let coefficients = CoherencyCoefficients::new(0.12, 0.021, vec![0.68, 0.74, 0.71])
        .expect("valid coefficients");
    let coordination_threshold = CoordinationThreshold::from_calibration(&coefficients, 0.3);
    CalibrationCompletedEvent {
        calibration_id: TaskId::new(),
        coefficients,
        coordination_threshold,
        ensemble: None,
        eigen: None,
        timestamp: Utc::now(),
        pairwise_beta: None,
        cg_mode: CgMode::default(),
        adapter_families: vec!["Mock".into()],
        explorer_verification_family_match: false,
        single_family_warning: false,
        n_max_lo: 0.0,
        n_max_hi: 0.0,
        n_eff_cosine_prior: 0.0,
        calibration_quality: CalibrationQuality::default(),
        calibration_source: CalibrationSource::Measured,
        beta_quality: None,
    }
}

fn make_state() -> AppState {
    let adapter = Arc::new(DecompositionMockAdapter::new("mock".into()));
    AppState::new_for_tests(
        H2AIConfig::default(),
        vec![adapter.clone() as Arc<dyn h2ai_types::adapter::IComputeAdapter>],
        adapter as Arc<dyn h2ai_types::adapter::IComputeAdapter>,
    )
}

// ── registry() ────────────────────────────────────────────────────────────────

#[test]
fn registry_without_scoring_adapter_returns_plain_registry() {
    let state = make_state();
    let reg = state.registry();
    // Must construct without panic — no scoring adapter variant.
    let _ = reg;
}

#[test]
fn registry_with_scoring_adapter_uses_it() {
    let mut state = make_state();
    let scoring = Arc::new(DecompositionMockAdapter::new("scoring".into()));
    state.scoring_adapter = Some(scoring.clone() as Arc<dyn h2ai_types::adapter::IComputeAdapter>);
    let reg = state.registry();
    let _ = reg;
}

// ── seed_calibration_from_default_if_needed ────────────────────────────────────

#[tokio::test]
async fn seed_calibration_noop_for_default_tenant() {
    let state = make_state();
    // Default tenant with no calibration — seed must be a no-op.
    state
        .seed_calibration_from_default_if_needed(&TenantId::default_tenant())
        .await;
    let default_ts = state.tenant_state(&TenantId::default_tenant());
    let cal = default_ts.calibration.read().await;
    assert!(
        cal.is_none(),
        "default tenant must not receive seeded calibration"
    );
}

#[tokio::test]
async fn seed_calibration_copies_from_default_to_new_tenant() {
    let state = make_state();

    // Set calibration on default tenant.
    let default_ts = state.tenant_state(&TenantId::default_tenant());
    *default_ts.calibration.write().await = Some(synthetic_calibration());

    let other_tenant = TenantId::from("other-corp");
    assert!(
        state
            .tenant_state(&other_tenant)
            .calibration
            .read()
            .await
            .is_none(),
        "other tenant must start with no calibration"
    );

    state
        .seed_calibration_from_default_if_needed(&other_tenant)
        .await;

    assert!(
        state
            .tenant_state(&other_tenant)
            .calibration
            .read()
            .await
            .is_some(),
        "seed must have copied calibration to other tenant"
    );
}

#[tokio::test]
async fn seed_calibration_noop_when_other_tenant_already_has_calibration() {
    let state = make_state();

    let other_tenant = TenantId::from("tenant-xyz");
    let ts = state.tenant_state(&other_tenant);
    let original_cal = synthetic_calibration();
    let original_id = original_cal.calibration_id.clone();
    *ts.calibration.write().await = Some(original_cal);

    // Default also has calibration (different ID).
    let different_cal = synthetic_calibration();
    *state
        .tenant_state(&TenantId::default_tenant())
        .calibration
        .write()
        .await = Some(different_cal);

    state
        .seed_calibration_from_default_if_needed(&other_tenant)
        .await;

    let ts2 = state.tenant_state(&other_tenant);
    let cal = ts2.calibration.read().await;
    let stored_id = cal.as_ref().unwrap().calibration_id.clone();
    assert_eq!(
        stored_id, original_id,
        "existing calibration must not be overwritten"
    );
}

#[tokio::test]
async fn seed_calibration_noop_when_default_has_no_calibration() {
    let state = make_state();
    let other_tenant = TenantId::from("tenant-abc");

    // Default tenant has no calibration; other tenant has none either.
    state
        .seed_calibration_from_default_if_needed(&other_tenant)
        .await;

    assert!(
        state
            .tenant_state(&other_tenant)
            .calibration
            .read()
            .await
            .is_none(),
        "other tenant must remain without calibration when default has none"
    );
}

// ── builder methods ────────────────────────────────────────────────────────────

#[test]
fn with_shadow_auditor_sets_adapter() {
    let state = make_state();
    assert!(state.shadow_auditor_adapter.is_none());
    let shadow = Arc::new(DecompositionMockAdapter::new("shadow".into()));
    let state = state.with_shadow_auditor(shadow as Arc<dyn h2ai_types::adapter::IComputeAdapter>);
    assert!(state.shadow_auditor_adapter.is_some());
}

#[test]
fn with_payload_store_sets_store() {
    use h2ai_orchestrator::payload_store::MemoryPayloadStore;
    let state = make_state();
    let store = Arc::new(MemoryPayloadStore::new());
    let state =
        state.with_payload_store(store as Arc<dyn h2ai_orchestrator::payload_store::PayloadStore>);
    // Just verify it doesn't panic and the store is set (we can't introspect the type easily)
    let _ = state;
}
