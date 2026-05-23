#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
//! Unit tests for `h2ai_api::bootstrap` — OPRO prior seeding.
//!
//! Uses `InMemoryStateBackend` so no NATS server is required.

use h2ai_api::bootstrap::{seed_all_bootstrap_priors, seed_bootstrap_prior};
use h2ai_config::{AdapterProfile, CalibrationBootstrapConfig, ProfileTier};
use h2ai_state::backend::OproStore;
use h2ai_state::in_memory::InMemoryStateBackend;
use h2ai_types::config::AdapterKind;

fn make_backend() -> InMemoryStateBackend {
    InMemoryStateBackend::new()
}

fn mock_kind() -> AdapterKind {
    AdapterKind::CloudGeneric {
        endpoint: String::new(),
        api_key_env: String::new(),
        model: None,
    }
}

// ── seed_bootstrap_prior ──────────────────────────────────────────────────────

#[tokio::test]
async fn seed_bootstrap_prior_capable_sets_j_eff_078() {
    let backend = make_backend();
    seed_bootstrap_prior("adapter-a", &ProfileTier::Capable, 10, &backend)
        .await
        .expect("seed must succeed");

    let state = backend
        .get_adapter_opro_state("adapter-a")
        .await
        .expect("get must succeed")
        .expect("state must exist");

    assert!(
        (state.j_eff_ema - 0.78).abs() < 1e-9,
        "Capable tier j_eff_ema must be 0.78, got {}",
        state.j_eff_ema
    );
    assert_eq!(state.adapter_name, "adapter-a");
    assert_eq!(state.n_tasks_total, 0);
}

#[tokio::test]
async fn seed_bootstrap_prior_standard_sets_j_eff_062() {
    let backend = make_backend();
    seed_bootstrap_prior("adapter-b", &ProfileTier::Standard, 10, &backend)
        .await
        .expect("seed must succeed");

    let state = backend
        .get_adapter_opro_state("adapter-b")
        .await
        .expect("get must succeed")
        .expect("state must exist");

    assert!(
        (state.j_eff_ema - 0.62).abs() < 1e-9,
        "Standard tier j_eff_ema must be 0.62, got {}",
        state.j_eff_ema
    );
}

#[tokio::test]
async fn seed_bootstrap_prior_fast_sets_j_eff_045() {
    let backend = make_backend();
    seed_bootstrap_prior("adapter-c", &ProfileTier::Fast, 10, &backend)
        .await
        .expect("seed must succeed");

    let state = backend
        .get_adapter_opro_state("adapter-c")
        .await
        .expect("get must succeed")
        .expect("state must exist");

    assert!(
        (state.j_eff_ema - 0.45).abs() < 1e-9,
        "Fast tier j_eff_ema must be 0.45, got {}",
        state.j_eff_ema
    );
}

#[tokio::test]
async fn seed_bootstrap_prior_is_idempotent() {
    let backend = make_backend();

    // First call: seeds with Capable (0.78).
    seed_bootstrap_prior("adapter-idem", &ProfileTier::Capable, 10, &backend)
        .await
        .expect("first seed must succeed");

    // Second call with a different tier: must be skipped (idempotency guard).
    seed_bootstrap_prior("adapter-idem", &ProfileTier::Fast, 10, &backend)
        .await
        .expect("second seed must succeed (noop)");

    let state = backend
        .get_adapter_opro_state("adapter-idem")
        .await
        .expect("get must succeed")
        .expect("state must exist");

    // j_eff_ema must still be 0.78 (from the first call), not 0.45.
    assert!(
        (state.j_eff_ema - 0.78).abs() < 1e-9,
        "idempotency guard must preserve original seeding; j_eff_ema = {}",
        state.j_eff_ema
    );
}

// ── seed_all_bootstrap_priors ─────────────────────────────────────────────────

#[tokio::test]
async fn seed_all_bootstrap_priors_seeds_every_profile() {
    let backend = make_backend();
    let profiles = vec![
        AdapterProfile {
            name: "p-capable".into(),
            kind: mock_kind(),
            tier: ProfileTier::Capable,
            is_reasoning_model: false,
        },
        AdapterProfile {
            name: "p-standard".into(),
            kind: mock_kind(),
            tier: ProfileTier::Standard,
            is_reasoning_model: false,
        },
        AdapterProfile {
            name: "p-fast".into(),
            kind: mock_kind(),
            tier: ProfileTier::Fast,
            is_reasoning_model: false,
        },
    ];
    let cfg = CalibrationBootstrapConfig::default();

    seed_all_bootstrap_priors(&profiles, &cfg, &backend).await;

    for (name, expected) in [("p-capable", 0.78), ("p-standard", 0.62), ("p-fast", 0.45)] {
        let state = backend
            .get_adapter_opro_state(name)
            .await
            .expect("get must succeed")
            .expect("state must exist after seeding");
        assert!(
            (state.j_eff_ema - expected).abs() < 1e-9,
            "{name}: expected j_eff_ema={expected}, got {}",
            state.j_eff_ema
        );
    }
}

#[tokio::test]
async fn seed_all_bootstrap_priors_empty_profiles_is_noop() {
    let backend = make_backend();
    let cfg = CalibrationBootstrapConfig::default();
    // Must not panic with an empty profile list.
    seed_all_bootstrap_priors(&[], &cfg, &backend).await;
}
