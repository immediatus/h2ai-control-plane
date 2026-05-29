#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
//! Unit tests for `h2ai_api::shadow_auditor`.
//!
//! Uses `AppState::new_for_tests()` so no NATS server is needed.
//! NATS-dependent paths (event publish, KV persistence) are skipped because
//! `AppState.nats` is `None` in the test harness.

use std::sync::Arc;

use h2ai_api::{shadow_auditor::ShadowAuditorAccumulator, state::AppState};
use h2ai_config::{H2AIConfig, SafetyConfig, ShadowAuditorConfig};
use h2ai_test_utils::decomposition_adapter;
use h2ai_types::{
    events::ShadowAuditorResultEvent,
    identity::{ExplorerId, TaskId},
};

fn make_state_with_shadow_cfg(cfg: ShadowAuditorConfig) -> AppState {
    let adapter = Arc::new(decomposition_adapter("mock"));
    let h2ai_cfg = H2AIConfig {
        safety: SafetyConfig {
            shadow_auditor: cfg,
            ..SafetyConfig::default()
        },
        ..H2AIConfig::default()
    };
    AppState::new_for_tests(
        h2ai_cfg,
        vec![adapter.clone() as Arc<dyn h2ai_types::adapter::IComputeAdapter>],
        adapter as Arc<dyn h2ai_types::adapter::IComputeAdapter>,
    )
}

fn make_state() -> AppState {
    let adapter = Arc::new(decomposition_adapter("mock"));
    AppState::new_for_tests(
        H2AIConfig::default(),
        vec![adapter.clone() as Arc<dyn h2ai_types::adapter::IComputeAdapter>],
        adapter as Arc<dyn h2ai_types::adapter::IComputeAdapter>,
    )
}

fn event(domain: &str, disagreement: bool) -> ShadowAuditorResultEvent {
    ShadowAuditorResultEvent {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        primary_approved: true,
        shadow_approved: !disagreement,
        disagreement,
        domain: domain.to_string(),
        primary_family: "FamilyA".into(),
        shadow_family: "FamilyB".into(),
        timestamp_ms: 0,
    }
}

// ── empty batch ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn process_empty_events_is_noop() {
    let state = Arc::new(make_state());
    let mut acc = ShadowAuditorAccumulator::new(state.clone());
    acc.process(vec![]).await;
    let promoted = state.promoted_audit_domains.read().await;
    assert!(
        promoted.is_empty(),
        "no domains should be promoted for empty input"
    );
}

// ── below promotion window ────────────────────────────────────────────────────

#[tokio::test]
async fn process_below_window_does_not_promote() {
    // Default window = 30; send 29 disagreements — must not promote.
    let state = Arc::new(make_state());
    let mut acc = ShadowAuditorAccumulator::new(state.clone());

    let events: Vec<_> = (0..29).map(|_| event("billing", true)).collect();
    acc.process(events).await;

    let promoted = state.promoted_audit_domains.read().await;
    assert!(
        !promoted.contains("billing"),
        "29 obs < window=30, must not promote"
    );
}

// ── promotion at threshold ────────────────────────────────────────────────────

#[tokio::test]
async fn process_at_window_with_high_rate_promotes_domain() {
    // Use a small window (3) and threshold (0.5) for speed.
    let state = Arc::new(make_state_with_shadow_cfg(ShadowAuditorConfig {
        enabled: true,
        promotion_threshold: 0.5,
        promotion_window: 3,
        auto_demotion: false,
        strict: false,
    }));
    let mut acc = ShadowAuditorAccumulator::new(state.clone());

    // 3 consecutive disagreements: rate = 1.0 > 0.5 → promotes.
    acc.process(vec![
        event("auth", true),
        event("auth", true),
        event("auth", true),
    ])
    .await;

    let promoted = state.promoted_audit_domains.read().await;
    assert!(
        promoted.contains("auth"),
        "domain 'auth' must be promoted after 3/3 disagreements"
    );
}

#[tokio::test]
async fn process_at_window_with_low_rate_does_not_promote() {
    // Window=4, threshold=0.5: 1/4 disagreements = 0.25 < 0.5 → no promotion.
    let state = Arc::new(make_state_with_shadow_cfg(ShadowAuditorConfig {
        enabled: true,
        promotion_threshold: 0.5,
        promotion_window: 4,
        auto_demotion: false,
        strict: false,
    }));
    let mut acc = ShadowAuditorAccumulator::new(state.clone());

    acc.process(vec![
        event("search", true),
        event("search", false),
        event("search", false),
        event("search", false),
    ])
    .await;

    let promoted = state.promoted_audit_domains.read().await;
    assert!(
        !promoted.contains("search"),
        "1/4 rate must not cross promotion threshold 0.5"
    );
}

// ── already promoted → no double-insert ──────────────────────────────────────

#[tokio::test]
async fn process_already_promoted_does_not_duplicate() {
    let state = Arc::new(make_state_with_shadow_cfg(ShadowAuditorConfig {
        enabled: true,
        promotion_threshold: 0.5,
        promotion_window: 2,
        auto_demotion: false,
        strict: false,
    }));
    let mut acc = ShadowAuditorAccumulator::new(state.clone());

    // First batch promotes the domain.
    acc.process(vec![event("db", true), event("db", true)])
        .await;
    assert!(state.promoted_audit_domains.read().await.contains("db"));

    // Second batch — domain already promoted, no change.
    acc.process(vec![event("db", true), event("db", true)])
        .await;
    let count = state.promoted_audit_domains.read().await.len();
    assert_eq!(count, 1, "must still have exactly 1 promoted domain");
}

// ── metrics update ────────────────────────────────────────────────────────────

#[tokio::test]
async fn process_updates_shadow_audit_metrics() {
    let state = Arc::new(make_state());
    let mut acc = ShadowAuditorAccumulator::new(state.clone());

    acc.process(vec![
        event("checkout", true),
        event("checkout", false),
        event("checkout", true),
    ])
    .await;

    let m = state.metrics.read().await;
    assert_eq!(m.shadow_audit_total, 3, "total must count all events");
    assert_eq!(
        m.shadow_audit_disagreements, 2,
        "disagreement count must be 2"
    );
    assert!(
        (m.shadow_audit_disagreement_rate - 2.0 / 3.0).abs() < 1e-9,
        "disagreement_rate must be 2/3, got {}",
        m.shadow_audit_disagreement_rate
    );
}

// ── auto-demotion branch ─────────────────────────────────────────────────────

#[tokio::test]
async fn auto_demotion_enabled_promoted_domain_stays_while_n_below_demotion_window() {
    // promotion_window=2, so demotion_window=4. After promotion the per-domain DomainWindow
    // is bounded to 2 items — n_observations never reaches 4, so no demotion fires.
    // This test exercises the `currently_promoted && auto_demotion` branch.
    let state = Arc::new(make_state_with_shadow_cfg(ShadowAuditorConfig {
        enabled: true,
        promotion_threshold: 0.5,
        promotion_window: 2,
        auto_demotion: true,
        strict: false,
    }));
    let mut acc = ShadowAuditorAccumulator::new(state.clone());

    // Promote: 2/2 disagreements.
    acc.process(vec![event("shipping", true), event("shipping", true)])
        .await;
    assert!(state
        .promoted_audit_domains
        .read()
        .await
        .contains("shipping"));

    // Feed 4 agree events — enters auto_demotion branch but n (=2) < demotion_window (=4).
    acc.process(vec![
        event("shipping", false),
        event("shipping", false),
        event("shipping", false),
        event("shipping", false),
    ])
    .await;

    assert!(
        state
            .promoted_audit_domains
            .read()
            .await
            .contains("shipping"),
        "domain must remain promoted when n_observations < demotion_window"
    );
}

// ── actual demotion fires ─────────────────────────────────────────────────────

#[tokio::test]
async fn auto_demotion_fires_when_conditions_are_met() {
    // Use promotion_window=0 → demotion_window=0; any n_observations >= 0 satisfies the
    // window check. Pre-populate domain as promoted so the demotion branch fires on first event.
    let state = Arc::new(make_state_with_shadow_cfg(ShadowAuditorConfig {
        enabled: true,
        promotion_threshold: 0.1, // demotion_threshold = 0.05
        promotion_window: 0,      // demotion_window = 0
        auto_demotion: true,
        strict: false,
    }));

    // Domain is already promoted before the accumulator starts running.
    state
        .promoted_audit_domains
        .write()
        .await
        .insert("cache".to_string());

    let mut acc = ShadowAuditorAccumulator::new(state.clone());

    // Single agree event: dr = 0.0 < demotion_threshold (0.05) → demotion fires.
    acc.process(vec![event("cache", false)]).await;

    assert!(
        !state.promoted_audit_domains.read().await.contains("cache"),
        "domain must be demoted when n_obs >= demotion_window and dr < demotion_threshold"
    );
}

// ── multi-domain isolation ────────────────────────────────────────────────────

#[tokio::test]
async fn process_multiple_domains_are_tracked_independently() {
    let state = Arc::new(make_state_with_shadow_cfg(ShadowAuditorConfig {
        enabled: true,
        promotion_threshold: 0.5,
        promotion_window: 2,
        auto_demotion: false,
        strict: false,
    }));
    let mut acc = ShadowAuditorAccumulator::new(state.clone());

    acc.process(vec![
        event("billing", true),
        event("billing", true), // 2/2 → promotes
        event("search", false),
        event("search", false), // 0/2 → no promotion
    ])
    .await;

    let promoted = state.promoted_audit_domains.read().await;
    assert!(promoted.contains("billing"), "billing must be promoted");
    assert!(!promoted.contains("search"), "search must not be promoted");
}
