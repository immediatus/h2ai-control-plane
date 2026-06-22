#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::unused_async,
    clippy::default_trait_access,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::similar_names,
    clippy::match_same_arms,
    clippy::literal_string_with_formatting_args,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::useless_format,
    clippy::option_if_let_else,
    clippy::map_unwrap_or,
    clippy::cloned_instead_of_copied,
    clippy::trivially_copy_pass_by_ref,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::needless_pass_by_value,
    clippy::explicit_iter_loop,
    clippy::needless_borrow,
    clippy::large_futures,
    clippy::manual_string_new,
    clippy::needless_lifetimes,
    clippy::elidable_lifetime_names,
    clippy::redundant_else,
    clippy::stable_sort_primitive,
    clippy::type_complexity,
    clippy::wildcard_imports,
    clippy::single_match_else,
    clippy::missing_fields_in_debug,
    clippy::doc_link_with_quotes,
    clippy::implicit_hasher,
    clippy::needless_collect,
    clippy::suboptimal_flops,
    clippy::missing_const_for_fn,
    clippy::needless_type_cast,
    clippy::unreadable_literal,
    clippy::no_effect_underscore_binding
)]
use h2ai_api::task_pipeline::{build_tao_config, is_network_error};
use h2ai_config::H2AIConfig;
use h2ai_types::config::ParetoWeights;
use h2ai_types::manifest::TaskManifest;
use h2ai_types::sizing::condorcet_quality;
use serde_json::json;

// ── compute_j_eff_raw invariants ──────────────────────────────────────────────
//
// compute_j_eff_raw is a private function in routes/tasks.rs, tested here by
// reimplementing the same formula using the public condorcet_quality function
// from h2ai-types. The formula is:
//   filter_ratio = n_valid / n_agents
//   j_eff = clamp(condorcet_quality(n_valid, filter_ratio, rho) /
//                 condorcet_quality(n_agents, p_mean, 0.0), 0, 1)
// Returns None when n_agents == 0 (q_ceiling == 0).

fn j_eff_raw(n_valid: usize, n_agents: usize, p_mean: f64, rho_mean: f64) -> Option<f64> {
    let filter_ratio = if n_agents > 0 {
        n_valid as f64 / n_agents as f64
    } else {
        0.0
    };
    let q_realized = condorcet_quality(n_valid, filter_ratio, rho_mean);
    let q_ceiling = condorcet_quality(n_agents, p_mean, 0.0);
    if q_ceiling > 0.0 {
        Some((q_realized / q_ceiling).clamp(0.0, 1.0))
    } else {
        None
    }
}

#[test]
fn j_eff_zero_valid_gives_zero() {
    let j = j_eff_raw(0, 4, 0.75, 0.3);
    assert_eq!(j, Some(0.0));
}

#[test]
fn j_eff_full_pass_at_most_one() {
    let j = j_eff_raw(4, 4, 0.75, 0.0).unwrap();
    assert!(j <= 1.0 + 1e-9, "j_eff={j} exceeds 1.0");
}

#[test]
fn j_eff_partial_pass_less_than_full() {
    let j_half = j_eff_raw(2, 4, 0.75, 0.3).unwrap();
    let j_full = j_eff_raw(4, 4, 0.75, 0.3).unwrap();
    assert!(
        j_half < j_full,
        "partial={j_half} should be < full={j_full}"
    );
}

#[test]
fn j_eff_zero_agents_gives_none() {
    let j = j_eff_raw(0, 0, 0.75, 0.0);
    assert!(j.is_none(), "expected None for n_agents=0");
}

#[test]
fn pareto_weights_must_sum_to_one() {
    assert!(ParetoWeights::new(0.5, 0.5, 0.5).is_err());
    assert!(ParetoWeights::new(0.2, 0.3, 0.5).is_ok());
}

#[test]
fn task_manifest_deserialises_from_api_shape() {
    let raw = json!({
        "description": "Propose stateless auth",
        "pareto_weights": {"diversity": 0.5, "containment": 0.3, "throughput": 0.2},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 4, "tau_min": 0.2, "tau_max": 0.9}
    });
    let m: TaskManifest = serde_json::from_value(raw).unwrap();
    assert_eq!(m.topology.kind, "ensemble");
    assert_eq!(m.explorers.count, 4);
}

#[test]
fn task_manifest_oracle_field_accessible() {
    // Verify oracle field exists and defaults to None — compile-time guard Phase 6.
    let raw = json!({
        "description": "test",
        "pareto_weights": {"diversity": 0.4, "containment": 0.3, "throughput": 0.3},
        "topology": {"kind": "ensemble"},
        "explorers": {"count": 2, "tau_min": 0.2, "tau_max": 0.9}
    });
    let m: TaskManifest = serde_json::from_value(raw).unwrap();
    assert!(m.oracle.is_none());
}

// ── is_network_error ──────────────────────────────────────────────────────────
//
// Regression: before the fix, "model hit max_tokens during thinking phase" was classified
// as is_network=true.  This caused the engine to stop silently instead of emitting a
// proper TaskFailed event (the adapter wraps budget errors in AdapterError::NetworkError).

#[test]
fn max_tokens_during_thinking_is_not_network_error() {
    assert!(
        !is_network_error(
            "adapter error: network error: model hit max_tokens during thinking phase; \
             increase max_tokens and retry"
        ),
        "max_tokens budget failure must NOT be treated as a network outage"
    );
}

#[test]
fn max_tokens_prefix_alone_is_not_network_error() {
    assert!(!is_network_error("max_tokens exceeded"));
}

#[test]
fn connection_refused_is_network_error() {
    assert!(is_network_error(
        "connection refused: host.docker.internal:8080"
    ));
}

#[test]
fn timed_out_is_network_error() {
    assert!(is_network_error(
        "network error: connection timed out after 30s"
    ));
}

#[test]
fn generic_network_error_without_max_tokens_is_network_error() {
    assert!(is_network_error("network error: unexpected EOF"));
}

// ── build_tao_config ─────────────────────────────────────────────────────────
//
// Regression: before the fix, `tao_config` was always built with `TaoConfig::default()`,
// ignoring `H2AIConfig.tao_per_turn_timeout_secs`.  The `[tao]` TOML section that scenario
// configs used was silently discarded by the config crate (H2AIConfig has no nested tao struct).

#[test]
fn tao_config_inherits_per_turn_timeout_from_h2ai_config() {
    let cfg = H2AIConfig {
        tao_per_turn_timeout_secs: 3600,
        ..Default::default()
    };
    let tao = build_tao_config(&cfg);
    assert_eq!(
        tao.per_turn_timeout_secs, 3600,
        "TaoConfig must inherit per_turn_timeout_secs from H2AIConfig; \
         before the fix it always used TaoConfig::default() = 600s regardless of config"
    );
}

#[test]
fn tao_config_default_is_600s() {
    let cfg = H2AIConfig::default();
    let tao = build_tao_config(&cfg);
    assert_eq!(
        tao.per_turn_timeout_secs, 600,
        "default tao_per_turn_timeout_secs in reference.toml must be 600s"
    );
}

#[test]
fn tao_config_inherits_timeout_retry_max_tokens_from_h2ai_config() {
    let cfg = H2AIConfig {
        tao_timeout_retry_max_tokens: 4096,
        ..Default::default()
    };
    let tao = build_tao_config(&cfg);
    assert_eq!(
        tao.timeout_retry_max_tokens, 4096,
        "TaoConfig must inherit timeout_retry_max_tokens from H2AIConfig; \
         before the fix it always used TaoConfig::default() = 512 which causes \
         thinking models to exhaust the retry budget and emit LlmAdapterUnavailable"
    );
}

#[test]
fn tao_config_default_timeout_retry_max_tokens_is_4096() {
    let cfg = H2AIConfig::default();
    let tao = build_tao_config(&cfg);
    assert_eq!(
        tao.timeout_retry_max_tokens, 4096,
        "default tao_timeout_retry_max_tokens in reference.toml must be 4096 (thinking-model safe)"
    );
}

#[test]
fn tao_config_preserves_retry_on_timeout_default() {
    let cfg = H2AIConfig {
        tao_per_turn_timeout_secs: 1800,
        ..Default::default()
    };
    let tao = build_tao_config(&cfg);
    let defaults = h2ai_types::config::TaoConfig::default();
    assert_eq!(tao.retry_on_timeout, defaults.retry_on_timeout);
}
