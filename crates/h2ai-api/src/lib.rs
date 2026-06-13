//! axum HTTP gateway, SSE event stream, and OPRO bandit.
//!
//! This is the sole HTTP-facing crate in the workspace. No other crate imports
//! it. All REST endpoints and the server-sent event stream live here. The main
//! binary (`h2ai-control-plane`) links this crate and calls into [`routes`] to
//! build the axum router.
//!
//! ## Key subsystems
//!
//! - [`routes`] — axum handlers: `POST /:tenant/tasks` (submit), `GET
//!   /:tenant/tasks/:id/events` (SSE), `POST …/merge`, `POST …/clarify`,
//!   `POST …/approve`, `POST /calibrate`, `GET /health`, `GET /ready`.
//! - [`task_pipeline`] — orchestrates the full task lifecycle: thinking loop,
//!   awareness probe, verification waves, autonomic retry, post-run attribution.
//! - [`bootstrap`] — OPRO prior seeding: `seed_bootstrap_prior` / `seed_all_bootstrap_priors`
//!   seed Thompson-sampling bandit priors for each adapter at startup.
//! - [`tenant_registry`] — per-tenant `Arc<TenantState>` cache; each tenant gets
//!   isolated calibration, bandit, and OPRO state.
//! - [`opro`] — Thompson-sampling bandit over adapter profiles; adapts prompt
//!   selection based on per-profile reward history.
//! - [`oracle`] / [`oracle_worker`] — optional oracle gate: LLM-driven post-hoc
//!   verification that runs asynchronously after the merge wave.
//! - [`shadow_auditor`] — `ShadowAuditorAccumulator`: tracks per-domain
//!   agree/disagree windows; promotes or demotes domains when disagreement rate
//!   crosses configured thresholds.
//! - [`recovery`] — `recover_in_flight_tasks`: scans NATS checkpoint KV at
//!   startup and resumes tasks left in-flight after a crash or restart.
//! - [`state`] — `AppState`: shared handle to NATS backend, config, tenant registry,
//!   and knowledge provider, cloned cheaply via `Arc`.
//! - [`rho_ema`] — `RhoEmaState`: rolling exponential moving average for the
//!   per-task quality signal used by the autonomic controller.
//! - [`debug_record`] — `TaskDebugRecord` struct and `append_debug_record` for
//!   persisting per-task debug artifacts (proposals, scores, verifier traces).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::unused_async,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::option_if_let_else,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::similar_names,
    clippy::match_same_arms,
    clippy::needless_pass_by_value,
    clippy::doc_markdown,
    clippy::doc_link_with_quotes,
    clippy::implicit_hasher,
    clippy::literal_string_with_formatting_args,
    clippy::missing_fields_in_debug,
    clippy::redundant_else,
    clippy::map_unwrap_or,
    clippy::type_complexity,
    clippy::float_cmp,
    clippy::large_futures,
    clippy::suboptimal_flops,
    clippy::needless_type_cast,
    clippy::elidable_lifetime_names
)]
pub mod bootstrap;
pub mod debug_record;
pub mod error;
pub mod metrics;
pub mod opro;
pub mod oracle;
pub mod oracle_worker;
pub mod recovery;
pub mod rho_ema;
pub mod routes;
pub mod shadow_auditor;
pub mod state;
pub mod task_pipeline;
pub mod tenant_registry;
