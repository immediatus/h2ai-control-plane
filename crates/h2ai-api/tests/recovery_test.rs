#![allow(clippy::missing_panics_doc, clippy::missing_errors_doc)]
//! Unit tests for `h2ai_api::recovery`.
//!
//! `recover_in_flight_tasks` and `spawn_resume` require a live NATS server and are
//! exercised in integration environments.  Only the NATS-free surface is tested here.

use h2ai_api::recovery::local_node_id;

// ── local_node_id ─────────────────────────────────────────────────────────────

#[test]
fn local_node_id_contains_colon_separator() {
    let id = local_node_id();
    assert!(
        id.contains(':'),
        "local_node_id must have 'hostname:PID' format, got: {id}"
    );
}

#[test]
fn local_node_id_pid_segment_is_numeric() {
    let id = local_node_id();
    let pid_part = id.rsplit(':').next().expect("colon must be present");
    pid_part
        .parse::<u32>()
        .unwrap_or_else(|_| panic!("PID segment must be numeric, got: {pid_part}"));
}

#[test]
fn local_node_id_is_stable_within_process() {
    assert_eq!(
        local_node_id(),
        local_node_id(),
        "local_node_id must return the same value on repeated calls within the same process"
    );
}
