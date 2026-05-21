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

use h2ai_orchestrator::oracle::{oracle_dispatch, OraclePublisher};
use h2ai_types::events::OraclePendingEvent;
use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::sizing::{OracleDomain, OracleSpec};
use std::sync::{Arc, Mutex};

// ── Mock ─────────────────────────────────────────────────────────────────────

struct MockPublisher {
    calls: Arc<Mutex<Vec<(String, Vec<u8>)>>>,
}

impl MockPublisher {
    fn new() -> Self {
        Self {
            calls: Arc::new(Mutex::new(vec![])),
        }
    }

    fn published(&self) -> Vec<(String, Vec<u8>)> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl OraclePublisher for MockPublisher {
    async fn publish_oracle(&self, subject: String, payload: bytes::Bytes) {
        self.calls.lock().unwrap().push((subject, payload.to_vec()));
    }
}

fn sample_spec() -> OracleSpec {
    OracleSpec {
        runner_uri: "http://oracle-sidecar:9090/evaluate".into(),
        timeout_ms: 5000,
        domain: OracleDomain::Code,
    }
}

// ── fire: publishes without error ─────────────────────────────────────────────

/// oracle_dispatch::fire must publish to the correct NATS subject.
#[tokio::test]
async fn fire_publishes_to_correct_subject() {
    let mock = MockPublisher::new();
    let task_id = TaskId::new();
    let tenant_id = TenantId::default_tenant();

    oracle_dispatch::fire(
        &mock,
        task_id.clone(),
        tenant_id.clone(),
        "output text",
        0.9,
        3,
        &sample_spec(),
    )
    .await;

    let calls = mock.published();
    assert_eq!(calls.len(), 1);
    let (subject, payload) = &calls[0];
    assert_eq!(
        subject,
        &format!("h2ai.oracle.{}.pending", tenant_id.as_ref())
    );

    let event: OraclePendingEvent = serde_json::from_slice(payload).expect("deserialize");
    assert_eq!(event.task_id, task_id);
    assert!((event.q_confidence - 0.9).abs() < 1e-9);
    assert_eq!(event.n_used, 3);
    assert_eq!(event.winning_output, "output text");
}

/// fire encodes the tenant_id in the NATS subject — different tenants get different subjects.
#[tokio::test]
async fn fire_subjects_are_tenant_scoped() {
    let mock_a = MockPublisher::new();
    let mock_b = MockPublisher::new();
    let task_id = TaskId::new();
    let tenant_a = TenantId("tenant_a".into());
    let tenant_b = TenantId("tenant_b".into());
    let spec = sample_spec();

    oracle_dispatch::fire(
        &mock_a,
        task_id.clone(),
        tenant_a.clone(),
        "output_a",
        0.8,
        2,
        &spec,
    )
    .await;
    oracle_dispatch::fire(
        &mock_b,
        task_id.clone(),
        tenant_b.clone(),
        "output_b",
        0.7,
        1,
        &spec,
    )
    .await;

    let calls_a = mock_a.published();
    let calls_b = mock_b.published();

    assert_eq!(calls_a.len(), 1);
    assert_eq!(calls_b.len(), 1);
    assert_eq!(
        calls_a[0].0,
        format!("h2ai.oracle.{}.pending", tenant_a.as_ref())
    );
    assert_eq!(
        calls_b[0].0,
        format!("h2ai.oracle.{}.pending", tenant_b.as_ref())
    );
    assert_ne!(calls_a[0].0, calls_b[0].0, "subjects must be tenant-scoped");
}

/// fire is fire-and-forget — must complete without panicking at edge confidence values.
#[tokio::test]
async fn fire_edge_confidence_values() {
    let mock = MockPublisher::new();
    let spec = sample_spec();

    oracle_dispatch::fire(
        &mock,
        TaskId::new(),
        TenantId::default_tenant(),
        "zero conf",
        0.0,
        1,
        &spec,
    )
    .await;
    oracle_dispatch::fire(
        &mock,
        TaskId::new(),
        TenantId::default_tenant(),
        "full conf",
        1.0,
        1,
        &spec,
    )
    .await;

    assert_eq!(
        mock.published().len(),
        2,
        "both edge-value calls must publish"
    );
}
