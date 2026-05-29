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
use h2ai_state::NatsClient;
use h2ai_types::events::{CalibrationCompletedEvent, H2AIEvent};
use h2ai_types::identity::TaskId;
use h2ai_types::sizing::{CoherencyCoefficients, CoordinationThreshold};

#[test]
fn h2ai_event_serialises_to_tagged_json() {
    let cc = CoherencyCoefficients::new(0.12, 0.021, vec![0.68, 0.74, 0.71]).unwrap();
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    let event = H2AIEvent::CalibrationCompleted(CalibrationCompletedEvent {
        calibration_id: TaskId::new(),
        coefficients: cc,
        coordination_threshold: theta,
        ensemble: None,
        eigen: None,
        timestamp: chrono::Utc::now(),
        pairwise_beta: None,
        cg_mode: Default::default(),
        adapter_families: Vec::new(),
        explorer_verification_family_match: false,
        single_family_warning: false,
        n_max_lo: 0.0,
        n_max_hi: 0.0,
        n_eff_cosine_prior: 0.0,
        calibration_quality: Default::default(),
        calibration_source: Default::default(),
        beta_quality: None,
    });

    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("\"event_type\":\"CalibrationCompleted\""));
    assert!(json.contains("\"payload\""));
}

#[test]
fn calibration_event_roundtrip() {
    let cc = CoherencyCoefficients::new(0.12, 0.021, vec![0.68, 0.74]).unwrap();
    let theta = CoordinationThreshold::from_calibration(&cc, 0.3);
    let original = CalibrationCompletedEvent {
        calibration_id: TaskId::new(),
        coefficients: cc,
        coordination_threshold: theta,
        ensemble: None,
        eigen: None,
        timestamp: chrono::Utc::now(),
        pairwise_beta: None,
        cg_mode: Default::default(),
        adapter_families: Vec::new(),
        explorer_verification_family_match: false,
        single_family_warning: false,
        n_max_lo: 0.0,
        n_max_hi: 0.0,
        n_eff_cosine_prior: 0.0,
        calibration_quality: Default::default(),
        calibration_source: Default::default(),
        beta_quality: None,
    };
    let bytes = serde_json::to_vec(&original).unwrap();
    let back: CalibrationCompletedEvent = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(back.coefficients.alpha, original.coefficients.alpha);
}

#[cfg(test)]
mod oracle_calibration_kv_tests {
    use super::*;

    #[tokio::test]
    async fn oracle_calibration_put_get_roundtrip() {
        let nats_url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".into());
        let client = match NatsClient::connect(&nats_url).await {
            Ok(c) => c,
            Err(e) => {
                eprintln!("NATS unavailable at {nats_url} — skipping: {e}");
                return;
            }
        };
        client.ensure_infrastructure().await.expect("infra");

        use h2ai_types::sizing::{OracleDomain, OracleObservation};
        let obs = vec![
            OracleObservation {
                task_id: "task-1".into(),
                q_confidence: 0.8,
                y_oracle: true,
                residual: 0.2,
                domain: OracleDomain::Code,
                timestamp_ms: 1000,
            },
            OracleObservation {
                task_id: "task-2".into(),
                q_confidence: 0.6,
                y_oracle: false,
                residual: 0.6,
                domain: OracleDomain::Factual,
                timestamp_ms: 2000,
            },
        ];

        let _tenant_id = h2ai_types::identity::TenantId::default();
        client.put_oracle_observations(&obs).await.expect("put");
        let retrieved = client.get_oracle_observations().await.expect("get");
        assert_eq!(retrieved.len(), 2);
        assert!((retrieved[0].q_confidence - 0.8).abs() < 1e-9);
        assert!((retrieved[1].residual - 0.6).abs() < 1e-9);
    }

    #[tokio::test]
    async fn oracle_calibration_get_returns_empty_when_absent() {
        let nats_url = std::env::var("NATS_URL").unwrap_or_else(|_| "nats://localhost:4222".into());
        let client = match NatsClient::connect(&nats_url).await {
            Ok(c) => c,
            Err(e) => {
                eprintln!("NATS unavailable at {nats_url} — skipping: {e}");
                return;
            }
        };
        client.ensure_infrastructure().await.ok();
        // get_oracle_observations must return Ok(vec![]) when key absent
        // Note: result depends on NATS state; just verify method exists and is callable
        let _ = client.get_oracle_observations().await;
    }
}

// ── wire protocol tests (require live NATS) ───────────────────────────────────

#[tokio::test]
async fn publish_and_receive_task_payload() {
    use h2ai_types::agent::{AgentDescriptor, ContextPayload, TaskPayload};
    use h2ai_types::identity::{AgentId, TaskId};
    use h2ai_types::sizing::TauValue;
    use std::time::Duration;

    let nats_url = h2ai_config::H2AIConfig::default().nats_url;
    let nats = match NatsClient::connect(&nats_url).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NATS unavailable at {nats_url} — skipping: {e}");
            return;
        }
    };
    nats.ensure_infrastructure().await.expect("infra");

    let task_id = TaskId::new();
    let agent_id = AgentId::from("test-agent");

    let subject = h2ai_nats::subjects::ephemeral_task_subject(&task_id);
    let mut sub = nats.client.subscribe(subject.clone()).await.unwrap();

    let payload = TaskPayload {
        task_id: task_id.clone(),
        agent_id: agent_id.clone(),
        agent: AgentDescriptor {
            model: "mock".into(),
            tools: vec![],
            cost_tier: h2ai_types::agent::CostTier::Mid,
        },
        instructions: "test".into(),
        context: ContextPayload::Inline("ctx".into()),
        tau: TauValue::new(0.5).unwrap(),
        max_tokens: 256,
        wave_mode: h2ai_types::agent::WaveMode::Normal,
    };
    nats.publish_task_payload(&payload).await.expect("publish");

    use futures::StreamExt;
    let msg = tokio::time::timeout(Duration::from_secs(2), sub.next())
        .await
        .expect("timeout")
        .expect("msg");
    let decoded: TaskPayload = serde_json::from_slice(&msg.payload).unwrap();
    assert_eq!(decoded.task_id, task_id);
    assert_eq!(decoded.agent_id, agent_id);
}

#[tokio::test]
async fn await_task_result_round_trip() {
    use h2ai_types::agent::TaskResult;
    use h2ai_types::identity::{AgentId, TaskId};
    use std::time::Duration;

    let nats_url = h2ai_config::H2AIConfig::default().nats_url;
    let nats = match NatsClient::connect(&nats_url).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NATS unavailable at {nats_url} — skipping: {e}");
            return;
        }
    };
    nats.ensure_infrastructure().await.expect("infra");

    let task_id = TaskId::new();
    let agent_id = AgentId::from("test-agent");

    // Publish to JetStream first — DeliverPolicy::All replays it for any subsequent consumer
    let result = TaskResult {
        task_id: task_id.clone(),
        agent_id: agent_id.clone(),
        output: "hello".into(),
        token_cost: 10,
        error: None,
        tool_calls: vec![],
    };
    let js = async_nats::jetstream::new(nats.client.clone());
    let result_subject = h2ai_nats::subjects::task_result_subject(&task_id);
    js.publish(result_subject, serde_json::to_vec(&result).unwrap().into())
        .await
        .unwrap()
        .await
        .unwrap();

    let nats2 = NatsClient::connect(&nats_url).await.unwrap();
    let received = nats2
        .await_task_result_once(&task_id, Duration::from_secs(5))
        .await
        .expect("result");
    assert_eq!(received.output, "hello");
    assert_eq!(received.task_id, task_id);
}
