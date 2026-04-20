use async_trait::async_trait;
use chrono::Utc;
use h2ai_telemetry::error::AuditError;
use h2ai_telemetry::provider::AuditProvider;
use h2ai_types::agent::AgentTelemetryEvent;
use h2ai_types::identity::{AgentId, TaskId};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct CountingAuditProvider {
    count: Arc<AtomicUsize>,
}

#[async_trait]
impl AuditProvider for CountingAuditProvider {
    async fn record_event(&self, _event: AgentTelemetryEvent) -> Result<(), AuditError> {
        self.count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn flush(&self) -> Result<(), AuditError> {
        Ok(())
    }
}

fn make_event() -> AgentTelemetryEvent {
    AgentTelemetryEvent::SystemError {
        task_id: TaskId::new(),
        agent_id: AgentId::from("agent-1"),
        error: "test error".into(),
        timestamp: Utc::now(),
    }
}

#[tokio::test]
async fn audit_provider_records_event() {
    let count = Arc::new(AtomicUsize::new(0));
    let provider = CountingAuditProvider {
        count: count.clone(),
    };
    provider.record_event(make_event()).await.unwrap();
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn audit_provider_records_multiple_events() {
    let count = Arc::new(AtomicUsize::new(0));
    let provider = CountingAuditProvider {
        count: count.clone(),
    };
    for _ in 0..3 {
        provider.record_event(make_event()).await.unwrap();
    }
    assert_eq!(count.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn audit_provider_flush_succeeds() {
    let count = Arc::new(AtomicUsize::new(0));
    let provider = CountingAuditProvider { count };
    assert!(provider.flush().await.is_ok());
}
