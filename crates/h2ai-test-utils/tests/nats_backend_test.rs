use h2ai_state::backend::NatsBackend;
use h2ai_state::backend::TaskDispatchBackend;
use h2ai_test_utils::{MockNatsBackend, MockTaskDispatchBackend};
use std::sync::Arc;

#[test]
fn mock_nats_backend_satisfies_nats_backend_trait() {
    let mock = MockNatsBackend::new();
    let _: Arc<dyn NatsBackend> = Arc::new(mock);
}

#[test]
fn mock_task_dispatch_backend_satisfies_trait() {
    let mock = MockTaskDispatchBackend::new();
    let _: Arc<dyn TaskDispatchBackend> = Arc::new(mock);
}
