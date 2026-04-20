#[test]
fn nats_dispatch_config_is_accessible_from_engine() {
    // Compile-time check: NatsDispatchConfig is re-exported from engine module
    let _size = std::mem::size_of::<h2ai_orchestrator::engine::NatsDispatchConfig>();
}
