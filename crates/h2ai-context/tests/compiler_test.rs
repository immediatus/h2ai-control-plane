use h2ai_config::H2AIConfig;
use h2ai_context::adr::{parse_adr, AdrConstraints};
use h2ai_context::compiler::{compile, ContextError};

fn cfg() -> H2AIConfig {
    H2AIConfig::default()
}

fn adr_budget() -> AdrConstraints {
    parse_adr(
        "ADR-004",
        r#"
## Constraints
- All budget mutations MUST use Redis Lua idempotency key
- No per-request state may be stored in service memory
"#,
    )
}

fn adr_grpc() -> AdrConstraints {
    parse_adr(
        "ADR-002",
        r#"
## Constraints
- Internal services MUST use gRPC
- REST is only permitted for external APIs
"#,
    )
}

#[test]
fn compile_returns_error_when_j_eff_below_threshold() {
    let result = compile(
        "design a caching layer",
        &[],
        "grpc latency budget redis",
        &cfg(),
    );
    assert!(matches!(result, Err(ContextError::ContextUnderflow { j_eff, .. }) if j_eff < 0.4));
}

#[test]
fn compile_succeeds_when_j_eff_at_or_above_threshold() {
    let corpus = vec![adr_budget(), adr_grpc()];
    let result = compile(
        "enforce budget pacing idempotency with redis and grpc internal services",
        &corpus,
        "budget redis idempotency grpc internal",
        &cfg(),
    );
    assert!(result.is_ok());
    let cr = result.unwrap();
    assert!(cr.j_eff >= 0.4);
}

#[test]
fn compiled_system_context_contains_adr_source_name() {
    let corpus = vec![adr_budget()];
    let result = compile(
        "prevent double-billing on restart using redis idempotency budget mutations memory",
        &corpus,
        "redis idempotency budget mutations memory",
        &cfg(),
    );
    let cr = result.unwrap();
    assert!(cr.system_context.contains("ADR-004"));
}

#[test]
fn compiled_system_context_contains_manifest() {
    let manifest =
        "prevent double-billing on restart using redis idempotency budget mutations memory";
    let corpus = vec![adr_budget()];
    let result = compile(
        manifest,
        &corpus,
        "redis idempotency budget mutations memory",
        &cfg(),
    );
    let cr = result.unwrap();
    assert!(cr.system_context.contains(manifest));
}

#[test]
fn j_eff_recorded_in_result() {
    let corpus = vec![adr_budget(), adr_grpc()];
    let result = compile(
        "budget redis idempotency grpc internal services memory",
        &corpus,
        "budget redis idempotency grpc internal",
        &cfg(),
    );
    let cr = result.unwrap();
    assert!(cr.j_eff > 0.0 && cr.j_eff <= 1.0);
}

#[test]
fn compile_respects_custom_j_eff_gate() {
    let mut custom = H2AIConfig::default();
    custom.j_eff_gate = 0.99;
    let corpus = vec![adr_budget()];
    let result = compile(
        "prevent double-billing on restart using redis idempotency budget mutations memory",
        &corpus,
        "redis idempotency budget mutations memory",
        &custom,
    );
    assert!(matches!(result, Err(ContextError::ContextUnderflow { .. })));
}
