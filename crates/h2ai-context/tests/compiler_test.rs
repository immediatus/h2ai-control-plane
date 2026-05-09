use h2ai_constraints::types::ConstraintDoc;

use h2ai_context::compiler::compile;

#[test]
fn compiled_system_context_contains_adr_source_name() {
    let doc = ConstraintDoc::new_llm_judge(
        "ADR-004",
        "All budget mutations MUST use a Redis Lua idempotency key. No per-request state may be stored in service memory.",
    );
    let result = compile(
        "prevent double-billing on restart using redis idempotency budget mutations memory",
        &[doc],
        true,
    );
    assert!(result.system_context.contains("ADR-004"));
}

#[test]
fn compiled_system_context_contains_manifest() {
    let manifest =
        "prevent double-billing on restart using redis idempotency budget mutations memory";
    let doc = ConstraintDoc::new_llm_judge(
        "ADR-004",
        "All budget mutations MUST use a Redis Lua idempotency key.",
    );
    let result = compile(manifest, &[doc], true);
    assert!(result.system_context.contains(manifest));
}

#[test]
fn compile_with_empty_corpus_uses_manifest_only() {
    let manifest = "redis idempotency budget mutations memory";
    let result = compile(manifest, &[], false);
    assert!(result.system_context.contains(manifest));
}

#[test]
fn compile_multiple_constraints_includes_all_ids() {
    let doc_a = ConstraintDoc::new_llm_judge(
        "ADR-001",
        "Use stateless JWT tokens for authentication. No server-side session state.",
    );
    let doc_b = ConstraintDoc::new_llm_judge(
        "ADR-002",
        "Internal services MUST use gRPC for inter-service communication. REST is not permitted internally.",
    );
    let result = compile("implement stateless jwt grpc auth", &[doc_a, doc_b], true);
    assert!(result.system_context.contains("ADR-001"));
    assert!(result.system_context.contains("ADR-002"));
}
