use h2ai_constraints::loader::parse_constraint_doc;
use h2ai_context::compiler::compile;

#[test]
fn compiled_system_context_contains_adr_source_name() {
    let doc = parse_constraint_doc(
        "ADR-004",
        "# ADR-004\n\n## Constraints\n- All budget mutations MUST use Redis Lua idempotency key\n- No per-request state may be stored in service memory\n",
    );
    let result = compile(
        "prevent double-billing on restart using redis idempotency budget mutations memory",
        &[doc],
    );
    assert!(result.system_context.contains("ADR-004"));
}

#[test]
fn compiled_system_context_contains_manifest() {
    let manifest =
        "prevent double-billing on restart using redis idempotency budget mutations memory";
    let doc = parse_constraint_doc(
        "ADR-004",
        "# ADR-004\n\n## Constraints\n- All budget mutations MUST use Redis Lua idempotency key\n",
    );
    let result = compile(manifest, &[doc]);
    assert!(result.system_context.contains(manifest));
}

#[test]
fn compile_with_empty_corpus_uses_manifest_only() {
    let manifest = "redis idempotency budget mutations memory";
    let result = compile(manifest, &[]);
    assert!(result.system_context.contains(manifest));
}

#[test]
fn compile_multiple_constraints_includes_all_ids() {
    let doc_a = parse_constraint_doc(
        "ADR-001",
        "# ADR-001\n\n## Constraints\n- Use stateless JWT tokens\n",
    );
    let doc_b = parse_constraint_doc(
        "ADR-002",
        "# ADR-002\n\n## Constraints\n- Internal services MUST use gRPC\n",
    );
    let result = compile("implement stateless jwt grpc auth", &[doc_a, doc_b]);
    assert!(result.system_context.contains("ADR-001"));
    assert!(result.system_context.contains("ADR-002"));
}
