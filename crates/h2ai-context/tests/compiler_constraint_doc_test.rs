use h2ai_constraints::loader::parse_constraint_doc;
use h2ai_context::compiler::compile;

#[test]
fn compile_accepts_constraint_docs() {
    let doc = parse_constraint_doc(
        "ADR-001",
        "# ADR-001\n\n## Constraints\npersonal data minimization privacy gdpr\n",
    );
    let result = compile("use personal data minimization techniques", &[doc]);
    assert!(result.system_context.contains("ADR-001"));
}

#[test]
fn compile_system_context_contains_constraint_id() {
    let doc = parse_constraint_doc(
        "GDPR-001",
        "# GDPR-001\n\n## Constraints\npersonal data privacy\n",
    );
    let result = compile("handle personal data carefully", &[doc]);
    assert!(
        result.system_context.contains("GDPR-001"),
        "system context must include constraint id"
    );
}
