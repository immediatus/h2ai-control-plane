use h2ai_constraints::types::ConstraintDoc;

use h2ai_context::compiler::compile;

#[test]
fn compile_accepts_constraint_docs() {
    let doc = ConstraintDoc::new_llm_judge(
        "ADR-001",
        "Personal data minimization — proposals must store minimum data required and must cite a GDPR legal basis.",
    );
    let result = compile("use personal data minimization techniques", &[doc], true);
    assert!(result.system_context.contains("ADR-001"));
}

#[test]
fn compile_system_context_contains_constraint_id() {
    let doc = ConstraintDoc::new_llm_judge(
        "GDPR-001",
        "Personal data privacy — proposals must not collect data beyond what the stated purpose requires.",
    );
    let result = compile("handle personal data carefully", &[doc], true);
    assert!(
        result.system_context.contains("GDPR-001"),
        "system context must include constraint id"
    );
}
