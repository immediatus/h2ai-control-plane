use h2ai_config::H2AIConfig;
use h2ai_constraints::loader::parse_constraint_doc;
use h2ai_context::compiler::compile;

fn cfg() -> H2AIConfig {
    H2AIConfig {
        j_eff_gate: 0.1,
        ..Default::default()
    }
}

#[tokio::test]
async fn compile_accepts_constraint_docs() {
    let doc = parse_constraint_doc(
        "ADR-001",
        "# ADR-001\n\n## Constraints\npersonal data minimization privacy gdpr\n",
    );
    let result = compile(
        "use personal data minimization techniques",
        &[doc],
        "personal data",
        &cfg(),
        None,
    )
    .await;
    assert!(result.is_ok());
    let cr = result.unwrap();
    assert!(cr.j_eff > 0.0);
}

#[tokio::test]
async fn compile_system_context_contains_constraint_id() {
    let doc = parse_constraint_doc(
        "GDPR-001",
        "# GDPR-001\n\n## Constraints\npersonal data privacy\n",
    );
    let result = compile(
        "handle personal data carefully",
        &[doc],
        "personal",
        &cfg(),
        None,
    )
    .await;
    let ctx = result.unwrap().system_context;
    assert!(
        ctx.contains("GDPR-001"),
        "system context must include constraint id"
    );
}
