use h2ai_config::H2AIConfig;
use h2ai_constraints::loader::parse_constraint_doc;
use h2ai_constraints::types::{
    ConstraintDoc, ConstraintPredicate, ConstraintSeverity, VocabularyMode,
};
use h2ai_context::compiler::{compile, ContextError};

fn cfg() -> H2AIConfig {
    H2AIConfig::default()
}

fn constraint_budget() -> ConstraintDoc {
    parse_constraint_doc(
        "ADR-004",
        r#"
## Constraints
- All budget mutations MUST use Redis Lua idempotency key
- No per-request state may be stored in service memory
"#,
    )
}

fn constraint_grpc() -> ConstraintDoc {
    parse_constraint_doc(
        "ADR-002",
        r#"
## Constraints
- Internal services MUST use gRPC
- REST is only permitted for external APIs
"#,
    )
}

#[tokio::test]
async fn compile_returns_error_when_j_eff_below_threshold() {
    let result = compile(
        "design a caching layer",
        &[],
        "grpc latency budget redis",
        &cfg(),
        None,
    )
    .await;
    assert!(matches!(result, Err(ContextError::ContextUnderflow { j_eff, .. }) if j_eff < 0.4));
}

#[tokio::test]
async fn compile_succeeds_when_j_eff_at_or_above_threshold() {
    let corpus = vec![constraint_budget(), constraint_grpc()];
    let result = compile(
        "enforce budget pacing idempotency with redis and grpc internal services",
        &corpus,
        "budget redis idempotency grpc internal",
        &cfg(),
        None,
    )
    .await;
    assert!(result.is_ok());
    let cr = result.unwrap();
    assert!(cr.j_eff >= 0.4);
}

#[tokio::test]
async fn compiled_system_context_contains_adr_source_name() {
    let corpus = vec![constraint_budget()];
    let result = compile(
        "prevent double-billing on restart using redis idempotency budget mutations memory",
        &corpus,
        "redis idempotency budget mutations memory",
        &cfg(),
        None,
    )
    .await;
    let cr = result.unwrap();
    assert!(cr.system_context.contains("ADR-004"));
}

#[tokio::test]
async fn compiled_system_context_contains_manifest() {
    let manifest =
        "prevent double-billing on restart using redis idempotency budget mutations memory";
    let corpus = vec![constraint_budget()];
    let result = compile(
        manifest,
        &corpus,
        "redis idempotency budget mutations memory",
        &cfg(),
        None,
    )
    .await;
    let cr = result.unwrap();
    assert!(cr.system_context.contains(manifest));
}

#[tokio::test]
async fn j_eff_recorded_in_result() {
    let corpus = vec![constraint_budget(), constraint_grpc()];
    let result = compile(
        "budget redis idempotency grpc internal services memory",
        &corpus,
        "budget redis idempotency grpc internal",
        &cfg(),
        None,
    )
    .await;
    let cr = result.unwrap();
    assert!(cr.j_eff > 0.0 && cr.j_eff <= 1.0);
}

#[tokio::test]
async fn compile_respects_custom_j_eff_gate() {
    let mut custom = H2AIConfig::default();
    custom.j_eff_gate = 0.99;
    let corpus = vec![constraint_budget()];
    let result = compile(
        "prevent double-billing on restart using redis idempotency budget mutations memory",
        &corpus,
        "redis idempotency budget mutations memory",
        &custom,
        None,
    )
    .await;
    assert!(matches!(result, Err(ContextError::ContextUnderflow { .. })));
}

#[tokio::test]
async fn compile_passes_when_j_eff_exactly_equals_gate() {
    let tokens = "redis idempotency budget mutations memory";
    let mut cfg_at_boundary = H2AIConfig::default();
    cfg_at_boundary.j_eff_gate = 0.4;
    let result = compile(tokens, &[], tokens, &cfg_at_boundary, None).await;
    assert!(result.is_ok(), "j_eff exactly at or above gate must pass");
}

#[tokio::test]
async fn compile_with_empty_corpus_uses_manifest_only() {
    let manifest = "redis idempotency budget mutations memory";
    let result = compile(manifest, &[], manifest, &cfg(), None).await;
    let cr = result.unwrap();
    assert!(cr.system_context.contains(manifest));
}

#[tokio::test]
async fn compile_with_empty_required_keywords_fails_gate() {
    let result = compile("anything here", &[], "", &cfg(), None).await;
    assert!(
        matches!(result, Err(ContextError::ContextUnderflow { j_eff, .. }) if j_eff == 0.0),
        "empty required keywords must produce J_eff=0.0 and fail gate"
    );
}

#[tokio::test]
async fn compile_empty_manifest_with_empty_required_keywords_fails_gate() {
    let result = compile("", &[], "", &cfg(), None).await;
    assert!(matches!(result, Err(ContextError::ContextUnderflow { j_eff, .. }) if j_eff == 0.0),);
}

// ── Signed J_eff / contamination tests ───────────────────────────────────────

fn adr_with_negative_keyword(id: &str, positive: &[&str], negative: &[&str]) -> ConstraintDoc {
    use h2ai_constraints::types::CompositeOp;
    ConstraintDoc {
        id: id.to_owned(),
        source_file: id.to_owned(),
        description: String::new(),
        severity: ConstraintSeverity::Hard { threshold: 0.8 },
        predicate: ConstraintPredicate::Composite {
            op: CompositeOp::And,
            children: vec![
                ConstraintPredicate::VocabularyPresence {
                    mode: VocabularyMode::AllOf,
                    terms: positive.iter().map(|s| s.to_string()).collect(),
                },
                ConstraintPredicate::NegativeKeyword {
                    terms: negative.iter().map(|s| s.to_string()).collect(),
                },
            ],
        },
        remediation_hint: None,
    }
}

#[tokio::test]
async fn contamination_is_zero_when_no_negative_vocab() {
    let corpus = vec![constraint_budget()];
    let result = compile(
        "enforce budget pacing idempotency with redis and grpc internal services",
        &corpus,
        "budget redis idempotency grpc internal",
        &cfg(),
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        result.contamination, 0.0,
        "no negative vocab → contamination must be 0.0"
    );
}

#[tokio::test]
async fn contamination_is_positive_when_manifest_uses_prohibited_term() {
    let corpus = vec![adr_with_negative_keyword(
        "ADR-006",
        &["zgc", "java", "virtual", "threads"],
        &["g1gc"],
    )];
    let mut permissive = cfg();
    permissive.j_eff_gate = 0.0;
    let result = compile(
        "configure the java service with g1gc heap settings zgc virtual threads",
        &corpus,
        "zgc java virtual threads",
        &permissive,
        None,
    )
    .await
    .unwrap();
    assert!(
        result.contamination > 0.0,
        "manifest with prohibited term 'g1gc' must have contamination > 0, got {}",
        result.contamination
    );
}

#[tokio::test]
async fn j_eff_is_lower_for_constraint_violating_manifest() {
    let corpus = vec![adr_with_negative_keyword(
        "ADR-006",
        &["zgc", "java"],
        &["g1gc"],
    )];
    let required = "zgc java heap threads";
    let mut permissive = cfg();
    permissive.j_eff_gate = 0.0;

    let compliant = compile(
        "configure zgc heap for java service with virtual threads",
        &corpus,
        required,
        &permissive,
        None,
    )
    .await
    .unwrap();

    let violating = compile(
        "configure g1gc heap for java service with virtual threads",
        &corpus,
        required,
        &permissive,
        None,
    )
    .await
    .unwrap();

    assert!(
        compliant.j_eff > violating.j_eff,
        "compliant manifest (j_eff={:.3}) must score higher than violating (j_eff={:.3})",
        compliant.j_eff,
        violating.j_eff
    );
    assert_eq!(
        compliant.contamination, 0.0,
        "compliant manifest must have zero contamination"
    );
    assert!(
        violating.contamination > 0.0,
        "violating manifest must have positive contamination"
    );
}

#[tokio::test]
async fn j_eff_penalises_multiple_prohibited_terms() {
    let corpus = vec![adr_with_negative_keyword(
        "ADR-004",
        &["redis", "atomic", "lua"],
        &["synchronised", "synchronized", "lock", "mutex"],
    )];
    let required = "redis atomic lua";
    let mut permissive = cfg();
    permissive.j_eff_gate = 0.0;

    let no_violation = compile(
        "redis atomic lua idempotent budget",
        &corpus,
        required,
        &permissive,
        None,
    )
    .await
    .unwrap();

    let heavy_violation = compile(
        "redis atomic lua synchronised mutex lock budget",
        &corpus,
        required,
        &permissive,
        None,
    )
    .await
    .unwrap();

    assert!(
        no_violation.j_eff > heavy_violation.j_eff,
        "manifest with multiple prohibited terms must score lower: no_violation={:.3} heavy_violation={:.3}",
        no_violation.j_eff,
        heavy_violation.j_eff
    );
    assert_eq!(
        no_violation.contamination, 0.0,
        "no-violation manifest must have zero contamination"
    );
    assert!(
        heavy_violation.contamination > 0.0,
        "heavy-violation manifest must have positive contamination"
    );
}
