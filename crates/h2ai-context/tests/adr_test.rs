use h2ai_context::adr::parse_adr;

const ADR_004: &str = r#"
# ADR-004: Budget Pacing with Idempotency

## Status
Accepted

## Context
Budget enforcement must survive server restarts.

## Decision
Use Redis Lua atomic check-and-set with 30s TTL.

## Constraints
- All budget mutations MUST use Redis Lua idempotency key
- No per-request state may be stored in service memory
- TTL must not exceed 60 seconds

## Consequences
Survives restarts. Slightly higher Redis load.
"#;

const ADR_NO_CONSTRAINTS: &str = r#"
# ADR-999: Example without constraints section

## Status
Accepted

## Decision
Use PostgreSQL.
"#;

#[test]
fn parse_adr_extracts_constraint_keywords() {
    let result = parse_adr("ADR-004", ADR_004);
    let vocab = result.vocabulary();
    assert!(vocab.contains("redis"));
    assert!(vocab.contains("idempotency"));
    assert!(vocab.contains("budget"));
}

#[test]
fn parse_adr_returns_empty_keywords_when_no_constraints_section() {
    let result = parse_adr("ADR-999", ADR_NO_CONSTRAINTS);
    assert!(result.vocabulary().is_empty());
}

#[test]
fn parse_adr_stores_source_name() {
    let result = parse_adr("ADR-004", ADR_004);
    assert_eq!(result.id, "ADR-004");
}

#[test]
fn parse_adr_only_extracts_constraints_section_not_context() {
    let result = parse_adr("ADR-004", ADR_004);
    // "enforcement" appears only in Context section, not Constraints
    assert!(!result.vocabulary().contains("enforcement"));
}
