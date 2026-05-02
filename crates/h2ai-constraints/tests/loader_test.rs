use h2ai_constraints::loader::{load_corpus, parse_constraint_doc};
use h2ai_constraints::types::ConstraintSeverity;

const ADR_MARKDOWN: &str = r#"
# ADR-001: Data Minimization

## Status
Accepted

## Constraints
personal data minimization privacy gdpr

## Context
We must comply with GDPR Article 5(1)(c).
"#;

const GENERIC_HARD: &str = r#"
# POLICY-001: No PII in Logs

## Hard Constraints
password pii secret token
"#;

const GENERIC_SOFT: &str = r#"
# POLICY-002: Prefer Structured Output

## Soft Constraints
json structured schema
"#;

const GENERIC_ADVISORY: &str = r#"
# POLICY-003: Performance Target

## Advisory
latency throughput benchmark
"#;

#[test]
fn adr_constraints_section_parsed_as_hard() {
    let doc = parse_constraint_doc("ADR-001", ADR_MARKDOWN);
    assert_eq!(doc.id, "ADR-001");
    match &doc.severity {
        ConstraintSeverity::Hard { threshold } => {
            assert!(
                (*threshold - 0.8).abs() < 1e-9,
                "default threshold must be 0.8"
            )
        }
        other => panic!("expected Hard, got {:?}", other),
    }
    let vocab = doc.vocabulary();
    assert!(vocab.contains("personal"), "vocab must contain 'personal'");
    assert!(vocab.contains("gdpr"), "vocab must contain 'gdpr'");
}

#[test]
fn hard_constraints_section_is_hard_severity() {
    let doc = parse_constraint_doc("POLICY-001", GENERIC_HARD);
    assert!(matches!(doc.severity, ConstraintSeverity::Hard { .. }));
    let vocab = doc.vocabulary();
    assert!(vocab.contains("password"));
}

#[test]
fn soft_constraints_section_is_soft_severity() {
    let doc = parse_constraint_doc("POLICY-002", GENERIC_SOFT);
    assert!(matches!(doc.severity, ConstraintSeverity::Soft { .. }));
}

#[test]
fn advisory_section_is_advisory_severity() {
    let doc = parse_constraint_doc("POLICY-003", GENERIC_ADVISORY);
    assert!(matches!(doc.severity, ConstraintSeverity::Advisory));
}

#[test]
fn no_constraints_section_produces_empty_vocabulary() {
    let doc = parse_constraint_doc("EMPTY-001", "# Empty Doc\n\nNo constraints here.");
    assert!(doc.vocabulary().is_empty());
}

#[test]
fn load_corpus_from_temp_dir() {
    use std::fs;
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("ADR-001.md"), ADR_MARKDOWN).unwrap();
    fs::write(dir.path().join("POLICY-001.md"), GENERIC_HARD).unwrap();
    let corpus = load_corpus(dir.path()).unwrap();
    assert_eq!(corpus.len(), 2);
    let ids: Vec<_> = corpus.iter().map(|d| d.id.as_str()).collect();
    assert!(ids.contains(&"ADR-001") || ids.contains(&"POLICY-001"));
}

#[test]
fn load_corpus_missing_dir_returns_empty() {
    let result = load_corpus("/tmp/h2ai-test-nonexistent-dir-xyz");
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}
