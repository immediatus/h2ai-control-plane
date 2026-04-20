use h2ai_context::jaccard::{jaccard, tokenize};

#[test]
fn tokenize_lowercases_and_splits_on_whitespace() {
    let tokens = tokenize("Budget pacing idempotency");
    assert!(tokens.contains("budget"));
    assert!(tokens.contains("pacing"));
    assert!(tokens.contains("idempotency"));
}

#[test]
fn tokenize_strips_punctuation() {
    let tokens = tokenize("ADR-004: stateless, auth.");
    assert!(tokens.contains("adr"));
    assert!(tokens.contains("stateless"));
    assert!(tokens.contains("auth"));
}

#[test]
fn jaccard_identical_sets_is_one() {
    let a = tokenize("budget pacing idempotency");
    let b = tokenize("budget pacing idempotency");
    assert!((jaccard(&a, &b) - 1.0).abs() < 1e-9);
}

#[test]
fn jaccard_disjoint_sets_is_zero() {
    let a = tokenize("budget pacing");
    let b = tokenize("grpc latency");
    assert!(jaccard(&a, &b).abs() < 1e-9);
}

#[test]
fn jaccard_partial_overlap() {
    let a = tokenize("a b");
    let b = tokenize("b c");
    assert!((jaccard(&a, &b) - 1.0 / 3.0).abs() < 1e-9);
}

#[test]
fn jaccard_empty_sets_is_zero() {
    let a = tokenize("");
    let b = tokenize("");
    assert_eq!(jaccard(&a, &b), 0.0);
}
