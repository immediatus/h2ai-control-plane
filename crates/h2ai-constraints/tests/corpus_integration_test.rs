/// End-to-end integration tests that load the real ads-platform constraint corpus
/// from `tests/e2e/constraints/` and verify the full pipeline:
/// YAML loading, BM25 retrieval, domain navigation, relation graph, and payload fetch.
use h2ai_constraints::resolver::ConstraintResolver;
use h2ai_constraints::source::FsConstraintStore;
use h2ai_constraints::types::ConstraintPredicate;
use h2ai_constraints::wiki::WikiCache;
use std::path::PathBuf;
use std::sync::Arc;

fn corpus_dir() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest)
        .join("../../tests/e2e/constraints")
        .canonicalize()
        .expect("ads-platform constraints directory must exist")
}

async fn load_corpus() -> (FsConstraintStore, WikiCache) {
    let (index, store) = FsConstraintStore::load(corpus_dir())
        .await
        .expect("corpus must load");
    let cache = WikiCache::from_docs(&store.all_docs_sorted());
    let _ = index; // index used via resolver in resolver tests
    (store, cache)
}

async fn make_resolver() -> ConstraintResolver {
    let (index, store) = FsConstraintStore::load(corpus_dir())
        .await
        .expect("corpus must load");
    ConstraintResolver::new(Arc::new(index), Arc::new(store))
}

// ── Loading ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn corpus_loads_all_yaml_constraints() {
    let (store, _) = load_corpus().await;
    let docs = store.all_docs_sorted();
    // 7 original ads-platform constraints + 4 CACHE constraints added for agent-comparison experiment
    assert_eq!(
        docs.len(),
        11,
        "ads-platform corpus must contain exactly 11 constraints; got {}: {:?}",
        docs.len(),
        docs.iter().map(|d| &d.id).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn corpus_all_constraints_have_llm_judge_predicate() {
    use h2ai_constraints::types::CompositeOp;
    let (store, _) = load_corpus().await;
    for doc in store.all_docs_sorted() {
        let has_llm_judge = match &doc.predicate {
            ConstraintPredicate::LlmJudge { .. } => true,
            // Composite(And([..., LlmJudge])) — numeric_checks present
            ConstraintPredicate::Composite { op, children } => {
                *op == CompositeOp::And
                    && children
                        .last()
                        .map(|c| matches!(c, ConstraintPredicate::LlmJudge { .. }))
                        .unwrap_or(false)
            }
            _ => false,
        };
        assert!(
            has_llm_judge,
            "constraint {} must have LlmJudge predicate (or Composite ending in LlmJudge); got non-LlmJudge",
            doc.id
        );
    }
}

#[tokio::test]
async fn corpus_rubrics_include_domain_context() {
    let (store, _) = load_corpus().await;
    for doc in store.all_docs_sorted() {
        if let ConstraintPredicate::LlmJudge { rubric } = &doc.predicate {
            assert!(
                rubric.contains("Domain:"),
                "constraint {} rubric must include 'Domain:' line; got: {}",
                doc.id,
                &rubric[..rubric.len().min(200)]
            );
        }
    }
}

#[tokio::test]
async fn corpus_rubrics_include_remediation_hint() {
    let (store, _) = load_corpus().await;
    for doc in store.all_docs_sorted() {
        if let ConstraintPredicate::LlmJudge { rubric } = &doc.predicate {
            assert!(
                rubric.contains("Remediation hint:"),
                "constraint {} rubric must include 'Remediation hint:'; got: {}",
                doc.id,
                &rubric[..rubric.len().min(200)]
            );
        }
    }
}

#[tokio::test]
async fn corpus_all_expected_ids_present() {
    let (store, _) = load_corpus().await;
    let ids: std::collections::HashSet<String> =
        store.all_docs_sorted().into_iter().map(|d| d.id).collect();
    for expected in [
        "CONSTRAINT-001",
        "CONSTRAINT-002",
        "CONSTRAINT-003",
        "CONSTRAINT-004",
        "CONSTRAINT-005",
        "CONSTRAINT-006",
        "CONSTRAINT-007",
    ] {
        assert!(ids.contains(expected), "corpus must contain {expected}");
    }
}

// ── NumericCheck predicate construction ──────────────────────────────────────

#[test]
fn yaml_numeric_checks_generate_composite_predicate() {
    use h2ai_constraints::types::{CompositeOp, ConstraintPredicate};
    use h2ai_constraints::yaml::parse_yaml_constraint;
    use std::path::Path;

    let yaml = r#"
id: TEST-NUM
title: "Numeric check test"
severity: hard
criteria:
  pass: "Global timeout ≤100ms."
  fail: "Global timeout exceeds 100ms."
numeric_checks:
  - pattern: "(?i)timeout[^0-9]*([0-9]+)"
    op: le
    value: 100.0
"#;
    let doc = parse_yaml_constraint(Path::new("TEST-NUM.yaml"), yaml).unwrap();
    match &doc.predicate {
        ConstraintPredicate::Composite { op, children } => {
            assert_eq!(*op, CompositeOp::And);
            assert_eq!(children.len(), 2, "one NumericThreshold + one LlmJudge");
            assert!(
                matches!(children[0], ConstraintPredicate::NumericThreshold { .. }),
                "first child must be NumericThreshold"
            );
            assert!(
                matches!(children[1], ConstraintPredicate::LlmJudge { .. }),
                "last child must be LlmJudge"
            );
        }
        other => panic!("expected Composite predicate, got {other:?}"),
    }
}

#[test]
fn yaml_without_numeric_checks_still_produces_llm_judge() {
    use h2ai_constraints::types::ConstraintPredicate;
    use h2ai_constraints::yaml::parse_yaml_constraint;
    use std::path::Path;

    let yaml = r#"
id: TEST-PLAIN
title: "Plain LLM judge"
severity: hard
criteria:
  pass: "Proposal is stateless."
  fail: "Proposal uses state."
"#;
    let doc = parse_yaml_constraint(Path::new("TEST-PLAIN.yaml"), yaml).unwrap();
    assert!(
        matches!(doc.predicate, ConstraintPredicate::LlmJudge { .. }),
        "no numeric_checks → plain LlmJudge"
    );
}

// ── Domain navigation ────────────────────────────────────────────────────────

#[tokio::test]
async fn navigate_by_domain_billing_returns_constraints_004_005_007() {
    let (_, cache) = load_corpus().await;
    let billing = cache.navigate_by_domain("billing");
    let ids: std::collections::HashSet<&str> = billing.iter().map(|m| m.id.as_str()).collect();
    assert!(
        ids.contains("CONSTRAINT-004"),
        "billing must include 004 (budget pacing)"
    );
    assert!(
        ids.contains("CONSTRAINT-005"),
        "billing must include 005 (audit log)"
    );
    assert!(
        ids.contains("CONSTRAINT-007"),
        "billing must include 007 (consistency)"
    );
}

#[tokio::test]
async fn navigate_by_domain_latency_returns_constraints_002_003_006() {
    let (_, cache) = load_corpus().await;
    let latency = cache.navigate_by_domain("latency");
    let ids: std::collections::HashSet<&str> = latency.iter().map(|m| m.id.as_str()).collect();
    assert!(
        ids.contains("CONSTRAINT-002"),
        "latency must include 002 (protocols)"
    );
    assert!(
        ids.contains("CONSTRAINT-003"),
        "latency must include 003 (RTB timeouts)"
    );
    assert!(
        ids.contains("CONSTRAINT-006"),
        "latency must include 006 (ZGC runtime)"
    );
}

// ── Relation graph navigation ─────────────────────────────────────────────────

#[tokio::test]
async fn navigate_related_004_reaches_005_and_007() {
    let (_, cache) = load_corpus().await;
    let related = cache.navigate_related("CONSTRAINT-004");
    let ids: std::collections::HashSet<&str> = related.iter().map(|m| m.id.as_str()).collect();
    assert!(
        ids.contains("CONSTRAINT-005"),
        "CONSTRAINT-004 must relate to 005 (audit log)"
    );
    assert!(
        ids.contains("CONSTRAINT-007"),
        "CONSTRAINT-004 must relate to 007 (consistency)"
    );
}

#[tokio::test]
async fn navigate_related_003_reaches_006() {
    let (_, cache) = load_corpus().await;
    let related = cache.navigate_related("CONSTRAINT-003");
    let ids: std::collections::HashSet<&str> = related.iter().map(|m| m.id.as_str()).collect();
    assert!(
        ids.contains("CONSTRAINT-006"),
        "CONSTRAINT-003 must relate to 006 (ZGC/GC pauses)"
    );
}

// ── BM25 semantic retrieval ───────────────────────────────────────────────────

#[tokio::test]
async fn bm25_query_budget_idempotency_ranks_004_first() {
    let (_, cache) = load_corpus().await;
    let hits = cache.search("budget idempotency atomic redis debit duplicate", 3);
    assert!(
        !hits.is_empty(),
        "BM25 must return results for budget/idempotency query"
    );
    assert_eq!(
        hits[0].id, "CONSTRAINT-004",
        "budget idempotency query must rank CONSTRAINT-004 first; got {}",
        hits[0].id
    );
}

#[tokio::test]
async fn bm25_query_grpc_protobuf_ranks_002_first() {
    let (_, cache) = load_corpus().await;
    let hits = cache.search("grpc protobuf internal service communication binary", 3);
    assert!(!hits.is_empty(), "BM25 must return results for gRPC query");
    assert_eq!(
        hits[0].id, "CONSTRAINT-002",
        "gRPC query must rank CONSTRAINT-002 first; got {}",
        hits[0].id
    );
}

#[tokio::test]
async fn bm25_query_kafka_audit_log_ranks_005_first() {
    let (_, cache) = load_corpus().await;
    let hits = cache.search("kafka clickhouse immutable audit log financial billing", 3);
    assert!(
        !hits.is_empty(),
        "BM25 must return results for audit log query"
    );
    assert_eq!(
        hits[0].id, "CONSTRAINT-005",
        "audit log query must rank CONSTRAINT-005 first; got {}",
        hits[0].id
    );
}

#[tokio::test]
async fn bm25_query_zgc_virtual_threads_ranks_006_first() {
    let (_, cache) = load_corpus().await;
    let hits = cache.search("zgc virtual threads garbage collection heap pause java", 3);
    assert!(!hits.is_empty(), "BM25 must return results for ZGC query");
    assert_eq!(
        hits[0].id, "CONSTRAINT-006",
        "ZGC query must rank CONSTRAINT-006 first; got {}",
        hits[0].id
    );
}

// ── Resolver (tag + BM25) ────────────────────────────────────────────────────

#[tokio::test]
async fn source_resolve_by_billing_tag_returns_billing_constraints() {
    let resolver = make_resolver().await;
    let docs = resolver.resolve(&[], &["billing".to_string()], "").await;
    let ids: std::collections::HashSet<&str> = docs.iter().map(|d| d.id.as_str()).collect();
    assert!(
        ids.contains("CONSTRAINT-004"),
        "billing tag must resolve 004"
    );
    assert!(
        ids.contains("CONSTRAINT-005"),
        "billing tag must resolve 005"
    );
    assert!(
        ids.contains("CONSTRAINT-007"),
        "billing tag must resolve 007"
    );
}

#[tokio::test]
async fn source_resolve_semantic_union_merges_tag_and_bm25() {
    let resolver = make_resolver().await;
    let docs = resolver
        .resolve(
            &[],
            &["billing".to_string()],
            "stateless sticky session node affinity",
        )
        .await;
    let ids: std::collections::HashSet<&str> = docs.iter().map(|d| d.id.as_str()).collect();
    assert!(
        ids.contains("CONSTRAINT-004"),
        "tag union must include 004 (billing)"
    );
    assert!(
        ids.contains("CONSTRAINT-001"),
        "BM25 union must include 001 (stateless)"
    );
}

#[tokio::test]
async fn source_load_payload_returns_llm_judge_with_remediation() {
    let resolver = make_resolver().await;
    let docs = resolver
        .resolve(&["CONSTRAINT-004".to_string()], &[], "")
        .await;
    assert_eq!(
        docs.len(),
        1,
        "must resolve exactly one doc for explicit ID"
    );
    assert_eq!(docs[0].id, "CONSTRAINT-004");
    if let ConstraintPredicate::LlmJudge { rubric } = &docs[0].predicate {
        assert!(
            rubric.contains("Remediation hint:"),
            "fetched rubric must include remediation hint"
        );
        assert!(
            rubric.contains("Domain:"),
            "fetched rubric must include domain context"
        );
    } else {
        panic!("CONSTRAINT-004 payload must be LlmJudge");
    }
}
