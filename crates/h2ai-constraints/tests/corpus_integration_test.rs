/// End-to-end integration tests that load the real ads-platform constraint corpus
/// from `docs/examples/ads-platform/constraints/` and verify the full pipeline:
/// YAML loading, BM25 retrieval, domain navigation, relation graph, and payload fetch.
///
/// These tests act as a quality gate — if the corpus data or parsing breaks, they fail.
use h2ai_constraints::source::{ConstraintSource, FsConstraintSource};
use h2ai_constraints::types::ConstraintPredicate;
use h2ai_constraints::wiki::WikiCache;
use std::path::PathBuf;

fn corpus_dir() -> PathBuf {
    // Locate ads-platform corpus relative to the workspace root.
    let manifest = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest)
        .join("../../docs/examples/ads-platform/constraints")
        .canonicalize()
        .expect("ads-platform constraints directory must exist")
}

// ── Loading ──────────────────────────────────────────────────────────────────

#[test]
fn corpus_loads_all_seven_yaml_constraints() {
    let source = FsConstraintSource::load(corpus_dir()).expect("corpus must load without error");
    let docs = source.all_docs();
    assert_eq!(
        docs.len(),
        7,
        "ads-platform corpus must contain exactly 7 constraints; got {}: {:?}",
        docs.len(),
        docs.iter().map(|d| &d.id).collect::<Vec<_>>()
    );
}

#[test]
fn corpus_all_constraints_have_llm_judge_predicate() {
    // All ads-platform constraints are YAML with `criteria` — must produce LlmJudge, never VocabularyPresence.
    let source = FsConstraintSource::load(corpus_dir()).expect("corpus must load");
    for doc in source.all_docs() {
        assert!(
            matches!(doc.predicate, ConstraintPredicate::LlmJudge { .. }),
            "constraint {} must have LlmJudge predicate (loaded from YAML criteria); got non-LlmJudge",
            doc.id
        );
    }
}

#[test]
fn corpus_rubrics_include_domain_context() {
    // Every constraint has domains — build_rubric() must append them.
    let source = FsConstraintSource::load(corpus_dir()).expect("corpus must load");
    for doc in source.all_docs() {
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

#[test]
fn corpus_rubrics_include_remediation_hint() {
    // All 7 YAML constraints now carry remediation_hint — must appear in rubric.
    let source = FsConstraintSource::load(corpus_dir()).expect("corpus must load");
    for doc in source.all_docs() {
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

#[test]
fn corpus_all_expected_ids_present() {
    let source = FsConstraintSource::load(corpus_dir()).expect("corpus must load");
    let ids: std::collections::HashSet<&str> =
        source.all_docs().iter().map(|d| d.id.as_str()).collect();
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

// ── Domain navigation ────────────────────────────────────────────────────────

#[test]
fn navigate_by_domain_billing_returns_constraints_004_005_007() {
    let source = FsConstraintSource::load(corpus_dir()).expect("corpus must load");
    let cache = WikiCache::from_docs(source.all_docs());

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

#[test]
fn navigate_by_domain_latency_returns_constraints_002_003_006() {
    let source = FsConstraintSource::load(corpus_dir()).expect("corpus must load");
    let cache = WikiCache::from_docs(source.all_docs());

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

#[test]
fn navigate_related_004_reaches_005_and_007() {
    let source = FsConstraintSource::load(corpus_dir()).expect("corpus must load");
    let cache = WikiCache::from_docs(source.all_docs());

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

#[test]
fn navigate_related_003_reaches_006() {
    let source = FsConstraintSource::load(corpus_dir()).expect("corpus must load");
    let cache = WikiCache::from_docs(source.all_docs());

    let related = cache.navigate_related("CONSTRAINT-003");
    let ids: std::collections::HashSet<&str> = related.iter().map(|m| m.id.as_str()).collect();
    assert!(
        ids.contains("CONSTRAINT-006"),
        "CONSTRAINT-003 must relate to 006 (ZGC/GC pauses)"
    );
}

// ── BM25 semantic retrieval ───────────────────────────────────────────────────

#[test]
fn bm25_query_budget_idempotency_ranks_004_first() {
    let source = FsConstraintSource::load(corpus_dir()).expect("corpus must load");
    let cache = WikiCache::from_docs(source.all_docs());

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

#[test]
fn bm25_query_grpc_protobuf_ranks_002_first() {
    let source = FsConstraintSource::load(corpus_dir()).expect("corpus must load");
    let cache = WikiCache::from_docs(source.all_docs());

    let hits = cache.search("grpc protobuf internal service communication binary", 3);
    assert!(!hits.is_empty(), "BM25 must return results for gRPC query");
    assert_eq!(
        hits[0].id, "CONSTRAINT-002",
        "gRPC query must rank CONSTRAINT-002 first; got {}",
        hits[0].id
    );
}

#[test]
fn bm25_query_kafka_audit_log_ranks_005_first() {
    let source = FsConstraintSource::load(corpus_dir()).expect("corpus must load");
    let cache = WikiCache::from_docs(source.all_docs());

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

#[test]
fn bm25_query_zgc_virtual_threads_ranks_006_first() {
    let source = FsConstraintSource::load(corpus_dir()).expect("corpus must load");
    let cache = WikiCache::from_docs(source.all_docs());

    let hits = cache.search("zgc virtual threads garbage collection heap pause java", 3);
    assert!(!hits.is_empty(), "BM25 must return results for ZGC query");
    assert_eq!(
        hits[0].id, "CONSTRAINT-006",
        "ZGC query must rank CONSTRAINT-006 first; got {}",
        hits[0].id
    );
}

// ── Source resolution (two-stage: tag + BM25) ────────────────────────────────

#[tokio::test]
async fn source_resolve_by_billing_tag_returns_billing_constraints() {
    let source = FsConstraintSource::load(corpus_dir()).expect("corpus must load");
    let metas = source
        .resolve_context(&["billing".to_string()], &[], "")
        .await;
    let ids: std::collections::HashSet<&str> = metas.iter().map(|m| m.id.as_str()).collect();
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
    // billing tag → 004, 005, 007; "stateless sticky session" BM25 → 001; both must appear.
    let source = FsConstraintSource::load(corpus_dir()).expect("corpus must load");
    let metas = source
        .resolve_context(
            &["billing".to_string()],
            &[],
            "stateless sticky session node affinity",
        )
        .await;
    let ids: std::collections::HashSet<&str> = metas.iter().map(|m| m.id.as_str()).collect();
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
    let source = FsConstraintSource::load(corpus_dir()).expect("corpus must load");
    let payload = source
        .load_payload("CONSTRAINT-004", "v1")
        .await
        .expect("payload fetch must succeed for known constraint");
    assert_eq!(payload.id, "CONSTRAINT-004");
    if let ConstraintPredicate::LlmJudge { rubric } = &payload.predicate {
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
