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

fn load_corpus() -> (FsConstraintStore, WikiCache) {
    let (index, store) = FsConstraintStore::load(corpus_dir()).expect("corpus must load");
    let cache = WikiCache::from_docs(&store.all_docs_sorted());
    let _ = index; // index used via resolver in resolver tests
    (store, cache)
}

fn make_resolver() -> ConstraintResolver {
    let (index, store) = FsConstraintStore::load(corpus_dir()).expect("corpus must load");
    ConstraintResolver::new(Arc::new(index), Arc::new(store))
}

// ── Loading ──────────────────────────────────────────────────────────────────

#[test]
fn corpus_loads_all_yaml_constraints() {
    let (store, _) = load_corpus();
    let docs = store.all_docs_sorted();
    // 8 original ads-platform constraints + 4 CACHE constraints + 3 saga constraints (C-009/C-010/C-011)
    assert_eq!(
        docs.len(),
        15,
        "ads-platform corpus must contain exactly 15 constraints; got {}: {:?}",
        docs.len(),
        docs.iter().map(|d| &d.id).collect::<Vec<_>>()
    );
}

#[test]
fn corpus_all_constraints_have_composite_predicate_ending_in_llm_judge() {
    use h2ai_constraints::types::CompositeOp;
    let (store, _) = load_corpus();
    for doc in store.all_docs_sorted() {
        match &doc.predicate {
            ConstraintPredicate::Composite { op, children } => {
                assert_eq!(*op, CompositeOp::And, "constraint {} must use And", doc.id);
                assert!(
                    children
                        .last()
                        .map(|c| matches!(c, ConstraintPredicate::LlmJudge { .. }))
                        .unwrap_or(false),
                    "constraint {} Composite must end with LlmJudge",
                    doc.id
                );
            }
            other => panic!(
                "constraint {} must be Composite(And([..., LlmJudge])); got {other:?}",
                doc.id
            ),
        }
    }
}

#[test]
fn corpus_rubrics_include_domain_context() {
    let (store, _) = load_corpus();
    for doc in store.all_docs_sorted() {
        if doc.domains.is_empty() {
            continue;
        }
        if let ConstraintPredicate::Composite { children, .. } = &doc.predicate {
            if let Some(ConstraintPredicate::LlmJudge { rubric }) = children.last() {
                assert!(
                    rubric.contains("Domain:"),
                    "constraint {} rubric must include 'Domain:' line; got: {}",
                    doc.id,
                    &rubric[..rubric.len().min(200)]
                );
            }
        }
    }
}

#[test]
fn corpus_rubrics_include_remediation_hint() {
    let (store, _) = load_corpus();
    for doc in store.all_docs_sorted() {
        if doc.remediation_hint.is_none() {
            continue;
        }
        if let ConstraintPredicate::Composite { children, .. } = &doc.predicate {
            if let Some(ConstraintPredicate::LlmJudge { rubric }) = children.last() {
                assert!(
                    rubric.contains("Remediation hint:"),
                    "constraint {} rubric must include 'Remediation hint:'; got: {}",
                    doc.id,
                    &rubric[..rubric.len().min(200)]
                );
            }
        }
    }
}

#[test]
fn corpus_all_expected_ids_present() {
    let (store, _) = load_corpus();
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
fn yaml_without_semantic_or_predicates_produces_composite_with_single_llm_judge() {
    use h2ai_constraints::types::{CompositeOp, ConstraintPredicate};
    use h2ai_constraints::yaml::parse_yaml_constraint;
    use std::path::Path;

    let yaml = r#"
id: TEST-PLAIN2
title: "Plain LLM judge"
severity: hard
criteria:
  pass: "Proposal is stateless."
  fail: "Proposal uses state."
"#;
    let doc = parse_yaml_constraint(Path::new("TEST-PLAIN2.yaml"), yaml).unwrap();
    match &doc.predicate {
        ConstraintPredicate::Composite { op, children } => {
            assert_eq!(*op, CompositeOp::And);
            assert_eq!(children.len(), 1, "no semantic gates → only LlmJudge child");
            assert!(matches!(&children[0], ConstraintPredicate::LlmJudge { .. }));
        }
        other => panic!("expected Composite(And([LlmJudge])), got {other:?}"),
    }
}

// ── Domain navigation ────────────────────────────────────────────────────────

#[test]
fn navigate_by_domain_billing_returns_constraints_004_005_007() {
    let (_, cache) = load_corpus();
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
    let (_, cache) = load_corpus();
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
    let (_, cache) = load_corpus();
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
    let (_, cache) = load_corpus();
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
    let (_, cache) = load_corpus();
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
    let (_, cache) = load_corpus();
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
    let (_, cache) = load_corpus();
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
    let (_, cache) = load_corpus();
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
    let resolver = make_resolver();
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
    let resolver = make_resolver();
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
async fn source_load_payload_has_remediation_hint_in_composite_rubric() {
    let resolver = make_resolver();
    let docs = resolver
        .resolve(&["CONSTRAINT-004".to_string()], &[], "")
        .await;
    assert_eq!(
        docs.len(),
        1,
        "must resolve exactly one doc for explicit ID"
    );
    assert_eq!(docs[0].id, "CONSTRAINT-004");
    if let ConstraintPredicate::Composite { children, .. } = &docs[0].predicate {
        if let Some(ConstraintPredicate::LlmJudge { rubric }) = children.last() {
            assert!(
                rubric.contains("Remediation hint:"),
                "rubric must include hint"
            );
            assert!(rubric.contains("Domain:"), "rubric must include domain");
        } else {
            panic!("CONSTRAINT-004 Composite must end with LlmJudge");
        }
    } else {
        panic!("CONSTRAINT-004 must be Composite predicate");
    }
}

#[test]
fn yaml_semantic_section_parses_exclusion_requirement_ordering() {
    use h2ai_constraints::types::{CompositeOp, ConstraintPredicate};
    use h2ai_constraints::yaml::parse_yaml_constraint;
    use std::path::Path;

    let yaml = r#"
id: TEST-SEM
title: "Semantic section test"
severity: hard
criteria:
  pass: "Uses Kafka."
  fail: "Does not use Kafka."
semantic:
  exclusions:
    - pattern: "direct DB write"
      passes: 3
  requirements:
    - concept: "Kafka topic"
      passes: 3
  orderings:
    - first: "debit"
      then: "publish"
      passes: 3
"#;
    let doc = parse_yaml_constraint(Path::new("TEST-SEM.yaml"), yaml).unwrap();
    match &doc.predicate {
        ConstraintPredicate::Composite { op, children } => {
            assert_eq!(*op, CompositeOp::And);
            assert_eq!(
                children.len(),
                4,
                "1 exclusion + 1 requirement + 1 ordering + LlmJudge"
            );
            assert!(matches!(
                &children[0],
                ConstraintPredicate::SemanticExclusion { .. }
            ));
            assert!(matches!(
                &children[1],
                ConstraintPredicate::SemanticPresence { .. }
            ));
            assert!(matches!(
                &children[2],
                ConstraintPredicate::SemanticOrdering { .. }
            ));
            assert!(matches!(&children[3], ConstraintPredicate::LlmJudge { .. }));
        }
        other => panic!("expected Composite, got {other:?}"),
    }
}

#[test]
fn yaml_both_semantic_and_predicates_returns_none() {
    use h2ai_constraints::yaml::parse_yaml_constraint;
    use std::path::Path;

    let yaml = r#"
id: TEST-COLLISION
title: "Key collision test"
severity: hard
criteria:
  pass: "Pass."
  fail: "Fail."
semantic:
  exclusions:
    - pattern: "bad pattern"
predicates:
  - type: semantic_ordering
    first: "a"
    then: "b"
"#;
    let result = parse_yaml_constraint(Path::new("TEST-COLLISION.yaml"), yaml);
    assert!(
        result.is_none(),
        "both semantic: and predicates: must result in None"
    );
}

#[test]
fn yaml_legacy_predicates_array_maps_to_composite() {
    use h2ai_constraints::types::{CompositeOp, ConstraintPredicate};
    use h2ai_constraints::yaml::parse_yaml_constraint;
    use std::path::Path;

    let yaml = r#"
id: TEST-LEGACY
title: "Legacy predicates test"
severity: hard
criteria:
  pass: "Ordering correct."
  fail: "Ordering wrong."
predicates:
  - type: semantic_ordering
    first: "account debit"
    then: "Kafka publish"
"#;
    let doc = parse_yaml_constraint(Path::new("TEST-LEGACY.yaml"), yaml).unwrap();
    match &doc.predicate {
        ConstraintPredicate::Composite { op, children } => {
            assert_eq!(*op, CompositeOp::And);
            assert_eq!(children.len(), 2, "1 SemanticOrdering + LlmJudge");
            if let ConstraintPredicate::SemanticOrdering { first, .. } = &children[0] {
                assert_eq!(first, "account debit");
            } else {
                panic!("first child must be SemanticOrdering");
            }
        }
        other => panic!("expected Composite, got {other:?}"),
    }
}

#[test]
fn new_llm_judge_produces_composite_not_bare_llm_judge() {
    use h2ai_constraints::types::{CompositeOp, ConstraintDoc, ConstraintPredicate};
    let doc = ConstraintDoc::new_llm_judge("C-TEST", "The proposal must be stateless.");
    match &doc.predicate {
        ConstraintPredicate::Composite { op, children } => {
            assert_eq!(*op, CompositeOp::And);
            assert_eq!(children.len(), 1);
            assert!(
                matches!(&children[0], ConstraintPredicate::LlmJudge { rubric } if rubric.contains("stateless"))
            );
        }
        other => panic!("new_llm_judge must produce Composite(And([LlmJudge])), got {other:?}"),
    }
}

#[test]
fn new_soft_llm_judge_produces_composite_not_bare_llm_judge() {
    use h2ai_constraints::types::{CompositeOp, ConstraintDoc, ConstraintPredicate};
    let doc = ConstraintDoc::new_soft_llm_judge("C-SOFT", "Soft requirement text.");
    match &doc.predicate {
        ConstraintPredicate::Composite { op, children } => {
            assert_eq!(*op, CompositeOp::And);
            assert_eq!(children.len(), 1);
            assert!(matches!(&children[0], ConstraintPredicate::LlmJudge { .. }));
        }
        other => panic!("new_soft_llm_judge must produce Composite, got {other:?}"),
    }
}

#[test]
fn constraint_004_has_exclusion_and_requirement_semantic_gates() {
    use h2ai_constraints::types::ConstraintPredicate;
    let (store, _) = load_corpus();
    let doc = store
        .all_docs_sorted()
        .into_iter()
        .find(|d| d.id == "CONSTRAINT-004")
        .unwrap();
    if let ConstraintPredicate::Composite { children, .. } = &doc.predicate {
        let exclusions = children
            .iter()
            .filter(|c| matches!(c, ConstraintPredicate::SemanticExclusion { .. }))
            .count();
        let requirements = children
            .iter()
            .filter(|c| matches!(c, ConstraintPredicate::SemanticPresence { .. }))
            .count();
        assert_eq!(
            exclusions, 1,
            "CONSTRAINT-004 must have 1 SemanticExclusion gate"
        );
        assert_eq!(
            requirements, 1,
            "CONSTRAINT-004 must have 1 SemanticPresence gate"
        );
    } else {
        panic!("CONSTRAINT-004 must be Composite");
    }
}

#[test]
fn constraint_005_has_exclusion_requirement_and_ordering_gates() {
    use h2ai_constraints::types::ConstraintPredicate;
    let (store, _) = load_corpus();
    let doc = store
        .all_docs_sorted()
        .into_iter()
        .find(|d| d.id == "CONSTRAINT-005")
        .unwrap();
    if let ConstraintPredicate::Composite { children, .. } = &doc.predicate {
        let exclusions = children
            .iter()
            .filter(|c| matches!(c, ConstraintPredicate::SemanticExclusion { .. }))
            .count();
        let requirements = children
            .iter()
            .filter(|c| matches!(c, ConstraintPredicate::SemanticPresence { .. }))
            .count();
        let orderings = children
            .iter()
            .filter(|c| matches!(c, ConstraintPredicate::SemanticOrdering { .. }))
            .count();
        assert_eq!(
            exclusions, 1,
            "CONSTRAINT-005 must have 1 SemanticExclusion gate"
        );
        assert_eq!(
            requirements, 1,
            "CONSTRAINT-005 must have 1 SemanticPresence gate"
        );
        assert_eq!(
            orderings, 1,
            "CONSTRAINT-005 must have 1 SemanticOrdering gate"
        );
    } else {
        panic!("CONSTRAINT-005 must be Composite");
    }
}

#[test]
fn constraint_008_has_exclusion_and_requirement_semantic_gates() {
    use h2ai_constraints::types::ConstraintPredicate;
    let (store, _) = load_corpus();
    let doc = store
        .all_docs_sorted()
        .into_iter()
        .find(|d| d.id == "CONSTRAINT-008")
        .unwrap();
    if let ConstraintPredicate::Composite { children, .. } = &doc.predicate {
        let exclusions = children
            .iter()
            .filter(|c| matches!(c, ConstraintPredicate::SemanticExclusion { .. }))
            .count();
        let requirements = children
            .iter()
            .filter(|c| matches!(c, ConstraintPredicate::SemanticPresence { .. }))
            .count();
        assert_eq!(
            exclusions, 1,
            "CONSTRAINT-008 must have 1 SemanticExclusion gate"
        );
        assert_eq!(
            requirements, 1,
            "CONSTRAINT-008 must have 1 SemanticPresence gate"
        );
    } else {
        panic!("CONSTRAINT-008 must be Composite");
    }
}

#[test]
fn constraint_009_has_exclusion_requirement_and_ordering_gates() {
    use h2ai_constraints::types::ConstraintPredicate;
    let (store, _) = load_corpus();
    let doc = store
        .all_docs_sorted()
        .into_iter()
        .find(|d| d.id == "CONSTRAINT-009")
        .expect("CONSTRAINT-009 must exist");
    if let ConstraintPredicate::Composite { children, .. } = &doc.predicate {
        let exclusions = children
            .iter()
            .filter(|c| matches!(c, ConstraintPredicate::SemanticExclusion { .. }))
            .count();
        let requirements = children
            .iter()
            .filter(|c| matches!(c, ConstraintPredicate::SemanticPresence { .. }))
            .count();
        let orderings = children
            .iter()
            .filter(|c| matches!(c, ConstraintPredicate::SemanticOrdering { .. }))
            .count();
        assert!(
            exclusions >= 2,
            "CONSTRAINT-009 must have ≥2 SemanticExclusion gates"
        );
        assert!(
            requirements >= 2,
            "CONSTRAINT-009 must have ≥2 SemanticPresence gates"
        );
        assert!(
            orderings >= 1,
            "CONSTRAINT-009 must have ≥1 SemanticOrdering gate"
        );
    } else {
        panic!("CONSTRAINT-009 must be Composite");
    }
}

#[test]
fn constraint_010_has_exclusion_and_requirement_semantic_gates() {
    use h2ai_constraints::types::ConstraintPredicate;
    let (store, _) = load_corpus();
    let doc = store
        .all_docs_sorted()
        .into_iter()
        .find(|d| d.id == "CONSTRAINT-010")
        .expect("CONSTRAINT-010 must exist");
    if let ConstraintPredicate::Composite { children, .. } = &doc.predicate {
        let exclusions = children
            .iter()
            .filter(|c| matches!(c, ConstraintPredicate::SemanticExclusion { .. }))
            .count();
        let requirements = children
            .iter()
            .filter(|c| matches!(c, ConstraintPredicate::SemanticPresence { .. }))
            .count();
        assert!(
            exclusions >= 2,
            "CONSTRAINT-010 must have ≥2 SemanticExclusion gates"
        );
        assert!(
            requirements >= 2,
            "CONSTRAINT-010 must have ≥2 SemanticPresence gates"
        );
    } else {
        panic!("CONSTRAINT-010 must be Composite");
    }
}

#[test]
fn constraint_011_has_exclusion_requirement_and_ordering_gates() {
    use h2ai_constraints::types::ConstraintPredicate;
    let (store, _) = load_corpus();
    let doc = store
        .all_docs_sorted()
        .into_iter()
        .find(|d| d.id == "CONSTRAINT-011")
        .expect("CONSTRAINT-011 must exist");
    if let ConstraintPredicate::Composite { children, .. } = &doc.predicate {
        let exclusions = children
            .iter()
            .filter(|c| matches!(c, ConstraintPredicate::SemanticExclusion { .. }))
            .count();
        let requirements = children
            .iter()
            .filter(|c| matches!(c, ConstraintPredicate::SemanticPresence { .. }))
            .count();
        let orderings = children
            .iter()
            .filter(|c| matches!(c, ConstraintPredicate::SemanticOrdering { .. }))
            .count();
        assert!(
            exclusions >= 2,
            "CONSTRAINT-011 must have ≥2 SemanticExclusion gates"
        );
        assert!(
            requirements >= 2,
            "CONSTRAINT-011 must have ≥2 SemanticPresence gates"
        );
        assert!(
            orderings >= 1,
            "CONSTRAINT-011 must have ≥1 SemanticOrdering gate"
        );
    } else {
        panic!("CONSTRAINT-011 must be Composite");
    }
}
