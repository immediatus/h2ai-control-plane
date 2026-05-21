use h2ai_constraints::retrieval::{resolve_with_retrieval, ConstraintRetriever};
use h2ai_constraints::types::{
    ConstraintDoc, ConstraintMeta, ConstraintPredicate, ConstraintSeverity, NumericOp,
    PredicateKind,
};
use std::collections::HashMap;

fn build_test_index() -> ConstraintRetriever {
    ConstraintRetriever::build(
        [
            (
                "CONSTRAINT-001",
                "stateless request service ttl cache eviction".to_string(),
            ),
            (
                "CONSTRAINT-002",
                "grpc protobuf internal service rest json external protocol".to_string(),
            ),
            (
                "CONSTRAINT-003",
                "rtb timeout adaptive per-dsp latency histogram auction".to_string(),
            ),
            (
                "CONSTRAINT-004",
                "budget idempotency redis atomic lua transaction debit".to_string(),
            ),
            (
                "CONSTRAINT-005",
                "financial audit log kafka clickhouse immutable append billing".to_string(),
            ),
            (
                "CONSTRAINT-006",
                "java zgc virtual threads heap garbage collection pause latency".to_string(),
            ),
            (
                "CONSTRAINT-007",
                "consistency budget billing cache redis cockroachdb hlc".to_string(),
            ),
        ]
        .into_iter(),
    )
}

#[test]
fn retrieves_correct_top_result_for_budget_query() {
    let idx = build_test_index();
    let results = idx.query("budget idempotency atomic debit", 3);
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "CONSTRAINT-004");
}

#[test]
fn retrieves_correct_top_result_for_grpc_query() {
    let idx = build_test_index();
    let results = idx.query("grpc protobuf service communication internal", 3);
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "CONSTRAINT-002");
}

#[test]
fn retrieves_correct_top_result_for_audit_query() {
    let idx = build_test_index();
    let results = idx.query("financial audit kafka billing immutable log", 3);
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "CONSTRAINT-005");
}

#[test]
fn top_k_limits_result_count() {
    let idx = build_test_index();
    let results = idx.query("service latency cache redis", 2);
    assert!(results.len() <= 2);
}

#[test]
fn empty_query_returns_empty() {
    let idx = build_test_index();
    let results = idx.query("the and for with", 10);
    assert!(
        results.is_empty(),
        "stop-words-only query should return nothing"
    );
}

#[test]
fn empty_index_returns_empty() {
    let idx = ConstraintRetriever::build(std::iter::empty());
    let results = idx.query("budget redis", 5);
    assert!(results.is_empty());
}

#[test]
fn scores_are_sorted_descending() {
    let idx = build_test_index();
    let results = idx.query("latency timeout cache service", 5);
    for w in results.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "results must be sorted descending by score"
        );
    }
}

#[test]
fn idf_weights_rare_terms_higher() {
    // "hlc" only appears in CONSTRAINT-007; querying for it should score 007 at top.
    let idx = build_test_index();
    let results = idx.query("hlc hybrid logical clock", 3);
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "CONSTRAINT-007");
}

// ── Lines 157-163: len() and is_empty() ─────────────────────────────────────

#[test]
fn len_and_is_empty_reflect_indexed_count() {
    let idx = build_test_index();
    assert_eq!(idx.len(), 7);
    assert!(!idx.is_empty());

    let empty_idx = ConstraintRetriever::build(std::iter::empty());
    assert_eq!(empty_idx.len(), 0);
    assert!(empty_idx.is_empty());
}

// ── Lines 172-173: extract_rubric_text for OracleExecution, NumericThreshold ──

fn make_doc(id: &str, pred: ConstraintPredicate) -> ConstraintDoc {
    ConstraintDoc {
        id: id.to_string(),
        source_file: format!("{id}.yaml"),
        description: format!("{id} description"),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: pred,
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
    }
}

#[test]
fn from_docs_with_oracle_and_numeric_predicates_builds_retriever() {
    // OracleExecution and NumericThreshold fall into the `_ => String::new()` arm
    // of extract_rubric_text — the entry is still indexed (just with empty rubric text).
    let docs = vec![
        make_doc(
            "ORC-001",
            ConstraintPredicate::OracleExecution {
                test_runner_uri: "http://localhost/run".into(),
                test_suite: "suite.py".into(),
                timeout_secs: 30,
            },
        ),
        make_doc(
            "NUM-001",
            ConstraintPredicate::NumericThreshold {
                field_pattern: r"latency[:\s]+(\d+)".into(),
                op: NumericOp::Lt,
                value: 200.0,
            },
        ),
    ];
    let idx = ConstraintRetriever::from_docs(&docs);
    assert_eq!(idx.len(), 2);
    // "ORC-001" text matches its own id
    let results = idx.query("ORC-001", 5);
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "ORC-001");
}

// ── Lines 172-173: NegativeKeyword and RegexMatch arms in extract_rubric_text ──

#[test]
fn from_docs_with_negative_keyword_and_regex_predicates() {
    use h2ai_constraints::types::VocabularyMode;
    let docs = vec![
        make_doc(
            "NEG-KW-001",
            ConstraintPredicate::NegativeKeyword {
                terms: vec!["password".into(), "secret".into()],
            },
        ),
        make_doc(
            "REGEX-001",
            ConstraintPredicate::RegexMatch {
                pattern: r"\bUUID\b".into(),
                must_match: true,
            },
        ),
        // VocabularyPresence also has its own arm (terms.join)
        make_doc(
            "VP-001",
            ConstraintPredicate::VocabularyPresence {
                mode: VocabularyMode::AllOf,
                terms: vec!["idempotency".into(), "atomic".into()],
            },
        ),
    ];
    let idx = ConstraintRetriever::from_docs(&docs);
    assert_eq!(idx.len(), 3);

    // NegativeKeyword terms are indexed — can search by them
    let neg_results = idx.query("password secret", 5);
    assert!(neg_results.iter().any(|c| c.id == "NEG-KW-001"));

    // RegexMatch: the id "REGEX-001" itself is indexed as a token
    let regex_results = idx.query("REGEX-001", 5);
    assert!(
        !regex_results.is_empty(),
        "REGEX-001 must be searchable by id"
    );
}

// ── Lines 282-293: resolve_with_retrieval ────────────────────────────────────

#[test]
fn resolve_with_retrieval_returns_matching_metas() {
    let docs = vec![
        make_doc(
            "C-BUDGET",
            ConstraintPredicate::LlmJudge {
                rubric: "atomic idempotency budget deduction redis lua".into(),
            },
        ),
        make_doc(
            "C-GRPC",
            ConstraintPredicate::LlmJudge {
                rubric: "grpc protobuf internal service communication".into(),
            },
        ),
    ];
    let retriever = ConstraintRetriever::from_docs(&docs);

    let mut metas: HashMap<String, ConstraintMeta> = HashMap::new();
    for doc in &docs {
        metas.insert(
            doc.id.clone(),
            ConstraintMeta {
                id: doc.id.clone(),
                summary: doc.description.clone(),
                severity: doc.severity.clone(),
                predicate_kind: PredicateKind::LlmJudge,
                domains: vec![],
                mandatory_for_tags: vec![],
                related_to: vec![],
                payload_version: "v1".into(),
                inline_predicate: None,
                source: None,
                last_updated_ms: None,
            },
        );
    }

    let results = resolve_with_retrieval("atomic idempotency budget redis", 5, &retriever, &metas);
    assert!(
        results.iter().any(|m| m.id == "C-BUDGET"),
        "budget constraint must be returned"
    );
}

#[test]
fn resolve_with_retrieval_returns_empty_for_no_match() {
    let docs = vec![make_doc(
        "C-ONE",
        ConstraintPredicate::LlmJudge {
            rubric: "grpc protobuf service".into(),
        },
    )];
    let retriever = ConstraintRetriever::from_docs(&docs);
    let metas: HashMap<String, ConstraintMeta> = HashMap::new();

    // No metas means even matching docs produce empty result
    let results = resolve_with_retrieval("grpc protobuf", 5, &retriever, &metas);
    assert!(results.is_empty());
}
