use h2ai_constraints::types::{
    ConstraintMeta, ConstraintPayload, ConstraintPredicate, ConstraintSeverity, PredicateKind,
    VocabularyMode,
};

#[test]
fn constraint_meta_roundtrip_json() {
    let meta = ConstraintMeta {
        id: "ADR-001".into(),
        summary: "All outputs must cite source.".into(),
        severity: ConstraintSeverity::Hard { threshold: 0.8 },
        predicate_kind: PredicateKind::Static,
        domains: vec!["internal".into()],
        mandatory_for_tags: vec!["audit".into()],
        related_to: vec!["ADR-002".into()],
        payload_version: "v1".into(),
        inline_predicate: Some(ConstraintPredicate::VocabularyPresence {
            mode: VocabularyMode::AnyOf,
            terms: vec!["source".into()],
        }),
        source: Some("internal/policy-42".into()),
        last_updated_ms: Some(1_700_000_000_000),
    };
    let json = serde_json::to_string(&meta).unwrap();
    let back: ConstraintMeta = serde_json::from_str(&json).unwrap();
    assert_eq!(back.id, "ADR-001");
    assert_eq!(back.predicate_kind, PredicateKind::Static);
    assert!(back.inline_predicate.is_some());
    assert_eq!(back.source.as_deref(), Some("internal/policy-42"));
    assert_eq!(back.last_updated_ms, Some(1_700_000_000_000));
    assert_eq!(back.related_to, vec!["ADR-002"]);
}

#[test]
fn constraint_meta_source_defaults_to_none() {
    // Constraints without provenance metadata must still deserialize cleanly.
    let json = r#"{"id":"X","summary":"s","severity":"Advisory","predicate_kind":"static","domains":[],"mandatory_for_tags":[],"payload_version":"v1"}"#;
    let meta: ConstraintMeta = serde_json::from_str(json).unwrap();
    assert!(meta.source.is_none());
    assert!(meta.last_updated_ms.is_none());
    assert!(
        meta.related_to.is_empty(),
        "related_to must default to empty"
    );
}

#[test]
fn constraint_payload_roundtrip_json() {
    let payload = ConstraintPayload {
        id: "GDPR-DPA-001".into(),
        version: "v2".into(),
        predicate: ConstraintPredicate::LlmJudge {
            rubric: "Verify data minimization principle is applied.".into(),
        },
    };
    let json = serde_json::to_string(&payload).unwrap();
    let back: ConstraintPayload = serde_json::from_str(&json).unwrap();
    assert_eq!(back.version, "v2");
    assert!(matches!(
        back.predicate,
        ConstraintPredicate::LlmJudge { .. }
    ));
}

#[test]
fn predicate_kind_from_tier_static() {
    use h2ai_constraints::types::ConstraintDoc;
    let doc = ConstraintDoc {
        id: "T1".into(),
        source_file: "t1.md".into(),
        description: "test".into(),
        severity: ConstraintSeverity::Advisory,
        predicate: ConstraintPredicate::LengthRange {
            min_chars: Some(10),
            max_chars: None,
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };
    let meta = ConstraintMeta::from_doc(&doc);
    assert_eq!(meta.predicate_kind, PredicateKind::Static);
    assert!(meta.inline_predicate.is_some());
}

#[test]
fn predicate_kind_from_tier_llm_judge() {
    use h2ai_constraints::types::ConstraintDoc;
    let doc = ConstraintDoc {
        id: "LJ-001".into(),
        source_file: "lj.md".into(),
        description: "LLM judge constraint.".into(),
        severity: ConstraintSeverity::Soft { weight: 1.0 },
        predicate: ConstraintPredicate::LlmJudge {
            rubric: "Verify the response is helpful.".into(),
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };
    let meta = ConstraintMeta::from_doc(&doc);
    assert_eq!(meta.predicate_kind, PredicateKind::LlmJudge);
    assert!(
        meta.inline_predicate.is_none(),
        "LlmJudge must not be inlined"
    );
}

#[test]
fn predicate_kind_from_tier_oracle() {
    use h2ai_constraints::types::ConstraintDoc;
    let doc = ConstraintDoc {
        id: "ORC-001".into(),
        source_file: "orc.md".into(),
        description: "Oracle constraint.".into(),
        severity: ConstraintSeverity::Hard { threshold: 1.0 },
        predicate: ConstraintPredicate::OracleExecution {
            test_runner_uri: "http://localhost:8080/run".into(),
            test_suite: "suite_a".into(),
            timeout_secs: 30,
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };
    let meta = ConstraintMeta::from_doc(&doc);
    assert_eq!(meta.predicate_kind, PredicateKind::Oracle);
    assert!(
        meta.inline_predicate.is_none(),
        "Oracle must not be inlined"
    );
}

#[test]
fn wiki_cache_resolves_by_tag() {
    use h2ai_constraints::wiki::WikiCache;

    let mut cache = WikiCache::default();
    cache
        .context_map
        .insert("eu_data".into(), vec!["GDPR-001".into(), "GDPR-002".into()]);
    cache.metas.insert(
        "GDPR-001".into(),
        ConstraintMeta {
            id: "GDPR-001".into(),
            summary: "Minimize data.".into(),
            severity: ConstraintSeverity::Hard { threshold: 0.8 },
            predicate_kind: PredicateKind::Static,
            domains: vec!["eu_data".into()],
            mandatory_for_tags: vec![],
            related_to: vec![],
            payload_version: "v1".into(),
            inline_predicate: None,
            source: None,
            last_updated_ms: None,
        },
    );
    cache.metas.insert(
        "GDPR-002".into(),
        ConstraintMeta {
            id: "GDPR-002".into(),
            summary: "Right to erasure.".into(),
            severity: ConstraintSeverity::Soft { weight: 1.0 },
            predicate_kind: PredicateKind::LlmJudge,
            domains: vec!["eu_data".into()],
            mandatory_for_tags: vec![],
            related_to: vec![],
            payload_version: "v1".into(),
            inline_predicate: None,
            source: None,
            last_updated_ms: None,
        },
    );

    let resolved = cache.resolve(&["eu_data".to_string()], &[]);
    assert_eq!(resolved.len(), 2);
}

#[test]
fn wiki_cache_explicit_ids_override() {
    use h2ai_constraints::wiki::WikiCache;

    let mut cache = WikiCache::default();
    cache.metas.insert(
        "ADR-001".into(),
        ConstraintMeta {
            id: "ADR-001".into(),
            summary: "Architecture rule.".into(),
            severity: ConstraintSeverity::Advisory,
            predicate_kind: PredicateKind::Static,
            domains: vec![],
            mandatory_for_tags: vec![],
            related_to: vec![],
            payload_version: "v1".into(),
            inline_predicate: None,
            source: None,
            last_updated_ms: None,
        },
    );

    let resolved = cache.resolve(&[], &["ADR-001".to_string()]);
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].id, "ADR-001");
}

#[test]
fn wiki_cache_deduplicates_tag_and_explicit() {
    use h2ai_constraints::wiki::WikiCache;

    let mut cache = WikiCache::default();
    cache
        .context_map
        .insert("internal".into(), vec!["ADR-001".into()]);
    cache.metas.insert(
        "ADR-001".into(),
        ConstraintMeta {
            id: "ADR-001".into(),
            summary: "Rule.".into(),
            severity: ConstraintSeverity::Advisory,
            predicate_kind: PredicateKind::Static,
            domains: vec!["internal".into()],
            mandatory_for_tags: vec![],
            related_to: vec![],
            payload_version: "v1".into(),
            inline_predicate: None,
            source: None,
            last_updated_ms: None,
        },
    );

    let resolved = cache.resolve(&["internal".to_string()], &["ADR-001".to_string()]);
    assert_eq!(resolved.len(), 1);
}

#[test]
fn wiki_cache_from_docs_builds_index() {
    use h2ai_constraints::types::ConstraintDoc;
    use h2ai_constraints::wiki::WikiCache;

    let doc = ConstraintDoc {
        id: "ADR-X".into(),
        source_file: "adr-x.md".into(),
        description: "Test constraint.".into(),
        severity: ConstraintSeverity::Advisory,
        predicate: ConstraintPredicate::VocabularyPresence {
            mode: VocabularyMode::AnyOf,
            terms: vec!["test".into()],
        },
        remediation_hint: None,
        domains: vec!["internal".into()],
        mandatory_for_tags: vec!["audit".into()],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    };
    let cache = WikiCache::from_docs(&[doc]);
    assert!(cache.metas.contains_key("ADR-X"));
    assert!(
        cache.context_map.contains_key("internal"),
        "domain 'internal' must be in context_map"
    );
    assert!(
        cache.context_map.contains_key("audit"),
        "mandatory_for_tag 'audit' must be in context_map"
    );
    assert_eq!(cache.context_map["internal"], vec!["ADR-X"]);
    assert_eq!(cache.context_map["audit"], vec!["ADR-X"]);
}

// ── New tests: relation graph navigation ─────────────────────────────────────

fn make_doc(
    id: &str,
    domains: &[&str],
    related_to: &[&str],
) -> h2ai_constraints::types::ConstraintDoc {
    h2ai_constraints::types::ConstraintDoc {
        id: id.into(),
        source_file: format!("{id}.yaml"),
        description: format!("Constraint {id}"),
        severity: ConstraintSeverity::Advisory,
        predicate: ConstraintPredicate::LlmJudge {
            rubric: format!("{id} rubric text for retrieval"),
        },
        remediation_hint: None,
        domains: domains
            .iter()
            .map(std::string::ToString::to_string)
            .collect(),
        mandatory_for_tags: vec![],
        related_to: related_to
            .iter()
            .map(std::string::ToString::to_string)
            .collect(),
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    }
}

#[test]
fn navigate_related_returns_declared_relations() {
    use h2ai_constraints::wiki::WikiCache;

    let docs = vec![
        make_doc("C-001", &["billing"], &["C-002", "C-003"]),
        make_doc("C-002", &["billing"], &[]),
        make_doc("C-003", &["compliance"], &[]),
    ];
    let cache = WikiCache::from_docs(&docs);

    let related = cache.navigate_related("C-001");
    let ids: std::collections::HashSet<&str> = related.iter().map(|m| m.id.as_str()).collect();
    assert!(ids.contains("C-002"), "C-002 must be in related");
    assert!(ids.contains("C-003"), "C-003 must be in related");
    assert_eq!(ids.len(), 2);
}

#[test]
fn navigate_related_returns_empty_for_no_relations() {
    use h2ai_constraints::wiki::WikiCache;

    let docs = vec![make_doc("C-001", &["billing"], &[])];
    let cache = WikiCache::from_docs(&docs);
    assert!(cache.navigate_related("C-001").is_empty());
    assert!(cache.navigate_related("UNKNOWN").is_empty());
}

#[test]
fn navigate_by_domain_returns_domain_constraints() {
    use h2ai_constraints::wiki::WikiCache;

    let docs = vec![
        make_doc("C-001", &["billing"], &[]),
        make_doc("C-002", &["billing", "compliance"], &[]),
        make_doc("C-003", &["latency"], &[]),
    ];
    let cache = WikiCache::from_docs(&docs);

    let billing = cache.navigate_by_domain("billing");
    let ids: std::collections::HashSet<&str> = billing.iter().map(|m| m.id.as_str()).collect();
    assert!(ids.contains("C-001"));
    assert!(ids.contains("C-002"));
    assert!(!ids.contains("C-003"));
}

#[test]
fn navigate_by_domain_unknown_returns_empty() {
    use h2ai_constraints::wiki::WikiCache;

    let docs = vec![make_doc("C-001", &["billing"], &[])];
    let cache = WikiCache::from_docs(&docs);
    assert!(cache.navigate_by_domain("nonexistent").is_empty());
}

// ── Lines 126, 147-148, 161, 205-207: WikiCache coverage ─────────────────────

#[test]
fn wiki_cache_search_returns_empty_when_retriever_is_none() {
    use h2ai_constraints::wiki::WikiCache;

    // Simulate a deserialized WikiCache (retriever is #[serde(skip)] → None)
    let mut cache = WikiCache::default();
    cache.metas.insert(
        "C-001".into(),
        ConstraintMeta {
            id: "C-001".into(),
            summary: "budget idempotency atomic redis".into(),
            severity: ConstraintSeverity::Advisory,
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
    // retriever is None (default) — search must return empty vec
    let hits = cache.search("budget atomic redis", 5);
    assert!(
        hits.is_empty(),
        "search with no retriever must return empty"
    );
}

#[test]
fn wiki_cache_rebuild_retriever_enables_search() {
    use h2ai_constraints::types::ConstraintDoc;
    use h2ai_constraints::wiki::WikiCache;

    let docs = vec![h2ai_constraints::types::ConstraintDoc {
        id: "C-REBUILD".into(),
        source_file: "c-rebuild.yaml".into(),
        description: "atomic idempotency budget redis lua debit".into(),
        severity: ConstraintSeverity::Advisory,
        predicate: ConstraintPredicate::LlmJudge {
            rubric: "idempotency key atomic budget deduction".into(),
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    }];

    // Build normally (retriever present)
    let mut cache = WikiCache::from_docs(&docs);
    // Simulate losing the retriever (as if deserialized from NATS KV)
    cache.retriever = None;
    assert!(
        cache.search("atomic idempotency", 5).is_empty(),
        "no retriever → no results"
    );

    // Rebuild
    cache.rebuild_retriever(&docs);
    let hits = cache.search("atomic idempotency budget", 5);
    assert!(
        !hits.is_empty(),
        "after rebuild_retriever, search must work"
    );
    // Suppress unused warning for ConstraintDoc
    let _ = ConstraintDoc::new_llm_judge("x", "y");
}

#[test]
fn wiki_resolve_with_semantic_explicit_ids_and_tags() {
    use h2ai_constraints::wiki::WikiCache;

    let docs = vec![
        h2ai_constraints::types::ConstraintDoc {
            id: "C-EXPLICIT".into(),
            source_file: "c-explicit.yaml".into(),
            description: "Explicitly requested constraint".into(),
            severity: ConstraintSeverity::Hard { threshold: 0.45 },
            predicate: ConstraintPredicate::LlmJudge {
                rubric: "explicit".into(),
            },
            remediation_hint: None,
            domains: vec![],
            mandatory_for_tags: vec![],
            related_to: vec![],
            binary_checks: vec![],
            version: 1,
            repair_provenance: None,
            pass_criteria: None,
        },
        h2ai_constraints::types::ConstraintDoc {
            id: "C-TAGGED".into(),
            source_file: "c-tagged.yaml".into(),
            description: "Tag-matched constraint".into(),
            severity: ConstraintSeverity::Advisory,
            predicate: ConstraintPredicate::LlmJudge {
                rubric: "tagged billing".into(),
            },
            remediation_hint: None,
            domains: vec!["billing".into()],
            mandatory_for_tags: vec![],
            related_to: vec![],
            binary_checks: vec![],
            version: 1,
            repair_provenance: None,
            pass_criteria: None,
        },
    ];
    let cache = WikiCache::from_docs(&docs);

    // Stage 1 only: explicit_ids + tags, empty query_text
    let resolved = cache.resolve_with_semantic(
        &["billing".to_string()],
        &["C-EXPLICIT".to_string()],
        "", // empty query → no BM25 stage
        5,
    );
    let ids: std::collections::HashSet<&str> = resolved.iter().map(|m| m.id.as_str()).collect();
    assert!(ids.contains("C-EXPLICIT"), "explicit id must be included");
    assert!(
        ids.contains("C-TAGGED"),
        "tagged constraint must be included"
    );
}

#[test]
fn wiki_resolve_with_semantic_query_only() {
    use h2ai_constraints::wiki::WikiCache;

    let docs = vec![h2ai_constraints::types::ConstraintDoc {
        id: "C-QUERY-ONLY".into(),
        source_file: "c-query.yaml".into(),
        description: "stateless service cache eviction ttl".into(),
        severity: ConstraintSeverity::Advisory,
        predicate: ConstraintPredicate::LlmJudge {
            rubric: "stateless request service ttl cache eviction sticky session".into(),
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: vec![],
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    }];
    let cache = WikiCache::from_docs(&docs);

    // No tags, no explicit_ids — only BM25 semantic search
    let resolved = cache.resolve_with_semantic(&[], &[], "stateless service cache eviction", 5);
    assert!(
        resolved.iter().any(|m| m.id == "C-QUERY-ONLY"),
        "BM25-only path must surface C-QUERY-ONLY"
    );
}

// ── New tests: BM25 semantic search ─────────────────────────────────────────

#[test]
fn wiki_search_returns_relevant_results() {
    use h2ai_constraints::wiki::WikiCache;

    // Override make_doc defaults with content-rich descriptions for BM25 relevance:
    let docs_rich = vec![
        h2ai_constraints::types::ConstraintDoc {
            id: "C-BUD".into(),
            source_file: "c-bud.yaml".into(),
            description: "Budget idempotency atomic redis lua debit constraint".into(),
            severity: ConstraintSeverity::Advisory,
            predicate: ConstraintPredicate::LlmJudge {
                rubric: "idempotency key atomic check budget deduction redis lua".into(),
            },
            remediation_hint: None,
            domains: vec!["billing".into()],
            mandatory_for_tags: vec![],
            related_to: vec![],
            binary_checks: vec![],
            version: 1,
            repair_provenance: None,
            pass_criteria: None,
        },
        h2ai_constraints::types::ConstraintDoc {
            id: "C-GRP".into(),
            source_file: "c-grp.yaml".into(),
            description: "gRPC protobuf internal service protocol constraint".into(),
            severity: ConstraintSeverity::Advisory,
            predicate: ConstraintPredicate::LlmJudge {
                rubric: "grpc protobuf binary internal service communication".into(),
            },
            remediation_hint: None,
            domains: vec!["communication".into()],
            mandatory_for_tags: vec![],
            related_to: vec![],
            binary_checks: vec![],
            version: 1,
            repair_provenance: None,
            pass_criteria: None,
        },
    ];
    let cache = WikiCache::from_docs(&docs_rich);

    let hits = cache.search("atomic idempotency budget redis deduction", 3);
    assert!(
        !hits.is_empty(),
        "BM25 search must return results for relevant query"
    );
    assert_eq!(
        hits[0].id, "C-BUD",
        "budget constraint must rank first for budget query"
    );
}

#[test]
fn wiki_search_returns_empty_for_stopword_only_query() {
    use h2ai_constraints::wiki::WikiCache;

    let docs = vec![make_doc("C-001", &["billing"], &[])];
    let cache = WikiCache::from_docs(&docs);
    let hits = cache.search("the and for with", 5);
    assert!(
        hits.is_empty(),
        "stop-words-only query must return no results"
    );
}

#[test]
fn wiki_resolve_with_semantic_merges_tag_and_bm25() {
    use h2ai_constraints::wiki::WikiCache;

    let docs = vec![
        h2ai_constraints::types::ConstraintDoc {
            id: "C-TAG".into(),
            source_file: "c-tag.yaml".into(),
            description: "Tag-based mandatory constraint".into(),
            severity: ConstraintSeverity::Hard { threshold: 0.45 },
            predicate: ConstraintPredicate::LlmJudge {
                rubric: "unrelated rubric topic".into(),
            },
            remediation_hint: None,
            domains: vec!["billing".into()],
            mandatory_for_tags: vec![],
            related_to: vec![],
            binary_checks: vec![],
            version: 1,
            repair_provenance: None,
            pass_criteria: None,
        },
        h2ai_constraints::types::ConstraintDoc {
            id: "C-SEM".into(),
            source_file: "c-sem.yaml".into(),
            description: "Semantic-only constraint for stateless service design".into(),
            severity: ConstraintSeverity::Advisory,
            predicate: ConstraintPredicate::LlmJudge {
                rubric: "stateless request service ttl cache eviction sticky session".into(),
            },
            remediation_hint: None,
            domains: vec!["availability".into()],
            mandatory_for_tags: vec![],
            related_to: vec![],
            binary_checks: vec![],
            version: 1,
            repair_provenance: None,
            pass_criteria: None,
        },
    ];
    let cache = WikiCache::from_docs(&docs);

    // billing tag → C-TAG; "stateless" text → C-SEM; result must contain both
    let resolved = cache.resolve_with_semantic(
        &["billing".to_string()],
        &[],
        "stateless service cache eviction",
        5,
    );
    let ids: std::collections::HashSet<&str> = resolved.iter().map(|m| m.id.as_str()).collect();
    assert!(
        ids.contains("C-TAG"),
        "tag-matched constraint must be included"
    );
    assert!(
        ids.contains("C-SEM"),
        "semantically-matched constraint must be included"
    );
}
