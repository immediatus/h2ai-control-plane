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
}

#[test]
fn constraint_meta_source_defaults_to_none() {
    // Constraints without provenance metadata must still deserialize cleanly.
    let json = r#"{"id":"X","summary":"s","severity":"Advisory","predicate_kind":"static","domains":[],"mandatory_for_tags":[],"payload_version":"v1"}"#;
    let meta: ConstraintMeta = serde_json::from_str(json).unwrap();
    assert!(meta.source.is_none());
    assert!(meta.last_updated_ms.is_none());
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
    };
    let meta = ConstraintMeta::from_doc(&doc);
    assert_eq!(meta.predicate_kind, PredicateKind::Static);
    assert!(meta.inline_predicate.is_some());
}

#[test]
fn predicate_kind_from_tier_llm_judge() {
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate};
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
    };
    let meta = ConstraintMeta::from_doc(&doc);
    assert_eq!(meta.predicate_kind, PredicateKind::LlmJudge);
    // LlmJudge predicates must NOT be inlined — they require a Predicate Store fetch.
    assert!(
        meta.inline_predicate.is_none(),
        "LlmJudge must not be inlined"
    );
}

#[test]
fn predicate_kind_from_tier_oracle() {
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate};
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
    };
    let meta = ConstraintMeta::from_doc(&doc);
    assert_eq!(meta.predicate_kind, PredicateKind::Oracle);
    // Oracle predicates must NOT be inlined — they require a Predicate Store fetch.
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
            payload_version: "v1".into(),
            inline_predicate: None,
            source: None,
            last_updated_ms: None,
        },
    );

    // ADR-001 matched by both tag and explicit — should appear once.
    let resolved = cache.resolve(&["internal".to_string()], &["ADR-001".to_string()]);
    assert_eq!(resolved.len(), 1);
}

#[test]
fn wiki_cache_from_docs_builds_index() {
    use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, VocabularyMode};
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
    };
    let cache = WikiCache::from_docs(&[doc]);
    assert!(cache.metas.contains_key("ADR-X"));
    // Verify context_map is populated from domains and mandatory_for_tags
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
