use h2ai_orchestrator::induction::{
    cluster_by_similarity, jaccard_shingles, normalize_for_shingling, trigram_shingles,
    InductionResult,
};

// ── normalize_for_shingling ───────────────────────────────────────────────────

#[test]
fn normalize_strips_version_tokens() {
    let out = normalize_for_shingling("ClickHouse v25.6 MergeTree");
    assert!(
        !out.contains("25"),
        "version digits must be stripped: got '{out}'"
    );
    assert!(
        !out.contains("v25"),
        "version token must be stripped: got '{out}'"
    );
}

#[test]
fn normalize_lowercases_and_splits_camel() {
    let out = normalize_for_shingling("ReplacingMergeTree");
    // CamelCase split: "replacing merge tree"
    assert_eq!(out, "replacing merge tree");
}

#[test]
fn normalize_strips_punctuation() {
    let out = normalize_for_shingling("CONSTRAINT-005: use append-only");
    // hyphens stripped, colon stripped, lowercased
    assert!(!out.contains('-'), "hyphens must be removed: got '{out}'");
    assert!(!out.contains(':'), "colons must be removed: got '{out}'");
}

// ── trigram_shingles ─────────────────────────────────────────────────────────

#[test]
fn trigram_shingles_empty_string() {
    let shingles = trigram_shingles("");
    assert!(shingles.is_empty());
}

#[test]
fn trigram_shingles_short_string() {
    // "ab" is too short for a trigram (needs 3 chars)
    let shingles = trigram_shingles("ab");
    assert!(shingles.is_empty());
}

#[test]
fn trigram_shingles_three_chars() {
    let shingles = trigram_shingles("abc");
    assert_eq!(shingles.len(), 1);
    assert_eq!(shingles[0], [b'a', b'b', b'c']);
}

// ── jaccard_shingles ─────────────────────────────────────────────────────────

#[test]
fn jaccard_identical_strings_is_one() {
    let a = trigram_shingles("billing audit log");
    let b = trigram_shingles("billing audit log");
    let j = jaccard_shingles(&a, &b);
    assert!((j - 1.0).abs() < 1e-9);
}

#[test]
fn jaccard_disjoint_strings_is_zero() {
    let a = trigram_shingles("aaa bbb");
    let b = trigram_shingles("zzz yyy");
    let j = jaccard_shingles(&a, &b);
    assert!((j - 0.0).abs() < 1e-9);
}

#[test]
fn jaccard_empty_is_zero() {
    let a: Vec<[u8; 3]> = vec![];
    let b: Vec<[u8; 3]> = vec![];
    let j = jaccard_shingles(&a, &b);
    assert_eq!(j, 0.0);
}

// ── cluster_by_similarity ────────────────────────────────────────────────────

#[test]
fn cluster_identical_strings_same_cluster() {
    let strings = vec![
        "billing audit log".to_string(),
        "billing audit log".to_string(),
    ];
    let labels = cluster_by_similarity(&strings, 0.3);
    assert_eq!(
        labels[0], labels[1],
        "identical strings must be in same cluster"
    );
}

#[test]
fn cluster_dissimilar_strings_different_clusters() {
    let strings = vec![
        "billing audit log".to_string(),
        "network timeout redis".to_string(),
    ];
    let labels = cluster_by_similarity(&strings, 0.3);
    assert_ne!(
        labels[0], labels[1],
        "dissimilar strings must be in different clusters"
    );
}

// ── InductionResult compatibility gate ───────────────────────────────────────

#[test]
fn induction_result_compatible_when_tags_overlap() {
    let result = InductionResult {
        patterns: vec![],
        trigger_tags: vec!["billing".to_string(), "audit-log".to_string()],
    };
    assert!(result.is_compatible_with(&["billing".to_string(), "C-005".to_string()]));
}

#[test]
fn induction_result_incompatible_when_no_tag_overlap() {
    let result = InductionResult {
        patterns: vec![],
        trigger_tags: vec!["auth".to_string()],
    };
    assert!(!result.is_compatible_with(&["billing".to_string(), "C-005".to_string()]));
}
