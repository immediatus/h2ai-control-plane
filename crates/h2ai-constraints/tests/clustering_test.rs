use h2ai_constraints::clustering::{build_clusters, cluster_check_indices};
use h2ai_constraints::types::{ConstraintDoc, ConstraintPredicate, ConstraintSeverity};

fn doc(id: &str, related: &[&str]) -> ConstraintDoc {
    ConstraintDoc {
        id: id.to_owned(),
        source_file: format!("{id}.yaml"),
        description: String::new(),
        severity: ConstraintSeverity::Hard { threshold: 0.5 },
        predicate: ConstraintPredicate::LlmJudge {
            rubric: String::new(),
        },
        remediation_hint: None,
        domains: vec![],
        mandatory_for_tags: vec![],
        related_to: related.iter().map(|s| s.to_string()).collect(),
        binary_checks: vec![],
        version: 1,
        repair_provenance: None,
        pass_criteria: None,
    }
}

#[test]
fn single_doc_produces_one_singleton_cluster() {
    let docs = vec![doc("A", &[])];
    let clusters = build_clusters(&docs);
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0], vec!["A"]);
}

#[test]
fn two_unrelated_docs_produce_two_singleton_clusters() {
    let docs = vec![doc("A", &[]), doc("B", &[])];
    let clusters = build_clusters(&docs);
    assert_eq!(clusters.len(), 2);
    let ids: std::collections::HashSet<String> =
        clusters.iter().flat_map(|c| c.iter().cloned()).collect();
    assert!(ids.contains("A"));
    assert!(ids.contains("B"));
}

#[test]
fn two_docs_with_direct_related_to_form_one_cluster() {
    let docs = vec![doc("A", &["B"]), doc("B", &[])];
    let clusters = build_clusters(&docs);
    assert_eq!(clusters.len(), 1);
    let ids: std::collections::HashSet<String> = clusters[0].iter().cloned().collect();
    assert!(ids.contains("A"));
    assert!(ids.contains("B"));
}

#[test]
fn transitive_chain_a_b_b_c_forms_one_cluster() {
    let docs = vec![doc("A", &["B"]), doc("B", &["C"]), doc("C", &[])];
    let clusters = build_clusters(&docs);
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].len(), 3);
}

#[test]
fn two_independent_pairs_produce_two_clusters() {
    let docs = vec![
        doc("A", &["B"]),
        doc("B", &[]),
        doc("C", &["D"]),
        doc("D", &[]),
    ];
    let clusters = build_clusters(&docs);
    assert_eq!(clusters.len(), 2);
}

#[test]
fn empty_corpus_produces_empty_clusters() {
    let clusters = build_clusters(&[]);
    assert!(clusters.is_empty());
}

#[test]
fn reference_to_unknown_id_does_not_create_phantom_node() {
    let docs = vec![doc("A", &["X"])];
    let clusters = build_clusters(&docs);
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0], vec!["A"]);
}

#[test]
fn cluster_check_indices_returns_indices_of_matching_ids() {
    let all_checks = vec![
        "A:check-one".to_owned(),
        "A:check-two".to_owned(),
        "B:check-one".to_owned(),
        "C:check-one".to_owned(),
    ];
    let cluster = vec!["A".to_owned()];
    let indices = cluster_check_indices(&cluster, &all_checks);
    assert_eq!(indices, vec![0, 1]);
}

#[test]
fn cluster_check_indices_empty_when_no_match() {
    let all_checks = vec!["B:check-one".to_owned()];
    let cluster = vec!["A".to_owned()];
    let indices = cluster_check_indices(&cluster, &all_checks);
    assert!(indices.is_empty());
}

#[test]
fn cluster_check_indices_multi_cluster_ids() {
    let all_checks = vec!["A:foo".to_owned(), "B:bar".to_owned(), "C:baz".to_owned()];
    let cluster = vec!["A".to_owned(), "C".to_owned()];
    let indices = cluster_check_indices(&cluster, &all_checks);
    assert_eq!(indices, vec![0, 2]);
}
