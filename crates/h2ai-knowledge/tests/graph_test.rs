use h2ai_knowledge::graph::ConstraintGraph;
use h2ai_knowledge::types::{CrossRef, KnowledgeNode, NodeDepth, NodeSource};

fn leaf(id: &str, related: Vec<&str>, cross_refs: Vec<&str>) -> KnowledgeNode {
    KnowledgeNode {
        id: id.to_string(),
        depth: NodeDepth::Leaf,
        synthesis: format!("constraint {id}"),
        invariants: vec![],
        failure_modes: vec![],
        domains: vec!["financial_systems".into()],
        entry_points: vec![],
        tensions: vec![],
        cross_references: cross_refs
            .into_iter()
            .map(|r| CrossRef {
                id: r.to_string(),
                domain: String::new(),
                reason: String::new(),
            })
            .collect(),
        related: related.into_iter().map(String::from).collect(),
        source: NodeSource::YamlConstraint { id: id.to_string() },
        importance: 0.7,
    }
}

#[test]
fn ppr_expands_related_constraints() {
    let nodes = vec![
        leaf("C-004", vec!["C-005", "C-008"], vec![]),
        leaf("C-005", vec![], vec!["C-007"]),
        leaf("C-007", vec![], vec![]),
        leaf("C-008", vec![], vec![]),
    ];
    let graph = ConstraintGraph::build(&nodes);
    let results = graph.ppr(&["C-004"], 0.15, 5, 20);
    let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
    assert!(
        !ids.contains(&"C-004"),
        "seeds must not appear in ppr results"
    );
    assert!(ids.contains(&"C-005"), "C-005 must be reachable from C-004");
    assert!(ids.contains(&"C-008"), "C-008 must be reachable from C-004");
}

#[test]
fn ppr_no_panic_isolated_node() {
    let nodes = vec![leaf("C-001", vec![], vec![])];
    let graph = ConstraintGraph::build(&nodes);
    let results = graph.ppr(&["C-001"], 0.15, 5, 20);
    assert!(results.is_empty(), "isolated node expands to nothing");
}

#[test]
fn ppr_unknown_seed_returns_empty() {
    let nodes = vec![leaf("C-004", vec!["C-005"], vec![])];
    let graph = ConstraintGraph::build(&nodes);
    let results = graph.ppr(&["UNKNOWN"], 0.15, 5, 20);
    assert!(results.is_empty());
}

#[test]
fn ppr_results_are_sorted_descending() {
    let nodes = vec![
        leaf("A", vec!["B", "C", "D"], vec![]),
        leaf("B", vec![], vec![]),
        leaf("C", vec![], vec![]),
        leaf("D", vec![], vec![]),
    ];
    let graph = ConstraintGraph::build(&nodes);
    let results = graph.ppr(&["A"], 0.15, 10, 20);
    for i in 1..results.len() {
        assert!(
            results[i - 1].1 >= results[i].1,
            "results must be sorted descending by score"
        );
    }
}
