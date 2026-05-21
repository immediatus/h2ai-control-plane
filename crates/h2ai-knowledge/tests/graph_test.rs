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

#[test]
fn ppr_empty_graph_returns_empty() {
    // n == 0 path: ConstraintGraph::ppr returns early
    let graph = ConstraintGraph::build(&[]);
    let results = graph.ppr(&["any"], 0.15, 5, 20);
    assert!(results.is_empty(), "empty graph must return empty results");
}

#[test]
fn build_graph_self_loop_ignored() {
    // A node that references itself in `related` — self-loop must be skipped
    let nodes = vec![
        leaf("C-001", vec!["C-001", "C-002"], vec![]),
        leaf("C-002", vec![], vec![]),
    ];
    let graph = ConstraintGraph::build(&nodes);
    // PPR from C-001 should reach C-002 but C-001 itself is excluded (seed)
    let results = graph.ppr(&["C-001"], 0.15, 5, 20);
    let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
    assert!(
        !ids.contains(&"C-001"),
        "C-001 is a seed and must not appear in results"
    );
    assert!(ids.contains(&"C-002"), "C-002 must be reachable from C-001");
}

#[test]
fn build_graph_unknown_neighbour_skipped() {
    // C-001 references C-999 which does not exist in the graph — must be silently skipped
    let nodes = vec![
        leaf("C-001", vec!["C-999", "C-002"], vec![]),
        leaf("C-002", vec![], vec![]),
    ];
    let graph = ConstraintGraph::build(&nodes);
    // Must not panic; C-002 is reachable, C-999 is absent
    let results = graph.ppr(&["C-001"], 0.15, 5, 20);
    let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
    assert!(!ids.contains(&"C-999"), "unknown node must not appear");
    assert!(ids.contains(&"C-002"));
}
