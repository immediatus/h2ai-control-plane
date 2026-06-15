use h2ai_types::thinking::ThinkingReport;

#[test]
fn thinking_report_new_fields_default_to_zero_empty() {
    let report = ThinkingReport::default();
    assert!(report.retrieved_node_ids.is_empty());
    assert_eq!(report.skill_nodes_used, 0);
}

#[test]
fn thinking_report_backwards_compat_deserializes_without_new_fields() {
    let json = r#"{"shared_understanding":"test","tensions":[],"coverage_score":0.8,"iteration":1,"prev_similarity":0.5}"#;
    let report: ThinkingReport = serde_json::from_str(json).unwrap();
    assert!(report.retrieved_node_ids.is_empty());
    assert_eq!(report.skill_nodes_used, 0);
}

#[test]
fn thinking_report_roundtrip_preserves_new_fields() {
    let report = ThinkingReport {
        retrieved_node_ids: vec!["node-1".to_string(), "node-2".to_string()],
        skill_nodes_used: 2,
        ..Default::default()
    };
    let json = serde_json::to_string(&report).unwrap();
    let restored: ThinkingReport = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.retrieved_node_ids, vec!["node-1", "node-2"]);
    assert_eq!(restored.skill_nodes_used, 2);
}
