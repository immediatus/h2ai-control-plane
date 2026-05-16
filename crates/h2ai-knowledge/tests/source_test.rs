use h2ai_knowledge::source::{KnowledgeSource, YamlDirSource};
use h2ai_knowledge::types::NodeDepth;
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/constraints")
}

#[tokio::test]
async fn yaml_dir_source_loads_constraints() {
    let source = YamlDirSource::new(fixture_dir());
    let items = source.all_items().await;
    assert_eq!(items.len(), 3, "should load C-004, C-005, C-008");
    let ids: Vec<&str> = items.iter().map(|i| i.id.as_str()).collect();
    assert!(ids.contains(&"C-004"));
    assert!(ids.contains(&"C-005"));
    assert!(ids.contains(&"C-008"));
}

#[tokio::test]
async fn yaml_dir_source_constraint_has_related() {
    let source = YamlDirSource::new(fixture_dir());
    let items = source.all_items().await;
    let c004 = items.iter().find(|i| i.id == "C-004").unwrap();
    assert!(c004.related.contains(&"C-005".to_string()));
    assert!(c004.related.contains(&"C-008".to_string()));
}

#[tokio::test]
async fn yaml_dir_source_loads_wiki_nodes() {
    let source = YamlDirSource::new(fixture_dir());
    let nodes = source.wiki_nodes().await;
    assert_eq!(nodes.len(), 1, "one topic node: financial-systems");
    assert_eq!(nodes[0].depth, NodeDepth::Topic);
    assert!(nodes[0].id.contains("financial-systems"));
    assert!(!nodes[0].synthesis.is_empty());
}

#[tokio::test]
async fn yaml_dir_source_global_node_present() {
    let source = YamlDirSource::new(fixture_dir());
    let global = source.global_node().await;
    assert!(global.is_some());
    let g = global.unwrap();
    assert_eq!(g.depth, NodeDepth::Global);
    assert!(!g.synthesis.is_empty());
}

#[tokio::test]
async fn yaml_dir_source_global_node_absent() {
    let tmp = tempfile::TempDir::new().unwrap();
    let source = YamlDirSource::new(tmp.path());
    let global = source.global_node().await;
    assert!(global.is_none());
}
