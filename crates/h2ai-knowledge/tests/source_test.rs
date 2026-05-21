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

#[tokio::test]
async fn yaml_dir_source_wiki_node_id_derived_from_stem_when_empty() {
    // A wiki YAML without an `id` field → id must be derived as "topic:<stem>"
    let tmp = tempfile::TempDir::new().unwrap();
    let wiki_dir = tmp.path().join("wiki");
    std::fs::create_dir(&wiki_dir).unwrap();
    std::fs::write(
        wiki_dir.join("my-domain.yaml"),
        r#"synthesis: "Domain synthesis text for testing"
domains:
  - my-domain
"#,
    )
    .unwrap();

    let source = YamlDirSource::new(tmp.path());
    let nodes = source.wiki_nodes().await;
    assert_eq!(nodes.len(), 1);
    assert_eq!(
        nodes[0].id, "topic:my-domain",
        "node id must be derived from file stem when id field is absent"
    );
}

#[tokio::test]
async fn yaml_dir_source_wiki_nodes_skips_non_yaml_files() {
    let tmp = tempfile::TempDir::new().unwrap();
    let wiki_dir = tmp.path().join("wiki");
    std::fs::create_dir(&wiki_dir).unwrap();
    std::fs::write(wiki_dir.join("readme.txt"), "not a yaml").unwrap();
    std::fs::write(
        wiki_dir.join("real-topic.yaml"),
        r#"id: topic:real-topic
synthesis: "Real synthesis"
domains:
  - real
"#,
    )
    .unwrap();

    let source = YamlDirSource::new(tmp.path());
    let nodes = source.wiki_nodes().await;
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].id, "topic:real-topic");
}

#[tokio::test]
async fn yaml_dir_source_all_items_returns_empty_for_missing_dir() {
    // Corpus dir does not exist — load_corpus fails, returns empty vec (no panic)
    let source = YamlDirSource::new("/nonexistent/path/corpus");
    let items = source.all_items().await;
    assert!(items.is_empty());
}
