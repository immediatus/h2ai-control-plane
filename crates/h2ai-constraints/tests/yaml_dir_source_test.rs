use h2ai_constraints::loader::YamlDirSource;
use h2ai_constraints::source::ConstraintSource;
use std::path::PathBuf;

fn corpus_dir() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest)
        .join("../../tests/e2e/constraints")
        .canonicalize()
        .expect("ads-platform constraints directory must exist")
}

#[test]
fn yaml_dir_source_loads_all_yaml_files() {
    let source = YamlDirSource::new(corpus_dir());
    let specs = source.load_all().expect("must load corpus");
    assert!(
        !specs.is_empty(),
        "YamlDirSource must load at least one constraint"
    );
    for s in &specs {
        assert!(!s.id.is_empty(), "every spec must have a non-empty id");
    }
}

#[test]
fn yaml_dir_source_nonexistent_dir_returns_empty() {
    let source = YamlDirSource::new("/nonexistent/path/that/does/not/exist");
    let specs = source
        .load_all()
        .expect("nonexistent dir must return empty vec, not error");
    assert!(specs.is_empty());
}

#[test]
fn yaml_dir_source_deduplicates_by_id() {
    // If we load from a real dir, IDs must be unique.
    let source = YamlDirSource::new(corpus_dir());
    let specs = source.load_all().expect("must load");
    let mut ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    for s in &specs {
        assert!(ids.insert(s.id.clone()), "duplicate ID found: {}", s.id);
    }
}
