use h2ai_constraints::source::{ConstraintSource, InMemorySource, RuntimeConstraintStore};
use h2ai_constraints::spec::SemanticSpec;

fn make_spec(id: &str) -> SemanticSpec {
    SemanticSpec::builder(id)
        .title(format!("Constraint {id}"))
        .rubric_pass("Pass.")
        .rubric_fail("Fail.")
        .build()
}

#[test]
fn in_memory_source_returns_all_specs() {
    let specs = vec![make_spec("C-T1"), make_spec("C-T2")];
    let source = InMemorySource { specs };
    let loaded = source.load_all().expect("InMemorySource must not fail");
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0].id, "C-T1");
    assert_eq!(loaded[1].id, "C-T2");
}

#[test]
fn runtime_constraint_store_from_source_loads_docs() {
    let specs = vec![make_spec("C-T3")];
    let source = InMemorySource { specs };
    let store = RuntimeConstraintStore::from_source(&source).expect("must load");
    let docs = store.all_docs_sorted();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0].id, "C-T3");
}

#[test]
fn fs_constraint_store_alias_still_works() {
    use h2ai_constraints::source::FsConstraintStore;
    let specs = vec![make_spec("C-T4")];
    let source = InMemorySource { specs };
    let _store: FsConstraintStore =
        RuntimeConstraintStore::from_source(&source).expect("must load");
}
