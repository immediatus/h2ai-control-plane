use h2ai_constraints::source::{
    metas_from_store, ConstraintSource, InMemorySource, RuntimeConstraintIndex,
    RuntimeConstraintStore,
};
use h2ai_constraints::spec::SemanticSpec;
use h2ai_constraints::store::ConstraintStore;

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

// ── Lines 63-64: RuntimeConstraintIndex with mandatory_for_tags ───────────────

#[test]
fn runtime_constraint_index_indexes_mandatory_for_tags() {
    use h2ai_constraints::index::ConstraintIndex;

    let spec = SemanticSpec::builder("C-TAGS")
        .title("Tagged Constraint")
        .rubric_pass("Pass.")
        .rubric_fail("Fail.")
        .mandatory_for_tag("billing")
        .mandatory_for_tag("audit")
        .build();
    let source = InMemorySource { specs: vec![spec] };
    let store = RuntimeConstraintStore::from_source(&source).expect("must load");
    let docs = store.all_docs_sorted();
    let index = RuntimeConstraintIndex::from_docs(&docs);

    // tags are indexed as "mandatory_for_tags" — find_by_tags must return the id
    let ids = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(index.find_by_tags(&["billing".to_string()]));
    assert!(
        ids.contains(&"C-TAGS".to_string()),
        "billing tag must resolve to C-TAGS"
    );
}

// ── Lines 131-133: all_docs() and all_docs_sorted() ──────────────────────────

#[test]
fn all_docs_returns_all_loaded_docs() {
    let specs = vec![make_spec("C-Z"), make_spec("C-A"), make_spec("C-M")];
    let source = InMemorySource { specs };
    let store = RuntimeConstraintStore::from_source(&source).expect("must load");

    let all = store.all_docs();
    assert_eq!(all.len(), 3);

    let sorted = store.all_docs_sorted();
    assert_eq!(sorted.len(), 3);
    assert_eq!(sorted[0].id, "C-A");
    assert_eq!(sorted[1].id, "C-M");
    assert_eq!(sorted[2].id, "C-Z");
}

// ── Line 148: RuntimeConstraintStore::load() returning Err for unknown ID ─────

#[tokio::test]
async fn constraint_store_load_unknown_id_returns_not_found() {
    let specs = vec![make_spec("C-KNOWN")];
    let source = InMemorySource { specs };
    let store = RuntimeConstraintStore::from_source(&source).expect("must load");

    let result = store.load("C-UNKNOWN").await;
    assert!(result.is_err(), "unknown id must return Err");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("C-UNKNOWN"),
        "error must mention the missing id"
    );
}

// ── Lines 153-159: metas_from_store ──────────────────────────────────────────

#[test]
fn metas_from_store_returns_one_meta_per_doc() {
    let specs = vec![make_spec("C-META-1"), make_spec("C-META-2")];
    let source = InMemorySource { specs };
    let store = RuntimeConstraintStore::from_source(&source).expect("must load");

    let metas = metas_from_store(&store);
    assert_eq!(metas.len(), 2);
    let ids: Vec<&str> = metas.iter().map(|m| m.id.as_str()).collect();
    assert!(ids.contains(&"C-META-1"));
    assert!(ids.contains(&"C-META-2"));
}
