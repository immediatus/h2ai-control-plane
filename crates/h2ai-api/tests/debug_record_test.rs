#![allow(clippy::missing_panics_doc)]

use h2ai_api::debug_record::{append_debug_record, TaskDebugRecord};
use std::io::BufRead;

// ── append_debug_record: happy path ──────────────────────────────────────────

#[test]
fn append_debug_record_writes_valid_json_line() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("debug.jsonl");
    let record = TaskDebugRecord::default();
    append_debug_record(path.to_str().unwrap(), &record);

    let content = std::fs::read_to_string(&path).unwrap();
    assert!(!content.is_empty(), "file must have content");
    let line = content.lines().next().unwrap();
    serde_json::from_str::<serde_json::Value>(line).expect("must be valid JSON");
}

// ── append_debug_record: creates file when absent ────────────────────────────

#[test]
fn append_debug_record_creates_file_if_absent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("new_debug.jsonl");
    assert!(!path.exists(), "file must not exist yet");

    append_debug_record(path.to_str().unwrap(), &TaskDebugRecord::default());

    assert!(path.exists(), "file must be created");
}

// ── append_debug_record: multiple calls → multiple lines ─────────────────────

#[test]
fn append_debug_record_appends_second_record_as_new_line() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi.jsonl");

    append_debug_record(path.to_str().unwrap(), &TaskDebugRecord::default());
    append_debug_record(path.to_str().unwrap(), &TaskDebugRecord::default());

    let file = std::fs::File::open(&path).unwrap();
    let line_count = std::io::BufReader::new(file).lines().count();
    assert_eq!(line_count, 2, "two records must produce two lines");
}

// ── append_debug_record: unwritable path logs and does not panic ─────────────

#[test]
fn append_debug_record_bad_path_does_not_panic() {
    let record = TaskDebugRecord::default();
    // Path into non-existent directory — open will fail
    append_debug_record("/nonexistent_h2ai_test_dir/debug.jsonl", &record);
    // Test passes if no panic
}
