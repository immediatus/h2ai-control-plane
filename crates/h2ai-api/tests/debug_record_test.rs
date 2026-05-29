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

// ── TaskDebugRecord::build ────────────────────────────────────────────────────

#[test]
fn build_produces_correct_task_id_and_resolved_output() {
    use chrono::Utc;
    use h2ai_api::debug_record::TaskDebugRecord;
    use h2ai_config::H2AIConfig;
    use h2ai_orchestrator::attribution::HarnessAttribution;
    use h2ai_orchestrator::coherence::CoherenceState;
    use h2ai_orchestrator::engine::EngineOutput;
    use h2ai_types::events::{
        SelectionResolvedEvent, TaskComplexityAssessedEvent, VerificationScoredEvent,
    };
    use h2ai_types::identity::{ExplorerId, TaskId};
    use h2ai_types::sizing::{MergeStrategy, PredictionBasis, ProbeSkipReason, TaskQuadrant};

    let task_id = TaskId::new();
    let output = EngineOutput {
        task_id: task_id.clone(),
        resolved_output: "the answer".into(),
        selection_resolved: SelectionResolvedEvent {
            task_id: task_id.clone(),
            valid_proposals: vec![],
            pruned_proposals: vec![],
            merge_strategy: MergeStrategy::ScoreOrdered,
            timestamp: Utc::now(),
            merge_elapsed_secs: None,
            n_input_proposals: 0,
            n_failed_proposals: 0,
        },
        attribution: HarnessAttribution {
            baseline_quality: 0.7,
            topology_gain: 0.1,
            verification_gain: 0.05,
            tao_gain: 0.02,
            q_confidence: 0.82,
            prediction_basis: PredictionBasis::Heuristic,
            q_measured: None,
            rho_adjusted: 0.3,
            case_b_flag: false,
            synthesis_gain: 0.0,
        },
        attribution_interval: None,
        verification_events: vec![VerificationScoredEvent {
            task_id: task_id.clone(),
            explorer_id: ExplorerId::new(),
            score: 0.9,
            reason: "ok".into(),
            passed: true,
            cache_hit: false,
            timestamp: Utc::now(),
        }],
        failed_proposals: vec![],
        talagrand: None,
        suggested_next_params: None,
        waste_ratio: 0.25,
        applied_optimizations: vec![],
        topology_retry_events: vec![],
        mode_collapse_count: 0,
        epistemic_yield: None,
        task_quadrant: Some(TaskQuadrant::Coverage),
        complexity_event: TaskComplexityAssessedEvent {
            task_id: task_id.clone(),
            tcc_structural: 0.5,
            tcc_empirical: None,
            tcc_effective: 0.5,
            n_eff_pool: None,
            task_quadrant: TaskQuadrant::Coverage,
            probe_skipped: true,
            probe_skip_reason: ProbeSkipReason::None,
            heavy_fraction: 0.0,
            tcc_mismatch: false,
            probe_cost_tokens: 0,
            n_informative_static: 0,
            timestamp: Utc::now(),
        },
        frontier_event: None,
        adapter_correctness: vec![],
        coherence_state: CoherenceState::default(),
        comparison_events: vec![],
        shadow_audit_events: vec![],
        correlated_warnings: vec![],
        researcher_grounding_events: vec![],
        diversity_degraded_event: None,
        srani_events: vec![],
        srani_ema_cfi_updated: 0.15,
        srani_count_updated: 3,
        oracle_gate_passed: None,
        leader_elected_events: vec![],
        socratic_diagnosis_events: vec![],
        consensus_agreement_rate: Some(0.9),
    };

    let cfg = H2AIConfig::default();
    let record = TaskDebugRecord::build("test description", 0.1, 2, &output, &cfg);

    let json = serde_json::to_value(&record).expect("must serialize");
    assert_eq!(json["task_id"], task_id.to_string());
    assert_eq!(json["resolved_output"], "the answer");
    assert!((json["q_confidence"].as_f64().unwrap() - 0.82).abs() < 1e-9);
    assert!((json["waste_ratio"].as_f64().unwrap() - 0.25).abs() < 1e-9);
    assert_eq!(json["srani_ema_before"].as_f64().unwrap(), 0.1);
    assert_eq!(json["srani_count_before"].as_u64().unwrap(), 2);
    assert_eq!(json["srani_ema_after"].as_f64().unwrap(), 0.15);
    assert_eq!(json["srani_count_after"].as_u64().unwrap(), 3);
    assert_eq!(json["description"], "test description");
    // One verification event mapped
    assert_eq!(json["verification_events"].as_array().unwrap().len(), 1);
    assert!(json["verification_events"][0]["passed"].as_bool().unwrap());
}
