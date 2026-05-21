use h2ai_types::identity::{TaskId, TenantId};
use h2ai_types::reasoning_checkpoint::{
    ArchetypeSelection, CompletedWave, ReasoningCheckpointPhase, TaskReasoningCheckpoint,
};

fn make_ids() -> (TaskId, TenantId) {
    (TaskId::new(), TenantId::from("test-tenant"))
}

// ── ReasoningCheckpointPhase ordering ────────────────────────────────────────

#[test]
fn phase_created_is_at_least_created() {
    let p = ReasoningCheckpointPhase::Created;
    assert!(p.is_at_least(&ReasoningCheckpointPhase::Created));
}

#[test]
fn phase_thinking_done_is_at_least_created() {
    let p = ReasoningCheckpointPhase::ThinkingDone;
    assert!(p.is_at_least(&ReasoningCheckpointPhase::Created));
}

#[test]
fn phase_wave_completed_is_at_least_thinking_done() {
    let p = ReasoningCheckpointPhase::WaveCompleted(0);
    assert!(p.is_at_least(&ReasoningCheckpointPhase::ThinkingDone));
}

#[test]
fn phase_wave_2_is_at_least_wave_0() {
    let p2 = ReasoningCheckpointPhase::WaveCompleted(2);
    let p0 = ReasoningCheckpointPhase::WaveCompleted(0);
    assert!(p2.is_at_least(&p0));
}

#[test]
fn phase_merge_done_is_at_least_wave_10() {
    let p = ReasoningCheckpointPhase::MergeDone;
    assert!(p.is_at_least(&ReasoningCheckpointPhase::WaveCompleted(10)));
}

#[test]
fn phase_resolved_is_at_least_merge_done() {
    let p = ReasoningCheckpointPhase::Resolved;
    assert!(p.is_at_least(&ReasoningCheckpointPhase::MergeDone));
}

#[test]
fn phase_created_is_not_at_least_thinking_done() {
    let p = ReasoningCheckpointPhase::Created;
    assert!(!p.is_at_least(&ReasoningCheckpointPhase::ThinkingDone));
}

#[test]
fn phase_wave_0_is_not_at_least_merge_done() {
    let p = ReasoningCheckpointPhase::WaveCompleted(0);
    assert!(!p.is_at_least(&ReasoningCheckpointPhase::MergeDone));
}

// ── ReasoningCheckpointPhase serde ───────────────────────────────────────────

#[test]
fn phase_created_serde_roundtrip() {
    let p = ReasoningCheckpointPhase::Created;
    let json = serde_json::to_string(&p).unwrap();
    let back: ReasoningCheckpointPhase = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, ReasoningCheckpointPhase::Created));
}

#[test]
fn phase_wave_completed_serde_roundtrip() {
    let p = ReasoningCheckpointPhase::WaveCompleted(3);
    let json = serde_json::to_string(&p).unwrap();
    let back: ReasoningCheckpointPhase = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, ReasoningCheckpointPhase::WaveCompleted(3)));
}

#[test]
fn phase_resolved_serde_roundtrip() {
    let p = ReasoningCheckpointPhase::Resolved;
    let json = serde_json::to_string(&p).unwrap();
    let back: ReasoningCheckpointPhase = serde_json::from_str(&json).unwrap();
    assert!(matches!(back, ReasoningCheckpointPhase::Resolved));
}

// ── TaskReasoningCheckpoint::new_created ─────────────────────────────────────

#[test]
fn new_created_sets_expected_defaults() {
    let (task_id, tenant_id) = make_ids();
    let cp = TaskReasoningCheckpoint::new_created(
        task_id.clone(),
        tenant_id.clone(),
        vec!["C-001".into(), "C-002".into()],
        Some("code".into()),
    );
    assert_eq!(cp.task_id, task_id);
    assert_eq!(cp.tenant_id, tenant_id);
    assert!(matches!(cp.phase, ReasoningCheckpointPhase::Created));
    assert_eq!(cp.constraint_tags, vec!["C-001", "C-002"]);
    assert_eq!(cp.domain, Some("code".into()));
    assert!(cp.shared_understanding.is_none());
    assert!(cp.tensions.is_none());
    assert!(cp.archetype_selection.is_none());
    assert!(cp.completed_waves.is_empty());
    assert_eq!(cp.retry_count, 0);
    assert_eq!(cp.hitl_timeouts_fired, 0);
    assert!(cp.task_quadrant.is_none());
    assert_eq!(cp.created_at, cp.last_updated);
}

#[test]
fn new_created_with_no_domain() {
    let (task_id, tenant_id) = make_ids();
    let cp = TaskReasoningCheckpoint::new_created(task_id, tenant_id, vec![], None);
    assert!(cp.domain.is_none());
}

// ── TaskReasoningCheckpoint serde roundtrip ───────────────────────────────────

#[test]
fn checkpoint_serde_roundtrip() {
    let (task_id, tenant_id) = make_ids();
    let mut cp = TaskReasoningCheckpoint::new_created(
        task_id.clone(),
        tenant_id,
        vec!["C-001".into()],
        Some("code".into()),
    );
    cp.phase = ReasoningCheckpointPhase::ThinkingDone;
    cp.shared_understanding = Some("The core problem is X.".into());
    cp.tensions = Some(vec!["performance vs correctness".into()]);
    cp.thinking_iterations = Some(3);
    cp.completed_waves.push(CompletedWave {
        wave_index: 0,
        adapter_outputs: vec![],
    });

    let json = serde_json::to_string(&cp).unwrap();
    let back: TaskReasoningCheckpoint = serde_json::from_str(&json).unwrap();
    assert_eq!(back.task_id, task_id);
    assert!(matches!(back.phase, ReasoningCheckpointPhase::ThinkingDone));
    assert_eq!(
        back.shared_understanding.as_deref(),
        Some("The core problem is X.")
    );
    assert_eq!(back.thinking_iterations, Some(3));
    assert_eq!(back.completed_waves.len(), 1);
}

// ── into_meta_state ───────────────────────────────────────────────────────────

#[test]
fn into_meta_state_returns_none_when_shared_understanding_absent() {
    let (task_id, tenant_id) = make_ids();
    let cp = TaskReasoningCheckpoint::new_created(task_id, tenant_id, vec![], None);
    assert!(cp.into_meta_state().is_none());
}

#[test]
fn into_meta_state_returns_some_with_full_thinking_artifacts() {
    let (task_id, tenant_id) = make_ids();
    let mut cp = TaskReasoningCheckpoint::new_created(
        task_id.clone(),
        tenant_id.clone(),
        vec!["C-001".into()],
        Some("code".into()),
    );
    cp.phase = ReasoningCheckpointPhase::Resolved;
    cp.shared_understanding = Some("The root cause is Y.".into());
    cp.tensions = Some(vec!["latency vs throughput".into()]);
    cp.archetype_selection = Some(vec![ArchetypeSelection {
        name: "microservices".into(),
        confidence: 0.8,
    }]);
    cp.thinking_iterations = Some(5);
    cp.retry_count = 1;

    let meta = cp.into_meta_state().expect("should produce meta state");
    assert_eq!(meta.task_id, task_id);
    assert_eq!(meta.tenant_id, tenant_id);
    assert_eq!(meta.shared_understanding, "The root cause is Y.");
    assert_eq!(meta.tensions, vec!["latency vs throughput"]);
    assert_eq!(meta.archetype_results.len(), 1);
    assert_eq!(meta.archetype_results[0].name, "microservices");
    assert!((meta.archetype_results[0].confidence - 0.8).abs() < 1e-9);
    assert_eq!(meta.thinking_iterations, 5);
    assert_eq!(meta.retry_count, 1);
    assert_eq!(meta.domain, Some("code".into()));
    assert_eq!(meta.constraint_tags, vec!["C-001"]);
}

#[test]
fn into_meta_state_tensions_defaults_to_empty_vec_when_none() {
    let (task_id, tenant_id) = make_ids();
    let mut cp = TaskReasoningCheckpoint::new_created(task_id, tenant_id, vec![], None);
    cp.shared_understanding = Some("understood".into());
    // tensions is None — should default to empty
    let meta = cp.into_meta_state().unwrap();
    assert!(meta.tensions.is_empty());
}

#[test]
fn into_meta_state_archetype_results_defaults_to_empty_when_none() {
    let (task_id, tenant_id) = make_ids();
    let mut cp = TaskReasoningCheckpoint::new_created(task_id, tenant_id, vec![], None);
    cp.shared_understanding = Some("understood".into());
    let meta = cp.into_meta_state().unwrap();
    assert!(meta.archetype_results.is_empty());
}

#[test]
fn into_meta_state_thinking_iterations_defaults_to_zero() {
    let (task_id, tenant_id) = make_ids();
    let mut cp = TaskReasoningCheckpoint::new_created(task_id, tenant_id, vec![], None);
    cp.shared_understanding = Some("understood".into());
    // thinking_iterations is None
    let meta = cp.into_meta_state().unwrap();
    assert_eq!(meta.thinking_iterations, 0);
}
