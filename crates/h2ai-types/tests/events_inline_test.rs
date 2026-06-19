use chrono::Utc;
use h2ai_types::events::*;
use h2ai_types::identity::TaskId;

// ── gap_f6_event_tests ────────────────────────────────────────────────────────

#[test]
fn awareness_probe_completed_event_round_trips() {
    let e = AwarenessProbeCompletedEvent {
        task_id: TaskId::new(),
        mode: "shadow".to_string(),
        degraded: false,
        n_items: 2,
        n_unjudged: 0,
        verdicts: vec![ProbeVerdictEntry {
            constraint_id: "C-1".into(),
            verdict: "ACKNOWLEDGED".into(),
            is_hard: true,
            gated: false,
            rationale: "plan mentions Lua".into(),
        }],
        re_iterated: false,
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&e).expect("serialize");
    let back: AwarenessProbeCompletedEvent = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.n_items, 2);
    assert_eq!(back.verdicts[0].constraint_id, "C-1");
    assert_eq!(back.mode, "shadow");

    let wrapped = H2AIEvent::AwarenessProbeCompleted(e);
    let json2 = serde_json::to_string(&wrapped).expect("serialize wrapped");
    let back2: H2AIEvent = serde_json::from_str(&json2).expect("deserialize wrapped");
    assert!(matches!(back2, H2AIEvent::AwarenessProbeCompleted(_)));
}

// ── gap_k1_event_tests ────────────────────────────────────────────────────────

#[test]
fn constraint_coherence_warning_round_trips() {
    let e = ConstraintCoherenceWarning {
        constraint_id: "C-1".into(),
        check_index: 0,
        consistency: 0.4,
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&e).unwrap();
    let back: ConstraintCoherenceWarning = serde_json::from_str(&json).unwrap();
    assert_eq!(back.check_index, 0);
}

#[test]
fn verifier_instability_event_round_trips() {
    let e = VerifierInstabilityEvent {
        task_id: TaskId::new(),
        constraint_id: "C-1".into(),
        instability_score: 0.034,
        wave_a: 1,
        wave_b: 2,
        divergent_reasons: vec!["reason A".into(), "reason B".into()],
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&e).unwrap();
    let back: VerifierInstabilityEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(back.divergent_reasons.len(), 2);
}

#[test]
fn constraint_version_created_round_trips() {
    let e = ConstraintVersionCreated {
        task_id: TaskId::new(),
        constraint_id: "C-1".into(),
        old_version: 1,
        new_version: 2,
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&e).unwrap();
    let back: ConstraintVersionCreated = serde_json::from_str(&json).unwrap();
    assert_eq!(back.old_version, 1);
    assert_eq!(back.new_version, 2);
}

#[test]
fn constraint_repair_attempted_round_trips() {
    let e = ConstraintRepairAttempted {
        task_id: TaskId::new(),
        constraint_id: "C-1".into(),
        check_index: 0,
        candidate_count: 3,
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&e).unwrap();
    let back: ConstraintRepairAttempted = serde_json::from_str(&json).unwrap();
    assert_eq!(back.candidate_count, 3);
}

#[test]
fn constraint_repair_failed_round_trips() {
    let e = ConstraintRepairFailed {
        task_id: TaskId::new(),
        constraint_id: "C-1".into(),
        check_index: 0,
        best_score: 0.42,
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&e).unwrap();
    let back: ConstraintRepairFailed = serde_json::from_str(&json).unwrap();
    assert_eq!(back.best_score, 0.42);
}

#[test]
fn complexity_probe_event_roundtrip() {
    let ev = ComplexityProbeEvent {
        task_id: TaskId::new(),
        complexity: 4,
        rationale: "formal proof".into(),
        decompose_recommended: true,
        probe_latency_ms: 250,
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&ev).unwrap();
    let back: ComplexityProbeEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(back.complexity, 4);
    assert!(back.decompose_recommended);
}

#[test]
fn verifier_reason_contradiction_beyond_budget_defaults() {
    // Existing events without the field must deserialise cleanly (backward compat).
    let json = r#"{"task_id":"00000000-0000-0000-0000-000000000001","wave":1,"constraint_id":"c1","reasons":[],"min_jaccard":0.1,"fallback_hint":null,"timestamp":"2026-01-01T00:00:00Z"}"#;
    let ev: VerifierReasonContradictionEvent = serde_json::from_str(json).unwrap();
    assert_eq!(ev.beyond_budget_count, 0);
}

#[test]
fn constraint_ambiguity_detected_event_roundtrip() {
    let evt = ConstraintAmbiguityDetectedEvent {
        task_id: TaskId::new(),
        constraint_id: "CONSTRAINT-005".into(),
        check_idx: Some(4),
        original_check_text: "Does the proposal use a dual-ledger model?".into(),
        suggested_rewrite: "Does the proposal use Redis as the sole charge-path ledger?".into(),
        evidence: vec!["term 'cockroachdb' negated in rubric guidance: FM-005-2".into()],
        final_score: 0.75,
        wave: 3,
        timestamp: Utc::now(),
    };
    let wrapped = H2AIEvent::ConstraintAmbiguityDetected(evt.clone());
    let json = serde_json::to_string(&wrapped).expect("serialize");
    let back: H2AIEvent = serde_json::from_str(&json).expect("deserialize");
    if let H2AIEvent::ConstraintAmbiguityDetected(inner) = back {
        assert_eq!(inner.constraint_id, evt.constraint_id);
        assert_eq!(inner.check_idx, Some(4));
        assert_eq!(inner.evidence.len(), 1);
    } else {
        panic!("wrong variant after roundtrip");
    }
}

// ── tiered_exit_event_tests ───────────────────────────────────────────────────

#[test]
fn tiered_exit_event_serializes() {
    let evt = TieredExitEvent {
        wave: 1,
        n: 3,
        k_required: 2,
        k_accepted: 3,
        acceptance_score: 0.85,
    };
    let json = serde_json::to_string(&evt).expect("serialize");
    assert!(json.contains("\"wave\":1"));
    assert!(json.contains("\"n\":3"));
    assert!(json.contains("\"k_required\":2"));
    assert!(json.contains("\"k_accepted\":3"));
}

#[test]
fn h2ai_event_tee_roundtrip() {
    let evt = TieredExitEvent {
        wave: 0,
        n: 1,
        k_required: 1,
        k_accepted: 1,
        acceptance_score: 0.9,
    };
    let wrapped = H2AIEvent::TieredExit(evt.clone());
    let json = serde_json::to_string(&wrapped).expect("serialize");
    let back: H2AIEvent = serde_json::from_str(&json).expect("deserialize");
    if let H2AIEvent::TieredExit(inner) = back {
        assert_eq!(inner.wave, 0);
        assert_eq!(inner.n, 1);
    } else {
        panic!("wrong variant");
    }
}

// ── cost_guard_event_tests ────────────────────────────────────────────────────

#[test]
fn cost_threshold_warning_event_serializes() {
    let evt = CostThresholdWarningEvent {
        task_id: TaskId::new(),
        tokens_used: 8_000,
        budget_tokens: 10_000,
        fraction_used: 0.80,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&evt).unwrap();
    let back: CostThresholdWarningEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.tokens_used, 8_000);
    assert!((back.fraction_used - 0.80).abs() < 1e-9);
}

#[test]
fn budget_exhausted_event_serializes() {
    let evt = BudgetExhaustedEvent {
        task_id: TaskId::new(),
        tokens_used: 10_500,
        budget_tokens: 10_000,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&evt).unwrap();
    let back: BudgetExhaustedEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.tokens_used, 10_500);
}

#[test]
fn convergence_gate_event_serializes() {
    let evt = ConvergenceGateEvent {
        task_id: TaskId::new(),
        wave: 1,
        n_live: 2,
        convergence_fraction: 1.0,
        theta_converge: 0.87,
        best_score: 0.83,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&evt).unwrap();
    let back: ConvergenceGateEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.wave, 1);
    assert!((back.convergence_fraction - 1.0).abs() < 1e-9);
}

#[test]
fn h2ai_event_wraps_cost_events() {
    let warn = CostThresholdWarningEvent {
        task_id: TaskId::new(),
        tokens_used: 8_000,
        budget_tokens: 10_000,
        fraction_used: 0.80,
        timestamp: Utc::now(),
    };
    let wrapped = H2AIEvent::CostThresholdWarning(warn);
    let s = serde_json::to_string(&wrapped).unwrap();
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    if let H2AIEvent::CostThresholdWarning(inner) = back {
        assert_eq!(inner.tokens_used, 8_000);
    } else {
        panic!("wrong variant");
    }
}

// ── cost_propagation_tests ────────────────────────────────────────────────────

#[test]
fn task_attribution_event_includes_tokens_used() {
    let json = r#"{
        "task_id": "00000000-0000-0000-0000-000000000001",
        "q_confidence": 0.85,
        "prediction_basis": "Heuristic",
        "waste_ratio": 1.0,
        "timestamp": "2026-06-06T00:00:00Z",
        "tokens_used": 4200
    }"#;
    let ev: TaskAttributionEvent = serde_json::from_str(json).unwrap();
    assert_eq!(ev.tokens_used, 4200);
}

#[test]
fn task_attribution_event_tokens_used_defaults_to_zero_for_old_events() {
    let json = r#"{
        "task_id": "00000000-0000-0000-0000-000000000002",
        "q_confidence": 0.75,
        "prediction_basis": "Heuristic",
        "waste_ratio": 1.0,
        "timestamp": "2026-06-06T00:00:00Z"
    }"#;
    let ev: TaskAttributionEvent = serde_json::from_str(json).unwrap();
    assert_eq!(ev.tokens_used, 0);
}

#[test]
fn task_attribution_event_backwards_compat_without_skill_nodes_injected() {
    // Minimal JSON that existed before skill_nodes_injected was added
    let json = r#"{
        "task_id": "00000000-0000-0000-0000-000000000001",
        "q_confidence": 0.8,
        "prediction_basis": "Heuristic",
        "waste_ratio": 1.0,
        "tokens_used": 0,
        "timestamp": "2026-01-01T00:00:00Z",
        "calibration_source": "Measured"
    }"#;
    let ev: TaskAttributionEvent = serde_json::from_str(json).unwrap();
    assert_eq!(
        ev.skill_nodes_injected, 0,
        "skill_nodes_injected must default to 0 for old events"
    );
}

#[test]
fn generation_knowledge_event_roundtrip() {
    let ev = GenerationKnowledgeEvent {
        task_id: h2ai_types::identity::TaskId::new(),
        knowledge_injected: true,
        skill_nodes_count: 3,
        q_confidence: 0.82,
        timestamp: Utc::now(),
    };
    let json = serde_json::to_string(&ev).unwrap();
    let restored: GenerationKnowledgeEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.skill_nodes_count, 3);
    assert!(restored.knowledge_injected);
}

// ── pipeline resilience: frozen verifier bypass tests ────────────────────────

#[test]
fn verifier_frozen_event_roundtrip() {
    let e = VerifierFrozenEvent {
        constraint_id: "C-007".to_string(),
        frozen_since_wave: 2,
        per_wave_scores: vec![0.0, 0.0, 0.0],
        sample_reason: "verifier returned identical rejection reason for 3 waves".to_string(),
        bypassed: true,
    };
    let json = serde_json::to_string(&e).expect("serialize");
    let back: VerifierFrozenEvent = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.constraint_id, "C-007");
    assert_eq!(back.frozen_since_wave, 2);
    assert_eq!(back.per_wave_scores.len(), 3);
    assert!(back.bypassed);
}

#[test]
fn verifier_frozen_event_wrapped_in_h2ai_event() {
    let e = VerifierFrozenEvent {
        constraint_id: "C-007".to_string(),
        frozen_since_wave: 1,
        per_wave_scores: vec![0.0, 0.0],
        sample_reason: "frozen".to_string(),
        bypassed: false,
    };
    let wrapped = H2AIEvent::VerifierFrozen(e);
    let json = serde_json::to_string(&wrapped).expect("serialize");
    let back: H2AIEvent = serde_json::from_str(&json).expect("deserialize");
    assert!(matches!(back, H2AIEvent::VerifierFrozen(_)));
}

#[test]
fn branch_pruned_event_bypass_reason_defaults_to_none() {
    use h2ai_types::sizing::RoleErrorCost;
    let e = BranchPrunedEvent {
        task_id: TaskId::new(),
        explorer_id: h2ai_types::identity::ExplorerId::new(),
        reason: "verifier score below threshold".to_string(),
        raw_output: "some output".to_string(),
        constraint_error_cost: RoleErrorCost::new(0.5).unwrap(),
        violated_constraints: vec![],
        timestamp: Utc::now(),
        retry_count: 0,
        bypass_reason: None,
    };
    assert!(e.bypass_reason.is_none());
    let json = serde_json::to_string(&e).expect("serialize");
    let back: BranchPrunedEvent = serde_json::from_str(&json).expect("deserialize");
    assert!(back.bypass_reason.is_none());
}

#[test]
fn branch_pruned_event_bypass_reason_old_json_defaults() {
    // Old JSON without bypass_reason must still parse
    let json = r#"{"task_id":"00000000-0000-0000-0000-000000000001","explorer_id":"00000000-0000-0000-0000-000000000002","reason":"pruned","raw_output":"","constraint_error_cost":0.0,"violated_constraints":[],"timestamp":"2026-01-01T00:00:00Z","retry_count":0}"#;
    let e: BranchPrunedEvent = serde_json::from_str(json).expect("deserialize old json");
    assert!(e.bypass_reason.is_none());
}

#[test]
fn coherence_incomplete_event_bypassed_verifier_constraint_ids_defaults_to_empty() {
    let e = CoherenceIncompleteEvent {
        task_id: TaskId::new(),
        uncovered_domains: vec!["billing".to_string()],
        active_contradictions: vec![],
        retries: 2,
        timestamp: Utc::now(),
        bypassed_verifier_constraint_ids: vec![],
    };
    assert!(e.bypassed_verifier_constraint_ids.is_empty());
    let json = serde_json::to_string(&e).expect("serialize");
    let back: CoherenceIncompleteEvent = serde_json::from_str(&json).expect("deserialize");
    assert!(back.bypassed_verifier_constraint_ids.is_empty());
}

#[test]
fn coherence_incomplete_event_old_json_defaults_bypassed_ids() {
    let json = r#"{"task_id":"00000000-0000-0000-0000-000000000001","uncovered_domains":["billing"],"active_contradictions":[],"retries":1,"timestamp":"2026-01-01T00:00:00Z"}"#;
    let e: CoherenceIncompleteEvent = serde_json::from_str(json).expect("deserialize old json");
    assert!(e.bypassed_verifier_constraint_ids.is_empty());
}

#[test]
fn constraint_violation_check_reasons_defaults_to_none() {
    let cv = ConstraintViolation {
        constraint_id: "C-001".to_string(),
        score: 0.0,
        severity_label: "Hard".to_string(),
        remediation_hint: None,
        constraint_description: String::new(),
        verifier_reason: None,
        check_verdicts: vec![],
        criteria_pass: None,
        check_reasons: None,
    };
    assert!(cv.check_reasons.is_none());
    let json = serde_json::to_string(&cv).expect("serialize");
    let back: ConstraintViolation = serde_json::from_str(&json).expect("deserialize");
    assert!(back.check_reasons.is_none());
}

#[test]
fn constraint_violation_check_reasons_old_json_defaults() {
    let json =
        r#"{"constraint_id":"C-001","score":0.0,"severity_label":"Hard","remediation_hint":null}"#;
    let cv: ConstraintViolation = serde_json::from_str(json).expect("deserialize old json");
    assert!(cv.check_reasons.is_none());
}

// ── GAP-D3: TerminalCause and TaskFailedEvent diagnostic fields ───────────────

#[test]
fn terminal_cause_serializes_to_variant_name() {
    let cause = TerminalCause::VerificationExhaustion;
    let s = serde_json::to_string(&cause).unwrap();
    assert_eq!(s, r#""VerificationExhaustion""#);
}

#[test]
fn terminal_cause_severity_ordering() {
    // LlmAdapterUnavailable (rank 0) < VerificationExhaustion (rank 1) in severity
    // The severity_rank() method returns lower number = higher severity
    assert!(
        TerminalCause::LlmAdapterUnavailable.severity_rank()
            < TerminalCause::VerificationExhaustion.severity_rank()
    );
    assert!(
        TerminalCause::VerificationExhaustion.severity_rank()
            < TerminalCause::Timeout.severity_rank()
    );
    assert!(TerminalCause::Timeout.severity_rank() < TerminalCause::Unknown.severity_rank());
}

#[test]
fn task_failed_event_has_required_diagnostic_fields() {
    let event = TaskFailedEvent {
        task_id: TaskId::new(),
        primary_cause: TerminalCause::VerificationExhaustion,
        contributing_causes: vec![TerminalCause::Timeout],
        top_violated_constraints: vec![("C-005".to_string(), 4), ("C-004".to_string(), 2)],
        last_selection_valid_count: Some(0),
        pruned_events: vec![],
        topologies_tried: vec![],
        tau_values_tried: vec![],
        multiplication_condition_failure: None,
        timestamp: Utc::now(),
    };
    assert_eq!(event.primary_cause, TerminalCause::VerificationExhaustion);
    assert_eq!(event.top_violated_constraints.len(), 2);
    assert_eq!(event.last_selection_valid_count, Some(0));
}

#[test]
fn task_failed_event_old_json_defaults_new_fields() {
    let old_json = r#"{
        "task_id": "00000000-0000-0000-0000-000000000001",
        "pruned_events": [],
        "topologies_tried": [],
        "tau_values_tried": [],
        "timestamp": "2026-01-01T00:00:00Z"
    }"#;
    let event: TaskFailedEvent = serde_json::from_str(old_json).expect("old json must deserialize");
    assert_eq!(event.primary_cause, TerminalCause::Unknown);
    assert!(event.contributing_causes.is_empty());
    assert!(event.top_violated_constraints.is_empty());
    assert!(event.last_selection_valid_count.is_none());
}
