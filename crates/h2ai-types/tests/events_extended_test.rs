/// Extended coverage tests for `h2ai_types::events` — complements the existing
/// `events_test.rs` file which already covers several structs.
use chrono::Utc;
use h2ai_types::events::*;
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::{PredictionBasis, ProbeSkipReason};

fn task_id() -> TaskId {
    TaskId::new()
}

fn explorer_id() -> ExplorerId {
    ExplorerId::new()
}

// ── Default impls ─────────────────────────────────────────────────────────────

#[test]
fn calibration_quality_default_is_domain() {
    assert_eq!(CalibrationQuality::default(), CalibrationQuality::Domain);
}

#[test]
fn cg_mode_default_is_constraint_profile() {
    assert_eq!(CgMode::default(), CgMode::ConstraintProfile);
}

#[test]
fn calibration_source_default_is_measured() {
    assert_eq!(CalibrationSource::default(), CalibrationSource::Measured);
}

#[test]
fn grounding_source_default_is_llm_researcher() {
    assert_eq!(GroundingSource::default(), GroundingSource::LlmResearcher);
}

// ── CalibrationQuality ────────────────────────────────────────────────────────

#[test]
fn calibration_quality_serde_roundtrip() {
    for v in [CalibrationQuality::Domain, CalibrationQuality::Bootstrap] {
        let s = serde_json::to_string(&v).unwrap();
        let back: CalibrationQuality = serde_json::from_str(&s).unwrap();
        assert_eq!(back, v);
    }
}

// ── CgMode ───────────────────────────────────────────────────────────────────

#[test]
fn cg_mode_serde_roundtrip() {
    for v in [CgMode::ConstraintProfile, CgMode::EmbeddingCosine] {
        let s = serde_json::to_string(&v).unwrap();
        let back: CgMode = serde_json::from_str(&s).unwrap();
        assert_eq!(back, v);
    }
}

// ── CalibrationSource ─────────────────────────────────────────────────────────

#[test]
fn calibration_source_serde_roundtrip_all_variants() {
    for v in [
        CalibrationSource::Measured,
        CalibrationSource::PartialFit,
        CalibrationSource::SyntheticPriors,
    ] {
        let s = serde_json::to_string(&v).unwrap();
        let back: CalibrationSource = serde_json::from_str(&s).unwrap();
        assert_eq!(back, v);
    }
}

// ── FailureMode ───────────────────────────────────────────────────────────────

#[test]
fn failure_mode_constrained_exploration_serde_roundtrip() {
    let v = FailureMode::ConstrainedExploration;
    let s = serde_json::to_string(&v).unwrap();
    let back: FailureMode = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, FailureMode::ConstrainedExploration));
}

#[test]
fn failure_mode_mode_collapse_serde_roundtrip() {
    let v = FailureMode::ModeCollapse;
    let s = serde_json::to_string(&v).unwrap();
    let back: FailureMode = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, FailureMode::ModeCollapse));
}

#[test]
fn failure_mode_correlated_hallucination_serde_roundtrip() {
    let v = FailureMode::CorrelatedHallucination {
        cv: 0.15,
        mean_jaccard_distance: 0.72,
    };
    let s = serde_json::to_string(&v).unwrap();
    let back: FailureMode = serde_json::from_str(&s).unwrap();
    match back {
        FailureMode::CorrelatedHallucination {
            cv,
            mean_jaccard_distance,
        } => {
            assert!((cv - 0.15).abs() < 1e-9);
            assert!((mean_jaccard_distance - 0.72).abs() < 1e-9);
        }
        other => panic!("wrong variant: {other:?}"),
    }
}

// ── GroundingSource ───────────────────────────────────────────────────────────

#[test]
fn grounding_source_serde_roundtrip_all_variants() {
    for v in [
        GroundingSource::SpecAnchor,
        GroundingSource::LlmResearcher,
        GroundingSource::WebSearch,
    ] {
        let s = serde_json::to_string(&v).unwrap();
        let back: GroundingSource = serde_json::from_str(&s).unwrap();
        assert_eq!(back, v);
    }
}

// ── EpistemicYieldEvent ───────────────────────────────────────────────────────

#[test]
fn epistemic_yield_event_serde_roundtrip() {
    let e = EpistemicYieldEvent {
        task_id: task_id(),
        n_eff_cosine_actual: 2.3,
        n_eff_prior: 2.0,
        yield_ratio: 0.77,
        adapters: vec!["adapter-a".into(), "adapter-b".into()],
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: EpistemicYieldEvent = serde_json::from_str(&s).unwrap();
    assert!((back.yield_ratio - 0.77).abs() < 1e-9);
    assert_eq!(back.adapters.len(), 2);
}

// ── TaskSnapshot ─────────────────────────────────────────────────────────────

#[test]
fn task_snapshot_serde_roundtrip() {
    let e = TaskSnapshot {
        task_id: task_id(),
        last_sequence: 42,
        task_state_json: r#"{"status":"running"}"#.into(),
        taken_at: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: TaskSnapshot = serde_json::from_str(&s).unwrap();
    assert_eq!(back.last_sequence, 42);
    assert!(back.task_state_json.contains("running"));
}

// ── ProposalFailureReason ─────────────────────────────────────────────────────

#[test]
fn proposal_failure_reason_serde_roundtrip_all_variants() {
    let cases: Vec<ProposalFailureReason> = vec![
        ProposalFailureReason::Timeout,
        ProposalFailureReason::OomPanic("SIGKILL".into()),
        ProposalFailureReason::AdapterError("connection reset".into()),
    ];
    for v in cases {
        let s = serde_json::to_string(&v).unwrap();
        let back: ProposalFailureReason = serde_json::from_str(&s).unwrap();
        // Verify the round-trip produces the same JSON
        let s2 = serde_json::to_string(&back).unwrap();
        assert_eq!(s, s2);
    }
}

// ── ProposalEvent ─────────────────────────────────────────────────────────────

#[test]
fn proposal_event_generation_defaults_to_zero() {
    let json = format!(
        r#"{{"task_id":"{}", "explorer_id":"{}", "tau":0.5,
            "raw_output":"out","token_cost":10,
            "adapter_kind":{{"CloudGeneric":{{"endpoint":"http://x","api_key_env":"K","model":null}}}},
            "timestamp":"2026-01-01T00:00:00Z"}}"#,
        task_id(),
        explorer_id()
    );
    let e: ProposalEvent = serde_json::from_str(&json).unwrap();
    assert_eq!(e.generation, 0);
}

// ── ProposalFailedEvent ───────────────────────────────────────────────────────

#[test]
fn proposal_failed_event_serde_roundtrip() {
    let e = ProposalFailedEvent {
        task_id: task_id(),
        explorer_id: explorer_id(),
        reason: ProposalFailureReason::Timeout,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: ProposalFailedEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back.reason, ProposalFailureReason::Timeout));
}

// ── GenerationPhaseCompletedEvent ─────────────────────────────────────────────

#[test]
fn generation_phase_completed_event_serde_roundtrip() {
    let e = GenerationPhaseCompletedEvent {
        task_id: task_id(),
        total_explorers: 4,
        successful: 3,
        failed: 1,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: GenerationPhaseCompletedEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.total_explorers, 4);
    assert_eq!(back.successful, 3);
    assert_eq!(back.failed, 1);
}

// ── ValidationEvent ───────────────────────────────────────────────────────────

#[test]
fn validation_event_serde_roundtrip() {
    let eid = explorer_id();
    let e = ValidationEvent {
        task_id: task_id(),
        explorer_id: eid.clone(),
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: ValidationEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.explorer_id, eid);
}

// ── ConstraintViolation ───────────────────────────────────────────────────────

#[test]
#[allow(clippy::float_cmp)]
fn constraint_violation_serde_roundtrip() {
    let v = ConstraintViolation {
        constraint_id: "ADR-004".into(),
        score: 0.0,
        severity_label: "Hard".into(),
        remediation_hint: Some("use stateless auth".into()),
        constraint_description: String::new(),
        verifier_reason: None,
        check_verdicts: vec![],
        criteria_pass: None,
        check_reasons: None,
    };
    let s = serde_json::to_string(&v).unwrap();
    let back: ConstraintViolation = serde_json::from_str(&s).unwrap();
    assert_eq!(back.constraint_id, "ADR-004");
    assert_eq!(back.score, 0.0);
    assert_eq!(back.severity_label, "Hard");
    assert!(back.remediation_hint.is_some());
}

// ── ZeroSurvivalEvent ─────────────────────────────────────────────────────────

#[test]
fn zero_survival_event_with_failure_mode_serde_roundtrip() {
    let e = ZeroSurvivalEvent {
        task_id: task_id(),
        retry_count: 1,
        timestamp: Utc::now(),
        n_eff_cosine_actual: Some(1.1),
        failure_mode: Some(FailureMode::ModeCollapse),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: ZeroSurvivalEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.retry_count, 1);
    assert!(back.n_eff_cosine_actual.is_some());
    assert!(matches!(back.failure_mode, Some(FailureMode::ModeCollapse)));
}

// ── ZeroCoordinationQualityEvent ──────────────────────────────────────────────

#[test]
fn zero_coordination_quality_event_serde_roundtrip() {
    let e = ZeroCoordinationQualityEvent {
        task_id: task_id(),
        cg_embed: 0.05,
        forced_n_max: 1,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: ZeroCoordinationQualityEvent = serde_json::from_str(&s).unwrap();
    assert!((back.cg_embed - 0.05).abs() < 1e-9);
    assert_eq!(back.forced_n_max, 1);
}

// ── MergeResolvedEvent ────────────────────────────────────────────────────────

#[test]
fn merge_resolved_event_serde_roundtrip_with_optional_fields() {
    let e = MergeResolvedEvent {
        task_id: task_id(),
        resolved_output: "final output".into(),
        j_eff: Some(0.87),
        timestamp: Utc::now(),
        oracle_gate_passed: Some(true),
        zone3_hints: Some("hint text".into()),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: MergeResolvedEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.resolved_output, "final output");
    assert!((back.j_eff.unwrap() - 0.87).abs() < 1e-9);
    assert_eq!(back.oracle_gate_passed, Some(true));
    assert!(back.zone3_hints.is_some());
}

#[test]
fn merge_resolved_event_oracle_gate_absent_when_none() {
    let e = MergeResolvedEvent {
        task_id: task_id(),
        resolved_output: "out".into(),
        j_eff: None,
        timestamp: Utc::now(),
        oracle_gate_passed: None,
        zone3_hints: None,
    };
    let s = serde_json::to_string(&e).unwrap();
    // oracle_gate_passed uses skip_serializing_if = "Option::is_none"
    assert!(
        !s.contains("oracle_gate_passed"),
        "should be absent in JSON"
    );
}

// ── SelectionResolvedEvent ────────────────────────────────────────────────────

#[test]
fn selection_resolved_event_serde_roundtrip_with_elapsed() {
    let e = SelectionResolvedEvent {
        task_id: task_id(),
        valid_proposals: vec![explorer_id()],
        pruned_proposals: vec![],
        merge_strategy: h2ai_types::sizing::MergeStrategy::ScoreOrdered,
        timestamp: Utc::now(),
        merge_elapsed_secs: Some(0.042),
        n_input_proposals: 3,
        n_failed_proposals: 1,
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: SelectionResolvedEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.valid_proposals.len(), 1);
    assert!((back.merge_elapsed_secs.unwrap() - 0.042).abs() < 1e-9);
    assert_eq!(back.n_input_proposals, 3);
    assert_eq!(back.n_failed_proposals, 1);
}

// ── CoherenceIncompleteEvent (already in events_test.rs — skip duplicate) ─────

// ── VerifierComparisonEvent (already in events_test.rs — skip duplicate) ──────

// ── ShadowAuditorResultEvent (already in events_test.rs — skip duplicate) ─────

// ── CorrelatedEnsembleWarning ─────────────────────────────────────────────────

#[test]
fn correlated_ensemble_warning_serde_roundtrip() {
    let e = CorrelatedEnsembleWarning {
        task_id: task_id(),
        cv: 0.08,
        mean_jaccard_distance: 0.12,
        retry_count: 2,
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: CorrelatedEnsembleWarning = serde_json::from_str(&s).unwrap();
    assert!((back.cv - 0.08).abs() < 1e-9);
    assert!((back.mean_jaccard_distance - 0.12).abs() < 1e-9);
    assert_eq!(back.retry_count, 2);
}

// ── CorrelatedFabricationEvent ────────────────────────────────────────────────

#[test]
fn correlated_fabrication_event_serde_roundtrip() {
    let e = CorrelatedFabricationEvent {
        task_id: task_id(),
        cfi: 0.75,
        injection_pressure: 0.52,
        shared_ungrounded_entities: vec!["ServiceMesh".into(), "ApiGateway".into()],
        proposal_count: 3,
        hint_injected: true,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: CorrelatedFabricationEvent = serde_json::from_str(&s).unwrap();
    assert!((back.cfi - 0.75).abs() < 1e-9);
    assert!(back.hint_injected);
    assert_eq!(back.shared_ungrounded_entities.len(), 2);
}

// ── ResearcherGroundingEvent ──────────────────────────────────────────────────

#[test]
fn researcher_grounding_event_serde_roundtrip_with_slot() {
    let e = ResearcherGroundingEvent {
        task_id: task_id(),
        shared_assumption: "load balancer is required".into(),
        literature_summary: "research shows X".into(),
        slot: Some("slot_1".into()),
        source: GroundingSource::SpecAnchor,
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: ResearcherGroundingEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.slot, Some("slot_1".into()));
    assert_eq!(back.source, GroundingSource::SpecAnchor);
}

#[test]
fn researcher_grounding_event_default_source_when_absent() {
    let json = r#"{
        "task_id": "00000000-0000-0000-0000-000000000000",
        "shared_assumption": "assumption",
        "literature_summary": "summary",
        "slot": null
    }"#;
    let e: ResearcherGroundingEvent = serde_json::from_str(json).unwrap();
    assert_eq!(e.source, GroundingSource::LlmResearcher);
}

// ── DiversityGuardDegradedEvent ───────────────────────────────────────────────

#[test]
fn diversity_guard_degraded_event_serde_roundtrip() {
    let e = DiversityGuardDegradedEvent {
        task_id: task_id(),
        reason: "coverage 0.25 < threshold 0.40".into(),
        coverage_score: 0.25,
        slot_domains: vec!["security".into(), "reliability".into()],
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: DiversityGuardDegradedEvent = serde_json::from_str(&s).unwrap();
    assert!((back.coverage_score - 0.25).abs() < 1e-9);
    assert_eq!(back.slot_domains.len(), 2);
}

// ── TaoIterationEvent ─────────────────────────────────────────────────────────

#[test]
fn tao_iteration_event_serde_roundtrip() {
    let e = TaoIterationEvent {
        task_id: task_id(),
        explorer_id: explorer_id(),
        turn: 2,
        observation: "pattern not matched on turn 2; retrying".into(),
        passed: false,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: TaoIterationEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.turn, 2);
    assert!(!back.passed);
}

// ── VerificationScoredEvent ───────────────────────────────────────────────────

#[test]
fn verification_scored_event_serde_roundtrip_cache_hit_default() {
    let e = VerificationScoredEvent {
        task_id: task_id(),
        explorer_id: explorer_id(),
        score: 0.95,
        reason: "all constraints satisfied".into(),
        passed: true,
        cache_hit: false,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: VerificationScoredEvent = serde_json::from_str(&s).unwrap();
    assert!((back.score - 0.95).abs() < 1e-9);
    assert!(back.passed);
    assert!(!back.cache_hit);
}

#[test]
fn verification_scored_event_cache_hit_defaults_false_when_absent() {
    let json = format!(
        r#"{{"task_id":"{}","explorer_id":"{}","score":0.8,"reason":"ok","passed":true,"timestamp":"2026-01-01T00:00:00Z"}}"#,
        task_id(),
        explorer_id()
    );
    let e: VerificationScoredEvent = serde_json::from_str(&json).unwrap();
    assert!(!e.cache_hit);
}

// ── SubtaskPlanCreatedEvent ───────────────────────────────────────────────────

#[test]
fn subtask_plan_created_event_serde_roundtrip() {
    let e = SubtaskPlanCreatedEvent {
        task_id: task_id(),
        plan_id: task_id(),
        subtask_count: 5,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: SubtaskPlanCreatedEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.subtask_count, 5);
}

// ── SubtaskPlanReviewedEvent ──────────────────────────────────────────────────

#[test]
fn subtask_plan_reviewed_event_approved_serde_roundtrip() {
    let e = SubtaskPlanReviewedEvent {
        task_id: task_id(),
        plan_id: task_id(),
        approved: true,
        reason: "plan looks good".into(),
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: SubtaskPlanReviewedEvent = serde_json::from_str(&s).unwrap();
    assert!(back.approved);
    assert_eq!(back.reason, "plan looks good");
}

// ── SubtaskStartedEvent ───────────────────────────────────────────────────────

#[test]
fn subtask_started_event_serde_roundtrip() {
    use h2ai_types::identity::SubtaskId;
    let e = SubtaskStartedEvent {
        task_id: task_id(),
        plan_id: task_id(),
        subtask_id: SubtaskId::new(),
        description: "implement auth module".into(),
        wave: 0,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: SubtaskStartedEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.description, "implement auth module");
    assert_eq!(back.wave, 0);
}

// ── SubtaskCompletedEvent ─────────────────────────────────────────────────────

#[test]
fn subtask_completed_event_serde_roundtrip() {
    use h2ai_types::identity::SubtaskId;
    let e = SubtaskCompletedEvent {
        task_id: task_id(),
        plan_id: task_id(),
        subtask_id: SubtaskId::new(),
        token_cost: 300,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: SubtaskCompletedEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.token_cost, 300);
}

// ── OptimizationKind ──────────────────────────────────────────────────────────

#[test]
fn optimization_kind_serde_roundtrip() {
    for v in [
        OptimizationKind::TauSpreadAdjusted,
        OptimizationKind::TopologyHintSet,
    ] {
        let s = serde_json::to_string(&v).unwrap();
        let back: OptimizationKind = serde_json::from_str(&s).unwrap();
        assert_eq!(back, v);
    }
}

// ── AppliedOptimization ───────────────────────────────────────────────────────

#[test]
fn applied_optimization_serde_roundtrip() {
    let a = AppliedOptimization {
        kind: OptimizationKind::TauSpreadAdjusted,
        reason: "waste ratio above threshold".into(),
        before: "tau_spread=0.3".into(),
        after: "tau_spread=0.5".into(),
    };
    let s = serde_json::to_string(&a).unwrap();
    let back: AppliedOptimization = serde_json::from_str(&s).unwrap();
    assert_eq!(back.kind, OptimizationKind::TauSpreadAdjusted);
    assert_eq!(back.before, "tau_spread=0.3");
}

// ── TaskAttributionEvent ──────────────────────────────────────────────────────

#[test]
fn task_attribution_event_waste_ratio_defaults_to_one() {
    // waste_ratio uses a custom default fn — omitting it must yield 1.0
    let json = format!(
        r#"{{"task_id":"{}","q_confidence":0.9,"prediction_basis":"Heuristic","timestamp":"2026-01-01T00:00:00Z"}}"#,
        task_id()
    );
    let e: TaskAttributionEvent = serde_json::from_str(&json).unwrap();
    assert!((e.waste_ratio - 1.0).abs() < 1e-9);
}

#[test]
fn task_attribution_event_full_serde_roundtrip() {
    let e = TaskAttributionEvent {
        task_id: task_id(),
        q_confidence: 0.82,
        q_measured: Some(0.90),
        q_interval_lo: Some(0.78),
        q_interval_hi: Some(0.86),
        prediction_basis: PredictionBasis::Heuristic,
        waste_ratio: 0.75,
        applied_optimizations: vec![AppliedOptimization {
            kind: OptimizationKind::TopologyHintSet,
            reason: "waste".into(),
            before: "Ensemble".into(),
            after: "Solo".into(),
        }],
        tokens_used: 0,
        skill_nodes_injected: 0,
        timestamp: Utc::now(),
        approval_decision: None,
        calibration_source: CalibrationSource::Measured,
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: TaskAttributionEvent = serde_json::from_str(&s).unwrap();
    assert!((back.q_confidence - 0.82).abs() < 1e-9);
    assert!((back.waste_ratio - 0.75).abs() < 1e-9);
    assert_eq!(back.applied_optimizations.len(), 1);
    assert_eq!(back.calibration_source, CalibrationSource::Measured);
}

// ── TaskComplexityAssessedEvent ───────────────────────────────────────────────

#[test]
fn task_complexity_assessed_event_serde_roundtrip() {
    let e = TaskComplexityAssessedEvent {
        task_id: task_id(),
        tcc_structural: 2.1,
        tcc_empirical: Some(2.4),
        tcc_effective: 2.7,
        n_eff_pool: Some(3.1),
        task_quadrant: h2ai_types::sizing::TaskQuadrant::Coverage,
        probe_skipped: false,
        probe_skip_reason: ProbeSkipReason::default(),
        heavy_fraction: 0.15,
        tcc_mismatch: true,
        probe_cost_tokens: 150,
        n_informative_static: 4,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: TaskComplexityAssessedEvent = serde_json::from_str(&s).unwrap();
    assert!((back.tcc_structural - 2.1).abs() < 1e-9);
    assert!(back.tcc_mismatch);
    assert_eq!(back.probe_cost_tokens, 150);
    assert_eq!(back.n_informative_static, 4);
}

// ── ConstraintFrontierEvent ───────────────────────────────────────────────────

#[test]
fn constraint_frontier_event_serde_roundtrip() {
    let e = ConstraintFrontierEvent {
        task_id: task_id(),
        satisfaction_matrix: vec![vec![0.9, 0.5], vec![0.1, 1.0]],
        constraint_ids: vec!["c1".into(), "c2".into()],
        explorer_ids: vec![explorer_id(), explorer_id()],
        pareto_coverage: 0.88,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: ConstraintFrontierEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.satisfaction_matrix.len(), 2);
    assert_eq!(back.constraint_ids.len(), 2);
    assert!((back.pareto_coverage - 0.88).abs() < 1e-9);
}

// ── OracleGateResultEvent ─────────────────────────────────────────────────────

#[test]
fn oracle_gate_result_event_serde_roundtrip() {
    let e = OracleGateResultEvent {
        task_id: "task-123".into(),
        gate_passed: true,
        confidence: 0.93,
        summary: "all test cases passed".into(),
        checked_proposals: 3,
        passed_proposals: 3,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: OracleGateResultEvent = serde_json::from_str(&s).unwrap();
    assert!(back.gate_passed);
    assert_eq!(back.passed_proposals, 3);
    assert_eq!(back.task_id, "task-123");
}

// ── PendingClarificationEvent ─────────────────────────────────────────────────

#[test]
fn pending_clarification_event_serde_roundtrip() {
    let e = PendingClarificationEvent {
        task_id: "task-abc".into(),
        question: "Should we use gRPC or REST?".into(),
        context: "The ADR requires low-latency communication.".into(),
        timeout_secs: 3600,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: PendingClarificationEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.question, "Should we use gRPC or REST?");
    assert_eq!(back.timeout_secs, 3600);
}

// ── ApprovalRiskLevel ─────────────────────────────────────────────────────────

#[test]
fn approval_risk_level_serde_roundtrip_all_variants() {
    for v in [
        ApprovalRiskLevel::Low,
        ApprovalRiskLevel::Medium,
        ApprovalRiskLevel::High,
    ] {
        let s = serde_json::to_string(&v).unwrap();
        let back: ApprovalRiskLevel = serde_json::from_str(&s).unwrap();
        assert_eq!(back, v);
    }
}

// ── ApprovalTrigger ───────────────────────────────────────────────────────────

#[test]
fn approval_trigger_serde_roundtrip_all_variants() {
    for v in [
        ApprovalTrigger::ManifestFlag,
        ApprovalTrigger::LowConfidence,
    ] {
        let s = serde_json::to_string(&v).unwrap();
        let back: ApprovalTrigger = serde_json::from_str(&s).unwrap();
        assert_eq!(back, v);
    }
}

// ── PendingApprovalEvent ──────────────────────────────────────────────────────

#[test]
fn pending_approval_event_serde_roundtrip() {
    let e = PendingApprovalEvent {
        task_id: task_id(),
        proposed_output: "final answer".into(),
        q_confidence: 0.55,
        prediction_basis: 0,
        n_used: 2,
        risk_level: ApprovalRiskLevel::High,
        triggered_by: ApprovalTrigger::LowConfidence,
        timeout_at_ms: 9_999_999,
        timestamp_ms: 1_000_000,
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: PendingApprovalEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.risk_level, ApprovalRiskLevel::High);
    assert_eq!(back.triggered_by, ApprovalTrigger::LowConfidence);
    assert!((back.q_confidence - 0.55).abs() < 1e-9);
}

// ── ApprovalResolvedEvent ─────────────────────────────────────────────────────

#[test]
fn approval_resolved_event_serde_roundtrip() {
    let e = ApprovalResolvedEvent {
        task_id: task_id(),
        approved: true,
        operator_id: "op-1".into(),
        reviewer_note: Some("looks good to me".into()),
        decided_at_ms: 2_000_000,
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: ApprovalResolvedEvent = serde_json::from_str(&s).unwrap();
    assert!(back.approved);
    assert_eq!(back.operator_id, "op-1");
    assert_eq!(back.reviewer_note.as_deref(), Some("looks good to me"));
}

// ── ThinkingLoopCompletedEvent ────────────────────────────────────────────────

#[test]
fn thinking_loop_completed_event_serde_roundtrip() {
    let e = ThinkingLoopCompletedEvent {
        task_id: task_id(),
        enabled: true,
        iterations_run: 3,
        coverage_score: 0.91,
        shared_understanding_len: 512,
        archetypes: vec!["researcher".into(), "skeptic".into()],
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: ThinkingLoopCompletedEvent = serde_json::from_str(&s).unwrap();
    assert!(back.enabled);
    assert_eq!(back.iterations_run, 3);
    assert_eq!(back.archetypes.len(), 2);
}

// ── OracleCalibrationPatchedEvent ─────────────────────────────────────────────

#[test]
fn oracle_calibration_patched_event_serde_roundtrip() {
    let e = OracleCalibrationPatchedEvent {
        task_id: task_id(),
        oracle_pass_rate: 0.78,
        n_observations: 15,
        p_mean_before: 0.70,
        p_mean_after: 0.78,
        rho_mean: 0.12,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: OracleCalibrationPatchedEvent = serde_json::from_str(&s).unwrap();
    assert!((back.oracle_pass_rate - 0.78).abs() < 1e-9);
    assert_eq!(back.n_observations, 15);
}

// ── OproTriggeredEvent ────────────────────────────────────────────────────────

#[test]
fn opro_triggered_event_serde_roundtrip() {
    let e = OproTriggeredEvent {
        adapter_name: "claude-3-opus".into(),
        prompt_key: "auditor_system".into(),
        j_eff_ema: 0.42,
        n_tasks_total: 100,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: OproTriggeredEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.adapter_name, "claude-3-opus");
    assert_eq!(back.n_tasks_total, 100);
}

// ── PromptVariantPromotedEvent ────────────────────────────────────────────────

#[test]
fn prompt_variant_promoted_event_serde_roundtrip() {
    let e = PromptVariantPromotedEvent {
        adapter_name: "gpt-4o".into(),
        prompt_key: "cot_rubric".into(),
        variant_id: "v3".into(),
        winning_score: 0.89,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: PromptVariantPromotedEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.variant_id, "v3");
    assert!((back.winning_score - 0.89).abs() < 1e-9);
}

// ── RotationReason ────────────────────────────────────────────────────────────

#[test]
fn rotation_reason_serde_roundtrip_all_variants() {
    for v in [RotationReason::FirstElection, RotationReason::Stagnation] {
        let s = serde_json::to_string(&v).unwrap();
        let back: RotationReason = serde_json::from_str(&s).unwrap();
        assert_eq!(back, v);
    }
}

// ── LeaderElectedEvent ────────────────────────────────────────────────────────

#[test]
fn leader_elected_event_serde_roundtrip() {
    let e = LeaderElectedEvent {
        task_id: task_id(),
        term: 1,
        leader_explorer_id: explorer_id(),
        q_confidence: 0.85,
        credibility_score: 0.91,
        rotation_reason: Some(RotationReason::Stagnation),
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: LeaderElectedEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.term, 1);
    assert_eq!(back.rotation_reason, Some(RotationReason::Stagnation));
}

#[test]
fn leader_elected_event_first_election_no_rotation_reason() {
    let e = LeaderElectedEvent {
        task_id: task_id(),
        term: 0,
        leader_explorer_id: explorer_id(),
        q_confidence: 0.80,
        credibility_score: 0.88,
        rotation_reason: None,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: LeaderElectedEvent = serde_json::from_str(&s).unwrap();
    assert!(back.rotation_reason.is_none());
}

// ── SocraticDiagnosisEvent ────────────────────────────────────────────────────

#[test]
fn socratic_diagnosis_event_serde_roundtrip() {
    let e = SocraticDiagnosisEvent {
        task_id: task_id(),
        term: 2,
        question: "Does the proposal handle partial network failures?".into(),
        violated_constraints: vec!["ADR-010".into(), "ADR-011".into()],
        eig_rank: 1,
        dedup_candidates_tried: 3,
        timestamp: Utc::now(),
    };
    let s = serde_json::to_string(&e).unwrap();
    let back: SocraticDiagnosisEvent = serde_json::from_str(&s).unwrap();
    assert_eq!(back.term, 2);
    assert_eq!(back.violated_constraints.len(), 2);
    assert_eq!(back.eig_rank, 1);
    assert_eq!(back.dedup_candidates_tried, 3);
}

// ── H2AIEvent::subject() ──────────────────────────────────────────────────────

#[test]
fn h2ai_event_subject_pending_approval_uses_task_specific_subject() {
    let tid = task_id();
    let e = H2AIEvent::PendingApproval(PendingApprovalEvent {
        task_id: tid.clone(),
        proposed_output: "out".into(),
        q_confidence: 0.5,
        prediction_basis: 0,
        n_used: 1,
        risk_level: ApprovalRiskLevel::Low,
        triggered_by: ApprovalTrigger::ManifestFlag,
        timeout_at_ms: 0,
        timestamp_ms: 0,
    });
    let subject = e.subject(&tid);
    assert_eq!(subject, format!("h2ai.tasks.{tid}.pending_approval"));
}

#[test]
fn h2ai_event_subject_approval_resolved_uses_task_specific_subject() {
    let tid = task_id();
    let e = H2AIEvent::ApprovalResolved(ApprovalResolvedEvent {
        task_id: tid.clone(),
        approved: false,
        operator_id: "timeout".into(),
        reviewer_note: None,
        decided_at_ms: 0,
    });
    let subject = e.subject(&tid);
    assert_eq!(subject, format!("h2ai.tasks.{tid}.approval_resolved"));
}

#[test]
fn h2ai_event_subject_other_variant_returns_generic_subject() {
    let tid = task_id();
    let e = H2AIEvent::CalibrationFailed {
        calibration_id: "cal-1".into(),
        reason: "adapter unreachable".into(),
    };
    let subject = e.subject(&tid);
    assert_eq!(subject, format!("h2ai.tasks.{tid}"));
}

#[test]
fn h2ai_event_subject_merge_resolved_uses_generic_subject() {
    let tid = task_id();
    let e = H2AIEvent::MergeResolved(MergeResolvedEvent {
        task_id: tid.clone(),
        resolved_output: "out".into(),
        j_eff: None,
        timestamp: Utc::now(),
        oracle_gate_passed: None,
        zone3_hints: None,
    });
    let subject = e.subject(&tid);
    assert_eq!(subject, format!("h2ai.tasks.{tid}"));
}

// ── H2AIEvent variants not covered by existing events_test.rs ─────────────────

#[test]
fn h2ai_event_calibration_failed_serde_roundtrip() {
    let e = H2AIEvent::CalibrationFailed {
        calibration_id: "cal-abc".into(),
        reason: "endpoint timeout".into(),
    };
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"CalibrationFailed\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    match back {
        H2AIEvent::CalibrationFailed {
            calibration_id,
            reason,
        } => {
            assert_eq!(calibration_id, "cal-abc");
            assert_eq!(reason, "endpoint timeout");
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn h2ai_event_thinking_loop_completed_serde_roundtrip() {
    let e = H2AIEvent::ThinkingLoopCompleted(ThinkingLoopCompletedEvent {
        task_id: task_id(),
        enabled: false,
        iterations_run: 0,
        coverage_score: 0.0,
        shared_understanding_len: 0,
        archetypes: vec![],
        timestamp: Utc::now(),
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"ThinkingLoopCompleted\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::ThinkingLoopCompleted(_)));
}

#[test]
fn h2ai_event_leader_elected_serde_roundtrip() {
    let e = H2AIEvent::LeaderElected(LeaderElectedEvent {
        task_id: task_id(),
        term: 0,
        leader_explorer_id: explorer_id(),
        q_confidence: 0.75,
        credibility_score: 0.80,
        rotation_reason: None,
        timestamp: Utc::now(),
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"LeaderElected\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::LeaderElected(_)));
}

#[test]
fn h2ai_event_socratic_diagnosis_serde_roundtrip() {
    let e = H2AIEvent::SocraticDiagnosis(SocraticDiagnosisEvent {
        task_id: task_id(),
        term: 1,
        question: "Is the timeout handled?".into(),
        violated_constraints: vec![],
        eig_rank: 1,
        dedup_candidates_tried: 0,
        timestamp: Utc::now(),
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"SocraticDiagnosis\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::SocraticDiagnosis(_)));
}

#[test]
fn h2ai_event_pending_approval_serde_roundtrip() {
    let e = H2AIEvent::PendingApproval(PendingApprovalEvent {
        task_id: task_id(),
        proposed_output: "answer".into(),
        q_confidence: 0.48,
        prediction_basis: 1,
        n_used: 2,
        risk_level: ApprovalRiskLevel::Medium,
        triggered_by: ApprovalTrigger::LowConfidence,
        timeout_at_ms: 5000,
        timestamp_ms: 1000,
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"PendingApproval\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::PendingApproval(_)));
}

#[test]
fn h2ai_event_approval_resolved_serde_roundtrip() {
    let e = H2AIEvent::ApprovalResolved(ApprovalResolvedEvent {
        task_id: task_id(),
        approved: true,
        operator_id: "human-1".into(),
        reviewer_note: None,
        decided_at_ms: 3000,
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"ApprovalResolved\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::ApprovalResolved(_)));
}

#[test]
fn h2ai_event_oracle_calibration_patched_serde_roundtrip() {
    let e = H2AIEvent::OracleCalibrationPatched(OracleCalibrationPatchedEvent {
        task_id: task_id(),
        oracle_pass_rate: 0.82,
        n_observations: 12,
        p_mean_before: 0.75,
        p_mean_after: 0.82,
        rho_mean: 0.10,
        timestamp: Utc::now(),
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"OracleCalibrationPatched\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::OracleCalibrationPatched(_)));
}

#[test]
fn h2ai_event_correlated_ensemble_serde_roundtrip() {
    let e = H2AIEvent::CorrelatedEnsemble(CorrelatedEnsembleWarning {
        task_id: task_id(),
        cv: 0.05,
        mean_jaccard_distance: 0.09,
        retry_count: 1,
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"CorrelatedEnsemble\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::CorrelatedEnsemble(_)));
}

#[test]
fn h2ai_event_researcher_grounding_serde_roundtrip() {
    let e = H2AIEvent::ResearcherGrounding(ResearcherGroundingEvent {
        task_id: task_id(),
        shared_assumption: "uses JWT".into(),
        literature_summary: "JWT is widely adopted".into(),
        slot: None,
        source: GroundingSource::WebSearch,
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"ResearcherGrounding\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::ResearcherGrounding(_)));
}

#[test]
fn h2ai_event_diversity_guard_degraded_serde_roundtrip() {
    let e = H2AIEvent::DiversityGuardDegraded(DiversityGuardDegradedEvent {
        task_id: task_id(),
        reason: "coverage 0.20 < threshold 0.40".into(),
        coverage_score: 0.20,
        slot_domains: vec![],
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"DiversityGuardDegraded\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::DiversityGuardDegraded(_)));
}

#[test]
fn h2ai_event_correlated_fabrication_serde_roundtrip() {
    let e = H2AIEvent::CorrelatedFabrication(CorrelatedFabricationEvent {
        task_id: task_id(),
        cfi: 0.6,
        injection_pressure: 0.45,
        shared_ungrounded_entities: vec!["Cache".into()],
        proposal_count: 2,
        hint_injected: false,
        timestamp: Utc::now(),
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"CorrelatedFabrication\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::CorrelatedFabrication(_)));
}

#[test]
fn h2ai_event_opro_triggered_serde_roundtrip() {
    let e = H2AIEvent::OproTriggered(OproTriggeredEvent {
        adapter_name: "llama3".into(),
        prompt_key: "cot".into(),
        j_eff_ema: 0.4,
        n_tasks_total: 50,
        timestamp: Utc::now(),
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"OproTriggered\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::OproTriggered(_)));
}

#[test]
fn h2ai_event_prompt_variant_promoted_serde_roundtrip() {
    let e = H2AIEvent::PromptVariantPromoted(PromptVariantPromotedEvent {
        adapter_name: "claude".into(),
        prompt_key: "eval".into(),
        variant_id: "v1".into(),
        winning_score: 0.95,
        timestamp: Utc::now(),
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"PromptVariantPromoted\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::PromptVariantPromoted(_)));
}

#[test]
fn h2ai_event_pending_clarification_serde_roundtrip() {
    let e = H2AIEvent::PendingClarification(PendingClarificationEvent {
        task_id: "task-x".into(),
        question: "gRPC or REST?".into(),
        context: "latency is key".into(),
        timeout_secs: 600,
        timestamp: Utc::now(),
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"PendingClarification\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::PendingClarification(_)));
}

#[test]
fn h2ai_event_oracle_gate_result_serde_roundtrip() {
    let e = H2AIEvent::OracleGateResult(OracleGateResultEvent {
        task_id: "t-ggg".into(),
        gate_passed: false,
        confidence: 0.3,
        summary: "2/3 tests failed".into(),
        checked_proposals: 3,
        passed_proposals: 1,
        timestamp: Utc::now(),
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"OracleGateResult\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::OracleGateResult(_)));
}

#[test]
fn h2ai_event_zero_coordination_quality_serde_roundtrip() {
    let e = H2AIEvent::CoherenceIncomplete(CoherenceIncompleteEvent {
        task_id: task_id(),
        uncovered_domains: vec!["reliability".into()],
        active_contradictions: vec![],
        retries: 1,
        timestamp: Utc::now(),
        bypassed_verifier_constraint_ids: vec![],
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"CoherenceIncomplete\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::CoherenceIncomplete(_)));
}

#[test]
fn h2ai_event_tao_iteration_serde_roundtrip() {
    let e = H2AIEvent::TaoIteration(TaoIterationEvent {
        task_id: task_id(),
        explorer_id: explorer_id(),
        turn: 1,
        observation: "ok".into(),
        passed: true,
        timestamp: Utc::now(),
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"TaoIteration\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::TaoIteration(_)));
}

#[test]
fn h2ai_event_verifier_comparison_serde_roundtrip() {
    let e = H2AIEvent::VerifierComparison(VerifierComparisonEvent {
        task_id: task_id(),
        explorer_id: explorer_id(),
        standard_score: 0.9,
        adversarial_score: 0.6,
        standard_passed: true,
        adversarial_passed: false,
        verifier_kind: "llmjudge".into(),
        timestamp: Utc::now(),
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"VerifierComparison\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::VerifierComparison(_)));
}

#[test]
fn h2ai_event_subtask_plan_created_serde_roundtrip() {
    let e = H2AIEvent::SubtaskPlanCreated(SubtaskPlanCreatedEvent {
        task_id: task_id(),
        plan_id: task_id(),
        subtask_count: 3,
        timestamp: Utc::now(),
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"SubtaskPlanCreated\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::SubtaskPlanCreated(_)));
}

#[test]
fn h2ai_event_task_attribution_serde_roundtrip() {
    let e = H2AIEvent::TaskAttribution(TaskAttributionEvent {
        task_id: task_id(),
        q_confidence: 0.88,
        q_measured: None,
        q_interval_lo: None,
        q_interval_hi: None,
        prediction_basis: PredictionBasis::Heuristic,
        waste_ratio: 1.0,
        applied_optimizations: vec![],
        tokens_used: 0,
        skill_nodes_injected: 0,
        timestamp: Utc::now(),
        approval_decision: None,
        calibration_source: CalibrationSource::SyntheticPriors,
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"TaskAttribution\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::TaskAttribution(_)));
}

#[test]
fn h2ai_event_epistemic_yield_serde_roundtrip() {
    let e = H2AIEvent::EpistemicYield(EpistemicYieldEvent {
        task_id: task_id(),
        n_eff_cosine_actual: 1.8,
        n_eff_prior: 2.0,
        yield_ratio: 0.6,
        adapters: vec!["a".into()],
    });
    let s = serde_json::to_string(&e).unwrap();
    assert!(s.contains("\"event_type\":\"EpistemicYield\""));
    let back: H2AIEvent = serde_json::from_str(&s).unwrap();
    assert!(matches!(back, H2AIEvent::EpistemicYield(_)));
}
