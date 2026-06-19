use chrono::Utc;
use h2ai_autonomic::retry::{RetryAction, RetryPolicy};
use h2ai_types::config::TopologyKind;
use h2ai_types::events::{BranchPrunedEvent, ConstraintViolation, ZeroSurvivalEvent};
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::sizing::RoleErrorCost;

// ── Shared builders ────────────────────────────────────────────────────────────

fn zero_event() -> ZeroSurvivalEvent {
    ZeroSurvivalEvent {
        task_id: TaskId::new(),
        retry_count: 0,
        timestamp: Utc::now(),
        n_eff_cosine_actual: None,
        failure_mode: None,
    }
}

fn pruned_event(reason: &str) -> BranchPrunedEvent {
    BranchPrunedEvent {
        task_id: TaskId::new(),
        explorer_id: ExplorerId::new(),
        reason: reason.into(),
        raw_output: String::new(),
        constraint_error_cost: RoleErrorCost::new(0.5).unwrap(),
        violated_constraints: vec![],
        timestamp: Utc::now(),
        retry_count: 0,
        bypass_reason: None,
    }
}

fn hard_violation(id: &str, score: f64, hint: Option<&str>) -> ConstraintViolation {
    ConstraintViolation {
        constraint_id: id.into(),
        score,
        severity_label: "Hard".into(),
        remediation_hint: hint.map(str::to_owned),
        constraint_description: format!("{id} must be satisfied"),
        verifier_reason: None,
        check_verdicts: vec![],
        criteria_pass: None,
        check_reasons: None,
    }
}

fn hard_violation_with_reason(id: &str, score: f64, reason: &str) -> ConstraintViolation {
    ConstraintViolation {
        constraint_id: id.into(),
        score,
        severity_label: "Hard".into(),
        remediation_hint: Some(format!("fix {id}")),
        constraint_description: format!("{id} must be satisfied"),
        verifier_reason: Some(reason.into()),
        check_verdicts: vec![],
        criteria_pass: None,
        check_reasons: None,
    }
}

fn pruned_with_violations(violations: Vec<ConstraintViolation>) -> BranchPrunedEvent {
    BranchPrunedEvent {
        violated_constraints: violations,
        ..pruned_event("constraint violation")
    }
}

// ── RetryPolicy::decide ────────────────────────────────────────────────────────

/// Tests for topology frontier traversal (Ensemble → HierarchicalTree → TeamSwarmHybrid → Fail).
mod topology_frontier {
    use super::*;

    #[test]
    fn selects_ensemble_when_no_topology_tried() {
        let action = RetryPolicy::decide(&zero_event(), &[], vec![], vec![], None);
        assert!(matches!(action, RetryAction::Retry(TopologyKind::Ensemble)));
    }

    #[test]
    fn selects_hierarchical_tree_after_ensemble_fails() {
        let tried = vec![TopologyKind::Ensemble];
        let action = RetryPolicy::decide(&zero_event(), &tried, vec![], vec![], None);
        assert!(matches!(
            action,
            RetryAction::Retry(TopologyKind::HierarchicalTree { .. })
        ));
    }

    #[test]
    fn selects_team_swarm_hybrid_after_ensemble_and_tree_fail() {
        let tried = vec![
            TopologyKind::Ensemble,
            TopologyKind::HierarchicalTree {
                branching_factor: Some(3),
            },
        ];
        let action = RetryPolicy::decide(&zero_event(), &tried, vec![], vec![], None);
        assert!(matches!(
            action,
            RetryAction::Retry(TopologyKind::TeamSwarmHybrid)
        ));
    }

    #[test]
    fn fails_after_all_three_topologies_exhausted() {
        let tried = vec![
            TopologyKind::Ensemble,
            TopologyKind::HierarchicalTree {
                branching_factor: None,
            },
            TopologyKind::TeamSwarmHybrid,
        ];
        let action = RetryPolicy::decide(&zero_event(), &tried, vec![], vec![], None);
        assert!(matches!(action, RetryAction::Fail(_)));
    }

    #[test]
    fn fail_event_records_all_tried_topologies() {
        let tried = vec![
            TopologyKind::Ensemble,
            TopologyKind::HierarchicalTree {
                branching_factor: None,
            },
            TopologyKind::TeamSwarmHybrid,
        ];
        let action = RetryPolicy::decide(&zero_event(), &tried, vec![], vec![], None);
        if let RetryAction::Fail(event) = action {
            assert_eq!(event.topologies_tried.len(), 3);
        } else {
            panic!("expected Fail");
        }
    }
}

/// Tests for hallucination-signal detection in pruned-event reasons.
/// Applies only when there are NO structured constraint violations.
mod hallucination_signal {
    use super::*;

    #[test]
    fn majority_hallucination_reasons_trigger_tau_reduction() {
        let events = vec![
            pruned_event("hallucination detected: output fabricated facts"),
            pruned_event("hallucination detected: invented citations"),
        ];
        let action = RetryPolicy::decide(&zero_event(), &[], events, vec![], None);
        assert!(matches!(action, RetryAction::RetryWithTauReduction { .. }));
    }

    #[test]
    fn single_hallucination_event_triggers_tau_reduction() {
        let events = vec![pruned_event("hallucination detected")];
        let action = RetryPolicy::decide(&zero_event(), &[], events, vec![], None);
        assert!(matches!(action, RetryAction::RetryWithTauReduction { .. }));
    }

    #[test]
    fn tau_reduction_factor_is_strictly_between_zero_and_one() {
        let events = vec![pruned_event("hallucination detected")];
        let action = RetryPolicy::decide(&zero_event(), &[], events, vec![], None);
        if let RetryAction::RetryWithTauReduction { tau_factor, .. } = action {
            assert!(
                tau_factor > 0.0 && tau_factor < 1.0,
                "tau_factor={tau_factor}"
            );
        } else {
            panic!("expected RetryWithTauReduction");
        }
    }

    #[test]
    fn non_hallucination_reasons_produce_plain_retry() {
        let events = vec![
            pruned_event("violated ADR-001 constraint"),
            pruned_event("missing required field"),
        ];
        let action = RetryPolicy::decide(&zero_event(), &[], events, vec![], None);
        assert!(matches!(action, RetryAction::Retry(_)));
    }

    #[test]
    fn empty_pruned_events_produces_plain_retry() {
        let action = RetryPolicy::decide(&zero_event(), &[], vec![], vec![], None);
        assert!(matches!(action, RetryAction::Retry(_)));
    }
}

/// Tests for structured constraint-violation handling.
/// Hard violations take priority over the hallucination keyword scan.
mod constraint_violations {
    use super::*;

    #[test]
    fn hard_violations_produce_retry_with_targets() {
        let event = pruned_with_violations(vec![hard_violation(
            "GDPR-001",
            0.0,
            Some("Include data minimization language."),
        )]);
        let action = RetryPolicy::decide(&zero_event(), &[], vec![event], vec![], None);
        assert!(matches!(action, RetryAction::RetryWithTargets { .. }));
    }

    #[test]
    fn hard_violations_carry_remediation_hint_to_repair_target() {
        let event = pruned_with_violations(vec![hard_violation(
            "GDPR-001",
            0.0,
            Some("Include data minimization language."),
        )]);
        let action = RetryPolicy::decide(&zero_event(), &[], vec![event], vec![], None);
        if let RetryAction::RetryWithTargets { targets, .. } = action {
            let target = targets
                .iter()
                .find(|t| t.constraint_id == "GDPR-001")
                .unwrap();
            assert!(target
                .remediation_hint
                .as_deref()
                .is_some_and(|h| h.contains("data minimization")));
        } else {
            panic!("expected RetryWithTargets");
        }
    }

    #[test]
    fn hard_violations_without_hints_still_produce_retry_with_targets() {
        let event = pruned_with_violations(vec![hard_violation("GDPR-001", 0.0, None)]);
        let action = RetryPolicy::decide(&zero_event(), &[], vec![event], vec![], None);
        assert!(
            matches!(action, RetryAction::RetryWithTargets { .. }),
            "hard violations override the hallucination path regardless of hint presence"
        );
    }

    #[test]
    fn hard_violations_override_hallucination_signals_in_same_event() {
        let mut event = pruned_event("hallucination detected: fabricated output");
        event.violated_constraints = vec![hard_violation("GDPR-001", 0.0, None)];
        let action = RetryPolicy::decide(&zero_event(), &[], vec![event], vec![], None);
        assert!(
            matches!(action, RetryAction::RetryWithTargets { .. }),
            "structured violations take priority over the hallucination keyword scan"
        );
    }

    #[test]
    fn no_violations_with_hallucination_signal_produces_tau_reduction() {
        let event = pruned_event("hallucination detected: fabricated output");
        let action = RetryPolicy::decide(&zero_event(), &[], vec![event], vec![], None);
        assert!(matches!(action, RetryAction::RetryWithTauReduction { .. }));
    }
}

/// Tests for verifier-reason aggregation (Jaccard similarity, contradiction detection).
/// Exercises the score_reason_pairs accumulation and min_pairwise_jaccard path.
mod verifier_reason_aggregation {
    use super::*;

    #[test]
    fn single_verifier_reason_propagates_to_verifier_reasons_vec() {
        let event = pruned_with_violations(vec![hard_violation_with_reason(
            "GDPR-001",
            0.2,
            "response does not minimize personal data collection",
        )]);
        let action = RetryPolicy::decide(&zero_event(), &[], vec![event], vec![], None);
        if let RetryAction::RetryWithTargets { targets, .. } = action {
            let target = targets
                .iter()
                .find(|t| t.constraint_id == "GDPR-001")
                .unwrap();
            assert_eq!(target.verifier_reasons.len(), 1);
            assert_eq!(
                target.verifier_reasons[0].1,
                "response does not minimize personal data collection",
            );
            assert!((target.verifier_reasons[0].0 - 0.2).abs() < 1e-9);
        } else {
            panic!("expected RetryWithTargets");
        }
    }

    #[test]
    fn multiple_proposals_returns_top_k_by_score() {
        // Two events violate the same constraint. retry_count=0 → k=1.
        // Only the highest-scoring reason is returned on the first retry wave.
        let e1 = pruned_with_violations(vec![hard_violation_with_reason(
            "GDPR-001",
            0.3,
            "response does not minimize personal data collection",
        )]);
        let e2 = pruned_with_violations(vec![hard_violation_with_reason(
            "GDPR-001",
            0.6,
            "the proposal fails to minimize personal data collection",
        )]);
        let action = RetryPolicy::decide(&zero_event(), &[], vec![e1, e2], vec![], None);
        if let RetryAction::RetryWithTargets { targets, .. } = action {
            let target = targets
                .iter()
                .find(|t| t.constraint_id == "GDPR-001")
                .unwrap();
            // k=1 (retry_count=0) → single highest-scoring reason
            assert_eq!(target.verifier_reasons.len(), 1);
            assert_eq!(
                target.verifier_reasons[0].1,
                "the proposal fails to minimize personal data collection",
                "highest-scoring proposal reason must be selected"
            );
        } else {
            panic!("expected RetryWithTargets");
        }
    }

    #[test]
    fn whitespace_only_reasons_deduped_by_jaccard_keeps_highest_scoring() {
        // "  " reasons both have Jaccard 1.0 with each other → second is deduped.
        // The highest-scoring proposal's reason survives.
        let e1 = pruned_with_violations(vec![ConstraintViolation {
            constraint_id: "GDPR-001".into(),
            score: 0.2,
            severity_label: "Hard".into(),
            remediation_hint: Some("fix it".into()),
            constraint_description: "GDPR-001 must be satisfied".into(),
            verifier_reason: Some("  ".into()),
            check_verdicts: vec![],
            criteria_pass: None,
            check_reasons: None,
        }]);
        let e2 = pruned_with_violations(vec![ConstraintViolation {
            constraint_id: "GDPR-001".into(),
            score: 0.5,
            severity_label: "Hard".into(),
            remediation_hint: Some("fix it".into()),
            constraint_description: "GDPR-001 must be satisfied".into(),
            verifier_reason: Some("  ".into()),
            check_verdicts: vec![],
            criteria_pass: None,
            check_reasons: None,
        }]);
        let action = RetryPolicy::decide(&zero_event(), &[], vec![e1, e2], vec![], None);
        if let RetryAction::RetryWithTargets { targets, .. } = action {
            let target = targets
                .iter()
                .find(|t| t.constraint_id == "GDPR-001")
                .unwrap();
            // k=1, dedup removes identical reason → single entry from highest scorer (0.5)
            assert_eq!(target.verifier_reasons.len(), 1);
            assert_eq!(target.verifier_reasons[0].1, "  ");
            assert!((target.verifier_reasons[0].0 - 0.5).abs() < 1e-9);
        } else {
            panic!("expected RetryWithTargets");
        }
    }

    #[test]
    fn divergent_reasons_use_top_k_highest_scoring_proposals() {
        // Two events with unrelated reasons. k=1 → highest-scoring reason returned.
        // No None fallback — always provides signal regardless of divergence.
        let e1 = pruned_with_violations(vec![hard_violation_with_reason(
            "AUTH-001",
            0.2,
            "missing required authentication header in the request",
        )]);
        let e2 = pruned_with_violations(vec![hard_violation_with_reason(
            "AUTH-001",
            0.4,
            "hallucinated database table name referenced in output",
        )]);
        let action = RetryPolicy::decide(&zero_event(), &[], vec![e1, e2], vec![], None);
        if let RetryAction::RetryWithTargets { targets, .. } = action {
            let target = targets
                .iter()
                .find(|t| t.constraint_id == "AUTH-001")
                .unwrap();
            assert_eq!(target.verifier_reasons.len(), 1);
            assert_eq!(
                target.verifier_reasons[0].1,
                "hallucinated database table name referenced in output",
                "highest-scoring proposal reason must be returned even when reasons diverge"
            );
            assert!(
                target.remediation_hint.is_some(),
                "static remediation_hint must be preserved"
            );
        } else {
            panic!("expected RetryWithTargets");
        }
    }

    #[test]
    fn second_retry_wave_returns_two_unique_reasons() {
        // retry_count=1 → k=2. Two distinct reasons should both appear.
        let e1 = pruned_with_violations(vec![hard_violation_with_reason(
            "BFT-001",
            0.3,
            "missing quorum size proof for Byzantine fault tolerance",
        )]);
        let e2 = pruned_with_violations(vec![hard_violation_with_reason(
            "BFT-001",
            0.6,
            "view-change mechanism does not terminate in O(f) rounds",
        )]);
        let tried = vec![TopologyKind::Ensemble];
        let wave2_event = ZeroSurvivalEvent {
            retry_count: 1,
            ..zero_event()
        };
        let action = RetryPolicy::decide(&wave2_event, &tried, vec![e1, e2], vec![], None);
        if let RetryAction::RetryWithTargets { targets, .. } = action {
            let target = targets
                .iter()
                .find(|t| t.constraint_id == "BFT-001")
                .unwrap();
            // k=2, both reasons are distinct (low Jaccard) → both returned
            assert_eq!(
                target.verifier_reasons.len(),
                2,
                "wave 2 must provide 2 reasons"
            );
            assert_eq!(
                target.verifier_reasons[0].1,
                "view-change mechanism does not terminate in O(f) rounds",
                "highest-scoring reason first"
            );
        } else {
            panic!("expected RetryWithTargets");
        }
    }

    #[test]
    fn three_divergent_reasons_exercises_structural_divergence_log_path() {
        // Three pruned events all violating "BFT-002" with structurally divergent reasons.
        // Computed Jaccard pairs:
        //   (a,b): shared={request} → 1/13 ≈ 0.077
        //   (a,c): shared={}       → 0/15 = 0.0
        //   (b,c): shared={}       → 0/13 = 0.0
        // min_j=0.0, mean_j≈0.026 → min_j < mean_j*0.5 → tracing::warn! branch entered.
        let e1 = pruned_with_violations(vec![hard_violation_with_reason(
            "BFT-002",
            0.2,
            "missing required auth header in the API request",
        )]);
        let e2 = pruned_with_violations(vec![hard_violation_with_reason(
            "BFT-002",
            0.3,
            "request lacks proper authentication token validation",
        )]);
        let e3 = pruned_with_violations(vec![hard_violation_with_reason(
            "BFT-002",
            0.4,
            "completely unrelated byzantine fault tolerance failure mode",
        )]);
        let action = RetryPolicy::decide(&zero_event(), &[], vec![e1, e2, e3], vec![], None);
        assert!(
            matches!(action, RetryAction::RetryWithTargets { .. }),
            "divergent reasons must still produce RetryWithTargets"
        );
    }
}
