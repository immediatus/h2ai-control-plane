use chrono::Utc;
use h2ai_types::config::TopologyKind;
use h2ai_types::events::{BranchPrunedEvent, TaskFailedEvent, ZeroSurvivalEvent};
use h2ai_types::physics::MultiplicationConditionFailure;

pub enum RetryAction {
    Retry(TopologyKind),
    /// Majority of pruned reasons indicate hallucination — reduce τ on next attempt
    /// to push explorers toward more grounded, less speculative outputs.
    RetryWithTauReduction {
        topology: TopologyKind,
        tau_factor: f64,
    },
    /// Structured constraint violations found — provide targeted remediation hints.
    RetryWithHints {
        topology: TopologyKind,
        /// One hint per violated Hard constraint that has a remediation_hint.
        hints: Vec<String>,
    },
    Fail(TaskFailedEvent),
}

pub struct RetryPolicy;

/// Pareto frontier order: Ensemble → HierarchicalTree → TeamSwarmHybrid.
const FRONTIER: &[fn() -> TopologyKind] = &[
    || TopologyKind::Ensemble,
    || TopologyKind::HierarchicalTree {
        branching_factor: None,
    },
    || TopologyKind::TeamSwarmHybrid,
];

/// Keywords in pruned proposal reasons that suggest explorers were too creative/speculative.
const HALLUCINATION_SIGNALS: &[&str] = &[
    "hallucination",
    "hallucinated",
    "fabricated",
    "fabrication",
    "invented",
    "made up",
    "does not exist",
    "incorrect",
    "inaccurate",
    "not factual",
    "fictional",
    "false claim",
];

impl RetryPolicy {
    pub fn decide(
        event: &ZeroSurvivalEvent,
        tried_topologies: &[TopologyKind],
        pruned_events: Vec<BranchPrunedEvent>,
        tau_values_tried: Vec<Vec<f64>>,
        multiplication_failure: Option<MultiplicationConditionFailure>,
    ) -> RetryAction {
        // Check structured constraint violations first — they provide targeted hints.
        let hints: Vec<String> = pruned_events
            .iter()
            .flat_map(|e| e.violated_constraints.iter())
            .filter(|v| v.severity_label == "Hard")
            .filter_map(|v| v.remediation_hint.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Fall back to keyword scan only when no structured violations carry hints.
        let hallucination_count = if hints.is_empty() {
            pruned_events
                .iter()
                .filter(|e| {
                    let r = e.reason.to_lowercase();
                    HALLUCINATION_SIGNALS.iter().any(|sig| r.contains(sig))
                })
                .count()
        } else {
            0
        };

        // Majority (>50%) of pruned reasons indicate hallucination → reduce τ.
        let reduce_tau = hints.is_empty()
            && !pruned_events.is_empty()
            && hallucination_count * 2 >= pruned_events.len();

        for make_topology in FRONTIER {
            let candidate = make_topology();
            if !Self::has_tried(tried_topologies, &candidate) {
                return if !hints.is_empty() {
                    RetryAction::RetryWithHints {
                        topology: candidate,
                        hints,
                    }
                } else if reduce_tau {
                    RetryAction::RetryWithTauReduction {
                        topology: candidate,
                        tau_factor: 0.7,
                    }
                } else {
                    RetryAction::Retry(candidate)
                };
            }
        }

        RetryAction::Fail(TaskFailedEvent {
            task_id: event.task_id.clone(),
            pruned_events,
            topologies_tried: tried_topologies.to_vec(),
            tau_values_tried,
            multiplication_condition_failure: multiplication_failure,
            timestamp: Utc::now(),
        })
    }

    fn has_tried(tried: &[TopologyKind], candidate: &TopologyKind) -> bool {
        tried.iter().any(|t| Self::same_variant(t, candidate))
    }

    fn same_variant(a: &TopologyKind, b: &TopologyKind) -> bool {
        matches!(
            (a, b),
            (TopologyKind::Ensemble, TopologyKind::Ensemble)
                | (
                    TopologyKind::HierarchicalTree { .. },
                    TopologyKind::HierarchicalTree { .. }
                )
                | (TopologyKind::TeamSwarmHybrid, TopologyKind::TeamSwarmHybrid)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use h2ai_types::identity::{ExplorerId, TaskId};
    use h2ai_types::physics::RoleErrorCost;

    fn pruned(reason: &str) -> BranchPrunedEvent {
        BranchPrunedEvent {
            task_id: TaskId::new(),
            explorer_id: ExplorerId::new(),
            reason: reason.into(),
            constraint_error_cost: RoleErrorCost::new(0.5).unwrap(),
            violated_constraints: vec![],
            timestamp: Utc::now(),
        }
    }

    fn zero_event() -> ZeroSurvivalEvent {
        ZeroSurvivalEvent {
            task_id: TaskId::new(),
            retry_count: 0,
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn hallucination_reasons_trigger_tau_reduction() {
        let pruned_events = vec![
            pruned("hallucination detected: output fabricated facts"),
            pruned("hallucination detected: invented citations"),
        ];
        let action = RetryPolicy::decide(&zero_event(), &[], pruned_events, vec![], None);
        assert!(
            matches!(action, RetryAction::RetryWithTauReduction { .. }),
            "majority hallucination reasons must trigger tau reduction"
        );
    }

    #[test]
    fn non_hallucination_reasons_use_plain_retry() {
        let pruned_events = vec![
            pruned("violated ADR-001 constraint"),
            pruned("missing required field"),
        ];
        let action = RetryPolicy::decide(&zero_event(), &[], pruned_events, vec![], None);
        assert!(matches!(action, RetryAction::Retry(_)));
    }

    #[test]
    fn empty_pruned_events_uses_plain_retry() {
        let action = RetryPolicy::decide(&zero_event(), &[], vec![], vec![], None);
        assert!(matches!(action, RetryAction::Retry(_)));
    }

    #[test]
    fn tau_reduction_factor_is_in_open_unit_interval() {
        let pruned_events = vec![pruned("hallucination detected")];
        let action = RetryPolicy::decide(&zero_event(), &[], pruned_events, vec![], None);
        if let RetryAction::RetryWithTauReduction { tau_factor, .. } = action {
            assert!(
                tau_factor > 0.0 && tau_factor < 1.0,
                "tau_factor must be in (0,1), got {tau_factor}"
            );
        }
    }

    #[test]
    fn violated_constraints_with_hints_produce_retry_with_hints() {
        use h2ai_types::events::ConstraintViolation;
        let mut event = pruned("constraint violation");
        event.violated_constraints = vec![ConstraintViolation {
            constraint_id: "GDPR-001".into(),
            score: 0.0,
            severity_label: "Hard".into(),
            remediation_hint: Some("Include explicit data minimization language.".into()),
        }];
        let action = RetryPolicy::decide(&zero_event(), &[], vec![event], vec![], None);
        assert!(
            matches!(action, RetryAction::RetryWithHints { .. }),
            "structured Hard violations with hints must produce RetryWithHints"
        );
        if let RetryAction::RetryWithHints { hints, .. } = action {
            assert!(hints.iter().any(|h| h.contains("data minimization")));
        }
    }

    #[test]
    fn violated_constraints_without_hints_fall_back_to_reason_scan() {
        use h2ai_types::events::ConstraintViolation;
        // Violation with no remediation_hint → no hints collected → falls back to reason scan
        let mut event = pruned("hallucination detected: fabricated output");
        event.violated_constraints = vec![ConstraintViolation {
            constraint_id: "GDPR-001".into(),
            score: 0.0,
            severity_label: "Hard".into(),
            remediation_hint: None, // no hint
        }];
        let action = RetryPolicy::decide(&zero_event(), &[], vec![event], vec![], None);
        // Since no hints, falls back to hallucination keyword scan → TauReduction
        assert!(
            matches!(action, RetryAction::RetryWithTauReduction { .. }),
            "no remediation hints + hallucination reason must still trigger tau reduction"
        );
    }
}
