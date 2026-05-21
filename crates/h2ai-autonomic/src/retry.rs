use chrono::Utc;
use h2ai_types::config::TopologyKind;
use h2ai_types::events::{BranchPrunedEvent, TaskFailedEvent, ZeroSurvivalEvent};
use h2ai_types::sizing::MultiplicationConditionFailure;

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
        /// One hint per violated Hard constraint that has a `remediation_hint`.
        hints: Vec<String>,
    },
    Fail(TaskFailedEvent),
}

pub struct RetryPolicy;

/// Pareto frontier order: Ensemble → `HierarchicalTree` → `TeamSwarmHybrid`.
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

    const fn same_variant(a: &TopologyKind, b: &TopologyKind) -> bool {
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
