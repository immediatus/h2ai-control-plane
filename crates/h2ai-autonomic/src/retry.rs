use chrono::Utc;
use h2ai_types::config::TopologyKind;
use h2ai_types::events::{BranchPrunedEvent, TaskFailedEvent, ZeroSurvivalEvent};
use h2ai_types::physics::MultiplicationConditionFailure;

pub enum RetryAction {
    Retry(TopologyKind),
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

impl RetryPolicy {
    pub fn decide(
        event: &ZeroSurvivalEvent,
        tried_topologies: &[TopologyKind],
        pruned_events: Vec<BranchPrunedEvent>,
        tau_values_tried: Vec<Vec<f64>>,
        multiplication_failure: Option<MultiplicationConditionFailure>,
    ) -> RetryAction {
        for make_topology in FRONTIER {
            let candidate = make_topology();
            if !Self::has_tried(tried_topologies, &candidate) {
                return RetryAction::Retry(candidate);
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
