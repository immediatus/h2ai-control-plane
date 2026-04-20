use chrono::Utc;
use h2ai_config::H2AIConfig;
use h2ai_types::config::{
    AdapterKind, AuditorConfig, ExplorerConfig, ParetoWeights, ReviewGate, RoleSpec, TopologyKind,
};
use h2ai_types::events::TopologyProvisionedEvent;
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::physics::{
    CoherencyCoefficients, CoordinationThreshold, MergeStrategy, RoleErrorCost, TauValue,
};

#[derive(Debug)]
pub struct ProvisionInput<'a> {
    pub task_id: TaskId,
    pub cc: &'a CoherencyCoefficients,
    pub pareto_weights: &'a ParetoWeights,
    pub role_specs: &'a [RoleSpec],
    pub review_gates: Vec<ReviewGate>,
    pub auditor_config: AuditorConfig,
    pub explorer_adapter: AdapterKind,
    pub retry_count: u32,
    pub cfg: &'a H2AIConfig,
}

pub struct TopologyPlanner;

impl TopologyPlanner {
    pub fn provision(input: ProvisionInput<'_>) -> TopologyProvisionedEvent {
        let kappa_eff = input.cc.kappa_eff();
        let n_max = input.cc.n_max();
        let topology_kind = Self::select_topology(input.pareto_weights, &input.review_gates, n_max);
        let coordination_threshold =
            CoordinationThreshold::from_calibration(input.cc, input.cfg.coordination_threshold_max);

        let role_error_costs: Vec<RoleErrorCost> = input
            .role_specs
            .iter()
            .map(|rs| {
                let v = rs
                    .role_error_cost
                    .unwrap_or_else(|| rs.role.default_role_error_cost());
                RoleErrorCost::new(v).expect("role_error_cost is a valid [0,1] value")
            })
            .collect();

        let merge_strategy =
            MergeStrategy::from_role_costs(&role_error_costs, input.cfg.bft_threshold);

        let explorer_configs: Vec<ExplorerConfig> = input
            .role_specs
            .iter()
            .map(|rs| {
                let tau = rs.tau.unwrap_or_else(|| {
                    TauValue::new(rs.role.default_tau())
                        .expect("role default_tau must be in (0, 1]")
                });
                ExplorerConfig {
                    explorer_id: ExplorerId::new(),
                    tau,
                    adapter: input.explorer_adapter.clone(),
                    role: Some(rs.role.clone()),
                }
            })
            .collect();

        TopologyProvisionedEvent {
            task_id: input.task_id,
            topology_kind,
            explorer_configs,
            auditor_config: input.auditor_config,
            n_max,
            interface_n_max: None,
            kappa_eff,
            role_error_costs,
            merge_strategy,
            coordination_threshold,
            review_gates: input.review_gates,
            retry_count: input.retry_count,
            timestamp: Utc::now(),
        }
    }

    fn select_topology(
        pareto_weights: &ParetoWeights,
        review_gates: &[ReviewGate],
        n_max: f64,
    ) -> TopologyKind {
        if !review_gates.is_empty() {
            return TopologyKind::TeamSwarmHybrid;
        }
        if pareto_weights.containment > pareto_weights.throughput
            && pareto_weights.containment > pareto_weights.diversity
        {
            let bf = (n_max.floor() as u8).max(2);
            return TopologyKind::HierarchicalTree {
                branching_factor: Some(bf),
            };
        }
        TopologyKind::Ensemble
    }
}
