use chrono::Utc;
use h2ai_config::H2AIConfig;
use h2ai_types::config::{
    AdapterKind, AuditorConfig, ExplorerConfig, ParetoWeights, ReviewGate, RoleSpec, TopologyKind,
};
use h2ai_types::events::{TopologyProvisionedEvent, ZeroCoordinationQualityEvent};
use h2ai_types::identity::{ExplorerId, TaskId};
use h2ai_types::physics::{
    CoherencyCoefficients, CoordinationThreshold, EigenCalibration, MergeStrategy, RoleErrorCost,
    TauValue,
};

/// All parameters required to provision a topology for a single task execution.
///
/// `ProvisionInput` bundles calibration state (`cc`, `eigen`), task identity, and
/// operator policy (`pareto_weights`, `role_specs`, `review_gates`) so that
/// `TopologyPlanner::provision` remains a pure function with no hidden state.
#[derive(Debug)]
pub struct ProvisionInput<'a> {
    /// Identifier that propagates into the emitted `TopologyProvisionedEvent`.
    pub task_id: TaskId,
    /// Coherency coefficients (α, β_eff, CG mean) from the most recent calibration run.
    pub cc: &'a CoherencyCoefficients,
    /// Operator-supplied (T, E, D) weights used to score Pareto-frontier topology candidates.
    pub pareto_weights: &'a ParetoWeights,
    /// Per-role specifications that drive explorer τ assignment and merge strategy selection.
    pub role_specs: &'a [RoleSpec],
    /// Review gates that, when non-empty, unconditionally force `TeamSwarmHybrid` topology.
    pub review_gates: Vec<ReviewGate>,
    /// Auditor configuration forwarded verbatim into the provisioned topology event.
    pub auditor_config: AuditorConfig,
    /// Adapter kind assigned to every explorer in the provisioned ensemble.
    pub explorer_adapter: AdapterKind,
    /// When set by the retry loop, overrides Pareto-based topology selection.
    pub force_topology: Option<TopologyKind>,
    /// Retry generation counter forwarded to the topology event for observability.
    pub retry_count: u32,
    /// Runtime configuration supplying CG collapse threshold, token budgets, and USL bounds.
    pub cfg: &'a H2AIConfig,
    /// When present, caps n_max at the eigenvalue-derived optimal adapter count.
    ///
    /// Pruning adapters beyond the spectral optimum avoids coherency overhead without
    /// sacrificing ensemble quality.
    pub eigen: Option<&'a EigenCalibration>,
}

/// Stateless planner that converts calibration state and operator policy into a
/// ready-to-use `TopologyProvisionedEvent`.
///
/// Topology selection follows a strict priority order: CG collapse guard → review-gate
/// override → Pareto-frontier scoring — ensuring safety constraints are never bypassed
/// by misconfigured weights.
pub struct TopologyPlanner;

impl TopologyPlanner {
    /// Provision a topology from calibration state and operator policy.
    ///
    /// Returns a `(TopologyProvisionedEvent, Option<ZeroCoordinationQualityEvent>)` pair.
    /// The first element is always present and carries the selected topology kind, explorer
    /// configs, merge strategy, and n_max (USL-derived, optionally capped by eigenvalue
    /// pruning via `input.eigen`).
    /// The second element is `Some` when CG mean falls below `cfg.cg_collapse_threshold`;
    /// in that case n_max is forced to 1 and the caller must publish the collapse event so
    /// the orchestrator can take corrective action.
    /// Topology selection order: CG collapse forces `n_max=1` → review gates force
    /// `TeamSwarmHybrid` → Pareto-frontier scoring over `(T, E, D)` weights picks the
    /// highest-weighted candidate from `{HierarchicalTree, TeamSwarmHybrid, Ensemble}`.
    pub fn provision(
        input: ProvisionInput<'_>,
    ) -> (
        TopologyProvisionedEvent,
        Option<ZeroCoordinationQualityEvent>,
    ) {
        let beta_eff = input.cc.beta_eff();
        let cg_mean = input.cc.cg_mean();

        // CG collapse guard: when coordination quality falls below the configured threshold,
        // no ensemble benefit is possible. Force N_max=1 and surface a diagnostic event.
        let cg_collapsed = cg_mean < input.cfg.cg_collapse_threshold;
        let collapse_event = if cg_collapsed {
            Some(ZeroCoordinationQualityEvent {
                task_id: input.task_id.clone(),
                cg_embed: cg_mean,
                forced_n_max: 1,
                timestamp: Utc::now(),
            })
        } else {
            None
        };

        let n_max_usl = if cg_collapsed {
            1.0
        } else {
            match input.cfg.max_context_tokens {
                Some(max_tokens) => input.cc.n_max_context_aware(
                    input.cfg.explorer_max_tokens as f64,
                    max_tokens as f64,
                    input.cfg.context_pressure_gamma,
                ),
                None => input.cc.n_max(),
            }
        };
        let n_max = if cg_collapsed {
            1.0
        } else {
            match input.eigen {
                Some(eigen) if eigen.n_pruned > 0 => n_max_usl.min(eigen.n_pruned as f64),
                _ => n_max_usl,
            }
        };
        let topology_kind = input.force_topology.clone().unwrap_or_else(|| {
            Self::select_topology(input.pareto_weights, &input.review_gates, n_max)
        });
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

        let merge_strategy = MergeStrategy::from_role_costs(
            &role_error_costs,
            input.cfg.bft_threshold,
            input.cfg.krum_threshold,
            input.cfg.krum_fault_tolerance,
        );

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

        let event = TopologyProvisionedEvent {
            task_id: input.task_id,
            topology_kind,
            explorer_configs,
            auditor_config: input.auditor_config,
            n_max,
            interface_n_max: None,
            beta_eff,
            role_error_costs,
            merge_strategy,
            coordination_threshold,
            review_gates: input.review_gates,
            retry_count: input.retry_count,
            timestamp: Utc::now(),
        };
        (event, collapse_event)
    }

    fn select_topology(
        pareto_weights: &ParetoWeights,
        review_gates: &[ReviewGate],
        n_max: f64,
    ) -> TopologyKind {
        if !review_gates.is_empty() {
            return TopologyKind::TeamSwarmHybrid;
        }

        // Pareto-frontier topologies with (T, E, D) scores from theory-to-implementation.md.
        struct Candidate {
            score_t: f64,
            score_e: f64,
            score_d: f64,
            make: fn(f64) -> TopologyKind,
        }
        let candidates: [Candidate; 3] = [
            Candidate {
                score_t: 0.96,
                score_e: 0.96,
                score_d: 0.60,
                make: |n| TopologyKind::HierarchicalTree {
                    branching_factor: Some((n.floor() as u8).max(2)),
                },
            },
            Candidate {
                score_t: 0.84,
                score_e: 0.91,
                score_d: 0.95,
                make: |_| TopologyKind::TeamSwarmHybrid,
            },
            // Ensemble is weakly dominated by TeamSwarmHybrid on E and D; retained as an
            // explicit architecture option for future score calibration or forced selection.
            Candidate {
                score_t: 0.84,
                score_e: 0.84,
                score_d: 0.90,
                make: |_| TopologyKind::Ensemble,
            },
        ];

        let wt = pareto_weights.throughput;
        let we = pareto_weights.containment;
        let wd = pareto_weights.diversity;

        let (best_idx, _) = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, wt * c.score_t + we * c.score_e + wd * c.score_d))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("candidates array is non-empty");

        (candidates[best_idx].make)(n_max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use h2ai_types::config::ParetoWeights;

    fn weights(t: f32, e: f32, d: f32) -> ParetoWeights {
        ParetoWeights::new(t as f64, e as f64, d as f64).unwrap()
    }

    #[test]
    fn select_topology_containment_heavy_gives_hierarchical() {
        let result = TopologyPlanner::select_topology(&weights(0.1, 0.8, 0.1), &[], 9.0);
        assert!(
            matches!(result, TopologyKind::HierarchicalTree { .. }),
            "containment-heavy weights → HierarchicalTree, got {:?}",
            result
        );
    }

    #[test]
    fn select_topology_diversity_heavy_gives_team_swarm() {
        let result = TopologyPlanner::select_topology(&weights(0.1, 0.1, 0.8), &[], 9.0);
        assert!(
            matches!(result, TopologyKind::TeamSwarmHybrid),
            "diversity-heavy weights → TeamSwarmHybrid, got {:?}",
            result
        );
    }

    #[test]
    fn select_topology_review_gates_override_weights() {
        let gate = ReviewGate {
            reviewer: "b".into(),
            blocks: "a".into(),
        };
        let result = TopologyPlanner::select_topology(&weights(0.9, 0.05, 0.05), &[gate], 9.0);
        assert!(
            matches!(result, TopologyKind::TeamSwarmHybrid),
            "review gates must force TeamSwarmHybrid"
        );
    }

    #[test]
    fn select_topology_equal_weights_gives_team_swarm() {
        // Pareto scores with equal weights (0.333 each):
        // HierarchicalTree: 0.333*0.96 + 0.333*0.96 + 0.334*0.60 = 0.840
        // TeamSwarmHybrid:  0.333*0.84 + 0.333*0.91 + 0.334*0.95 = 0.900  ← wins
        // Ensemble:         0.333*0.84 + 0.333*0.84 + 0.334*0.90 = 0.860
        let result = TopologyPlanner::select_topology(&weights(0.333, 0.333, 0.334), &[], 9.0);
        assert!(
            matches!(result, TopologyKind::TeamSwarmHybrid),
            "equal weights → TeamSwarmHybrid (score 0.900), got {:?}",
            result
        );
    }
}
