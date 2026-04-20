use crate::attribution::{AttributionInput, HarnessAttribution};
use h2ai_config::H2AIConfig;

pub struct SuggestInput<'a> {
    pub current: &'a OptimizerParams,
    pub history: &'a [QualityMeasurement],
    pub n_max_ceiling: u32,
    /// Condorcet-optimal ensemble size (from EnsembleCalibration::n_optimal).
    /// When present, used as the upper target for agent-count suggestions instead of
    /// n_max_ceiling. n_max_ceiling (Amdahl ceiling) remains the hard cap.
    pub n_optimal: Option<u32>,
    pub p_mean: f64,
    pub rho_mean: f64,
    pub filter_ratio: f64,
    pub cfg: &'a H2AIConfig,
}

/// Current harness parameters the self-optimizer may adjust.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerParams {
    pub n_agents: u32,
    pub max_turns: u32,
    pub verify_threshold: f64,
}

/// One historical measurement: the params used and the resulting total quality.
#[derive(Debug, Clone)]
pub struct QualityMeasurement {
    pub params: OptimizerParams,
    pub q_total: f64,
}

pub struct SelfOptimizer;

impl SelfOptimizer {
    /// Suggest improved params given current params, history, and the N_max ceiling.
    ///
    /// Strategy (matches Proposition 8 MAPE-K guidance):
    /// 1. If max_turns < 4 and adding a TAO turn is predicted to raise Q_total
    ///    more than adding an agent → raise max_turns first.
    /// 2. Else if verify_threshold > 0.3 and tightening threshold is predicted
    ///    to raise Q_total → lower verify_threshold by 0.1.
    /// 3. Else if n_agents < n_max_ceiling → raise n_agents by 1.
    /// 4. If nothing improves (already at ceiling on all axes) → return current.
    ///
    /// The `history` slice can be empty; if non-empty, any suggestion that was
    /// already tried (params in history with no Q improvement) is skipped.
    pub fn suggest(input: SuggestInput<'_>) -> OptimizerParams {
        let SuggestInput {
            current,
            history,
            n_max_ceiling,
            n_optimal,
            p_mean,
            rho_mean,
            filter_ratio,
            cfg,
        } = input;
        // Effective upper bound: prefer n_optimal (Condorcet target) if available and
        // below the Amdahl ceiling; fall back to n_max_ceiling.
        let n_upper = n_optimal
            .map(|n| n.min(n_max_ceiling))
            .unwrap_or(n_max_ceiling);
        let tpf = cfg.tao_per_turn_factor;
        let current_q = Self::predict_q(current, p_mean, rho_mean, filter_ratio, tpf);

        // Option A: raise TAO turns
        if current.max_turns < 4 {
            let candidate = OptimizerParams {
                max_turns: current.max_turns + 1,
                ..current.clone()
            };
            let candidate_q = Self::predict_q(&candidate, p_mean, rho_mean, filter_ratio, tpf);
            if candidate_q > current_q && !Self::already_tried(&candidate, history) {
                // Also check if TAO gain > agent gain to prefer TAO
                let agent_candidate = OptimizerParams {
                    n_agents: (current.n_agents + 1).min(n_upper),
                    ..current.clone()
                };
                let agent_q =
                    Self::predict_q(&agent_candidate, p_mean, rho_mean, filter_ratio, tpf);
                if candidate_q >= agent_q {
                    return candidate;
                }
            }
        }

        // Option B: tighten verify_threshold
        if current.verify_threshold > cfg.optimizer_threshold_floor {
            let candidate = OptimizerParams {
                verify_threshold: (current.verify_threshold - cfg.optimizer_threshold_step)
                    .max(cfg.optimizer_threshold_floor),
                ..current.clone()
            };
            let candidate_q = Self::predict_q(&candidate, p_mean, rho_mean, filter_ratio, tpf);
            if candidate_q > current_q && !Self::already_tried(&candidate, history) {
                return candidate;
            }
        }

        // Option C: add an agent (up to n_upper = min(n_optimal, n_max_ceiling))
        if current.n_agents < n_upper {
            let candidate = OptimizerParams {
                n_agents: current.n_agents + 1,
                ..current.clone()
            };
            if !Self::already_tried(&candidate, history) {
                return candidate;
            }
        }

        // Nothing improves: return current
        current.clone()
    }

    fn predict_q(
        params: &OptimizerParams,
        p_mean: f64,
        rho_mean: f64,
        filter_ratio: f64,
        tao_per_turn_factor: f64,
    ) -> f64 {
        let attr = HarnessAttribution::compute(&AttributionInput {
            p_mean,
            rho_mean,
            n_agents: params.n_agents,
            verification_filter_ratio: filter_ratio,
            tao_turns_mean: params.max_turns as f64,
            tao_per_turn_factor,
        });
        attr.total_quality
    }

    fn already_tried(candidate: &OptimizerParams, history: &[QualityMeasurement]) -> bool {
        history.iter().any(|m| &m.params == candidate)
    }
}
