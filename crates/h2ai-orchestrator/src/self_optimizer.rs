use crate::attribution::{AttributionInput, HarnessAttribution};
use h2ai_config::H2AIConfig;

pub struct SuggestInput<'a> {
    pub current: &'a OptimizerParams,
    pub history: &'a [QualityMeasurement],
    pub n_max_ceiling: u32,
    pub alpha: f64,
    pub kappa_eff: f64,
    pub baseline_c_i: f64,
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
        let SuggestInput { current, history, n_max_ceiling, alpha, kappa_eff, baseline_c_i, filter_ratio, cfg } = input;
        let current_q = Self::predict_q(current, alpha, kappa_eff, baseline_c_i, filter_ratio);

        // Option A: raise TAO turns
        if current.max_turns < 4 {
            let candidate = OptimizerParams {
                max_turns: current.max_turns + 1,
                ..current.clone()
            };
            let candidate_q = Self::predict_q(&candidate, alpha, kappa_eff, baseline_c_i, filter_ratio);
            if candidate_q > current_q && !Self::already_tried(&candidate, history) {
                // Also check if TAO gain > agent gain to prefer TAO
                let agent_candidate = OptimizerParams {
                    n_agents: (current.n_agents + 1).min(n_max_ceiling),
                    ..current.clone()
                };
                let agent_q = Self::predict_q(&agent_candidate, alpha, kappa_eff, baseline_c_i, filter_ratio);
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
            let candidate_q = Self::predict_q(&candidate, alpha, kappa_eff, baseline_c_i, filter_ratio);
            if candidate_q > current_q && !Self::already_tried(&candidate, history) {
                return candidate;
            }
        }

        // Option C: add an agent
        if current.n_agents < n_max_ceiling {
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
        alpha: f64,
        kappa_eff: f64,
        baseline_c_i: f64,
        filter_ratio: f64,
    ) -> f64 {
        let attr = HarnessAttribution::compute(&AttributionInput {
            baseline_c_i,
            n_agents: params.n_agents,
            alpha,
            kappa_eff,
            verification_filter_ratio: filter_ratio,
            tao_turns_mean: params.max_turns as f64,
        });
        attr.total_quality
    }

    fn already_tried(candidate: &OptimizerParams, history: &[QualityMeasurement]) -> bool {
        history.iter().any(|m| &m.params == candidate)
    }
}
