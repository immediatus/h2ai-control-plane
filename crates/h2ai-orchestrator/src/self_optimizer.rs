use crate::attribution::{AttributionInput, HarnessAttribution};
use h2ai_config::H2AIConfig;
use h2ai_types::sizing::PredictionBasis;

const TAU_EMA_ALPHA: f64 = 0.1;

/// Online EMA estimator for τ spread (`tau_min`, `tau_max`).
///
/// Updated when `SelfOptimizer` suggests τ adjustments on wasteful successful tasks.
/// User-specified τ bounds in task manifests always override this estimator.
#[derive(Debug, Clone)]
pub struct TauSpreadEstimator {
    tau_min_ema: f64,
    tau_max_ema: f64,
}

impl TauSpreadEstimator {
    #[must_use]
    pub const fn new(initial_tau_min: f64, initial_tau_max: f64) -> Self {
        Self {
            tau_min_ema: initial_tau_min.clamp(0.0, 1.0),
            tau_max_ema: initial_tau_max.clamp(0.0, 1.0),
        }
    }

    /// Update EMA with a suggested (`tau_min`, `tau_max`) pair.
    pub fn update(&mut self, suggested_min: f64, suggested_max: f64) {
        self.tau_min_ema = TAU_EMA_ALPHA
            .mul_add(
                suggested_min.clamp(0.0, 1.0),
                (1.0 - TAU_EMA_ALPHA) * self.tau_min_ema,
            )
            .clamp(0.0, 1.0);
        self.tau_max_ema = TAU_EMA_ALPHA
            .mul_add(
                suggested_max.clamp(0.0, 1.0),
                (1.0 - TAU_EMA_ALPHA) * self.tau_max_ema,
            )
            .clamp(0.0, 1.0);
    }

    #[must_use]
    pub const fn tau_min(&self) -> f64 {
        self.tau_min_ema
    }
    #[must_use]
    pub const fn tau_max(&self) -> f64 {
        self.tau_max_ema
    }
}

pub struct SuggestInput<'a> {
    pub current: &'a OptimizerParams,
    pub history: &'a [QualityMeasurement],
    pub n_max_ceiling: u32,
    /// Condorcet-optimal ensemble size (from `EnsembleCalibration::n_optimal`).
    /// When present, used as the upper target for agent-count suggestions instead of
    /// `n_max_ceiling`. `n_max_ceiling` (Amdahl ceiling) remains the hard cap.
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

/// One historical measurement: the params used and the resulting confidence estimate.
#[derive(Debug, Clone)]
pub struct QualityMeasurement {
    pub params: OptimizerParams,
    pub q_confidence: f64,
}

pub struct SelfOptimizer;

impl SelfOptimizer {
    /// Suggest improved params given current params, history, and the `N_max` ceiling.
    ///
    /// Strategy (matches Proposition 8 MAPE-K guidance):
    /// 1. If `max_turns` < 4 and adding a TAO turn is predicted to raise `q_confidence`
    ///    more than adding an agent → raise `max_turns` first.
    /// 2. Else if `verify_threshold` > 0.3 and tightening threshold is predicted
    ///    to raise `q_confidence` → lower `verify_threshold` by 0.1.
    /// 3. Else if `n_agents` < `n_max_ceiling` → raise `n_agents` by 1.
    /// 4. If nothing improves (already at ceiling on all axes) → return current.
    ///
    /// The `history` slice can be empty; if non-empty, any suggestion that was
    /// already tried (params in history with no Q improvement) is skipped.
    #[must_use]
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
        let n_upper = n_optimal.map_or(n_max_ceiling, |n| n.min(n_max_ceiling));
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
        // When filter_ratio == 0 (ZeroSurvival — no proposal survived), predict_q returns
        // the same value for any threshold because the model doesn't see the threshold as an
        // input. Force a reduction: lowering the threshold is the only knob that can let
        // proposals through without a full topology change.
        let zero_survival = filter_ratio < 1e-9;
        if current.verify_threshold > cfg.optimizer_threshold_floor {
            let candidate = OptimizerParams {
                verify_threshold: (current.verify_threshold - cfg.optimizer_threshold_step)
                    .max(cfg.optimizer_threshold_floor),
                ..current.clone()
            };
            let should_lower = zero_survival || {
                let candidate_q = Self::predict_q(&candidate, p_mean, rho_mean, filter_ratio, tpf);
                candidate_q > current_q
            };
            if should_lower && !Self::already_tried(&candidate, history) {
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
            tao_turns_mean: f64::from(params.max_turns),
            tao_per_turn_factor,
            prediction_basis: PredictionBasis::Heuristic,
            talagrand_state: None,
            eigen_calibration: None,
        });
        attr.q_confidence
    }

    fn already_tried(candidate: &OptimizerParams, history: &[QualityMeasurement]) -> bool {
        history.iter().any(|m| &m.params == candidate)
    }
}
