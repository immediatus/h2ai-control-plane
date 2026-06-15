use crate::engine::{EngineError, EngineInput};
use crate::phases::StepResult;
use crate::task_store::TaskPhase;
use h2ai_autonomic::planner::{ProvisionInput, TopologyPlanner};
use h2ai_types::config::{AdapterKind, AgentRole, RoleSpec, TopologyKind};
use h2ai_types::events::TopologyProvisionedEvent;
use h2ai_types::sizing::TauValue;
use h2ai_types::sizing::{MergeStrategy, PredictionBasis, TaskQuadrant};

pub struct Input<'a> {
    pub engine_input: &'a EngineInput<'a>,
    pub force_topology: Option<TopologyKind>,
    pub tau_reduction_factor: f64,
    pub tau_spread_factor: f64,
    pub retry_count: u32,
    pub assessed_quadrant: TaskQuadrant,
    pub cg_mean: f64,
    pub n_max_ceiling: u32,
    pub explorer_adapter_kind: &'a AdapterKind,
    /// Pending constraint tombstone to attach to provisioned topology.
    pub pending_tombstone: Option<String>,
    /// Current `n_agents` value from MAPE-K optimizer params (used when roles not specified).
    pub n_agents: u32,
}

pub struct Output {
    pub provisioned: TopologyProvisionedEvent,
    pub explorer_count: u32,
    pub p_mean: f64,
    pub rho_mean: f64,
    pub attribution_basis: PredictionBasis,
}

/// Run Phase 2: Topology Provisioning.
///
/// Builds role specs with τ-spread expansion, calls `TopologyPlanner::provision`,
/// applies the constraint tombstone, and checks the `OutlierResistant` quorum guard.
/// Also derives `p_mean`, `rho_mean`, and `attribution_basis` from ensemble calibration
/// (or `cg_mean` heuristic when absent).
///
/// Returns `StepResult::Done(Output)` on success.
/// Returns `StepResult::Fatal(EngineError::InsufficientQuorum { ... })` when the
/// Krum quorum guard fails.
/// Never returns `StepResult::EarlyExit`.
#[must_use]
pub fn run(input: Input<'_>) -> StepResult<Output> {
    let engine_input = input.engine_input;
    let retry_count = input.retry_count;
    let assessed_quadrant = input.assessed_quadrant;
    let task_id = engine_input.task_id.clone();

    // ── Phase 2: Topology Provisioning ─────────────────────────────
    // Phase 1.5 Precision override (non-shadow): use within-family τ-spread
    // with 2–3 slots and the calibration_tau_spread bounds. The explorer_adapter
    // is already within-family (single AdapterKind); τ diversity decorrelates
    // the samples structurally without crossing family boundaries.
    // TODO(gap-a1-multi-family): when multiple calibrated families are available,
    // select the family with the highest EnsembleCalibration::p_mean here instead
    // of the single explorer_adapter family. Requires per-family p_mean tracking
    // in CalibrationCompletedEvent (not yet implemented).
    let precision_active = !engine_input.cfg.task_complexity.shadow_mode
        && assessed_quadrant == TaskQuadrant::Precision;
    let role_specs: Vec<RoleSpec> = if engine_input.manifest.explorers.roles.is_empty() {
        let count = if precision_active {
            // 2–3 slots: more than 1 provides synthesis benefit; cap at 3 to stay
            // within the Self-MoA budget where within-family wins.
            (input.n_max_ceiling as usize).clamp(3, engine_input.cfg.precision_mode_max_slots)
                as u32
        } else {
            input.n_agents.max(1)
        };
        let (tau_min_manifest, tau_max_manifest) = if precision_active {
            // Use calibration τ-spread bounds for Precision — not the manifest
            // values, which are set for multi-family Coverage tasks.
            let s = engine_input.cfg.calibration_tau_spread;
            (s[0].clamp(0.05, 0.95), s[1].clamp(0.05, 0.95))
        } else {
            (
                engine_input.manifest.explorers.tau_min.unwrap_or(0.2),
                engine_input.manifest.explorers.tau_max.unwrap_or(0.9),
            )
        };
        // Apply τ-spread expansion (Talagrand U-curve feedback) around the manifest centre.
        let tau_center = f64::midpoint(tau_max_manifest, tau_min_manifest);
        let half_spread = (tau_max_manifest - tau_min_manifest) / 2.0;
        let max_half = tau_center.min(1.0 - tau_center); // can't exceed [0,1]
        let expanded_half = (half_spread * input.tau_spread_factor).min(max_half);
        let tau_min = tau_center - expanded_half;
        let tau_max = tau_center + expanded_half;
        let step = if count > 1 {
            (tau_max - tau_min) / f64::from(count - 1)
        } else {
            0.0
        };
        (0..count)
            .map(|i| RoleSpec {
                agent_id: format!("exp_{}", (b'A' + (i % 26) as u8) as char),
                role: AgentRole::Executor,
                tau: Some(
                    TauValue::new(
                        ((tau_min + step * f64::from(i)) * input.tau_reduction_factor)
                            .clamp(0.05, 0.95),
                    )
                    .unwrap_or_else(|_| TauValue::new(0.05).unwrap()),
                ),
                role_error_cost: None,
            })
            .collect()
    } else {
        engine_input.manifest.explorers.roles.clone()
    };

    engine_input
        .store
        .set_phase(&task_id, TaskPhase::Provisioning, 0, retry_count);

    // In shadow_mode: quadrant is observational only — pass None to preserve
    // current topology selection. When armed (shadow_mode=false), pass the
    // assessed quadrant so TopologyPlanner can apply self-MoA for Precision tasks.
    let effective_quadrant = if engine_input.cfg.task_complexity.shadow_mode {
        None
    } else {
        Some(assessed_quadrant)
    };

    let (mut provisioned, _cg_collapse) = TopologyPlanner::provision(ProvisionInput {
        task_id,
        cc: &engine_input.calibration.coefficients,
        pareto_weights: &engine_input.manifest.pareto_weights,
        role_specs: &role_specs,
        review_gates: engine_input.manifest.explorers.review_gates.clone(),
        auditor_config: engine_input.auditor_config.clone(),
        explorer_adapter: input.explorer_adapter_kind.clone(),
        force_topology: input.force_topology.clone(),
        retry_count,
        cfg: engine_input.cfg,
        eigen: engine_input.calibration.eigen.as_ref(),
        ensemble: engine_input.calibration.ensemble.as_ref(),
        task_quadrant: effective_quadrant,
    });
    provisioned.constraint_tombstone = input.pending_tombstone.clone();

    // Guard: OutlierResistant requires n ≥ 2f+3. Fail early rather than silently falling back.
    if let MergeStrategy::OutlierResistant { f } | MergeStrategy::MultiOutlierResistant { f, .. } =
        &provisioned.merge_strategy
    {
        let f = *f;
        let n = provisioned.explorer_configs.len();
        let required = MergeStrategy::min_krum_quorum(f);
        if n < required {
            return StepResult::Fatal(EngineError::InsufficientQuorum { n, f, required });
        }
    }

    let explorer_count = provisioned.explorer_configs.len() as u32;

    // Derive p_mean, rho_mean, and prediction_basis from EnsembleCalibration when available.
    // Fallback proxies when calibration is absent (Heuristic basis):
    //   p = 0.5 + CG_mean / 2  (accuracy proxy from output similarity)
    //   ρ = 1 - CG_mean        (correlation proxy from output similarity)
    //
    // GAP-B5 operational convention: ρ = 1 − CG_mean assumes low constraint-agreement (diverse
    // profiles) implies lower error correlation. This assumes error patterns track constraint
    // specialisation — an unvalidated assumption. The proxy is replaced by the empirical ρ_EMA
    // estimator once ≥ 30 pairwise task observations have accumulated.
    let (p_mean, rho_mean, attribution_basis) = match &engine_input.calibration.ensemble {
        Some(ec) => (ec.p_mean, ec.rho_mean, ec.prediction_basis),
        None => (
            0.5 + input.cg_mean / 2.0,
            (1.0 - input.cg_mean).clamp(0.0, 1.0),
            PredictionBasis::Heuristic,
        ),
    };

    StepResult::Done(Output {
        provisioned,
        explorer_count,
        p_mean,
        rho_mean,
        attribution_basis,
    })
}
