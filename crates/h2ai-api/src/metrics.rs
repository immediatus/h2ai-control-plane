/// In-memory Prometheus-format metric state for the /metrics endpoint.
///
/// Shared via `Arc<RwLock<MetricsState>>` in `AppState`.
#[derive(Debug)]
pub struct MetricsState {
    /// Pool-level `N_eff` from last calibration (gauge).
    pub n_eff_prior: f64,
    /// Task-level `N_eff` from last `EpistemicYieldEvent` (gauge).
    pub n_eff_actual: f64,
    /// `yield_ratio` from last `EpistemicYieldEvent` (gauge).
    pub epistemic_yield_ratio: f64,
    /// Cumulative `ModeCollapse` interventions (counter).
    pub mapek_mode_collapse_count: u64,
    /// Cumulative `ConstrainedExploration` interventions (counter).
    pub mapek_constrained_exploration_count: u64,
    /// Phase 1.5 routing quadrant distribution (counters by quadrant).
    /// Used to validate `θ_tcc` and `θ_neff` thresholds during `shadow_mode` monitoring.
    pub phase15_quadrant_precision: u64,
    pub phase15_quadrant_coverage: u64,
    pub phase15_quadrant_complex: u64,
    pub phase15_quadrant_degenerate: u64,
    /// Current ECE (Expected Calibration Error). Target < 0.05, warn > 0.15.
    pub oracle_ece: f64,
    /// Rolling count of oracle observations (target ≥ 30 for conformal intervals).
    pub oracle_n_observations: u64,
    /// Fraction of tasks that carried an `OracleSpec` (oracle coverage).
    pub oracle_coverage_rate: f64,
    /// Rolling oracle pass rate (last 200 observations).
    pub oracle_pass_rate: f64,
    /// P90 of calibration residuals (proxy for conformal interval width).
    pub oracle_residual_p90: f64,
    /// Current `PredictionBasis`: 0=Heuristic, 1=Bootstrap, 2=Conformal.
    pub oracle_calibration_basis: u8,
    /// Cumulative count of successfully resolved tasks (used for coverage rate denominator).
    pub oracle_tasks_total: u64,
    /// Cumulative count of resolved tasks that carried an `OracleSpec`.
    pub oracle_tasks_with_spec: u64,
    /// Current calibration source label: "measured", "`partial_fit`", or "`synthetic_priors`".
    /// Empty string when no calibration has run yet (renders all gauges as 0).
    pub calibration_source_label: String,
    /// Total shadow audit observations (counter).
    pub shadow_audit_total: u64,
    /// Shadow audit observations where primary and shadow disagreed (counter).
    pub shadow_audit_disagreements: u64,
    /// Domains currently in two-auditor AND-vote mode (gauge).
    pub shadow_audit_promoted_domains: usize,
    /// Rolling disagreement rate across all domains (gauge, 0–1).
    pub shadow_audit_disagreement_rate: f64,
    /// Active safety profile name: "development" | "production" | "strict" | "custom".
    pub safety_profile_name: String,
    /// Krum fault tolerance setting (f in Krum algorithm).
    pub safety_krum_fault_tolerance: u64,
    /// Diversity threshold setting.
    pub safety_diversity_threshold: f64,
    /// Shadow auditor enabled (1=yes, 0=no).
    pub safety_shadow_auditor_enabled: u8,
    /// Bivariate CG check required (1=yes, 0=no).
    pub safety_require_bivariate_cg: u8,
}

impl Default for MetricsState {
    fn default() -> Self {
        Self {
            n_eff_prior: 0.0,
            n_eff_actual: 0.0,
            epistemic_yield_ratio: 0.0,
            mapek_mode_collapse_count: 0,
            mapek_constrained_exploration_count: 0,
            phase15_quadrant_precision: 0,
            phase15_quadrant_coverage: 0,
            phase15_quadrant_complex: 0,
            phase15_quadrant_degenerate: 0,
            oracle_ece: 0.0,
            oracle_n_observations: 0,
            oracle_coverage_rate: 0.0,
            oracle_pass_rate: 0.0,
            oracle_residual_p90: 0.0,
            oracle_calibration_basis: 0,
            oracle_tasks_total: 0,
            oracle_tasks_with_spec: 0,
            calibration_source_label: String::new(),
            shadow_audit_total: 0,
            shadow_audit_disagreements: 0,
            shadow_audit_promoted_domains: 0,
            shadow_audit_disagreement_rate: 0.0,
            safety_profile_name: "development".to_string(),
            safety_krum_fault_tolerance: 0,
            safety_diversity_threshold: 0.0,
            safety_shadow_auditor_enabled: 0,
            safety_require_bivariate_cg: 0,
        }
    }
}

impl MetricsState {
    /// Render all metrics in Prometheus text exposition format.
    #[must_use]
    pub fn to_prometheus_text(&self) -> String {
        format!(
            "# HELP h2ai_n_eff_prior Effective independent adapters from calibration (cosine N_eff prior)\n\
             # TYPE h2ai_n_eff_prior gauge\n\
             h2ai_n_eff_prior {n_eff_prior}\n\
             # HELP h2ai_n_eff_actual Effective independent adapters from last task (cosine N_eff actual)\n\
             # TYPE h2ai_n_eff_actual gauge\n\
             h2ai_n_eff_actual {n_eff_actual}\n\
             # HELP h2ai_epistemic_yield_ratio n_eff_actual / N_requested from last task\n\
             # TYPE h2ai_epistemic_yield_ratio gauge\n\
             h2ai_epistemic_yield_ratio {epistemic_yield_ratio}\n\
             # HELP h2ai_mapek_interventions_total MAPE-K failure mode interventions\n\
             # TYPE h2ai_mapek_interventions_total counter\n\
             h2ai_mapek_interventions_total{{failure_mode=\"mode_collapse\"}} {mode_collapse}\n\
             h2ai_mapek_interventions_total{{failure_mode=\"constrained_exploration\"}} {constrained_exploration}\n\
             # HELP h2ai_phase15_task_quadrant_total Phase 1.5 task routing quadrant distribution\n\
             # TYPE h2ai_phase15_task_quadrant_total counter\n\
             h2ai_phase15_task_quadrant_total{{quadrant=\"precision\"}} {q_precision}\n\
             h2ai_phase15_task_quadrant_total{{quadrant=\"coverage\"}} {q_coverage}\n\
             h2ai_phase15_task_quadrant_total{{quadrant=\"complex\"}} {q_complex}\n\
             h2ai_phase15_task_quadrant_total{{quadrant=\"degenerate\"}} {q_degenerate}\n\
             # HELP h2ai_oracle_ece_gauge Current ECE (target < 0.05, warn > 0.15)\n\
             # TYPE h2ai_oracle_ece_gauge gauge\n\
             h2ai_oracle_ece_gauge {oracle_ece}\n\
             # HELP h2ai_oracle_n_observations_total Rolling oracle observation count\n\
             # TYPE h2ai_oracle_n_observations_total gauge\n\
             h2ai_oracle_n_observations_total {oracle_n_obs}\n\
             # HELP h2ai_oracle_coverage_rate Fraction of tasks with oracle spec\n\
             # TYPE h2ai_oracle_coverage_rate gauge\n\
             h2ai_oracle_coverage_rate {oracle_coverage}\n\
             # HELP h2ai_oracle_pass_rate Rolling oracle pass rate\n\
             # TYPE h2ai_oracle_pass_rate gauge\n\
             h2ai_oracle_pass_rate {oracle_pass}\n\
             # HELP h2ai_oracle_residual_p90 P90 of calibration residuals\n\
             # TYPE h2ai_oracle_residual_p90 gauge\n\
             h2ai_oracle_residual_p90 {oracle_p90}\n\
             # HELP h2ai_calibration_basis PredictionBasis (0=Heuristic 1=Bootstrap 2=Conformal)\n\
             # TYPE h2ai_calibration_basis gauge\n\
             h2ai_calibration_basis {oracle_basis}\n\
             # HELP h2ai_oracle_tasks_total Total successfully resolved tasks\n\
             # TYPE h2ai_oracle_tasks_total counter\n\
             h2ai_oracle_tasks_total {oracle_tasks_total}\n\
             # HELP h2ai_oracle_tasks_with_spec_total Tasks that carried an OracleSpec\n\
             # TYPE h2ai_oracle_tasks_with_spec_total counter\n\
             h2ai_oracle_tasks_with_spec_total {oracle_tasks_with_spec}\n\
             # HELP h2ai_calibration_source Current calibration source (1 = active variant)\n\
             # TYPE h2ai_calibration_source gauge\n\
             h2ai_calibration_source{{source=\"measured\"}} {cal_src_measured}\n\
             h2ai_calibration_source{{source=\"partial_fit\"}} {cal_src_partial}\n\
             h2ai_calibration_source{{source=\"synthetic_priors\"}} {cal_src_synthetic}\n\
             # HELP h2ai_shadow_audit_total Total Phase 4 shadow auditor observations\n\
             # TYPE h2ai_shadow_audit_total counter\n\
             h2ai_shadow_audit_total {shadow_total}\n\
             # HELP h2ai_shadow_audit_disagreements_total Shadow audit observations where auditors disagreed\n\
             # TYPE h2ai_shadow_audit_disagreements_total counter\n\
             h2ai_shadow_audit_disagreements_total {shadow_disagreements}\n\
             # HELP h2ai_shadow_audit_promoted_domains Domains in two-auditor AND-vote mode\n\
             # TYPE h2ai_shadow_audit_promoted_domains gauge\n\
             h2ai_shadow_audit_promoted_domains {shadow_promoted}\n\
             # HELP h2ai_shadow_audit_disagreement_rate Rolling disagreement rate across all domains (0-1)\n\
             # TYPE h2ai_shadow_audit_disagreement_rate gauge\n\
             h2ai_shadow_audit_disagreement_rate {shadow_rate}\n\
             # HELP h2ai_safety_profile Active safety profile (1 = this profile is active)\n\
             # TYPE h2ai_safety_profile gauge\n\
             h2ai_safety_profile{{profile=\"{profile_name}\"}} 1\n\
             # HELP h2ai_safety_krum_fault_tolerance Krum fault tolerance setting\n\
             # TYPE h2ai_safety_krum_fault_tolerance gauge\n\
             h2ai_safety_krum_fault_tolerance {krum_ft}\n\
             # HELP h2ai_safety_diversity_threshold Diversity threshold setting\n\
             # TYPE h2ai_safety_diversity_threshold gauge\n\
             h2ai_safety_diversity_threshold {diversity}\n\
             # HELP h2ai_safety_shadow_auditor_enabled Shadow auditor enabled (1=yes, 0=no)\n\
             # TYPE h2ai_safety_shadow_auditor_enabled gauge\n\
             h2ai_safety_shadow_auditor_enabled {shadow_enabled}\n\
             # HELP h2ai_safety_require_bivariate_cg Bivariate CG check required (1=yes, 0=no)\n\
             # TYPE h2ai_safety_require_bivariate_cg gauge\n\
             h2ai_safety_require_bivariate_cg {bivariate}\n",
            n_eff_prior = self.n_eff_prior,
            n_eff_actual = self.n_eff_actual,
            epistemic_yield_ratio = self.epistemic_yield_ratio,
            mode_collapse = self.mapek_mode_collapse_count,
            constrained_exploration = self.mapek_constrained_exploration_count,
            q_precision = self.phase15_quadrant_precision,
            q_coverage = self.phase15_quadrant_coverage,
            q_complex = self.phase15_quadrant_complex,
            q_degenerate = self.phase15_quadrant_degenerate,
            oracle_ece = self.oracle_ece,
            oracle_n_obs = self.oracle_n_observations,
            oracle_coverage = self.oracle_coverage_rate,
            oracle_pass = self.oracle_pass_rate,
            oracle_p90 = self.oracle_residual_p90,
            oracle_basis = self.oracle_calibration_basis,
            oracle_tasks_total = self.oracle_tasks_total,
            oracle_tasks_with_spec = self.oracle_tasks_with_spec,
            cal_src_measured = i32::from(self.calibration_source_label == "measured"),
            cal_src_partial = i32::from(self.calibration_source_label == "partial_fit"),
            cal_src_synthetic = i32::from(self.calibration_source_label == "synthetic_priors"),
            shadow_total = self.shadow_audit_total,
            shadow_disagreements = self.shadow_audit_disagreements,
            shadow_promoted = self.shadow_audit_promoted_domains,
            shadow_rate = self.shadow_audit_disagreement_rate,
            profile_name = self.safety_profile_name,
            krum_ft = self.safety_krum_fault_tolerance,
            diversity = self.safety_diversity_threshold,
            shadow_enabled = self.safety_shadow_auditor_enabled,
            bivariate = self.safety_require_bivariate_cg,
        )
    }
}
