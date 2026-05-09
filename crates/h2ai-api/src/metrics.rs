/// In-memory Prometheus-format metric state for the /metrics endpoint.
///
/// Shared via `Arc<RwLock<MetricsState>>` in AppState.
#[derive(Debug, Default)]
pub struct MetricsState {
    /// Pool-level N_eff from last calibration (gauge).
    pub n_eff_prior: f64,
    /// Task-level N_eff from last EpistemicYieldEvent (gauge).
    pub n_eff_actual: f64,
    /// yield_ratio from last EpistemicYieldEvent (gauge).
    pub epistemic_yield_ratio: f64,
    /// Cumulative ModeCollapse interventions (counter).
    pub mapek_mode_collapse_count: u64,
    /// Cumulative ConstrainedExploration interventions (counter).
    pub mapek_constrained_exploration_count: u64,
    /// Phase 1.5 routing quadrant distribution (counters by quadrant).
    /// Used to validate θ_tcc and θ_neff thresholds during shadow_mode monitoring.
    pub phase15_quadrant_precision: u64,
    pub phase15_quadrant_coverage: u64,
    pub phase15_quadrant_complex: u64,
    pub phase15_quadrant_degenerate: u64,
    /// Current ECE (Expected Calibration Error). Target < 0.05, warn > 0.15.
    pub oracle_ece: f64,
    /// Rolling count of oracle observations (target ≥ 30 for conformal intervals).
    pub oracle_n_observations: u64,
    /// Fraction of tasks that carried an OracleSpec (oracle coverage).
    pub oracle_coverage_rate: f64,
    /// Rolling oracle pass rate (last 200 observations).
    pub oracle_pass_rate: f64,
    /// P90 of calibration residuals (proxy for conformal interval width).
    pub oracle_residual_p90: f64,
    /// Current PredictionBasis: 0=Heuristic, 1=Bootstrap, 2=Conformal.
    pub oracle_calibration_basis: u8,
    /// Cumulative count of successfully resolved tasks (used for coverage rate denominator).
    pub oracle_tasks_total: u64,
    /// Cumulative count of resolved tasks that carried an OracleSpec.
    pub oracle_tasks_with_spec: u64,
    /// Current calibration source label: "measured", "partial_fit", or "synthetic_priors".
    /// Empty string when no calibration has run yet (renders all gauges as 0).
    pub calibration_source_label: String,
}

impl MetricsState {
    /// Render all metrics in Prometheus text exposition format.
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
             h2ai_calibration_source{{source=\"synthetic_priors\"}} {cal_src_synthetic}\n",
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
            cal_src_measured = if self.calibration_source_label == "measured" { 1 } else { 0 },
            cal_src_partial = if self.calibration_source_label == "partial_fit" { 1 } else { 0 },
            cal_src_synthetic = if self.calibration_source_label == "synthetic_priors" { 1 } else { 0 },
        )
    }
}

#[cfg(test)]
mod metrics_tests {
    use super::*;

    #[test]
    fn metrics_state_formats_prometheus_text() {
        let m = MetricsState {
            n_eff_prior: 2.5,
            n_eff_actual: 2.1,
            epistemic_yield_ratio: 0.7,
            mapek_mode_collapse_count: 3,
            mapek_constrained_exploration_count: 1,
            phase15_quadrant_precision: 10,
            phase15_quadrant_coverage: 42,
            phase15_quadrant_complex: 5,
            phase15_quadrant_degenerate: 1,
            oracle_ece: 0.0,
            oracle_n_observations: 0,
            oracle_coverage_rate: 0.0,
            oracle_pass_rate: 0.0,
            oracle_residual_p90: 0.0,
            oracle_calibration_basis: 0,
            oracle_tasks_total: 0,
            oracle_tasks_with_spec: 0,
            calibration_source_label: String::new(),
        };
        let text = m.to_prometheus_text();
        assert!(text.contains("h2ai_n_eff_prior 2.5"));
        assert!(text.contains("h2ai_n_eff_actual 2.1"));
        assert!(text.contains("h2ai_epistemic_yield_ratio 0.7"));
        assert!(text.contains(r#"h2ai_mapek_interventions_total{failure_mode="mode_collapse"} 3"#));
        assert!(text.contains(
            r#"h2ai_mapek_interventions_total{failure_mode="constrained_exploration"} 1"#
        ));
        assert!(text.contains(r#"h2ai_phase15_task_quadrant_total{quadrant="precision"} 10"#));
        assert!(text.contains(r#"h2ai_phase15_task_quadrant_total{quadrant="coverage"} 42"#));
        assert!(text.contains(r#"h2ai_phase15_task_quadrant_total{quadrant="complex"} 5"#));
        assert!(text.contains(r#"h2ai_phase15_task_quadrant_total{quadrant="degenerate"} 1"#));
    }

    #[test]
    fn metrics_state_renders_calibration_source_measured() {
        let m = MetricsState {
            calibration_source_label: "measured".to_string(),
            ..Default::default()
        };
        let text = m.to_prometheus_text();
        assert!(text.contains(r#"h2ai_calibration_source{source="measured"} 1"#));
        assert!(text.contains(r#"h2ai_calibration_source{source="partial_fit"} 0"#));
        assert!(text.contains(r#"h2ai_calibration_source{source="synthetic_priors"} 0"#));
    }

    #[test]
    fn metrics_state_renders_calibration_source_synthetic() {
        let m = MetricsState {
            calibration_source_label: "synthetic_priors".to_string(),
            ..Default::default()
        };
        let text = m.to_prometheus_text();
        assert!(text.contains(r#"h2ai_calibration_source{source="measured"} 0"#));
        assert!(text.contains(r#"h2ai_calibration_source{source="synthetic_priors"} 1"#));
    }

    #[test]
    fn metrics_state_renders_calibration_source_partial_fit() {
        let m = MetricsState {
            calibration_source_label: "partial_fit".to_string(),
            ..Default::default()
        };
        let text = m.to_prometheus_text();
        assert!(text.contains(r#"h2ai_calibration_source{source="measured"} 0"#));
        assert!(text.contains(r#"h2ai_calibration_source{source="partial_fit"} 1"#));
        assert!(text.contains(r#"h2ai_calibration_source{source="synthetic_priors"} 0"#));
    }

    #[test]
    fn oracle_metrics_render_in_prometheus_text() {
        let m = MetricsState {
            n_eff_prior: 2.5,
            n_eff_actual: 2.1,
            epistemic_yield_ratio: 0.7,
            mapek_mode_collapse_count: 3,
            mapek_constrained_exploration_count: 1,
            phase15_quadrant_precision: 10,
            phase15_quadrant_coverage: 42,
            phase15_quadrant_complex: 5,
            phase15_quadrant_degenerate: 1,
            oracle_ece: 0.08,
            oracle_n_observations: 45,
            oracle_coverage_rate: 0.6,
            oracle_pass_rate: 0.75,
            oracle_residual_p90: 0.35,
            oracle_calibration_basis: 2,
            oracle_tasks_total: 0,
            oracle_tasks_with_spec: 0,
            calibration_source_label: String::new(),
        };
        let text = m.to_prometheus_text();
        assert!(text.contains("h2ai_oracle_ece_gauge 0.08"));
        assert!(text.contains("h2ai_oracle_n_observations_total 45"));
        assert!(text.contains("h2ai_oracle_coverage_rate 0.6"));
        assert!(text.contains("h2ai_oracle_pass_rate 0.75"));
        assert!(text.contains("h2ai_oracle_residual_p90 0.35"));
        assert!(text.contains("h2ai_calibration_basis 2"));
        assert!(text.contains("h2ai_oracle_tasks_total 0"));
        assert!(text.contains("h2ai_oracle_tasks_with_spec_total 0"));
    }
}
