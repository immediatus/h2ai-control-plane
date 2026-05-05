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
             h2ai_mapek_interventions_total{{failure_mode=\"constrained_exploration\"}} {constrained_exploration}\n",
            n_eff_prior = self.n_eff_prior,
            n_eff_actual = self.n_eff_actual,
            epistemic_yield_ratio = self.epistemic_yield_ratio,
            mode_collapse = self.mapek_mode_collapse_count,
            constrained_exploration = self.mapek_constrained_exploration_count,
        )
    }
}

#[cfg(test)]
mod metrics_tests {
    use super::*;

    #[test]
    fn metrics_state_formats_prometheus_text() {
        let mut m = MetricsState::default();
        m.n_eff_prior = 2.5;
        m.n_eff_actual = 2.1;
        m.epistemic_yield_ratio = 0.7;
        m.mapek_mode_collapse_count = 3;
        m.mapek_constrained_exploration_count = 1;
        let text = m.to_prometheus_text();
        assert!(text.contains("h2ai_n_eff_prior 2.5"));
        assert!(text.contains("h2ai_n_eff_actual 2.1"));
        assert!(text.contains("h2ai_epistemic_yield_ratio 0.7"));
        assert!(text.contains(r#"h2ai_mapek_interventions_total{failure_mode="mode_collapse"} 3"#));
        assert!(text.contains(
            r#"h2ai_mapek_interventions_total{failure_mode="constrained_exploration"} 1"#
        ));
    }
}
