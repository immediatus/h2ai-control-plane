use serde::{Deserialize, Serialize};

/// Source of the epistemic probe task.
///
/// `Same` uses the production prompt (truncated to the probe token budget).
/// `Synthetic` compiles a deterministic task from constraint YAML fields
/// (`criteria.pass` + `predicates`), enabling stationary k-regression
/// independent of user payload entropy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProbeSource {
    Same,
    Synthetic,
}

/// Three-state circuit breaker for the LlmJudge auditor.
///
/// `Closed` — LlmJudge active, normal path.
/// `Open` — LlmJudge bypassed; all verification routes through deterministic
///   `PredicateChecker` implementations. β₀ effectively spikes, capping N_max.
/// `HalfOpen` — recovery probe in progress; exactly one thread holds the
///   probe lease (via NATS KV CAS) and may call LlmJudge. All others fall back
///   to `PredicateChecker`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuditorCircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Per-(adapter_profile, constraint_id) calibration snapshot stored in NATS KV.
///
/// Key format: `calibration.{adapter_profile}` for the aggregate record,
/// or `calibration.{adapter_profile}.{constraint_id}` for per-constraint records.
/// The aggregate record uses `constraint_id: None`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationRecord {
    pub adapter_profile: String,
    pub constraint_id: Option<String>,
    pub alpha: f32,
    pub alpha_measured: f32,
    pub beta_0: f32,
    pub k: f32,
    /// Ring buffer of (N_useful, N_max, unix_minutes). Last 100 waves.
    /// unix_minutes (u32 covers ~8000 years) is used instead of millis to keep
    /// the tuple compact (6 bytes/entry vs 10). Used by k-regression to exclude
    /// Open-interval entries where yields were measured against PredicateChecker.
    pub n_useful_history: Vec<(u8, u8, u32)>,
    pub probe_source: ProbeSource,
    pub fingerprint: Option<Vec<f32>>,
    pub circuit_state: AuditorCircuitState,
}

/// Auditor circuit breaker health snapshot.
///
/// Stored in NATS KV under `auditor.health.{adapter_profile}`.
/// `last_probe_cg` is the CG score (range [0.0, 1.0]) from the most recent
/// inverted probe run — 0.0 means the auditor passed the flawed proposal
/// (catastrophic failure), 1.0 means it correctly rejected all flawed proposals.
/// `tripped_at` is unix milliseconds since epoch.
/// `recovery_probe_count` is the number of successful half-open probes that
/// contributed to the current Open→HalfOpen→Closed recovery cycle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditorHealth {
    pub state: AuditorCircuitState,
    pub last_probe_cg: f32,
    /// Unix milliseconds since epoch, or None if circuit has never tripped.
    pub tripped_at: Option<u64>,
    pub recovery_probe_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_record_round_trips_json() {
        let record = CalibrationRecord {
            adapter_profile: "capable".to_string(),
            constraint_id: Some("CONSTRAINT-005".to_string()),
            alpha: 0.12,
            alpha_measured: 0.08,
            beta_0: 0.039,
            k: 2.0,
            n_useful_history: vec![(3, 5, 1_000_000), (4, 5, 1_000_001)],
            probe_source: ProbeSource::Same,
            fingerprint: None,
            circuit_state: AuditorCircuitState::Closed,
        };
        let json = serde_json::to_string(&record).unwrap();
        let back: CalibrationRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.adapter_profile, "capable");
        assert_eq!(back.constraint_id, Some("CONSTRAINT-005".to_string()));
        assert!((back.beta_0 - 0.039).abs() < 1e-6);
        assert_eq!(back.n_useful_history.len(), 2);
        assert_eq!(back.n_useful_history[0], (3, 5, 1_000_000));
        assert_eq!(back.circuit_state, AuditorCircuitState::Closed);
    }

    #[test]
    fn auditor_health_round_trips_json() {
        let health = AuditorHealth {
            state: AuditorCircuitState::HalfOpen,
            last_probe_cg: 0.65,
            tripped_at: None,
            recovery_probe_count: 2,
        };
        let json = serde_json::to_string(&health).unwrap();
        let back: AuditorHealth = serde_json::from_str(&json).unwrap();
        assert_eq!(back.state, AuditorCircuitState::HalfOpen);
        assert!((back.last_probe_cg - 0.65).abs() < 1e-6);
        assert_eq!(back.recovery_probe_count, 2);
    }
}
