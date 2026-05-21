use serde::{Deserialize, Serialize};

/// Source of the epistemic probe task.
///
/// `Same` uses the production prompt (truncated to the probe token budget).
/// `Synthetic` compiles a deterministic task from constraint YAML fields
/// (`criteria.pass` + `predicates`), enabling stationary k-regression
/// independent of user payload entropy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProbeSource {
    Same,
    Synthetic,
}

/// Three-state circuit breaker for the `LlmJudge` auditor.
///
/// `Closed` — `LlmJudge` active, normal path.
/// `Open` — `LlmJudge` bypassed; all verification routes through deterministic
///   `PredicateChecker` implementations. β₀ effectively spikes, capping `N_max`.
/// `HalfOpen` — recovery probe in progress; exactly one thread holds the
///   probe lease (via NATS KV CAS) and may call `LlmJudge`. All others fall back
///   to `PredicateChecker`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditorCircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Per-(`adapter_profile`, `constraint_id`) calibration snapshot stored in NATS KV.
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
    /// Ring buffer of (`N_useful`, `N_max`, `unix_minutes`). Last 100 waves.
    /// `unix_minutes` (u32 covers ~8000 years) is used instead of millis to keep
    /// the tuple compact (6 bytes/entry vs 10). Used by k-regression to exclude
    /// Open-interval entries where yields were measured against `PredicateChecker`.
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
