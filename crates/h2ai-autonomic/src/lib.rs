//! MAPE-K control loop for H2AI task orchestration.
//!
//! - [`checker::MultiplicationChecker`] — pre-flight USL physics validation
//! - [`planner::TopologyPlanner`] — topology selection and explorer config derivation
//! - [`merger::MergeEngine`] — semilattice + BFT proposal resolution
//! - [`retry::RetryPolicy`] — Pareto-frontier topology retry on zero survival
//! - [`audit_channel::AuditChannelBuilder`] — builds Zone 3 audit findings from
//!   `ConstraintViolation` IR; Zone 3 contains only `constraint_id` and
//!   `remediation_hint`, never raw proposal text.
//! - [`calibration::CalibrationHarness`] — drives USL calibration rounds; also
//!   exports `compute_conflict_rate` and `beta_from_merge_spans` pure helpers.
//! - [`coherence_probe::CoherenceProbe`] — `ExampleBased` / `SelfConsistency`
//!   probe modes; returns `ProbeResult` with coherence score.
//! - [`complexity_probe::ComplexityProbe`] — rates task complexity 1–5 before
//!   dispatch; 2 is the conservative safe default on LLM failure.
//! - [`drift`] — `DriftMonitor` combining DDM fast-layer (Gama et al. 2004) and
//!   BOCPD structural shift detection for calibration drift.
//! - [`epistemic`] — `compute_n_eff_cosine` (effective independent adapters via
//!   cosine matrix eigenvalues), `classify_failure_mode`, `ConstraintRepairPlan`.
//! - [`knowledge_gap`] — `detect_cold_checks` flags under-exercised constraint
//!   checks; `build_gap_queries` produces retrieval queries from gap evidence.
//! - [`repair`] — `PartialPass` and `select_orthogonal_partials` for
//!   character-budget-bounded partial proposal repair on pruned branches.
//! - [`retry_accumulator::RetryAccumulator`] — per-task leaky accumulator for
//!   per-criterion violation rates across MAPE-K retries (EWA update rule).
//! - [`spec_repair::SpecRepairAdvisor`] — coherence-probe-driven spec rewriting;
//!   returns `RepairOutcome::Repaired` (with accepted rewrite) or `Unchanged`.

pub mod audit_channel;
pub mod calibration;
pub mod checker;
pub mod coherence_probe;
pub mod complexity_probe;
pub mod drift;
pub mod epistemic;
pub mod knowledge_gap;
pub mod merger;
pub mod planner;
pub mod repair;
pub mod retry;
pub mod retry_accumulator;
pub mod spec_repair;
