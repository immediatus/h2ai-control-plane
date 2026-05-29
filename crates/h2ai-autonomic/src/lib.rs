//! MAPE-K control loop for H2AI task orchestration.
//!
//! - [`checker::MultiplicationChecker`] — pre-flight USL physics validation
//! - [`planner::TopologyPlanner`] — topology selection and explorer config derivation
//! - [`merger::MergeEngine`] — semilattice + BFT proposal resolution
//! - [`retry::RetryPolicy`] — Pareto-frontier topology retry on zero survival

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
