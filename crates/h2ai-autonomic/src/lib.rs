//! MAPE-K control loop for H2AI task orchestration.
//!
//! - [`checker::MultiplicationChecker`] — pre-flight USL physics validation
//! - [`planner::TopologyPlanner`] — topology selection and explorer config derivation
//! - [`merger::MergeEngine`] — semilattice + BFT proposal resolution
//! - [`retry::RetryPolicy`] — Pareto-frontier topology retry on zero survival

pub mod audit_channel;
pub mod calibration;
pub mod checker;
pub mod epistemic;
pub mod merger;
pub mod planner;
pub mod repair;
pub mod retry;
pub mod retry_accumulator;
