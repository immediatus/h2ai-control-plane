use chrono::Utc;
use h2ai_constraints::ambiguity::jaccard;
use h2ai_types::config::TopologyKind;
use h2ai_types::events::{BranchPrunedEvent, TaskFailedEvent, ZeroSurvivalEvent};
use h2ai_types::sizing::MultiplicationConditionFailure;
use std::collections::HashMap;

pub enum RetryAction {
    Retry(TopologyKind),
    /// Majority of pruned reasons indicate hallucination — reduce τ on next attempt
    /// to push explorers toward more grounded, less speculative outputs.
    RetryWithTauReduction {
        topology: TopologyKind,
        tau_factor: f64,
    },
    /// Structured constraint violations found — provide targeted remediation hints.
    /// Legacy path kept for call sites that pre-date RepairTarget.
    RetryWithHints {
        topology: TopologyKind,
        /// One hint per violated Hard constraint that has a `remediation_hint`.
        hints: Vec<String>,
    },
    /// Structured repair targets with per-constraint description, hint, and dynamic verifier reason.
    /// Produced by RetryPolicy::decide() when ConstraintViolation carries rich metadata.
    RetryWithTargets {
        topology: TopologyKind,
        targets: Vec<crate::repair::RepairTarget>,
    },
    Fail(TaskFailedEvent),
}

pub struct RetryPolicy;

/// Pareto frontier order: Ensemble → `HierarchicalTree` → `TeamSwarmHybrid`.
const FRONTIER: &[fn() -> TopologyKind] = &[
    || TopologyKind::Ensemble,
    || TopologyKind::HierarchicalTree {
        branching_factor: None,
    },
    || TopologyKind::TeamSwarmHybrid,
];

/// Keywords in pruned proposal reasons that suggest explorers were too creative/speculative.
const HALLUCINATION_SIGNALS: &[&str] = &[
    "hallucination",
    "hallucinated",
    "fabricated",
    "fabrication",
    "invented",
    "made up",
    "does not exist",
    "incorrect",
    "inaccurate",
    "not factual",
    "fictional",
    "false claim",
];

/// Min pairwise Jaccard across all reason pairs. Returns 1.0 when fewer than two reasons.
fn min_pairwise_jaccard(reasons: &[String]) -> f64 {
    let mut min = 1.0_f64;
    for i in 0..reasons.len() {
        for j in (i + 1)..reasons.len() {
            min = min.min(jaccard(&reasons[i], &reasons[j]));
        }
    }
    min
}

/// Mean pairwise Jaccard across all reason pairs. Returns 1.0 when fewer than two reasons.
fn mean_pairwise_jaccard(reasons: &[String]) -> f64 {
    if reasons.len() < 2 {
        return 1.0;
    }
    let mut sum = 0.0_f64;
    let mut count = 0usize;
    for i in 0..reasons.len() {
        for j in (i + 1)..reasons.len() {
            sum += jaccard(&reasons[i], &reasons[j]);
            count += 1;
        }
    }
    if count == 0 {
        1.0
    } else {
        sum / count as f64
    }
}

/// Return up to `k` unique (score, reason) pairs from `pairs`, sorted score-descending.
///
/// Uniqueness is enforced by Jaccard deduplication: a candidate is dropped when it exceeds
/// `dedup_threshold` similarity with any already-selected reason. This prevents identical
/// failure signals from bloating the repair prompt while preserving genuinely distinct diagnoses.
pub(crate) fn top_k_unique_reasons(
    pairs: &[(f64, String)],
    k: usize,
    dedup_threshold: f64,
) -> Vec<(f64, String)> {
    let mut sorted = pairs.to_vec();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut result: Vec<(f64, String)> = Vec::new();
    for (score, reason) in sorted {
        let is_dup = result
            .iter()
            .any(|(_, r)| jaccard(r, &reason) > dedup_threshold);
        if !is_dup {
            result.push((score, reason));
        }
        if result.len() >= k {
            break;
        }
    }
    result
}

impl RetryPolicy {
    pub fn decide(
        event: &ZeroSurvivalEvent,
        tried_topologies: &[TopologyKind],
        pruned_events: Vec<BranchPrunedEvent>,
        tau_values_tried: Vec<Vec<f64>>,
        multiplication_failure: Option<MultiplicationConditionFailure>,
    ) -> RetryAction {
        // Aggregate per-constraint data from all pruned events.
        // For each hard-failing constraint: collect (score, verifier_reason, constraint_description, remediation_hint).
        struct ConstraintEntry {
            constraint_description: String,
            remediation_hint: Option<String>,
            criteria_pass: Option<String>,
            // Pairs of (score, verifier_reason) from each proposal that violated this constraint.
            score_reason_pairs: Vec<(f64, String)>,
        }
        let mut per_constraint: HashMap<String, ConstraintEntry> = HashMap::new();

        for pruned in &pruned_events {
            for v in pruned
                .violated_constraints
                .iter()
                .filter(|v| v.severity_label == "Hard")
            {
                let entry = per_constraint
                    .entry(v.constraint_id.clone())
                    .or_insert_with(|| ConstraintEntry {
                        constraint_description: v.constraint_description.clone(),
                        remediation_hint: v.remediation_hint.clone(),
                        criteria_pass: v.criteria_pass.clone(),
                        score_reason_pairs: Vec::new(),
                    });
                if let Some(ref r) = v.verifier_reason {
                    if !r.is_empty() {
                        entry.score_reason_pairs.push((v.score, r.clone()));
                    }
                }
            }
        }

        let has_violations = !per_constraint.is_empty();

        // Build RepairTargets from aggregated per-constraint data.
        let targets: Vec<crate::repair::RepairTarget> = per_constraint
            .into_iter()
            .map(|(constraint_id, entry)| {
                let reasons: Vec<String> = entry
                    .score_reason_pairs
                    .iter()
                    .map(|(_, r)| r.clone())
                    .collect();

                // Log structural divergence when min Jaccard falls below half the mean.
                // Self-calibrates to domain vocabulary — technical text naturally has low
                // absolute Jaccard but may still carry consistent failure signal.
                if reasons.len() >= 3 {
                    let min_j = min_pairwise_jaccard(&reasons);
                    let mean_j = mean_pairwise_jaccard(&reasons);
                    if min_j < mean_j * 0.5 {
                        tracing::warn!(
                            target: "h2ai.retry",
                            constraint_id = %constraint_id,
                            min_jaccard = min_j,
                            mean_jaccard = mean_j,
                            reason_count = reasons.len(),
                            "verifier reasons show structural divergence — \
                            escalating to top-k breadth selection"
                        );
                    }
                }

                // Progressive signal escalation: k grows with retry_count so repair prompts
                // receive more diverse failure context as single-reason repair keeps failing.
                // k = retry_count + 1: wave 0 → 1 reason, wave 1 → 2, wave 2 → 3, …
                let k = event.retry_count as usize + 1;
                crate::repair::RepairTarget {
                    constraint_id,
                    constraint_description: entry.constraint_description,
                    remediation_hint: entry.remediation_hint,
                    criteria_pass: entry.criteria_pass,
                    verifier_reasons: top_k_unique_reasons(&entry.score_reason_pairs, k, 0.7),
                }
            })
            .collect();

        // Fall back to hallucination-signal keyword scan only when no structured violations found.
        let hallucination_count = if has_violations {
            0
        } else {
            pruned_events
                .iter()
                .filter(|e| {
                    let r = e.reason.to_lowercase();
                    HALLUCINATION_SIGNALS.iter().any(|sig| r.contains(sig))
                })
                .count()
        };

        let reduce_tau = !has_violations
            && !pruned_events.is_empty()
            && hallucination_count * 2 >= pruned_events.len();

        for make_topology in FRONTIER {
            let candidate = make_topology();
            if !Self::has_tried(tried_topologies, &candidate) {
                return if has_violations {
                    RetryAction::RetryWithTargets {
                        topology: candidate,
                        targets,
                    }
                } else if reduce_tau {
                    RetryAction::RetryWithTauReduction {
                        topology: candidate,
                        tau_factor: 0.7,
                    }
                } else {
                    RetryAction::Retry(candidate)
                };
            }
        }

        RetryAction::Fail(TaskFailedEvent {
            task_id: event.task_id.clone(),
            pruned_events,
            topologies_tried: tried_topologies.to_vec(),
            tau_values_tried,
            multiplication_condition_failure: multiplication_failure,
            timestamp: Utc::now(),
        })
    }

    fn has_tried(tried: &[TopologyKind], candidate: &TopologyKind) -> bool {
        tried.iter().any(|t| Self::same_variant(t, candidate))
    }

    const fn same_variant(a: &TopologyKind, b: &TopologyKind) -> bool {
        matches!(
            (a, b),
            (TopologyKind::Ensemble, TopologyKind::Ensemble)
                | (
                    TopologyKind::HierarchicalTree { .. },
                    TopologyKind::HierarchicalTree { .. }
                )
                | (TopologyKind::TeamSwarmHybrid, TopologyKind::TeamSwarmHybrid)
        )
    }
}
