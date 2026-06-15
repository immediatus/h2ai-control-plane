use h2ai_types::events::BranchPrunedEvent;
use std::collections::{HashMap, HashSet};

/// A constraint that was passing at `passed_wave` but failed at `failed_wave`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DivergenceEvent {
    pub constraint_id: String,
    /// Last wave at which this constraint was NOT violated.
    pub passed_wave: u32,
    /// Wave at which it appeared as a violation for the first time after `passed_wave`.
    pub failed_wave: u32,
}

/// Returns a map of `constraint_id → Vec<wave>` listing all waves at which each
/// constraint appeared as a violation in a pruned event.
pub fn wave_violation_history_from_pruned(
    pruned: &[BranchPrunedEvent],
) -> HashMap<String, Vec<u32>> {
    let mut history: HashMap<String, Vec<u32>> = HashMap::new();
    for event in pruned {
        for viol in &event.violated_constraints {
            history
                .entry(viol.constraint_id.clone())
                .or_default()
                .push(event.retry_count);
        }
    }
    history
}

/// Returns regressions: constraints that appear as violations at wave N+1 but
/// did NOT appear at any earlier wave (new failure = divergence from prior passing state).
pub fn divergence_events_from_pruned(pruned: &[BranchPrunedEvent]) -> Vec<DivergenceEvent> {
    let history = wave_violation_history_from_pruned(pruned);

    let mut all_waves: Vec<u32> = pruned.iter().map(|e| e.retry_count).collect();
    all_waves.sort();
    all_waves.dedup();

    if all_waves.len() < 2 {
        return vec![];
    }

    let mut divergences: Vec<DivergenceEvent> = Vec::new();
    for window in all_waves.windows(2) {
        let w0 = window[0];
        let w1 = window[1];

        let violated_at_w0: HashSet<&str> = pruned
            .iter()
            .filter(|e| e.retry_count == w0)
            .flat_map(|e| {
                e.violated_constraints
                    .iter()
                    .map(|v| v.constraint_id.as_str())
            })
            .collect();

        let violated_at_w1: HashSet<&str> = pruned
            .iter()
            .filter(|e| e.retry_count == w1)
            .flat_map(|e| {
                e.violated_constraints
                    .iter()
                    .map(|v| v.constraint_id.as_str())
            })
            .collect();

        for cid in violated_at_w1.difference(&violated_at_w0) {
            // Only flag if w1 is its first appearance in all waves
            let first_seen = history
                .get(*cid)
                .and_then(|waves| waves.iter().copied().min())
                .unwrap_or(w1);
            if first_seen == w1 {
                divergences.push(DivergenceEvent {
                    constraint_id: cid.to_string(),
                    passed_wave: w0,
                    failed_wave: w1,
                });
            }
        }
    }
    divergences
}

/// Produces a balancing instruction injected into the DPPM merge-step prompt.
/// Returns an empty string when there is nothing to report.
pub fn build_balancing_instruction(
    oscillating_pairs: &[(String, String)],
    divergences: &[DivergenceEvent],
) -> String {
    if oscillating_pairs.is_empty() && divergences.is_empty() {
        return String::new();
    }

    let mut out = String::with_capacity(512);
    out.push_str("--- META-OBSERVER FINDINGS ---\n");

    if !oscillating_pairs.is_empty() {
        out.push_str("OSCILLATION DETECTED — the following constraint pairs cycle:\n");
        for (a, b) in oscillating_pairs {
            out.push_str(&format!(
                "  • {a} ↔ {b}: satisfying one broke the other across repair waves.\n"
            ));
        }
        out.push_str(
            "Resolution mandate: the unified proposal MUST satisfy BOTH simultaneously.\n\n",
        );
    }

    if !divergences.is_empty() {
        out.push_str(
            "REGRESSION DETECTED — the following constraints newly failed after prior waves:\n",
        );
        for d in divergences {
            out.push_str(&format!(
                "  • {} passed at wave {} but failed at wave {} — do not lose this constraint.\n",
                d.constraint_id, d.passed_wave, d.failed_wave
            ));
        }
        out.push('\n');
    }

    out.push_str("--- END META-OBSERVER FINDINGS ---");
    out
}

/// Filters `prior_instruction` to lines that mention at least one constraint
/// in `cluster_ids`. Returns empty string if none match.
pub fn sharpen_balancing_instruction(prior_instruction: &str, cluster_ids: &[String]) -> String {
    if prior_instruction.is_empty() || cluster_ids.is_empty() {
        return String::new();
    }

    let relevant: Vec<&str> = prior_instruction
        .lines()
        .filter(|line| cluster_ids.iter().any(|id| line.contains(id.as_str())))
        .collect();

    if relevant.is_empty() {
        return String::new();
    }

    relevant.join("\n")
}
