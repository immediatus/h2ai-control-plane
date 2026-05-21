#![allow(clippy::cast_precision_loss)]
use h2ai_types::events::ConstraintViolation;
use h2ai_types::sizing::OspConfig;
use std::collections::HashMap;

const CONCORDANCE_EPSILON: f64 = 1e-5;

/// Builds Zone 3 (Audit Findings) from structured `ConstraintViolation` IR.
///
/// Zone 3 contains only `constraint_id` and `remediation_hint` — never raw proposal text.
/// This prevents the Pink Elephant attention trap (spec §A.1): transformer attention heads
/// never compute representations for hallucinated content from failed proposals.
pub struct AuditChannelBuilder;

impl AuditChannelBuilder {
    /// Adaptive concordance threshold `τ(N_f)` from Hoeffding's inequality.
    ///
    /// τ = clamp(0.5 + 0.5·√(−ln(α) / (`2·N_f`)), 0.5, 1.0)
    /// At α=0.1: τ(1)=1.0, τ(2)≈0.96, τ(5)≈0.77, τ(10)≈0.66.
    #[must_use]
    pub fn adaptive_threshold(n_f: usize, alpha: f64) -> f64 {
        if n_f == 0 {
            return 1.0;
        }
        let inner = -alpha.ln() / (2.0 * n_f as f64);
        0.5f64.mul_add(inner.sqrt(), 0.5).clamp(0.5, 1.0)
    }

    /// Build Zone 3 audit-findings text from violation IR.
    ///
    /// `violations`: flat slice of `ConstraintViolation` from all `N_f` failed proposals.
    ///   These are structured IR records — no raw proposal output present.
    /// `n_f`: number of failed proposals (denominator for concordance rate).
    /// `n_v`: number of valid proposals (gravity-well guard).
    /// `retry_count`: current retry index.
    ///
    /// Returns `None` when injection conditions are not met.
    /// Returns `Some(text)` with content drawn only from `constraint_id` + `remediation_hint`.
    #[must_use]
    pub fn build_zone3(
        violations: &[ConstraintViolation],
        n_f: usize,
        n_v: usize,
        retry_count: u32,
        config: &OspConfig,
    ) -> Option<String> {
        if n_f == 0 || violations.is_empty() {
            return None;
        }
        if n_v > config.max_n_v_for_zone3 {
            return None;
        }
        if n_f == 1 && retry_count > 3 {
            return None;
        }

        let threshold = Self::adaptive_threshold(n_f, config.concordance_alpha);
        // n_f >= 1 guaranteed by the early return above; epsilon provides float comparison tolerance.
        let denominator = n_f as f64;

        // Per-criterion violation counts and best hint
        let mut counts: HashMap<&str, usize> = HashMap::new();
        let mut hints: HashMap<&str, &str> = HashMap::new();
        for v in violations {
            *counts.entry(v.constraint_id.as_str()).or_insert(0) += 1;
            if let Some(h) = v.remediation_hint.as_deref() {
                hints.entry(v.constraint_id.as_str()).or_insert(h);
            }
        }

        let mut concordant: Vec<(&str, f64, Option<&str>)> = counts
            .iter()
            .filter_map(|(&cid, &count)| {
                let c_k = count as f64 / denominator;
                if c_k + CONCORDANCE_EPSILON >= threshold {
                    Some((cid, c_k, hints.get(cid).copied()))
                } else {
                    None
                }
            })
            .collect();

        if concordant.is_empty() {
            return None;
        }

        // Sort descending by concordance for deterministic, highest-signal-first output
        concordant.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Positive evaluation-metadata framing — never prohibition
        let mut lines = vec![
            "AUDIT FINDINGS: The following criteria showed consistent difficulty in prior drafts:"
                .to_string(),
        ];
        for (cid, c_k, hint) in &concordant {
            lines.push(format!(
                "- {}: observed in {:.0}% of drafts",
                cid,
                c_k * 100.0
            ));
            if let Some(h) = hint {
                lines.push(format!("  Guidance: {h}"));
            }
        }

        Some(lines.join("\n"))
    }
}
