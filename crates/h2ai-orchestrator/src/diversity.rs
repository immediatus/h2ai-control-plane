use h2ai_constraints::types::ComplianceResult;
use h2ai_types::events::ProposalEvent;

pub use crate::verification::SatisfactionFingerprint;

pub enum DiversityResult {
    Diverse,
    Collapsed,
}

pub struct DiversityGuard;

impl DiversityGuard {
    /// Check whether passing proposals are too similar in constraint-satisfaction space.
    ///
    /// Computes the mean pairwise Hamming distance between satisfaction fingerprints
    /// (one `bool` per constraint: `true` = hard gate passed). Returns `Collapsed` when
    /// the mean distance is below `threshold`, signalling collective hallucination.
    ///
    /// Fails open (returns `Diverse`) when fewer than 2 proposals, any fingerprint is
    /// empty, or fingerprint lengths are inconsistent (constraint corpus changed mid-run).
    #[must_use]
    pub fn check(
        passed: &[(ProposalEvent, Vec<ComplianceResult>, bool)],
        threshold: f64,
    ) -> DiversityResult {
        if passed.len() < 2 {
            return DiversityResult::Diverse;
        }

        let fingerprints: Vec<SatisfactionFingerprint> = passed
            .iter()
            .map(|(_, results, _)| {
                results
                    .iter()
                    .map(h2ai_constraints::types::ComplianceResult::hard_passes)
                    .collect()
            })
            .collect();

        let len = fingerprints[0].len();
        if len == 0 || fingerprints.iter().any(|f| f.len() != len) {
            return DiversityResult::Diverse;
        }

        let mut total = 0.0;
        let mut pairs = 0u32;
        for i in 0..fingerprints.len() {
            for j in (i + 1)..fingerprints.len() {
                total += hamming(&fingerprints[i], &fingerprints[j]);
                pairs += 1;
            }
        }

        if (total / f64::from(pairs)) < threshold {
            DiversityResult::Collapsed
        } else {
            DiversityResult::Diverse
        }
    }
}

fn hamming(a: &[bool], b: &[bool]) -> f64 {
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count() as f64 / a.len() as f64
}
