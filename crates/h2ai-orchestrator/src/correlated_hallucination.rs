use std::collections::HashSet;

fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() > 1)
        .collect()
}

fn token_jaccard_distance(a: &str, b: &str) -> f64 {
    let ta = tokenize(a);
    let tb = tokenize(b);
    if ta.is_empty() && tb.is_empty() {
        return 0.0;
    }
    let intersection = ta.intersection(&tb).count() as f64;
    let union_count = ta.union(&tb).count() as f64;
    1.0 - (intersection / union_count)
}

/// Signal returned by `compute_cv` when at least 2 proposals are provided.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrelationSignal {
    /// Coefficient of variation: std_dev / mean of pairwise Jaccard distances.
    /// Low value (near 0) = proposals are tightly clustered = correlated hallucination risk.
    pub cv: f64,
    /// Mean pairwise Jaccard distance across all proposal pairs.
    pub mean_jaccard_distance: f64,
}

/// Compute CV of pairwise Jaccard distances across `proposals`.
///
/// Returns `None` when fewer than 2 proposals are provided.
/// Returns `Some(signal)` with `cv = 0.0` when all proposals are identical
/// or only two proposals exist (single-point distribution).
pub fn compute_cv(proposals: &[&str]) -> Option<CorrelationSignal> {
    let n = proposals.len();
    if n < 2 {
        return None;
    }
    let mut distances = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            distances.push(token_jaccard_distance(proposals[i], proposals[j]));
        }
    }
    let mean = distances.iter().sum::<f64>() / distances.len() as f64;
    if mean == 0.0 {
        // All pairs identical — definite correlation signal regardless of N.
        return Some(CorrelationSignal {
            cv: 0.0,
            mean_jaccard_distance: 0.0,
        });
    }
    // With only 1 pairwise distance (N=2) and non-zero mean, CV is always 0 by definition
    // (a single-point distribution has no variance). This is statistically meaningless —
    // cv=0 cannot distinguish correlated from diverse when there's only one data point.
    // Return None so the caller falls through without a spurious warning.
    if distances.len() < 2 {
        return None;
    }
    let variance =
        distances.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / distances.len() as f64;
    let cv = variance.sqrt() / mean;
    Some(CorrelationSignal {
        cv,
        mean_jaccard_distance: mean,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_proposal_returns_none() {
        assert!(compute_cv(&["some text"]).is_none());
    }

    #[test]
    fn empty_returns_none() {
        assert!(compute_cv(&[]).is_none());
    }

    #[test]
    fn identical_proposals_return_zero_cv() {
        let s = compute_cv(&[
            "the quick brown fox",
            "the quick brown fox",
            "the quick brown fox",
        ])
        .unwrap();
        assert_eq!(s.cv, 0.0);
        assert_eq!(s.mean_jaccard_distance, 0.0);
    }

    #[test]
    fn diverse_proposals_return_high_mean_distance() {
        let proposals = &[
            "quantum entanglement photon polarization measurement",
            "sql database transaction isolation deadlock prevention",
            "rust borrow checker lifetime ownership memory safety",
            "neural network backpropagation gradient descent loss",
        ];
        let s = compute_cv(proposals).unwrap();
        assert!(
            s.mean_jaccard_distance > 0.5,
            "diverse proposals should have large distances, got {}",
            s.mean_jaccard_distance
        );
    }

    #[test]
    fn similar_proposals_return_low_mean_distance() {
        let proposals = &[
            "stateless JWT authentication token validation",
            "stateless JWT authentication bearer token",
            "stateless token based JWT authentication scheme",
        ];
        let s = compute_cv(proposals).unwrap();
        assert!(
            s.mean_jaccard_distance < 0.6,
            "similar proposals should have small distances: got {}",
            s.mean_jaccard_distance
        );
    }

    #[test]
    fn two_diverse_proposals_returns_none() {
        // N=2 with non-zero distance: single-point distribution → CV meaningless → None
        assert!(compute_cv(&["foo bar baz", "foo bar qux"]).is_none());
    }

    #[test]
    fn two_identical_proposals_returns_zero_cv() {
        // N=2 identical: mean=0 → definite correlation signal
        let s = compute_cv(&["foo bar baz", "foo bar baz"]).unwrap();
        assert_eq!(s.cv, 0.0);
        assert_eq!(s.mean_jaccard_distance, 0.0);
    }
}
