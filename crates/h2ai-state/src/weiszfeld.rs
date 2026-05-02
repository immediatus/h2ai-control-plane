//! Weiszfeld iterative geometric median for Byzantine-robust proposal selection.
//!
//! ## Algorithm
//!
//! The geometric median minimises the sum of Euclidean distances to all input
//! vectors. Weiszfeld's iterative re-weighted least squares algorithm converges
//! to it in O(1/t) per iteration (Weiszfeld 1937, Pillutla et al. 2019).
//!
//! ## Byzantine resilience
//!
//! Breakdown point: ⌊n/2⌋ − 1 (50%). Tolerates up to half the input vectors
//! being adversarially placed without shifting the geometric median beyond the
//! convex hull of the honest vectors.
//!
//! ## When to use
//!
//! - Cluster incoherent (cluster assumption violated) AND embedding model present.
//! - Embedding space is Euclidean so the Weiszfeld proof applies directly.
//! - Use [`weiszfeld_select`] in the merger fallback chain when
//!   [`crate::krum::cluster_coherent`] returns `false`.
//!
//! ## Numerical stability
//!
//! Internally operates in f64 regardless of the f32 input vectors.
//! Distances are clamped to ≥ 1e-8 to avoid division-by-zero at the exact
//! median point. Convergence is unconditional for 20 iterations at n ≤ 9.
//!
//! ## Reference
//!
//! Pillutla, V., Kakade, S. M., & Harchaoui, Z. (2019). Robust aggregation for
//! federated learning. arXiv:1912.13445.

/// Selects the proposal closest to the geometric median of the embedding vectors.
///
/// Byzantine resilience: tolerates ⌊n/2⌋−1 corrupted vectors (breakdown point 1/2).
/// Convergence: O(1/t) in T iterations; T=20 is sufficient for n≤9.
///
/// Returns the index into `embeddings` of the proposal closest (cosine distance)
/// to the geometric median. Returns 0 if `embeddings` is empty.
///
/// Reference: Pillutla et al. (2019), arXiv:1912.13445.
pub fn weiszfeld_select(embeddings: &[Vec<f32>], max_iter: usize) -> usize {
    if embeddings.is_empty() {
        return 0;
    }
    if embeddings.len() == 1 {
        return 0;
    }

    let dim = embeddings[0].len();
    if dim == 0 {
        return 0;
    }

    // Initialize median as mean of all embeddings
    let mut median: Vec<f64> = vec![0.0; dim];
    for emb in embeddings {
        for (i, &v) in emb.iter().enumerate() {
            median[i] += v as f64;
        }
    }
    let n = embeddings.len() as f64;
    for v in &mut median {
        *v /= n;
    }

    // Weiszfeld iterations
    for _ in 0..max_iter {
        // Compute L2 distances from current median to each embedding
        let dists: Vec<f64> = embeddings
            .iter()
            .map(|emb| {
                let d: f64 = emb
                    .iter()
                    .zip(median.iter())
                    .map(|(&a, &b)| (a as f64 - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                d.max(1e-8) // clamp to avoid division by zero
            })
            .collect();

        let weight_sum: f64 = dists.iter().map(|d| 1.0 / d).sum();
        if weight_sum < 1e-12 {
            break;
        }

        // Weighted mean update
        let mut new_median = vec![0.0f64; dim];
        for (emb, &dist) in embeddings.iter().zip(dists.iter()) {
            let w = 1.0 / dist;
            for (i, &v) in emb.iter().enumerate() {
                new_median[i] += w * v as f64;
            }
        }
        for v in &mut new_median {
            *v /= weight_sum;
        }
        median = new_median;
    }

    // Return index of embedding with minimum cosine distance to final median
    let median_norm: f64 = median
        .iter()
        .map(|v| v.powi(2))
        .sum::<f64>()
        .sqrt()
        .max(1e-12);

    embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| {
            let emb_norm: f64 = emb
                .iter()
                .map(|&v| (v as f64).powi(2))
                .sum::<f64>()
                .sqrt()
                .max(1e-12);
            let dot: f64 = emb
                .iter()
                .zip(median.iter())
                .map(|(&a, &b)| a as f64 * b)
                .sum();
            let cosine_sim = dot / (emb_norm * median_norm);
            let cosine_dist = 1.0 - cosine_sim;
            (i, cosine_dist)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
